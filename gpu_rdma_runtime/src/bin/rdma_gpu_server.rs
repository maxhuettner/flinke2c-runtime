mod linux_impl {
    use std::io::{BufRead, BufReader, Write};
    use std::mem::size_of;
    use std::net::{Ipv6Addr, SocketAddr, TcpListener};
    use std::thread;
    use std::time::Duration;

    use anyhow::{Context, Result};
    use clap::Parser;

    use gpu_rdma_runtime::control_helpers::{recv_json, send_json, CliMtu};
    use gpu_rdma_runtime::control_protocol::{EndpointBootstrap, RdmaDestination, Slot};
    use gpu_rdma_runtime::rdma::{self, RdmaEndpoint};
    use sideway::ibverbs::device_context::Mtu;
    use sideway::ibverbs::queue_pair::QueuePair;

    #[derive(Parser, Debug)]
    #[command(name = "rdma-gpu-server")]
    #[command(about = "RDMA GPU server exposing persistent-kernel buffers to a CPU client")]
    struct Args {
        #[clap(long, short = 'p', default_value_t = 50001)]
        port: u16,
        #[arg(long)]
        ib_device: Option<String>,
        #[arg(long, default_value_t = 1)]
        ib_port: u8,
        #[arg(long, short = 'g', default_value_t = 0)]
        gid_index: u8,
        #[arg(long, short = 'm', default_value_t = CliMtu(Mtu::Mtu1024))]
        mtu: CliMtu,
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();

        let mut rdma_endpoint = RdmaEndpoint::build(args.ib_device.as_deref(), args.ib_port)?;
        let port_attr = rdma_endpoint.ctx.query_port(args.ib_port)?;
        let active_mtu = port_attr.active_mtu();

        let gid = rdma_endpoint.ctx.query_gid(args.ib_port, args.gid_index.into())?;
        let pckt_seq_num = rand::random::<u32>() & 0xffffff;

        let addr = SocketAddr::from((Ipv6Addr::UNSPECIFIED, args.port));
        let listener = TcpListener::bind(addr).with_context(|| format!("bind {}", addr))?;
        let (mut stream, _) = listener.accept()?;
        println!("bound bootstrap listener on {}", addr);

        let local = EndpointBootstrap {
            dest: RdmaDestination {
                gid,
                qp_number: rdma_endpoint.qp.qp_number(),
                packet_seq_num: pckt_seq_num,
            },
            writable: rdma_endpoint.memory_region_info(),
        };

        println!("server bootstrap local: {:?}", local);
        println!("server sending bootstrap");
        send_json(&mut stream, &local)?;
        println!("server waiting for client bootstrap");
        let remote: EndpointBootstrap = recv_json(&mut stream)?;
        println!("server received remote bootstrap: {:?}", remote);

        rdma_endpoint.connect(&remote.dest, args.ib_port, pckt_seq_num, active_mtu, 0, args.gid_index)?;
        println!("server QP connected");

        println!("server waiting for client RDMA write");

        let mut slot = loop {
            let v = rdma_endpoint.read_slot_local(0);
            if v.seq == 1 {
                break v;
            }
            std::hint::spin_loop();
        };

        println!("server received value: {slot}");

        slot.seq += 1;
        let msg = b"Hello from server";
        slot.len = msg.len() as u32;
        slot.value[..msg.len()].copy_from_slice(msg);
        rdma_endpoint.write_slot_local(0, slot);
        println!("server posting RDMA write back to client");
        let write_id = 0;
        rdma_endpoint.write_slot_remote(&remote.writable, write_id, 0)?;
        rdma_endpoint.wait_for_completion(write_id)?;
        println!("server RDMA write completion received");

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    linux_impl::run()
}
