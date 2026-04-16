mod linux_impl {

    use std::net::TcpStream;

    use std::time::Instant;

    use anyhow::{Context, Result};
    use clap::Parser;
    use gpu_rdma_runtime::control_helpers::{recv_json, send_json, CliMtu};
    use gpu_rdma_runtime::control_protocol::{EndpointBootstrap, RdmaDestination, MAX_ITEM_SIZE};
    use gpu_rdma_runtime::rdma::RdmaEndpoint;
    use gpu_rdma_runtime::ring_buffer::slot::Slot;
    use sideway::ibverbs::device_context::Mtu;
    use sideway::ibverbs::queue_pair::QueuePair;

    #[derive(Parser, Debug)]
    #[command(name = "rdma-test-client")]
    #[command(about = "CPU RDMA test client writing requests directly into remote GPU memory")]
    struct Args {
        #[arg(long)]
        server: String,
        #[arg(long)]
        ib_device: Option<String>,
        #[arg(long, default_value_t = 1)]
        ib_port: u8,
        #[arg(long, short = 's', default_value_t = 1024)]
        size: u32,
        #[arg(long, short = 'g', default_value_t = 0)]
        gid_index: u8,
        #[arg(long, short = 'm', default_value_t = CliMtu(Mtu::Mtu1024))]
        mtu: CliMtu,
    }

    pub fn run() -> Result<()> {
        let args = Args::parse();
        let mut stream = TcpStream::connect(&args.server).with_context(|| format!("connect {}", args.server))?;

        let mut rdma_endpoint = RdmaEndpoint::build(args.ib_device.as_deref(), args.ib_port)?;
        let port_attr = rdma_endpoint.ctx.query_port(args.ib_port)?;
        let active_mtu = port_attr.active_mtu();

        println!("client waiting for server bootstrap");
        let server: EndpointBootstrap = recv_json(&mut stream)?;
        println!("client received server bootstrap: {:?}", server);

        let gid = rdma_endpoint.ctx.query_gid(args.ib_port, args.gid_index.into())?;
        let pckt_seq_num = rand::random::<u32>() & 0x00ff_ffff;

        let local = EndpointBootstrap {
            dest: RdmaDestination {
                gid,
                qp_number: rdma_endpoint.qp.qp_number(),
                packet_seq_num: pckt_seq_num,
            },
            writable: rdma_endpoint.memory_region_info(),
        };

        println!("client sending bootstrap: {:?}", local);
        send_json(&mut stream, &local)?;

        rdma_endpoint.connect(&server.dest, args.ib_port, pckt_seq_num, active_mtu, 0, args.gid_index)?;
        println!("client QP connected");

        let msg = b"Hello from THE client";
        let mut value = [0u8; MAX_ITEM_SIZE];
        value[..msg.len()].copy_from_slice(msg);

        let slot = Slot {
            len: msg.len() as u32,
            value,
        };

        let start_time = Instant::now();
        rdma_endpoint.write_slot_local(slot)?;
        println!("client posting RDMA write");
        let wr_id = rdma_endpoint.write_slot_remote(&server.writable)?;
        rdma_endpoint.wait_for_completion(wr_id)?;
        rdma_endpoint.complete_sent_slot_local();
        println!("client RDMA write completed");

        let result = loop {
            match rdma_endpoint.read_slot_local() {
                Some(slot) => break slot,
                None => std::hint::spin_loop(),
            }
        };

        let elapsed = start_time.elapsed();
        println!("round-trip time: {:.2?}", elapsed);

        println!("result from server: {result}");

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    linux_impl::run()
}
