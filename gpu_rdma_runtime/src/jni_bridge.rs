//! Lightweight JNI interface for the existing raw-verbs client endpoint.

use std::collections::VecDeque;
use std::net::TcpStream;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use jni::objects::{JByteArray, JByteBuffer, JClass, JIntArray, JObject, JString};
use jni::sys::{jint, jlong, jobjectArray};
use jni::JNIEnv;
use rand::Rng;
use sideway::ibverbs::device_context::Mtu;
use sideway::ibverbs::queue_pair::QueuePair;

use crate::control_helpers::{mtu_value, recv_json, send_json};
use crate::control_protocol::{BootstrapHello, ClientRole, EndpointBootstrap, InputDone, MemoryRegionInfo, ProcessingSpec, RdmaDestination, MAX_ITEM_SIZE};
use crate::constants::RING_BUFFER_ELEMENTS;
use crate::rdma::{RdmaEndpoint, RdmaReceiver, RdmaSender};
use crate::ring_buffer::slot::Slot;

struct Session {
    role: ClientRole,
    sender: Option<Mutex<RdmaSender>>,
    receiver: Option<Mutex<RdmaReceiver>>,
    control: Mutex<TcpStream>,
    pending_wrs: Mutex<VecDeque<u64>>,
    published_slots: Mutex<u64>,
    remote_input: MemoryRegionInfo,
    pending_output: Mutex<VecDeque<u32>>,
}

fn fail(env: &mut JNIEnv<'_>, message: impl std::fmt::Display) {
    let _ = env.throw_new("java/io/IOException", message.to_string());
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

fn mtu(value: u32) -> Result<Mtu> {
    Ok(match value {
        256 => Mtu::Mtu256,
        512 => Mtu::Mtu512,
        1024 => Mtu::Mtu1024,
        2048 => Mtu::Mtu2048,
        4096 => Mtu::Mtu4096,
        other => anyhow::bail!("unsupported RDMA MTU {other}"),
    })
}

fn open_session(host: String, port: u16, device: Option<String>, ib_port: u8, gid_index: u8, role: ClientRole, processing: ProcessingSpec) -> Result<Box<Session>> {
    eprintln!("[RDMA-JNI] opening role={role:?} to {host}:{port}");
    let mut stream = TcpStream::connect((host.as_str(), port))
        .with_context(|| format!("connect to RDMA bootstrap {host}:{port}"))?;
    eprintln!("[RDMA-JNI] TCP connected as {role:?}; sending role hello");
    send_json(&mut stream, &BootstrapHello { role }).context("send Java RDMA client role")?;
    let mut endpoint = RdmaEndpoint::build(device.as_deref(), ib_port)?;
    let active_mtu = endpoint.ctx.query_port(ib_port)?.active_mtu();
    let server: EndpointBootstrap = recv_json(&mut stream).context("receive GPU RDMA bootstrap")?;
    let path_mtu = active_mtu.min(mtu(server.path_mtu)?);
    let gid = endpoint.ctx.query_gid(ib_port, gid_index.into())?;
    let psn = rand::thread_rng().gen::<u32>() & 0x00ff_ffff;
    let local = EndpointBootstrap {
        dest: RdmaDestination {
            gid,
            qp_number: endpoint.qp.qp_number(),
            packet_seq_num: psn,
        },
        writable: endpoint.memory_region_info(),
        path_mtu: mtu_value(active_mtu),
        processing,
    };
    send_json(&mut stream, &local).context("send Java RDMA bootstrap")?;
    endpoint.connect(&server.dest, ib_port, psn, path_mtu, 0, gid_index)?;
    let (sender, receiver) = endpoint.split();
    Ok(Box::new(Session {
        role,
        sender: Some(Mutex::new(sender)),
        receiver: (role == ClientRole::Post).then_some(Mutex::new(receiver)),
        control: Mutex::new(stream),
        pending_wrs: Mutex::new(VecDeque::new()),
        published_slots: Mutex::new(0),
        remote_input: server.writable,
        pending_output: Mutex::new(VecDeque::new()),
    }))
}

fn require_pre(session: &Session) -> Result<&Mutex<RdmaSender>> {
    anyhow::ensure!(session.role == ClientRole::Pre, "RDMA session is not a pre client");
    session.sender.as_ref().context("RDMA session is not a pre client")
}

fn require_post(session: &Session) -> Result<&Mutex<RdmaReceiver>> {
    session.receiver.as_ref().context("RDMA session is not a post client")
}

fn session<'a>(handle: jlong) -> Result<&'a Session> {
    if handle == 0 {
        anyhow::bail!("RDMA session is closed")
    }
    Ok(unsafe { &*(handle as *const Session) })
}

#[no_mangle]
pub extern "system" fn Java_org_apache_flink_table_runtime_functions_table_externalruntime_RustRdmaNative_open(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    host: JString<'_>,
    port: jint,
    device: JString<'_>,
    ib_port: jint,
    gid_index: jint,
    role: JString<'_>,
    processing: JString<'_>,
) -> jlong {
    let result = (|| -> Result<jlong> {
        let host: String = env.get_string(&host)?.into();
        let device: String = env.get_string(&device)?.into();
        let role: String = env.get_string(&role)?.into();
        let processing_json: String = env.get_string(&processing)?.into();
        let processing: ProcessingSpec = serde_json::from_str(&processing_json).context("parse processing spec")?;
        let role = match role.to_ascii_lowercase().as_str() {
            "pre" => ClientRole::Pre,
            "post" => ClientRole::Post,
            other => anyhow::bail!("invalid RDMA client role {other}; expected pre or post"),
        };
        let session = open_session(
            host,
            port.try_into().context("invalid RDMA bootstrap port")?,
            (!device.is_empty()).then_some(device),
            ib_port.try_into().context("invalid IB port")?,
            gid_index.try_into().context("invalid GID index")?,
            role,
            processing,
        )?;
        Ok(Box::into_raw(session) as jlong)
    })();
    match result {
        Ok(handle) => handle,
        Err(error) => {
            fail(&mut env, format!("{error:#}"));
            0
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_flink_table_runtime_functions_table_externalruntime_RustRdmaNative_writeSlot(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    value: JByteArray<'_>,
) {
    let result = (|| -> Result<()> {
        let value = env.convert_byte_array(value)?;
        anyhow::ensure!(value.len() <= MAX_ITEM_SIZE, "RDMA slot exceeds {MAX_ITEM_SIZE} bytes");
        let mut payload = [0u8; MAX_ITEM_SIZE];
        payload[..value.len()].copy_from_slice(&value);
        let s = session(handle)?;
        let mut sender = require_pre(s)?.lock().unwrap();
        while sender.available_send_slots() == 0 {
            let credit: u32 = recv_json(&mut s.control.lock().unwrap()).context("receive GPU input credit")?;
            anyhow::ensure!(credit > 0 && credit as u64 <= sender.posted_slots(), "invalid GPU input credit {credit}");
            sender.complete_round_trips(credit as usize)?;
        }
        sender.write_slot_local(Slot {
            len: value.len() as u32,
            timestamp_ns: now_ns(),
            value: payload,
        })?;
        Ok(())
    })();
    if let Err(error) = result {
        fail(&mut env, error);
    }
}

/// Batched counterpart to `writeSlot`: the Java caller has already framed every
/// row in the batch into one shared direct `ByteBuffer` (fixed `MAX_ITEM_SIZE`
/// stride per row, real length given per-entry in `frame_lengths`), so this
/// takes a single JNI crossing and a zero-copy read of that buffer instead of
/// the one-crossing-plus-array-copy-per-row the row-at-a-time path pays for
/// every element of a batch.
#[no_mangle]
pub extern "system" fn Java_org_apache_flink_table_runtime_functions_table_externalruntime_RustRdmaNative_writeBatch(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    batch: JByteBuffer<'_>,
    frame_lengths: JIntArray<'_>,
    count: jint,
) {
    let result = (|| -> Result<()> {
        let count: usize = count.try_into().context("invalid batch count")?;
        let base_ptr = env.get_direct_buffer_address(&batch)?;
        let capacity = env.get_direct_buffer_capacity(&batch)?;
        let mut lengths = vec![0i32; count];
        env.get_int_array_region(&frame_lengths, 0, &mut lengths)?;

        let s = session(handle)?;
        let mut sender = require_pre(s)?.lock().unwrap();
        for (i, &frame_len) in lengths.iter().enumerate() {
            anyhow::ensure!(frame_len >= 0, "invalid negative frame length {frame_len}");
            let frame_len = frame_len as usize;
            anyhow::ensure!(frame_len <= MAX_ITEM_SIZE, "RDMA slot exceeds {MAX_ITEM_SIZE} bytes");
            let base = i * MAX_ITEM_SIZE;
            anyhow::ensure!(base + frame_len <= capacity, "batch buffer too small for frame {i}");
            // SAFETY: base_ptr/capacity come from GetDirectBufferAddress/Capacity for
            // the direct buffer the Java caller allocated and does not touch again
            // until this call returns; base+frame_len is bounds-checked above.
            let frame = unsafe { std::slice::from_raw_parts(base_ptr.add(base), frame_len) };
            let mut payload = [0u8; MAX_ITEM_SIZE];
            payload[..frame_len].copy_from_slice(frame);

            while sender.available_send_slots() == 0 {
                let credit: u32 = recv_json(&mut s.control.lock().unwrap()).context("receive GPU input credit")?;
                anyhow::ensure!(credit > 0 && credit as u64 <= sender.posted_slots(), "invalid GPU input credit {credit}");
                sender.complete_round_trips(credit as usize)?;
            }
            sender.write_slot_local(Slot {
                len: frame_len as u32,
                timestamp_ns: now_ns(),
                value: payload,
            })?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        fail(&mut env, error);
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_flink_table_runtime_functions_table_externalruntime_RustRdmaNative_publish(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
    count: jint,
) {
    let result = (|| -> Result<()> {
        let count: usize = count.try_into().context("invalid batch count")?;
        let s = session(handle)?;
        let mut sender = require_pre(s)?.lock().unwrap();
        while sender.posted_slots() + count as u64 >= RING_BUFFER_ELEMENTS as u64 {
            let credit: u32 = recv_json(&mut s.control.lock().unwrap()).context("receive GPU input credit")?;
            anyhow::ensure!(credit > 0 && credit as u64 <= sender.posted_slots(), "invalid GPU input credit {credit}");
            sender.complete_round_trips(credit as usize)?;
        }
        let wr_id = sender.write_slots_remote(&s.remote_input, count)?;
        *s.published_slots.lock().unwrap() += count as u64;
        let mut pending_wrs = s.pending_wrs.lock().unwrap();
        pending_wrs.push_back(wr_id);
        if pending_wrs.len() >= 256 {
            let oldest = pending_wrs.pop_front().context("missing pending send")?;
            sender.wait_for_completion(oldest)?;
        }
        Ok(())
    })();
    if let Err(error) = result {
        fail(&mut env, error);
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_flink_table_runtime_functions_table_externalruntime_RustRdmaNative_receive(
    mut env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) -> jobjectArray {
    let result = (|| -> Result<jobjectArray> {
        let s = session(handle)?;
        let count = {
            let mut pending = s.pending_output.lock().unwrap();
            if let Some(count) = pending.pop_front() {
                count
            } else {
                let mut receiver = require_post(s)?.lock().unwrap();
                loop {
                    let batches = receiver.poll_output_batches()?;
                    if let Some(count) = batches.first() {
                        pending.extend(batches.iter().copied().skip(1));
                        break *count;
                    }
                    std::hint::spin_loop();
                }
            }
        } as usize;
        let class = env.find_class("[B")?;
        let output = env.new_object_array(count as i32, class, JObject::null())?;
        let mut receiver = require_post(s)?.lock().unwrap();
        for index in 0..count {
            let slot = receiver.read_slot_local().context("completed RDMA slot missing")?;
            let length = (slot.len as usize).min(MAX_ITEM_SIZE);
            let bytes = env.byte_array_from_slice(&slot.value[..length])?;
            env.set_object_array_element(&output, index as i32, bytes)?;
        }
        if s.role == ClientRole::Post {
            s.sender
                .as_ref()
                .context("post client notification QP is unavailable")?
                .lock()
                .unwrap()
                .replenish_output_notifications(1)?;
        }
        send_json(&mut s.control.lock().unwrap(), &(count as u32)).context("send post-client output credit")?;
        Ok(output.into_raw())
    })();
    match result {
        Ok(array) => array,
        Err(error) => {
            fail(&mut env, error);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_apache_flink_table_runtime_functions_table_externalruntime_RustRdmaNative_close(
    _env: JNIEnv<'_>,
    _class: JClass<'_>,
    handle: jlong,
) {
    if handle != 0 {
        unsafe {
            let session = Box::from_raw(handle as *mut Session);
            if session.role == ClientRole::Pre {
                let result = (|| -> Result<()> {
                    let mut sender = require_pre(&session)?.lock().unwrap();
                    let slots = *session.published_slots.lock().unwrap();
                    {
                        let mut pending = session.pending_wrs.lock().unwrap();
                        while let Some(wr_id) = pending.pop_front() {
                            sender.wait_for_completion(wr_id)?;
                        }
                    }
                    send_json(&mut session.control.lock().unwrap(), &InputDone { done: true, slots })
                        .context("send PRE input-done message")?;
                    while sender.posted_slots() > 0 {
                        let credit: u32 = recv_json(&mut session.control.lock().unwrap())
                            .context("receive final GPU input credit")?;
                        anyhow::ensure!(
                            credit > 0 && credit as u64 <= sender.posted_slots(),
                            "invalid final GPU input credit {credit}"
                        );
                        sender.complete_round_trips(credit as usize)?;
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    eprintln!("[RDMA-JNI] failed to drain PRE session on close: {error:#}");
                }
            }
            drop(session);
        }
    }
}
