use anyhow::{bail, Context, Result};
use clap::Parser;
use std::collections::{BTreeMap, HashSet};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::thread;

use crate::java_udf::{InputColumn, JavaArg, UdfHandle};

mod java_udf;

const DEFAULT_MAX_FRAME_SIZE: usize = 64 * 1024 * 1024; // 64 MiB hard safety cap

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    listen_host: String,
    #[arg(long)]
    in_port: u16,

    #[arg(long, default_value = "262144")]
    buf_size: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = num_cpus::get())]
    workers: usize,
    #[arg(long, default_value_t = 0)]
    max_in_flight: usize,

    #[arg(long, default_value_t = DEFAULT_MAX_FRAME_SIZE)]
    max_frame_size: usize,

    #[arg(long, value_delimiter = ',', default_value = "jar/flinke2c.jar")]
    udf_jars: Vec<PathBuf>,
    #[arg(long, default_value = "org.example.proxy.ScalarFunctionAdapter")]
    udf_adapter_class: String,
    #[arg(long, default_value = "evalBatch")]
    udf_method: String,
    #[arg(long, default_value = "([[Ljava/lang/String;)V")]
    udf_sig: String,

    #[arg(long, default_value_t = 0)]
    debug_sample_rows: usize,
    #[arg(long, default_value_t = 1)]
    debug_sample_batches: usize,
}

#[derive(Debug, serde::Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FunctionArg {
    name: String,
    #[serde(rename = "type", alias = "wireType")]
    arg_type: Option<String>,
}

#[derive(Debug, serde::Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FunctionResult {
    output_name: String,
    #[serde(default, rename = "outputType", alias = "outputWireType", alias = "wireType")]
    output_type: Option<String>,
}

#[derive(Debug, serde::Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FieldSpec {
    name: String,
    #[serde(default, rename = "wireType", alias = "type")]
    field_type: Option<String>,
}

#[derive(Debug, serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ConfigMessage {
    role: String,
    #[serde(default)]
    function_class: Option<String>,
    #[serde(default)]
    function_kind: Option<String>,
    #[serde(default)]
    function_args: Vec<FunctionArg>,
    #[serde(default)]
    function_results: Vec<FunctionResult>,
    #[serde(default)]
    reorder_responses: bool,
    #[serde(default)]
    pre_fields: Vec<FieldSpec>,
    #[serde(default)]
    post_fields: Vec<FieldSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FieldType {
    String,
    Boolean,
    Int64,
    Int32,
    Int16,
    Int8,
    Float64,
    Float32,
    Decimal { precision: Option<u8>, scale: i8 },
    DecimalUnscaledI64,
    DecimalUnscaledBytes,
    TimestampMillis, // on wire: int64 millis
    Date,            // on wire: int32 days (Flink), internally UDF uses millis
    Bytes,
    Unknown(String),
}

impl FieldType {
    fn decimal_scale(&self) -> Option<i8> {
        match self {
            FieldType::Decimal { scale, .. } => Some(*scale),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
struct SessionConfig {
    reorder_responses: bool,
    expected_input_len: usize,
    output_row_len: usize,
    post_field_sources: Vec<PostFieldSource>,
    passthrough_identity: bool,

    arg_positions: Vec<usize>,
    arg_names: Vec<String>,
    arg_types: Vec<FieldType>,

    output_positions: Vec<usize>,
    output_names: Vec<String>,
    output_types: Vec<FieldType>,

    // payload fields (excluding __op/__rowId), in config order, mapped to row positions
    pre_payload_positions: Vec<usize>,
    pre_payload_types: Vec<FieldType>,
    post_payload_positions: Vec<usize>,
    post_payload_types: Vec<FieldType>,
    /// Direct columnar source for each post-payload field (parallel to post_payload_positions/types).
    post_payload_sources: Vec<PayloadSource>,
}

#[derive(Clone, Debug)]
enum PostFieldSourceKind {
    Op,
    RowId,
    InputPos(usize),
    Output,
}

#[derive(Clone, Debug)]
struct PostFieldSource {
    pos: usize,
    kind: PostFieldSourceKind,
}

/// Maps each post-payload slot to its data source for zero-alloc columnar encode.
#[derive(Clone, Debug)]
enum PayloadSource {
    /// Value comes from the input row at the given position.
    InputAt(usize),
    /// Value comes from the UDF output column at the given index.
    OutputAt(usize),
}

#[derive(Clone, Debug)]
enum OutputSlotKind {
    Op,
    RowId,
    InputPos(usize),
    Output,
    Nil,
}

#[derive(Clone, Debug)]
enum V {
    Null,
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
    DecimalI128(i128), // unscaled
}

#[derive(Debug)]
pub struct TcpSessionConfig<'a> {
    pre: TcpStream,
    pre_cfg: ConfigMessage,
    post: TcpStream,
    post_cfg: ConfigMessage,
    udf: &'a mut Option<UdfHandle>,
    current_udf_class: &'a mut Option<String>,
    current_udf_types: &'a mut Option<Vec<String>>,
    args: &'a Args,
}

#[derive(Debug)]
struct UdfConfig<'a, W: Write> {
    writer: &'a mut W,
    rows: &'a [Vec<V>],
    session: &'a SessionConfig,
    udf: &'a mut UdfHandle,
    method: &'a str,
    debug_sample_rows: usize,
    debug_batches_remaining: &'a mut usize,
    max_frame_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let listener = TcpListener::bind((args.listen_host.as_str(), args.in_port)).context("bind in-port")?;
    let mut current_udf_class: Option<String> = None;
    let mut current_udf_types: Option<Vec<String>> = None;
    let mut udf: Option<UdfHandle> = None;

    println!("Waiting for PRE/POST on {}:{} ...", args.listen_host, args.in_port);

    loop {
        let session = match accept_pair(&listener) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("Failed to accept session: {err:#}");
                continue;
            }
        };

        let (pre, pre_cfg, post, post_cfg) = session;
        if let Err(err) = run_session(
            TcpSessionConfig {
                pre,
                pre_cfg,
                post,
                post_cfg,
                udf: &mut udf,
                current_udf_class: &mut current_udf_class,
                current_udf_types: &mut current_udf_types,
                args: &args,
            }
        ) {
            eprintln!("Session ended with error: {err:#}");
        }
    }
}

fn read_config(stream: &mut TcpStream) -> Result<ConfigMessage> {
    let len = read_i32_be_stream(stream).context("read config length")? as i64;
    if len <= 0 {
        bail!("empty/invalid config message length: {}", len);
    }
    let len = len as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).context("read config")?;
    let cfg: ConfigMessage = serde_json::from_slice(&buf).context("parse config json")?;
    println!("Received config: {:?}", cfg);
    Ok(cfg)
}

fn accept_pair(listener: &TcpListener) -> Result<(TcpStream, ConfigMessage, TcpStream, ConfigMessage)> {
    let mut pre: Option<(TcpStream, ConfigMessage)> = None;
    let mut post: Option<(TcpStream, ConfigMessage)> = None;

    while pre.is_none() || post.is_none() {
        let (mut stream, addr) = listener.accept().context("accept connection")?;
        let cfg = read_config(&mut stream).context("read config")?;
        match cfg.role.as_str() {
            "pre" => {
                if pre.is_some() {
                    bail!("duplicate PRE connection from {}", addr);
                }
                println!("PRE connected from {}", addr);
                pre = Some((stream, cfg));
            }
            "post" => {
                if post.is_some() {
                    bail!("duplicate POST connection from {}", addr);
                }
                println!("POST connected from {}", addr);
                post = Some((stream, cfg));
            }
            other => bail!("unknown role {}", other),
        }
    }

    let (pre_stream, pre_cfg) = pre.expect("pre connection");
    let (post_stream, post_cfg) = post.expect("post connection");
    Ok((pre_stream, pre_cfg, post_stream, post_cfg))
}

fn run_session(
    config: TcpSessionConfig,
) -> Result<()> {
    let pre_fn = config.pre_cfg.function_class.as_deref().unwrap_or("<none>");
    let post_fn = config.post_cfg.function_class.as_deref().unwrap_or("<none>");
    let pre_kind = config.pre_cfg.function_kind.as_deref().unwrap_or("<none>");
    let pre_cfg = &config.pre_cfg;
    let post_kind = config.post_cfg.function_kind.as_deref().unwrap_or("<none>");
    let post_cfg = &config.post_cfg;
    let pre = config.pre;
    let post = config.post;
    println!(
        "PRE config: functionKind={}, functionClass={}, functionArgs={}, functionResults={}",
        pre_kind,
        pre_fn,
        pre_cfg.function_args.len(),
        pre_cfg.function_results.len()
    );
    println!(
        "POST config: functionKind={}, functionClass={}, functionArgs={}, functionResults={}",
        post_kind,
        post_fn,
        post_cfg.function_args.len(),
        post_cfg.function_results.len()
    );

    let session_cfg = build_session_config(pre_cfg)?;
    println!(
        "Resolved {} UDF args at positions {:?}",
        session_cfg.arg_positions.len(),
        session_cfg.arg_positions
    );
    if !session_cfg.output_positions.is_empty() {
        println!(
            "Resolved {} UDF results at positions {:?}",
            session_cfg.output_positions.len(),
            session_cfg.output_positions
        );
    }

    let desired_udf_class = pre_cfg
        .function_class
        .as_deref()
        .context("functionClass missing from PRE config")?
        .to_string();
    let desired_udf_types: Vec<String> = pre_cfg
        .function_args
        .iter()
        .map(|arg| arg.arg_type.clone().unwrap_or_else(|| "<missing>".to_string()))
        .collect();

    let class_changed = config.current_udf_class
        .as_deref()
        .map(|current| current != desired_udf_class)
        .unwrap_or(true);
    let types_changed = config.current_udf_types
        .as_ref()
        .map(|current| current != &desired_udf_types)
        .unwrap_or(true);

    if class_changed || types_changed {
        let current_class_display = config.current_udf_class.as_deref().unwrap_or("<unset>");
        let current_types_display = config.current_udf_types
            .as_ref()
            .map(|v| format!("{v:?}"))
            .unwrap_or_else(|| "<unset>".to_string());
        println!(
            "Switching UDF config: class {} -> {}, types {} -> {:?}",
            current_class_display, desired_udf_class, current_types_display, desired_udf_types
        );
        *config.current_udf_class = Some(desired_udf_class);
        *config.current_udf_types = Some(desired_udf_types);
    }

    let worker_count = config.args.workers.max(1);

    // ------------------------------------------------------------------
    // Single-thread mode
    // ------------------------------------------------------------------
    if worker_count == 1 {
        if class_changed || types_changed || config.udf.is_none() {
            *config.udf = Some(UdfHandle::new_with_args(
                &config.args.udf_jars,
                &config.args.udf_adapter_class,
                "(Ljava/lang/String;[Ljava/lang/String;)V",
                &[
                    JavaArg::String(config.current_udf_class.as_ref().context("missing UDF class")?.clone()),
                    JavaArg::StringArray(config.current_udf_types.as_ref().context("missing UDF types")?.clone()),
                ],
            )?);
        }

        let udf_handle = config.udf.as_mut().context("UDF handle not initialized")?;
        if udf_handle.reload_if_changed()? {
            println!("Reloaded UDF classes after jar change");
        }

        let mut debug_batches_remaining = config.args.debug_sample_batches;
        let mut reader = BufReader::with_capacity(config.args.buf_size, pre);
        let mut writer = BufWriter::with_capacity(config.args.buf_size, post);

        let batch_size = config.args.batch_size.max(1);
        let mut batch_rows: Vec<Vec<V>> = Vec::with_capacity(batch_size);

        let mut scratch = Vec::<u8>::new();

        while let Some(row) = read_framed_row(&mut reader, &session_cfg, config.args.max_frame_size, &mut scratch)? {
            validate_row_len(&row, session_cfg.expected_input_len)?;
            batch_rows.push(row);
            if batch_rows.len() >= batch_size {
                let mut udf_config = UdfConfig {
                    writer: &mut writer,
                    rows: &batch_rows,
                    session: &session_cfg,
                    udf: udf_handle,
                    method: &config.args.udf_method,
                    debug_sample_rows: config.args.debug_sample_rows,
                    debug_batches_remaining: &mut debug_batches_remaining,
                    max_frame_size: config.args.max_frame_size,
                };
                apply_udf_to_rows_stream(&mut udf_config)?;
                writer.flush().ok();
                batch_rows.clear();
            }
        }

        if !batch_rows.is_empty() {
            let mut udf_config = UdfConfig {
                writer: &mut writer,
                rows: &batch_rows,
                session: &session_cfg,
                udf: udf_handle,
                method: &config.args.udf_method,
                debug_sample_rows: config.args.debug_sample_rows,
                debug_batches_remaining: &mut debug_batches_remaining,
                max_frame_size: config.args.max_frame_size,
            };
            apply_udf_to_rows_stream(&mut udf_config)?;
            writer.flush().ok();
        }

        writer.flush().ok();
        return Ok(());
    }

    // ------------------------------------------------------------------
    // Parallel mode
    // ------------------------------------------------------------------
    if config.args.debug_sample_rows > 0 {
        eprintln!("Debug sampling disabled in parallel mode");
    }

    let max_in_flight = if config.args.max_in_flight == 0 {
        worker_count * 2
    } else {
        config.args.max_in_flight
    };

    let (result_tx, result_rx) = mpsc::channel::<WorkResult>();
    let mut senders = Vec::with_capacity(worker_count);
    let mut worker_handles = Vec::with_capacity(worker_count);
    let inflight = Arc::new((std::sync::Mutex::new(0usize), std::sync::Condvar::new()));

    for _ in 0..worker_count {
        let (tx, rx) = mpsc::channel::<Option<WorkItem>>();
        senders.push(tx);

        let result_tx = result_tx.clone();
        let session_cfg = session_cfg.clone();
        let udf_jars = config.args.udf_jars.clone();
        let udf_adapter = config.args.udf_adapter_class.clone();
        let udf_method = config.args.udf_method.clone();
        let udf_class = config.current_udf_class.as_ref().context("missing UDF class")?.clone();
        let udf_types = config.current_udf_types.as_ref().context("missing UDF types")?.clone();

        let handle = thread::spawn(move || {
            let udf_handle = UdfHandle::new_with_args(
                &udf_jars,
                &udf_adapter,
                "(Ljava/lang/String;[Ljava/lang/String;)V",
                &[JavaArg::String(udf_class), JavaArg::StringArray(udf_types)],
            );
            let mut udf_handle = match udf_handle {
                Ok(handle) => handle,
                Err(err) => {
                    let _ = result_tx.send(WorkResult {
                        seq: 0,
                        result: Err(err),
                    });
                    return;
                }
            };

            let mut debug_batches_remaining = 0usize;
            for msg in rx {
                let Some(work) = msg else { break };
                let result = apply_udf_to_rows(
                    &work.rows,
                    &session_cfg,
                    &mut udf_handle,
                    &udf_method,
                    0,
                    &mut debug_batches_remaining,
                );
                if result_tx.send(WorkResult { seq: work.seq, result }).is_err() {
                    break;
                }
            }
        });
        worker_handles.push(handle);
    }
    drop(result_tx);

    let inflight_writer = Arc::clone(&inflight);
    let writer_buf_size = config.args.buf_size;
    let writer_session = session_cfg.clone();
    let writer_max_frame = config.args.max_frame_size;
    let writer_handle = thread::spawn(move || -> Result<()> {
        let mut writer = BufWriter::with_capacity(writer_buf_size, post);
        let mut pending: BTreeMap<usize, Vec<Vec<V>>> = BTreeMap::new();
        let mut next_seq = 0usize;
        let mut payload_buf = Vec::with_capacity(256);

        while let Ok(work) = result_rx.recv() {
            let WorkResult { seq, result } = work;
            let rows = result?;
            if seq == next_seq {
                write_framed_rows(&mut writer, &rows, &writer_session, writer_max_frame, &mut payload_buf)?;
                writer.flush().ok();
                next_seq += 1;
                while let Some(next_rows) = pending.remove(&next_seq) {
                    write_framed_rows(&mut writer, &next_rows, &writer_session, writer_max_frame, &mut payload_buf)?;
                    writer.flush().ok();
                    next_seq += 1;
                }
            } else {
                pending.insert(seq, rows);
            }

            let (lock, cvar) = &*inflight_writer;
            let mut count = lock.lock().expect("lock inflight");
            *count = count.saturating_sub(1);
            cvar.notify_one();
        }

        if !pending.is_empty() {
            bail!("writer ended with {} pending batches", pending.len());
        }

        writer.flush().ok();
        Ok(())
    });

    let mut dispatched = 0usize;
    let mut send_index = 0usize;
    let mut reader = BufReader::with_capacity(config.args.buf_size, pre);

    let batch_size = config.args.batch_size.max(1);
    let mut batch_rows: Vec<Vec<V>> = Vec::with_capacity(batch_size);
    let mut scratch = Vec::<u8>::new();

    loop {
        match read_framed_row(&mut reader, &session_cfg, config.args.max_frame_size, &mut scratch)? {
            Some(row) => {
                validate_row_len(&row, session_cfg.expected_input_len)?;
                batch_rows.push(row);
                if batch_rows.len() < batch_size {
                    continue;
                }
            }
            None => {
                if batch_rows.is_empty() {
                    break;
                }
            }
        }

        let rows = std::mem::take(&mut batch_rows);
        let (lock, cvar) = &*inflight;
        let mut count = lock.lock().expect("lock inflight");
        while *count >= max_in_flight {
            count = cvar.wait(count).expect("wait inflight");
        }
        *count += 1;
        drop(count);

        let sender = &senders[send_index % senders.len()];
        sender
            .send(Some(WorkItem { seq: dispatched, rows }))
            .context("dispatch batch to worker")?;
        dispatched += 1;
        send_index += 1;
    }

    for sender in &senders {
        let _ = sender.send(None);
    }
    drop(senders);

    for handle in worker_handles {
        let _ = handle.join();
    }

    writer_handle.join().expect("writer thread panicked")?;
    Ok(())
}

fn build_session_config(pre_cfg: &ConfigMessage) -> Result<SessionConfig> {
    let reorder_responses = pre_cfg.reorder_responses;

    if pre_cfg.pre_fields.is_empty() {
        bail!("preFields missing from PRE config");
    }
    if pre_cfg.post_fields.is_empty() {
        bail!("postFields missing from PRE config");
    }
    let base_offset = 1 + if reorder_responses { 1 } else { 0 };

    let mut pre_name_to_pos = std::collections::HashMap::new();
    let mut pre_payload_positions = Vec::new();
    let mut pre_payload_types = Vec::new();
    let mut pre_payload_idx = 0usize;

    for field in &pre_cfg.pre_fields {
        match field.name.as_str() {
            "__op" => {
                if pre_name_to_pos.insert(field.name.clone(), 0).is_some() {
                    bail!("duplicate preField name {}", field.name);
                }
            }
            "__rowId" => {
                if !reorder_responses {
                    bail!("preFields contains __rowId but reorderResponses=false");
                }
                if pre_name_to_pos.insert(field.name.clone(), 1).is_some() {
                    bail!("duplicate preField name {}", field.name);
                }
            }
            _ => {
                let pos = base_offset + pre_payload_idx;
                pre_payload_idx += 1;
                if pre_name_to_pos.insert(field.name.clone(), pos).is_some() {
                    bail!("duplicate preField name {}", field.name);
                }
                let ft_str = field
                    .field_type
                    .as_deref()
                    .with_context(|| format!("preField wireType missing for {}", field.name))?;
                pre_payload_positions.push(pos);
                pre_payload_types.push(parse_field_type(ft_str));
            }
        }
    }

    let mut post_field_positions = Vec::with_capacity(pre_cfg.post_fields.len());
    let mut post_payload_positions = Vec::new();
    let mut post_payload_types = Vec::new();
    let mut post_payload_idx = 0usize;

    for field in &pre_cfg.post_fields {
        let pos = match field.name.as_str() {
            "__op" => 0,
            "__rowId" => {
                if !reorder_responses {
                    bail!("postFields contains __rowId but reorderResponses=false");
                }
                1
            }
            _ => {
                let pos = base_offset + post_payload_idx;
                post_payload_idx += 1;
                let ft_str = field
                    .field_type
                    .as_deref()
                    .with_context(|| format!("postField wireType missing for {}", field.name))?;
                post_payload_positions.push(pos);
                post_payload_types.push(parse_field_type(ft_str));
                pos
            }
        };
        post_field_positions.push(pos);
    }

    let expected_input_len = base_offset + pre_payload_idx;
    let output_row_len = base_offset + post_payload_idx;

    let mut arg_positions = Vec::with_capacity(pre_cfg.function_args.len());
    let mut arg_names = Vec::with_capacity(pre_cfg.function_args.len());
    let mut arg_types = Vec::with_capacity(pre_cfg.function_args.len());
    for arg in &pre_cfg.function_args {
        let pos = pre_name_to_pos
            .get(&arg.name)
            .copied()
            .with_context(|| format!("functionArg {} not found in preFields", arg.name))?;
        arg_positions.push(pos);
        arg_names.push(arg.name.clone());
        let arg_type = arg
            .arg_type
            .as_deref()
            .with_context(|| format!("functionArg type missing for {}", arg.name))?;
        arg_types.push(parse_field_type(arg_type));
    }

    let mut post_name_to_pos = std::collections::HashMap::new();
    for (idx, field) in pre_cfg.post_fields.iter().enumerate() {
        let pos = post_field_positions[idx];
        if post_name_to_pos.insert(field.name.clone(), pos).is_some() {
            bail!("duplicate postField name {}", field.name);
        }
    }

    let mut output_positions = Vec::with_capacity(pre_cfg.function_results.len());
    let mut output_names = Vec::with_capacity(pre_cfg.function_results.len());
    let mut output_types = Vec::with_capacity(pre_cfg.function_results.len());
    for result in &pre_cfg.function_results {
        let pos = post_name_to_pos
            .get(&result.output_name)
            .copied()
            .with_context(|| format!("functionResult {} not found in postFields", result.output_name))?;
        output_positions.push(pos);
        output_names.push(result.output_name.clone());
        let output_type = result
            .output_type
            .as_deref()
            .map(parse_field_type)
            .with_context(|| format!("functionResult outputType/wireType missing for {}", result.output_name))?;
        output_types.push(output_type);
    }

    let output_name_set: HashSet<&str> = output_names.iter().map(|s| s.as_str()).collect();
    let mut post_field_sources = Vec::with_capacity(pre_cfg.post_fields.len());
    let mut output_slots = vec![OutputSlotKind::Nil; output_row_len];

    if output_row_len > 0 {
        output_slots[0] = OutputSlotKind::Op;
    }
    if reorder_responses && output_row_len > 1 {
        output_slots[1] = OutputSlotKind::RowId;
    }

    let mut output_pos_to_idx = vec![None; output_row_len];
    for (idx, pos) in output_positions.iter().enumerate() {
        if *pos >= output_row_len {
            bail!("functionResult outputIndex {} is out of range", pos);
        }
        if output_pos_to_idx[*pos].replace(idx).is_some() {
            bail!("duplicate functionResult target at position {}", pos);
        }
    }

    let mut passthrough_identity = true;
    for (idx, field) in pre_cfg.post_fields.iter().enumerate() {
        let pos = post_field_positions[idx];
        let (kind, slot) = if field.name == "__op" {
            if pos != 0 {
                passthrough_identity = false;
            }
            (PostFieldSourceKind::Op, OutputSlotKind::Op)
        } else if field.name == "__rowId" {
            if !reorder_responses || pos != 1 {
                passthrough_identity = false;
            }
            (PostFieldSourceKind::RowId, OutputSlotKind::RowId)
        } else if let Some(pre_pos) = pre_name_to_pos.get(&field.name).copied() {
            if pre_pos != pos {
                passthrough_identity = false;
            }
            (
                PostFieldSourceKind::InputPos(pre_pos),
                OutputSlotKind::InputPos(pre_pos),
            )
        } else if output_name_set.contains(field.name.as_str()) {
            passthrough_identity = false;
            output_pos_to_idx[pos]
                .with_context(|| format!("missing output column for postField {}", field.name))?;
            (PostFieldSourceKind::Output, OutputSlotKind::Output)
        } else {
            bail!("postField {} not found in preFields", field.name);
        };
        post_field_sources.push(PostFieldSource { pos, kind });
        output_slots[pos] = slot;
    }

    if expected_input_len != output_row_len {
        passthrough_identity = false;
    }

    for (pos, slot) in output_slots.iter().enumerate() {
        match slot {
            OutputSlotKind::Nil => {
                bail!("output position {} not mapped by postFields", pos);
            }
            OutputSlotKind::InputPos(pre_pos) => {
                if *pre_pos != pos {
                    passthrough_identity = false;
                }
            }
            OutputSlotKind::Op => {
                if pos != 0 {
                    passthrough_identity = false;
                }
            }
            OutputSlotKind::RowId => {
                if !reorder_responses || pos != 1 {
                    passthrough_identity = false;
                }
            }
            OutputSlotKind::Output => {}
        }
    }

    // Build per-payload-slot source map for direct columnar encode.
    let mut output_name_to_idx = std::collections::HashMap::new();
    for (i, name) in output_names.iter().enumerate() {
        output_name_to_idx.insert(name.as_str(), i);
    }
    let mut post_payload_sources = Vec::with_capacity(post_payload_positions.len());
    for field in &pre_cfg.post_fields {
        if field.name == "__op" || field.name == "__rowId" {
            continue;
        }
        if let Some(&out_idx) = output_name_to_idx.get(field.name.as_str()) {
            post_payload_sources.push(PayloadSource::OutputAt(out_idx));
        } else if let Some(&pre_pos) = pre_name_to_pos.get(&field.name) {
            post_payload_sources.push(PayloadSource::InputAt(pre_pos));
        } else {
            bail!("postField {} has no source", field.name);
        }
    }

    Ok(SessionConfig {
        reorder_responses,
        expected_input_len,
        output_row_len,
        post_field_sources,
        passthrough_identity,
        arg_positions,
        arg_names,
        arg_types,
        output_positions,
        output_names,
        output_types,
        pre_payload_positions,
        pre_payload_types,
        post_payload_positions,
        post_payload_types,
        post_payload_sources,
    })
}

// -----------------------------------------------------------------------------
// Work structs
// -----------------------------------------------------------------------------

struct WorkItem {
    seq: usize,
    rows: Vec<Vec<V>>,
}

struct WorkResult {
    seq: usize,
    result: Result<Vec<Vec<V>>>,
}

// -----------------------------------------------------------------------------
// Binary framing + codec
// -----------------------------------------------------------------------------

fn read_framed_row<R: Read>(
    reader: &mut R,
    session: &SessionConfig,
    max_frame_size: usize,
    scratch: &mut Vec<u8>,
) -> Result<Option<Vec<V>>> {
    let len = match read_i32_be_stream_opt(reader)? {
        Some(v) => v,
        None => return Ok(None),
    };
    if len < 0 {
        bail!("invalid negative frame length: {}", len);
    }
    let len = len as usize;
    if len > max_frame_size {
        bail!("frame length {} exceeds max_frame_size {}", len, max_frame_size);
    }

    scratch.resize(len, 0);
    reader.read_exact(scratch).context("read frame payload")?;

    decode_payload_to_row(scratch, session).map(Some)
}

fn decode_payload_to_row(payload: &[u8], session: &SessionConfig) -> Result<Vec<V>> {
    let mut p = 0usize;
    if payload.len() < 4 {
        bail!("truncated payload: missing __op");
    }
    let op = read_i32_be(payload, &mut p)?;
    let mut row = vec![V::Null; session.expected_input_len];
    row[0] = V::I32(op);

    if session.reorder_responses {
        let row_id = read_i64_be(payload, &mut p)?;
        if session.expected_input_len > 1 {
            row[1] = V::I64(row_id);
        }
    }

    let n_fields = session.pre_payload_positions.len();
    let null_bytes = (n_fields + 7) >> 3;

    if p + null_bytes > payload.len() {
        bail!("truncated payload: missing nullBitmap");
    }
    let null_pos = p;
    p += null_bytes;

    for (i, (&row_pos, ftype)) in session
        .pre_payload_positions
        .iter()
        .zip(session.pre_payload_types.iter())
        .enumerate()
    {
        if is_null_bit_set(payload, null_pos, i) {
            if row_pos < row.len() {
                row[row_pos] = V::Null;
            }
            continue;
        }
        let v = decode_field(payload, &mut p, ftype)
            .with_context(|| format!("decode field {} at row_pos {}", i, row_pos))?;
        if row_pos < row.len() {
            row[row_pos] = v;
        }
    }

    if p > payload.len() {
        bail!("payload decode overran buffer");
    }
    Ok(row)
}

fn decode_field(buf: &[u8], p: &mut usize, t: &FieldType) -> Result<V> {
    match t {
        FieldType::Boolean => {
            let b = read_u8(buf, p)?;
            Ok(V::Bool(b != 0))
        }
        FieldType::Int64 | FieldType::TimestampMillis => Ok(V::I64(read_i64_be(buf, p)?)),
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => Ok(V::I32(read_i32_be(buf, p)?)),
        FieldType::Float32 => {
            let bits = read_u32_be(buf, p)?;
            Ok(V::F32(f32::from_bits(bits)))
        }
        FieldType::Float64 => {
            let bits = read_u64_be(buf, p)?;
            Ok(V::F64(f64::from_bits(bits)))
        }
        FieldType::String | FieldType::Unknown(_) => {
            let bytes = read_len_bytes(buf, p)?;
            let s = String::from_utf8(bytes).context("invalid UTF-8 STRING")?;
            Ok(V::String(s))
        }
        FieldType::Bytes => {
            let bytes = read_len_bytes(buf, p)?;
            Ok(V::Bytes(bytes))
        }
        FieldType::Date => {
            // wire: int32 days since epoch (Flink); internally store millis for UDF convenience
            let days = read_i32_be(buf, p)? as i64;
            let millis = days.checked_mul(86_400_000).context("DATE days->millis overflow")?;
            Ok(V::I64(millis))
        }
        FieldType::Decimal { precision, .. } => {
            let prec = precision.unwrap_or(38);
            if prec <= 18 {
                let unscaled = read_i64_be(buf, p)? as i128;
                Ok(V::DecimalI128(unscaled))
            } else {
                let bytes = read_len_bytes(buf, p)?;
                let unscaled = twos_complement_be_to_i128(&bytes).context("decimal bytes too large for i128")?;
                Ok(V::DecimalI128(unscaled))
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = read_i64_be(buf, p)? as i128;
            Ok(V::DecimalI128(unscaled))
        }
        FieldType::DecimalUnscaledBytes => {
            let bytes = read_len_bytes(buf, p)?;
            let unscaled = twos_complement_be_to_i128(&bytes).context("decimal bytes too large for i128")?;
            Ok(V::DecimalI128(unscaled))
        }
    }
}

fn write_framed_rows<W: Write>(
    writer: &mut W,
    rows: &[Vec<V>],
    session: &SessionConfig,
    max_frame_size: usize,
    payload: &mut Vec<u8>,
) -> Result<()> {
    for row in rows {
        write_framed_row(writer, row, session, max_frame_size, payload)?;
    }
    Ok(())
}

fn write_framed_row<W: Write>(
    writer: &mut W,
    output_row: &[V],
    session: &SessionConfig,
    max_frame_size: usize,
    payload: &mut Vec<u8>,
) -> Result<()> {
    payload.clear();

    // __op
    let op = match output_row.first() {
        Some(V::I32(v)) => *v,
        Some(V::I64(v)) => *v as i32,
        _ => 0,
    };
    write_i32_be_vec(payload, op);

    // __rowId
    if session.reorder_responses {
        let row_id = match output_row.get(1) {
            Some(V::I64(v)) => *v,
            Some(V::I32(v)) => *v as i64,
            _ => 0,
        };
        write_i64_be_vec(payload, row_id);
    }

    // null bitmap for payload fields (post payload only)
    let n_fields = session.post_payload_positions.len();
    let null_bytes = (n_fields + 7) >> 3;
    let null_pos = payload.len();
    payload.resize(null_pos + null_bytes, 0);

    // encode values in postFields order (excluding __op/__rowId)
    for (i, (&row_pos, ftype)) in session
        .post_payload_positions
        .iter()
        .zip(session.post_payload_types.iter())
        .enumerate()
    {
        let v = output_row.get(row_pos).unwrap_or(&V::Null);
        if matches!(v, V::Null) {
            set_null_bit(&mut payload[null_pos..null_pos + null_bytes], i);
            continue;
        }
        encode_field(payload, v, ftype)
            .with_context(|| format!("encode post field {} at row_pos {}", i, row_pos))?;
    }

    if payload.len() > max_frame_size {
        bail!(
            "row payload exceeds max_frame_size: {} > {}",
            payload.len(),
            max_frame_size
        );
    }

    // frame: [len][payload]
    write_i32_be_stream(writer, payload.len() as i32)?;
    writer.write_all(payload.as_slice()).context("write payload")?;
    Ok(())
}

fn encode_field(out: &mut Vec<u8>, v: &V, t: &FieldType) -> Result<()> {
    match t {
        FieldType::Boolean => {
            let b = v_to_bool(v)?;
            out.push(if b { 1 } else { 0 });
        }
        FieldType::Int64 | FieldType::TimestampMillis => {
            let x = v_to_i64(v)?;
            write_i64_be_vec(out, x);
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let x = v_to_i64(v)?;
            write_i32_be_vec(out, x as i32);
        }
        FieldType::Float32 => {
            let x = v_to_f64(v)? as f32;
            write_u32_be_vec(out, x.to_bits());
        }
        FieldType::Float64 => {
            let x = v_to_f64(v)?;
            write_u64_be_vec(out, x.to_bits());
        }
        FieldType::String | FieldType::Unknown(_) => {
            let s = v_to_string(v)?;
            let bytes = s.as_bytes();
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(bytes);
        }
        FieldType::Bytes => {
            let bytes = match v {
                V::Bytes(b) => b.as_slice(),
                V::String(s) => s.as_bytes(),
                _ => bail!("cannot encode BYTES from {:?}", v),
            };
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(bytes);
        }
        FieldType::Date => {
            // wire expects int32 days since epoch.
            let millis = v_to_i64(v)?;
            let days = (millis / 86_400_000) as i32;
            write_i32_be_vec(out, days);
        }
        FieldType::Decimal { precision, .. } => {
            let unscaled = v_to_decimal_i128(v)?;
            let prec = precision.unwrap_or(38);
            if prec <= 18 {
                let as_i64 = i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
                write_i64_be_vec(out, as_i64);
            } else {
                let bytes = i128_to_twos_complement_be_minimal(unscaled);
                write_i32_be_vec(out, bytes.len() as i32);
                out.extend_from_slice(&bytes);
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = v_to_decimal_i128(v)?;
            let as_i64 = i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
            write_i64_be_vec(out, as_i64);
        }
        FieldType::DecimalUnscaledBytes => {
            let unscaled = v_to_decimal_i128(v)?;
            let bytes = i128_to_twos_complement_be_minimal(unscaled);
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(&bytes);
        }
    }
    Ok(())
}

/// Encodes the value at `row` from an InputColumn directly into the wire buffer.
/// Returns `true` if the value is null.
fn encode_input_column_at(out: &mut Vec<u8>, col: &InputColumn, row: usize, ftype: &FieldType) -> Result<bool> {
    if column_is_null(col, row) {
        return Ok(true);
    }
    match ftype {
        FieldType::Boolean => {
            let b = input_column_to_bool(col, row, ftype)?;
            out.push(if b { 1 } else { 0 });
        }
        FieldType::Int64 | FieldType::TimestampMillis => {
            let x = input_column_to_i64(col, row, ftype)?;
            write_i64_be_vec(out, x);
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let x = input_column_to_i64(col, row, ftype)?;
            write_i32_be_vec(out, x as i32);
        }
        FieldType::Float32 => {
            let x = input_column_to_f64(col, row, ftype)? as f32;
            write_u32_be_vec(out, x.to_bits());
        }
        FieldType::Float64 => {
            let x = input_column_to_f64(col, row, ftype)?;
            write_u64_be_vec(out, x.to_bits());
        }
        FieldType::String | FieldType::Unknown(_) => {
            let s = input_column_to_string(col, row, ftype)?;
            let bytes = s.as_bytes();
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(bytes);
        }
        FieldType::Bytes => {
            let s = input_column_to_string(col, row, ftype)?;
            let bytes = s.into_bytes();
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(&bytes);
        }
        FieldType::Date => {
            let millis = input_column_to_i64(col, row, ftype)?;
            let days = (millis / 86_400_000) as i32;
            write_i32_be_vec(out, days);
        }
        FieldType::Decimal { precision, .. } => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let prec = precision.unwrap_or(38);
            if prec <= 18 {
                let as_i64 = i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
                write_i64_be_vec(out, as_i64);
            } else {
                let bytes = i128_to_twos_complement_be_minimal(unscaled);
                write_i32_be_vec(out, bytes.len() as i32);
                out.extend_from_slice(&bytes);
            }
        }
        FieldType::DecimalUnscaledI64 => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let as_i64 = i64::try_from(unscaled).map_err(|_| anyhow::anyhow!("DECIMAL_UNSCALED_I64 overflow"))?;
            write_i64_be_vec(out, as_i64);
        }
        FieldType::DecimalUnscaledBytes => {
            let unscaled = input_column_to_decimal(col, row, ftype)?;
            let bytes = i128_to_twos_complement_be_minimal(unscaled);
            write_i32_be_vec(out, bytes.len() as i32);
            out.extend_from_slice(&bytes);
        }
    }
    Ok(false)
}

// bitmap helpers (LSB-first)
fn set_null_bit(bitmap: &mut [u8], field_index: usize) {
    let byte = field_index >> 3;
    let bit = field_index & 7;
    bitmap[byte] |= 1u8 << bit;
}
fn is_null_bit_set(payload: &[u8], bitmap_pos: usize, field_index: usize) -> bool {
    let byte = bitmap_pos + (field_index >> 3);
    let bit = field_index & 7;
    (payload[byte] & (1u8 << bit)) != 0
}

// endian + bounds helpers
fn read_u8(buf: &[u8], p: &mut usize) -> Result<u8> {
    if *p + 1 > buf.len() {
        bail!("truncated u8");
    }
    let v = buf[*p];
    *p += 1;
    Ok(v)
}
fn read_u32_be(buf: &[u8], p: &mut usize) -> Result<u32> {
    if *p + 4 > buf.len() {
        bail!("truncated u32");
    }
    let b = &buf[*p..*p + 4];
    *p += 4;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}
fn read_u64_be(buf: &[u8], p: &mut usize) -> Result<u64> {
    if *p + 8 > buf.len() {
        bail!("truncated u64");
    }
    let b = &buf[*p..*p + 8];
    *p += 8;
    Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
}
fn read_i32_be(buf: &[u8], p: &mut usize) -> Result<i32> {
    Ok(read_u32_be(buf, p)? as i32)
}
fn read_i64_be(buf: &[u8], p: &mut usize) -> Result<i64> {
    Ok(read_u64_be(buf, p)? as i64)
}
fn read_len_bytes(buf: &[u8], p: &mut usize) -> Result<Vec<u8>> {
    let len = read_i32_be(buf, p)?;
    if len < 0 {
        bail!("negative length");
    }
    let len = len as usize;
    if *p + len > buf.len() {
        bail!(
            "truncated len-bytes: need {}, have {}",
            len,
            buf.len().saturating_sub(*p)
        );
    }
    let out = buf[*p..*p + len].to_vec();
    *p += len;
    Ok(out)
}
fn write_u32_be_vec(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn write_u64_be_vec(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn write_i32_be_vec(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn write_i64_be_vec(out: &mut Vec<u8>, v: i64) {
    out.extend_from_slice(&v.to_be_bytes());
}
fn write_i32_be_stream<W: Write>(w: &mut W, v: i32) -> Result<()> {
    w.write_all(&v.to_be_bytes()).context("write i32 be")
}

fn read_i32_be_stream<R: Read>(r: &mut R) -> Result<i32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).context("read i32 be")?;
    Ok(i32::from_be_bytes(b))
}

fn read_i32_be_stream_opt<R: Read>(r: &mut R) -> Result<Option<i32>> {
    let mut b = [0u8; 4];
    match r.read_exact(&mut b) {
        Ok(()) => Ok(Some(i32::from_be_bytes(b))),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e).context("read i32 be opt"),
    }
}

// DECIMAL helpers: Java BigInteger.toByteArray() compatible representation
fn i128_to_twos_complement_be_minimal(v: i128) -> Vec<u8> {
    let bytes = v.to_be_bytes(); // 16 bytes
    let mut start = 0usize;
    while start < 15 {
        let b0 = bytes[start];
        let b1 = bytes[start + 1];
        // drop leading 0x00 if next byte sign bit is 0, or leading 0xFF if next sign bit is 1
        if (b0 == 0x00 && (b1 & 0x80) == 0x00) || (b0 == 0xFF && (b1 & 0x80) == 0x80) {
            start += 1;
            continue;
        }
        break;
    }
    bytes[start..].to_vec()
}

fn twos_complement_be_to_i128(bytes: &[u8]) -> Result<i128> {
    if bytes.is_empty() {
        return Ok(0);
    }
    if bytes.len() > 16 {
        bail!("too many bytes for i128: {}", bytes.len());
    }
    let sign_extend = if (bytes[0] & 0x80) != 0 { 0xFFu8 } else { 0x00u8 };
    let mut out = [sign_extend; 16];
    let start = 16 - bytes.len();
    out[start..].copy_from_slice(bytes);
    Ok(i128::from_be_bytes(out))
}

// -----------------------------------------------------------------------------
// Row validation
// -----------------------------------------------------------------------------

fn validate_row_len(row: &[V], expected_min: usize) -> Result<()> {
    if row.len() < expected_min {
        bail!("row has {} values but expected at least {}", row.len(), expected_min);
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// UDF application (same shape as before, but using V instead of msgpack Value)
// -----------------------------------------------------------------------------

fn apply_udf_to_rows(
    rows: &[Vec<V>],
    session: &SessionConfig,
    udf: &mut UdfHandle,
    method: &str,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) -> Result<Vec<Vec<V>>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let input_columns = build_input_columns(rows, &session.arg_positions, &session.arg_types)?;
    let output_columns = call_udf_to_columns(udf, method, &input_columns, &session.output_names)?;

    let mut out_rows = Vec::with_capacity(rows.len());
    for row_idx in 0..rows.len() {
        let source_row = rows.get(row_idx).context("missing source row")?;

        let mut out_row = if session.passthrough_identity && source_row.len() == session.output_row_len {
            source_row.clone()
        } else {
            let mut row = vec![V::Null; session.output_row_len];
            if let Some(op) = source_row.first() {
                row[0] = op.clone();
            }
            if session.reorder_responses {
                if let Some(row_id) = source_row.get(1) {
                    if session.output_row_len > 1 {
                        row[1] = row_id.clone();
                    }
                }
            }

            for source in &session.post_field_sources {
                match source.kind {
                    PostFieldSourceKind::Op => {
                        if source.pos < row.len() {
                            row[source.pos] = row[0].clone();
                        }
                    }
                    PostFieldSourceKind::RowId => {
                        if session.reorder_responses && source.pos < row.len() {
                            let row_id = row.get(1).cloned().unwrap_or(V::Null);
                            row[source.pos] = row_id;
                        }
                    }
                    PostFieldSourceKind::InputPos(pre_pos) => {
                        if source.pos < row.len() {
                            row[source.pos] = source_row.get(pre_pos).cloned().unwrap_or(V::Null);
                        }
                    }
                    PostFieldSourceKind::Output => {}
                }
            }

            row
        };

        if output_columns.len() < session.output_positions.len() {
            bail!(
                "UDF returned {} columns but functionResults resolved {} targets",
                output_columns.len(),
                session.output_positions.len()
            );
        }

        for (idx, pos) in session.output_positions.iter().enumerate() {
            if *pos >= out_row.len() {
                bail!("functionResult outputIndex {} is out of range", pos);
            }
            let v = output_column_to_v(
                &output_columns[idx],
                row_idx,
                session.output_types.get(idx).unwrap_or(&FieldType::String),
            )?;
            out_row[*pos] = v;
        }

        out_rows.push(out_row);
    }

    maybe_print_debug_rows(rows, session, &out_rows, debug_sample_rows, debug_batches_remaining);
    Ok(out_rows)
}

fn apply_udf_to_rows_stream<W: Write>(
    config: &mut UdfConfig<W>,
) -> Result<()> {
    let rows = config.rows;
    let session = config.session;
    let method = config.method;
    let max_frame_size = config.max_frame_size;
    let debug_sample_rows = config.debug_sample_rows;

    if rows.is_empty() {
        return Ok(());
    }

    // Debug path uses materialized output rows.
    if debug_sample_rows > 0 && *config.debug_batches_remaining > 0 {
        let out_rows = apply_udf_to_rows(
            rows, session, &mut *config.udf, method,
            debug_sample_rows, &mut *config.debug_batches_remaining,
        )?;
        let mut payload_buf = Vec::with_capacity(256);
        write_framed_rows(&mut *config.writer, &out_rows, session, max_frame_size, &mut payload_buf)?;
        return Ok(());
    }

    let input_columns = build_input_columns(rows, &session.arg_positions, &session.arg_types)?;
    let output_columns = call_udf_to_columns(&mut *config.udf, method, &input_columns, &session.output_names)?;

    if output_columns.len() < session.output_positions.len() {
        bail!(
            "UDF returned {} columns but functionResults resolved {} targets",
            output_columns.len(),
            session.output_positions.len()
        );
    }

    // Zero-alloc streaming encoder: encode each row directly from input/output columns.
    let n_fields = session.post_payload_positions.len();
    let null_bytes = (n_fields + 7) >> 3;
    let mut payload = Vec::with_capacity(256);
    let writer = &mut *config.writer;

    for (row_idx, source_row) in rows.iter().enumerate() {
        payload.clear();

        // __op
        let op = match source_row.first() {
            Some(V::I32(v)) => *v,
            Some(V::I64(v)) => *v as i32,
            _ => 0,
        };
        write_i32_be_vec(&mut payload, op);

        // __rowId
        if session.reorder_responses {
            let row_id = match source_row.get(1) {
                Some(V::I64(v)) => *v,
                Some(V::I32(v)) => *v as i64,
                _ => 0,
            };
            write_i64_be_vec(&mut payload, row_id);
        }

        // null bitmap placeholder
        let null_pos = payload.len();
        payload.resize(null_pos + null_bytes, 0);

        // encode payload fields directly from columnar sources
        for (i, source) in session.post_payload_sources.iter().enumerate() {
            let ftype = &session.post_payload_types[i];
            let is_null = match source {
                PayloadSource::InputAt(pre_pos) => {
                    let v = source_row.get(*pre_pos).unwrap_or(&V::Null);
                    if matches!(v, V::Null) {
                        true
                    } else {
                        encode_field(&mut payload, v, ftype)?;
                        false
                    }
                }
                PayloadSource::OutputAt(col_idx) => {
                    encode_input_column_at(&mut payload, &output_columns[*col_idx], row_idx, ftype)?
                }
            };
            if is_null {
                set_null_bit(&mut payload[null_pos..null_pos + null_bytes], i);
            }
        }

        if payload.len() > max_frame_size {
            bail!(
                "row payload exceeds max_frame_size: {} > {}",
                payload.len(),
                max_frame_size
            );
        }

        write_i32_be_stream(writer, payload.len() as i32)?;
        writer.write_all(payload.as_slice()).context("write payload")?;
    }

    Ok(())
}

fn call_udf_to_columns(
    udf: &mut UdfHandle,
    method: &str,
    input_columns: &[InputColumn],
    output_names: &[String],
) -> Result<Vec<InputColumn>> {
    let columns_method = if method.ends_with("Fast") || method.ends_with("ToColumnsTypedOut") {
        method.to_string()
    } else {
        format!("{method}Fast")
    };

    if output_names.is_empty() {
        return udf.call_typed_columns_to_typed_results(&columns_method, input_columns);
    }

    let named_method = if columns_method.ends_with("Named") {
        columns_method.clone()
    } else {
        format!("{columns_method}Named")
    };
    udf.call_typed_columns_to_named_results(&named_method, input_columns, output_names)
}

// -----------------------------------------------------------------------------
// Build input columns from decoded rows
// -----------------------------------------------------------------------------

fn build_input_columns(rows: &[Vec<V>], arg_positions: &[usize], arg_types: &[FieldType]) -> Result<Vec<InputColumn>> {
    let mut columns = Vec::with_capacity(arg_positions.len());
    for (idx, arg_index) in arg_positions.iter().enumerate() {
        let target_type = arg_types
            .get(idx)
            .with_context(|| format!("missing arg type at {}", idx))?;
        let column = build_input_column(rows, *arg_index, target_type)?;
        columns.push(column);
    }
    Ok(columns)
}

fn build_input_column(rows: &[Vec<V>], index: usize, target_type: &FieldType) -> Result<InputColumn> {
    match target_type {
        FieldType::String | FieldType::Unknown(_) => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                if matches!(v, V::Null) {
                    values.push(None);
                } else {
                    values.push(Some(v_to_string(v)?));
                }
            }
            Ok(InputColumn::String(values))
        }
        FieldType::Bytes => {
            // for UDF, bytes are represented as String column (base64 would be better; keeping simple)
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                if matches!(v, V::Null) {
                    values.push(None);
                } else {
                    match v {
                        V::Bytes(b) => values.push(Some(String::from_utf8_lossy(b).to_string())),
                        _ => values.push(Some(v_to_string(v)?)),
                    }
                }
            }
            Ok(InputColumn::String(values))
        }
        FieldType::Boolean => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(false);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_bool(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::Bool { values, is_null: nulls })
        }
        FieldType::Int64 | FieldType::TimestampMillis | FieldType::Date => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    // timestamps are already millis; dates were converted to millis in decode_field
                    values.push(v_to_i64(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::I64 { values, is_null: nulls })
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    let x = v_to_i64(v)?;
                    let casted = i32::try_from(x).map_err(|_| anyhow::anyhow!("value {} overflows i32", x))?;
                    values.push(casted);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::I32 { values, is_null: nulls })
        }
        FieldType::Float64 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0.0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_f64(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::F64 { values, is_null: nulls })
        }
        FieldType::Float32 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0.0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_f64(v)? as f32);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::F32 { values, is_null: nulls })
        }
        FieldType::Decimal { .. } | FieldType::DecimalUnscaledI64 | FieldType::DecimalUnscaledBytes => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let v = row.get(index).unwrap_or(&V::Null);
                let prev_len = values.len();
                if matches!(v, V::Null) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(v_to_decimal_i128(v)?);
                    if let Some(n) = nulls.as_mut() {
                        n.push(false);
                    }
                }
            }
            Ok(InputColumn::Decimal128 { values, is_null: nulls })
        }
    }
}

fn push_null(nulls: &mut Option<Vec<bool>>, previous_len: usize, is_null: bool) {
    if let Some(nulls) = nulls.as_mut() {
        nulls.push(is_null);
        return;
    }
    if is_null {
        let mut vec = vec![false; previous_len];
        vec.push(true);
        *nulls = Some(vec);
    }
}

fn column_is_null(column: &InputColumn, row: usize) -> bool {
    match column {
        InputColumn::String(values) => values.get(row).map(|v| v.is_none()).unwrap_or(true),
        InputColumn::I64 { is_null, .. }
        | InputColumn::I32 { is_null, .. }
        | InputColumn::F64 { is_null, .. }
        | InputColumn::F32 { is_null, .. }
        | InputColumn::Bool { is_null, .. }
        | InputColumn::Decimal128 { is_null, .. } => is_null_at(is_null.as_deref(), row),
    }
}

fn is_null_at(nulls: Option<&[bool]>, idx: usize) -> bool {
    nulls
        .map(|vals| vals.get(idx).copied().unwrap_or(false))
        .unwrap_or(false)
}

fn output_column_to_v(column: &InputColumn, row: usize, output_type: &FieldType) -> Result<V> {
    if column_is_null(column, row) {
        return Ok(V::Null);
    }
    match output_type {
        FieldType::String | FieldType::Unknown(_) => Ok(V::String(input_column_to_string(column, row, output_type)?)),
        FieldType::Bytes => Ok(V::Bytes(input_column_to_string(column, row, output_type)?.into_bytes())),
        FieldType::Boolean => Ok(V::Bool(input_column_to_bool(column, row, output_type)?)),
        FieldType::Int64 | FieldType::TimestampMillis | FieldType::Date => {
            Ok(V::I64(input_column_to_i64(column, row, output_type)?))
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            Ok(V::I32(input_column_to_i64(column, row, output_type)? as i32))
        }
        FieldType::Float64 => Ok(V::F64(input_column_to_f64(column, row, output_type)?)),
        FieldType::Float32 => Ok(V::F32(input_column_to_f64(column, row, output_type)? as f32)),
        FieldType::Decimal { .. } | FieldType::DecimalUnscaledI64 | FieldType::DecimalUnscaledBytes => {
            Ok(V::DecimalI128(input_column_to_decimal(column, row, output_type)?))
        }
    }
}

fn input_column_to_string(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<String> {
    match column {
        InputColumn::String(values) => values
            .get(row)
            .and_then(|v| v.clone())
            .ok_or_else(|| anyhow::anyhow!("null string value")),
        InputColumn::Bool { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::I64 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::I32 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::F64 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::F32 { values, .. } => Ok(values.get(row).map(|v| v.to_string()).unwrap_or_default()),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_string(value, scale))
        }
    }
}

fn input_column_to_bool(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<bool> {
    match column {
        InputColumn::Bool { values, .. } => Ok(*values.get(row).unwrap_or(&false)),
        InputColumn::I64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0) != 0),
        InputColumn::I32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0) != 0),
        InputColumn::F64 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) != 0.0),
        InputColumn::F32 { values, .. } => Ok(values.get(row).copied().unwrap_or(0.0) != 0.0),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_i64(value, scale)? != 0)
        }
        InputColumn::String(values) => values
            .get(row)
            .and_then(|v| v.as_deref())
            .map(parse_bool_string)
            .unwrap_or(Ok(false)),
    }
}

fn input_column_to_i64(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<i64> {
    match column {
        InputColumn::I64 { values, .. } => Ok(*values.get(row).unwrap_or(&0)),
        InputColumn::I32 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as i64),
        InputColumn::F64 { values, .. } => Ok(*values.get(row).unwrap_or(&0.0) as i64),
        InputColumn::F32 { values, .. } => Ok(*values.get(row).unwrap_or(&0.0) as i64),
        InputColumn::Bool { values, .. } => Ok(if *values.get(row).unwrap_or(&false) { 1 } else { 0 }),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            decimal_to_i64(value, scale)
        }
        InputColumn::String(values) => {
            let value = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_with(value, "i64")
        }
    }
}

fn input_column_to_f64(column: &InputColumn, row: usize, source_type: &FieldType) -> Result<f64> {
    match column {
        InputColumn::F64 { values, .. } => Ok(*values.get(row).unwrap_or(&0.0)),
        InputColumn::F32 { values, .. } => Ok(*values.get(row).unwrap_or(&0.0) as f64),
        InputColumn::I64 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as f64),
        InputColumn::I32 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as f64),
        InputColumn::Bool { values, .. } => Ok(if *values.get(row).unwrap_or(&false) { 1.0 } else { 0.0 }),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_f64(value, scale))
        }
        InputColumn::String(values) => {
            let value = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_with(value, "f64")
        }
    }
}

fn input_column_to_decimal(column: &InputColumn, row: usize, _source_type: &FieldType) -> Result<i128> {
    match column {
        InputColumn::Decimal128 { values, .. } => Ok(values.get(row).copied().unwrap_or(0)),
        InputColumn::I64 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as i128),
        InputColumn::I32 { values, .. } => Ok(*values.get(row).unwrap_or(&0) as i128),
        InputColumn::F64 { values, .. } => {
            parse_decimal_to_i128(&values.get(row).copied().unwrap_or(0.0).to_string(), 0)
        }
        InputColumn::F32 { values, .. } => {
            parse_decimal_to_i128(&values.get(row).copied().unwrap_or(0.0).to_string(), 0)
        }
        InputColumn::Bool { values, .. } => Ok(if *values.get(row).unwrap_or(&false) { 1 } else { 0 }),
        InputColumn::String(values) => {
            let s = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_decimal_to_i128(s, 0)
        }
    }
}

// -----------------------------------------------------------------------------
// V conversions
// -----------------------------------------------------------------------------

fn v_to_string(v: &V) -> Result<String> {
    match v {
        V::String(s) => Ok(s.clone()),
        V::Bytes(b) => Ok(String::from_utf8_lossy(b).to_string()),
        V::I32(x) => Ok(x.to_string()),
        V::I64(x) => Ok(x.to_string()),
        V::Bool(b) => Ok(b.to_string()),
        V::F32(x) => Ok(x.to_string()),
        V::F64(x) => Ok(x.to_string()),
        V::DecimalI128(x) => Ok(x.to_string()),
        V::Null => bail!("value is null"),
    }
}

fn v_to_bool(v: &V) -> Result<bool> {
    match v {
        V::Bool(b) => Ok(*b),
        V::I32(x) => Ok(*x != 0),
        V::I64(x) => Ok(*x != 0),
        V::F32(x) => Ok(*x != 0.0),
        V::F64(x) => Ok(*x != 0.0),
        V::String(s) => parse_bool_string(s),
        V::Bytes(b) => parse_bool_string(&String::from_utf8_lossy(b)),
        V::DecimalI128(x) => Ok(*x != 0),
        V::Null => bail!("value is null"),
    }
}

fn v_to_i64(v: &V) -> Result<i64> {
    match v {
        V::I64(x) => Ok(*x),
        V::I32(x) => Ok(*x as i64),
        V::Bool(b) => Ok(if *b { 1 } else { 0 }),
        V::F32(x) => Ok(*x as i64),
        V::F64(x) => Ok(*x as i64),
        V::String(s) => parse_with(s, "i64"),
        V::Bytes(b) => parse_with(&String::from_utf8_lossy(b), "i64"),
        V::DecimalI128(x) => i64::try_from(*x).map_err(|_| anyhow::anyhow!("decimal overflow for i64")),
        V::Null => bail!("value is null"),
    }
}

fn v_to_f64(v: &V) -> Result<f64> {
    match v {
        V::F64(x) => Ok(*x),
        V::F32(x) => Ok(*x as f64),
        V::I64(x) => Ok(*x as f64),
        V::I32(x) => Ok(*x as f64),
        V::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        V::String(s) => parse_with(s, "f64"),
        V::Bytes(b) => parse_with(&String::from_utf8_lossy(b), "f64"),
        V::DecimalI128(x) => Ok(*x as f64),
        V::Null => bail!("value is null"),
    }
}

fn v_to_decimal_i128(v: &V) -> Result<i128> {
    match v {
        V::DecimalI128(x) => Ok(*x),
        V::I64(x) => Ok(*x as i128),
        V::I32(x) => Ok(*x as i128),
        V::String(s) => parse_decimal_to_i128(s, 0),
        V::Bytes(b) => parse_decimal_to_i128(&String::from_utf8_lossy(b), 0),
        _ => bail!("cannot convert {:?} to decimal i128", v),
    }
}

// -----------------------------------------------------------------------------
// Debug printing (kept similar)
// -----------------------------------------------------------------------------

fn maybe_print_debug_rows(
    input_rows: &[Vec<V>],
    session: &SessionConfig,
    output_rows: &[Vec<V>],
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) {
    if debug_sample_rows == 0 || *debug_batches_remaining == 0 || input_rows.is_empty() {
        return;
    }

    let sample_rows = debug_sample_rows.min(input_rows.len());
    println!("Debug sample ({} of {} rows):", sample_rows, input_rows.len());
    for (row_idx, row) in input_rows.iter().take(sample_rows).enumerate() {
        let mut input_parts = Vec::with_capacity(session.arg_names.len());
        for (arg_pos, name) in session.arg_names.iter().enumerate() {
            let idx = session.arg_positions.get(arg_pos).copied().unwrap_or(0);
            let value = row.get(idx).unwrap_or(&V::Null);
            input_parts.push(format!("{}={}", name, debug_v(value)));
        }

        let mut output_parts = Vec::new();
        if !session.output_positions.is_empty() {
            for (idx, name) in session.output_names.iter().enumerate() {
                let pos = session.output_positions.get(idx).copied().unwrap_or(0);
                let value = output_rows.get(row_idx).and_then(|r| r.get(pos)).unwrap_or(&V::Null);
                output_parts.push(format!("{}={}", name, debug_v(value)));
            }
        }

        println!(
            "  row {}: {} -> {}",
            row_idx,
            input_parts.join(", "),
            output_parts.join(", ")
        );
    }

    *debug_batches_remaining = debug_batches_remaining.saturating_sub(1);
}

fn debug_v(v: &V) -> String {
    match v {
        V::Null => "<null>".to_string(),
        V::Bool(b) => b.to_string(),
        V::I32(x) => x.to_string(),
        V::I64(x) => x.to_string(),
        V::F32(x) => x.to_string(),
        V::F64(x) => x.to_string(),
        V::String(s) => s.clone(),
        V::Bytes(b) => format!("{:?}", b),
        V::DecimalI128(x) => x.to_string(),
    }
}

// -----------------------------------------------------------------------------
// Parsing + helpers (reused from your original code where relevant)
// -----------------------------------------------------------------------------

fn parse_bool_string(value: &str) -> Result<bool> {
    let normalized = value.trim().to_lowercase();
    match normalized.as_str() {
        "true" | "t" | "1" | "yes" | "y" => Ok(true),
        "false" | "f" | "0" | "no" | "n" => Ok(false),
        _ => bail!("failed to parse boolean from {}", value),
    }
}

fn parse_with<T>(value: &str, type_name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("failed to parse {type_name} from {value}: {err}"))
}

fn parse_decimal_to_i128(value: &str, scale: i8) -> Result<i128> {
    let scale = scale.max(0) as usize;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        bail!("decimal value is empty");
    }
    let negative = trimmed.starts_with('-');
    let unsigned = trimmed.strip_prefix('-').unwrap_or(trimmed);
    let mut parts = unsigned.splitn(2, '.');
    let int_part = parts.next().unwrap_or("");
    let frac_part = parts.next().unwrap_or("");

    let mut digits = String::with_capacity(int_part.len() + scale);
    digits.push_str(int_part);
    if scale > 0 {
        if frac_part.len() >= scale {
            digits.push_str(&frac_part[..scale]);
        } else {
            digits.push_str(frac_part);
            digits.push_str(&"0".repeat(scale - frac_part.len()));
        }
    }
    if digits.is_empty() {
        digits.push('0');
    }

    let mut v = digits
        .parse::<i128>()
        .map_err(|err| anyhow::anyhow!("failed to parse decimal from {value}: {err}"))?;
    if negative {
        v = -v;
    }
    Ok(v)
}

fn convert_decimal_scale(value: i128, source_scale: i8, target_scale: i8) -> Result<i128> {
    if source_scale == target_scale {
        return Ok(value);
    }
    if source_scale < target_scale {
        let factor = pow10_i128(target_scale - source_scale)?;
        return value.checked_mul(factor).context("decimal scale overflow");
    }
    let factor = pow10_i128(source_scale - target_scale)?;
    Ok(value / factor)
}

fn pow10_i128(scale: i8) -> Result<i128> {
    if scale <= 0 {
        return Ok(1);
    }
    let mut value: i128 = 1;
    for _ in 0..scale {
        value = value.checked_mul(10).context("decimal scale overflow")?;
    }
    Ok(value)
}

fn decimal_to_i64(value: i128, scale: i8) -> Result<i64> {
    let scaled = convert_decimal_scale(value, scale, 0)?;
    i64::try_from(scaled).map_err(|_| anyhow::anyhow!("decimal overflow for i64"))
}

fn decimal_to_f64(value: i128, scale: i8) -> f64 {
    if scale == 0 {
        return value as f64;
    }
    let factor = 10f64.powi(scale as i32);
    (value as f64) / factor
}

fn decimal_to_string(value: i128, scale: i8) -> String {
    let negative = value < 0;
    let mut digits = value.abs().to_string();
    let scale = scale.max(0) as usize;
    if scale > 0 {
        if digits.len() <= scale {
            let pad = scale + 1 - digits.len();
            digits = format!("{}{}", "0".repeat(pad), digits);
        }
        let split = digits.len() - scale;
        let (int_part, frac_part) = digits.split_at(split);
        let s = format!("{}.{}", int_part, frac_part);
        if negative {
            format!("-{}", s)
        } else {
            s
        }
    } else if negative {
        format!("-{}", digits)
    } else {
        digits
    }
}

fn parse_field_type(type_str: &str) -> FieldType {
    let base = base_type(type_str);
    match base.as_str() {
        "STRING" | "VARCHAR" | "CHAR" | "TEXT" => FieldType::String,
        "BOOLEAN" | "BOOL" => FieldType::Boolean,
        "BIGINT" | "LONG" | "INT64" => FieldType::Int64,
        "INT" | "INTEGER" | "INT32" => FieldType::Int32,
        "SMALLINT" | "INT16" => FieldType::Int16,
        "TINYINT" | "INT8" => FieldType::Int8,
        "DOUBLE" | "FLOAT64" => FieldType::Float64,
        "FLOAT" | "REAL" | "FLOAT32" => FieldType::Float32,
        "DECIMAL_UNSCALED_I64" => FieldType::DecimalUnscaledI64,
        "DECIMAL_UNSCALED_BYTES" => FieldType::DecimalUnscaledBytes,
        "DECIMAL" | "NUMERIC" => {
            let (precision, scale) = parse_decimal_precision_scale(type_str);
            FieldType::Decimal { precision, scale }
        }
        "TIMESTAMP" | "TIMESTAMP_MILLIS" => FieldType::TimestampMillis,
        "DATE" => FieldType::Date,
        "BIN" | "BINARY" | "VARBINARY" | "BYTES" => FieldType::Bytes,
        other => FieldType::Unknown(other.to_string()),
    }
}

fn parse_decimal_precision_scale(type_str: &str) -> (Option<u8>, i8) {
    let trimmed = type_str.trim();
    let start = trimmed.find('(');
    let end = trimmed.find(')');
    if let (Some(start), Some(end)) = (start, end) {
        let inner = &trimmed[start + 1..end];
        let mut parts = inner.split(',');
        let precision = parts.next().and_then(|p| p.trim().parse::<u8>().ok());
        let scale = parts.next().and_then(|s| s.trim().parse::<i8>().ok()).unwrap_or(0);
        (precision, scale)
    } else {
        (None, 0)
    }
}

fn base_type(type_str: &str) -> String {
    let trimmed = type_str.trim();
    let base = trimmed.split('(').next().unwrap_or(trimmed);
    base.trim().to_uppercase()
}
