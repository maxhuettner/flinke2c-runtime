use anyhow::{bail, Context, Result};
use arrow_array::array::{
    Array, ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int16Array,
    Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray, TimestampMicrosecondArray,
    TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt32Array,
    UInt64Array,
};
use arrow_array::RecordBatch;
use arrow_array::builder::{
    Decimal128Builder, Float32Builder, Float64Builder, Int32Builder, Int64Builder,
    LargeStringBuilder, StringBuilder, TimestampMicrosecondBuilder, TimestampMillisecondBuilder,
    TimestampNanosecondBuilder, TimestampSecondBuilder, UInt32Builder, UInt64Builder,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, TimeUnit};
use clap::Parser;
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use rmpv::decode as msgpack_decode;
use rmpv::encode as msgpack_encode;
use rmpv::{Utf8String, Value};
use serde::Deserialize;
use std::collections::{BTreeMap, HashSet};
use std::io::{BufReader, BufWriter, ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::thread;
use crate::java_udf::{InputColumn, JavaArg, UdfHandle};

mod java_udf;

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

    #[arg(long, value_delimiter = ',', default_value = "jar/flinke2c.jar")]
    udf_jars: Vec<PathBuf>,
    #[arg(long, default_value = "org.example.proxy.ScalarFunctionAdapter")]
    udf_adapter_class: String,
    #[arg(long, default_value = "evalBatch")]
    udf_method: String,
    #[arg(long, default_value = "([[Ljava/lang/String;)V")]
    udf_sig: String,

    // Debug: print a small sample of input/output rows per session.
    #[arg(long, default_value_t = 0)]
    debug_sample_rows: usize,
    #[arg(long, default_value_t = 1)]
    debug_sample_batches: usize,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FunctionArg {
    name: String,
    #[serde(rename = "type", alias = "wireType")]
    arg_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FunctionResult {
    output_name: String,
    #[serde(
        default,
        rename = "outputType",
        alias = "outputWireType",
        alias = "wireType"
    )]
    output_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FieldSpec {
    name: String,
    #[serde(default, rename = "wireType", alias = "type")]
    field_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
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
    Timestamp { unit: TimeUnit },
    Date,
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
    pre_name_to_pos: std::collections::HashMap<String, usize>,
    post_fields: Vec<FieldSpec>,
    post_field_positions: Vec<usize>,
    post_field_sources: Vec<PostFieldSource>,
    passthrough_identity: bool,
    output_slots: Vec<OutputSlotKind>,
    arg_positions: Vec<usize>,
    arg_names: Vec<String>,
    arg_types: Vec<FieldType>,
    output_positions: Vec<usize>,
    output_names: Vec<String>,
    output_types: Vec<FieldType>,
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

#[derive(Clone, Debug)]
enum OutputSlotKind {
    Op,
    RowId,
    InputPos(usize),
    Output(usize),
    Nil,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let listener = TcpListener::bind((args.listen_host.as_str(), args.in_port))
        .context("bind in-port")?;
    let mut current_udf_class: Option<String> = None;
    let mut current_udf_types: Option<Vec<String>> = None;
    let mut udf: Option<UdfHandle> = None;

    println!(
        "Waiting for PRE/POST on {}:{} ...",
        args.listen_host, args.in_port
    );
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
            pre,
            pre_cfg,
            post,
            post_cfg,
            &mut udf,
            &mut current_udf_class,
            &mut current_udf_types,
            &args,
        ) {
            eprintln!("Session ended with error: {err:#}");
        }
    }
}

fn read_config(stream: &mut TcpStream) -> Result<ConfigMessage> {
    let mut len_bytes = [0u8; 4];
    stream
        .read_exact(&mut len_bytes)
        .context("read config length")?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    if len == 0 {
        bail!("empty config message");
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).context("read config")?;
    let cfg: ConfigMessage = serde_json::from_slice(&buf).context("parse config json")?;
    println!("Received config: {:?}", cfg);
    Ok(cfg)
}

fn accept_pair(
    listener: &TcpListener,
) -> Result<(TcpStream, ConfigMessage, TcpStream, ConfigMessage)> {
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

fn build_session_config(pre_cfg: &ConfigMessage) -> Result<SessionConfig> {
    let reorder_responses = pre_cfg.reorder_responses;

    if pre_cfg.pre_fields.is_empty() {
        bail!("preFields missing from PRE config");
    }
    if pre_cfg.post_fields.is_empty() {
        bail!("postFields missing from PRE config");
    }

    let has_op_in_pre = pre_cfg
        .pre_fields
        .first()
        .map(|f| f.name == "__op")
        .unwrap_or(false);
    let has_op_in_post = pre_cfg
        .post_fields
        .first()
        .map(|f| f.name == "__op")
        .unwrap_or(false);

    let mut pre_name_to_pos = std::collections::HashMap::new();
    for (idx, field) in pre_cfg.pre_fields.iter().enumerate() {
        let pos = field_row_pos(idx, has_op_in_pre, reorder_responses);
        if pre_name_to_pos.insert(field.name.clone(), pos).is_some() {
            bail!("duplicate preField name {}", field.name);
        }
    }

    let mut post_field_positions = Vec::with_capacity(pre_cfg.post_fields.len());
    for (idx, _field) in pre_cfg.post_fields.iter().enumerate() {
        let pos = field_row_pos(idx, has_op_in_post, reorder_responses);
        post_field_positions.push(pos);
    }

    let expected_input_len = if has_op_in_pre {
        pre_cfg.pre_fields.len() + if reorder_responses { 1 } else { 0 }
    } else {
        pre_cfg.pre_fields.len() + 1 + if reorder_responses { 1 } else { 0 }
    };

    let output_row_len = if has_op_in_post {
        pre_cfg.post_fields.len() + if reorder_responses { 1 } else { 0 }
    } else {
        pre_cfg.post_fields.len() + 1 + if reorder_responses { 1 } else { 0 }
    };

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
            .with_context(|| {
                format!("functionResult {} not found in postFields", result.output_name)
            })?;
        output_positions.push(pos);
        output_names.push(result.output_name.clone());
        let output_type = result
            .output_type
            .as_deref()
            .map(parse_field_type)
            .with_context(|| {
                format!(
                    "functionResult outputType/wireType missing for {}",
                    result.output_name
                )
            })?;
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
            let output_idx = output_pos_to_idx[pos]
                .with_context(|| format!("missing output column for postField {}", field.name))?;
            (
                PostFieldSourceKind::Output,
                OutputSlotKind::Output(output_idx),
            )
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
                passthrough_identity = false;
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
            OutputSlotKind::Output(_) => {}
        }
    }

    Ok(SessionConfig {
        reorder_responses,
        expected_input_len,
        output_row_len,
        pre_name_to_pos,
        post_fields: pre_cfg.post_fields.clone(),
        post_field_positions,
        post_field_sources,
        passthrough_identity,
        output_slots,
        arg_positions,
        arg_names,
        arg_types,
        output_positions,
        output_names,
        output_types,
    })
}

fn field_row_pos(field_idx: usize, has_op: bool, reorder: bool) -> usize {
    if has_op {
        if reorder {
            if field_idx == 0 {
                0
            } else {
                field_idx + 1
            }
        } else {
            field_idx
        }
    } else {
        let base = 1 + if reorder { 1 } else { 0 };
        base + field_idx
    }
}

fn run_session(
    pre: TcpStream,
    pre_cfg: ConfigMessage,
    post: TcpStream,
    post_cfg: ConfigMessage,
    udf: &mut Option<UdfHandle>,
    current_udf_class: &mut Option<String>,
    current_udf_types: &mut Option<Vec<String>>,
    args: &Args,
) -> Result<()> {
    let pre_fn = pre_cfg.function_class.as_deref().unwrap_or("<none>");
    let post_fn = post_cfg.function_class.as_deref().unwrap_or("<none>");
    let pre_kind = pre_cfg.function_kind.as_deref().unwrap_or("<none>");
    let post_kind = post_cfg.function_kind.as_deref().unwrap_or("<none>");
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

    let session_cfg = build_session_config(&pre_cfg)?;
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
        .map(|arg| {
            arg.arg_type
                .clone()
                .unwrap_or_else(|| "<missing>".to_string())
        })
        .collect();

    let class_changed = current_udf_class
        .as_deref()
        .map(|current| current != desired_udf_class)
        .unwrap_or(true);
    let types_changed = current_udf_types
        .as_ref()
        .map(|current| current != &desired_udf_types)
        .unwrap_or(true);
    if class_changed || types_changed {
        let current_class_display = current_udf_class.as_deref().unwrap_or("<unset>");
        let current_types_display = current_udf_types
            .as_ref()
            .map(|v| format!("{v:?}"))
            .unwrap_or_else(|| "<unset>".to_string());
        println!(
            "Switching UDF config: class {} -> {}, types {} -> {:?}",
            current_class_display, desired_udf_class, current_types_display, desired_udf_types
        );
        *current_udf_class = Some(desired_udf_class);
        *current_udf_types = Some(desired_udf_types);
    }

    let worker_count = args.workers.max(1);
    if worker_count == 1 {
        if class_changed || types_changed || udf.is_none() {
            *udf = Some(UdfHandle::new_with_args(
                &args.udf_jars,
                &args.udf_adapter_class,
                "(Ljava/lang/String;[Ljava/lang/String;)V",
                &[
                    JavaArg::String(
                        current_udf_class
                            .as_ref()
                            .context("missing UDF class")?
                            .clone(),
                    ),
                    JavaArg::StringArray(
                        current_udf_types
                            .as_ref()
                            .context("missing UDF types")?
                            .clone(),
                    ),
                ],
            )?);
        }
        let udf_handle = udf.as_mut().context("UDF handle not initialized")?;
        if udf_handle.reload_if_changed()? {
            println!("Reloaded UDF classes after jar change");
        }

        let mut debug_batches_remaining = args.debug_sample_batches;
        let mut reader = BufReader::with_capacity(args.buf_size, pre);
        let mut writer = BufWriter::with_capacity(args.buf_size, post);
        let batch_size = args.batch_size.max(1);
        let mut batch_rows: Vec<Vec<Value>> = Vec::with_capacity(batch_size);

        loop {
            match read_msgpack_row(&mut reader)? {
                Some(row) => {
                    validate_row_len(&row, session_cfg.expected_input_len)?;
                    batch_rows.push(row);
                    if batch_rows.len() >= batch_size {
                        apply_udf_to_rows_stream(
                            &mut writer,
                            &batch_rows,
                            &session_cfg,
                            udf_handle,
                            &args.udf_method,
                            args.debug_sample_rows,
                            &mut debug_batches_remaining,
                        )?;
                        writer.flush().ok();
                        batch_rows.clear();
                    }
                }
                None => break,
            }
        }

        if !batch_rows.is_empty() {
            apply_udf_to_rows_stream(
                &mut writer,
                &batch_rows,
                &session_cfg,
                udf_handle,
                &args.udf_method,
                args.debug_sample_rows,
                &mut debug_batches_remaining,
            )?;
            writer.flush().ok();
        }

        writer.flush().ok();
        return Ok(());
    }

    if args.debug_sample_rows > 0 {
        eprintln!("Debug sampling disabled in parallel mode");
    }

    let max_in_flight = if args.max_in_flight == 0 {
        worker_count * 2
    } else {
        args.max_in_flight
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
        let udf_jars = args.udf_jars.clone();
        let udf_adapter = args.udf_adapter_class.clone();
        let udf_method = args.udf_method.clone();
        let udf_class = current_udf_class
            .as_ref()
            .context("missing UDF class")?
            .clone();
        let udf_types = current_udf_types
            .as_ref()
            .context("missing UDF types")?
            .clone();

        let handle = thread::spawn(move || {
            let udf_handle = UdfHandle::new_with_args(
                &udf_jars,
                &udf_adapter,
                "(Ljava/lang/String;[Ljava/lang/String;)V",
                &[
                    JavaArg::String(udf_class),
                    JavaArg::StringArray(udf_types),
                ],
            );
            let udf_handle = match udf_handle {
                Ok(handle) => handle,
                Err(err) => {
                    let _ = result_tx.send(WorkResult { seq: 0, result: Err(err) });
                    return;
                }
            };

            let mut debug_batches_remaining = 0usize;
            for msg in rx {
                let Some(work) = msg else {
                    break;
                };
                let result = apply_udf_to_rows(
                    &work.rows,
                    &session_cfg,
                    &udf_handle,
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
    let writer_buf_size = args.buf_size;
    let writer_handle = thread::spawn(move || -> Result<()> {
        let mut writer = BufWriter::with_capacity(writer_buf_size, post);
        let mut pending: BTreeMap<usize, Vec<Vec<Value>>> = BTreeMap::new();
        let mut next_seq = 0usize;

        while let Ok(work) = result_rx.recv() {
            let WorkResult { seq, result } = work;
            let rows = result?;
            if seq == next_seq {
                write_msgpack_rows(&mut writer, &rows)?;
                writer.flush().ok();
                next_seq += 1;
                while let Some(next_rows) = pending.remove(&next_seq) {
                    write_msgpack_rows(&mut writer, &next_rows)?;
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
    let mut reader = BufReader::with_capacity(args.buf_size, pre);
    let batch_size = args.batch_size.max(1);
    let mut batch_rows: Vec<Vec<Value>> = Vec::with_capacity(batch_size);

    loop {
        match read_msgpack_row(&mut reader)? {
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

    writer_handle
        .join()
        .expect("writer thread panicked")?;
    Ok(())
}


fn resolve_function_arg_indices(
    schema: &arrow_schema::Schema,
    pre_cfg: &ConfigMessage,
) -> Result<Vec<usize>> {
    let args = &pre_cfg.function_args;
    if !args.is_empty() {
        let mut indices = Vec::with_capacity(args.len());
        for arg in args {
            let name = arg.name.as_str();
            let resolved_idx = schema
                .fields()
                .iter()
                .enumerate()
                .find(|(_, field)| field.name() == name)
                .map(|(idx, _)| idx)
                .with_context(|| format!("functionArg {} not found in Arrow schema", name))?;
            indices.push(resolved_idx);
        }
        return Ok(indices);
    }

    bail!("No functionArgs provided in PRE config");
}

fn resolve_function_result_indices(
    schema: &arrow_schema::Schema,
    pre_cfg: &ConfigMessage,
) -> Result<Option<Vec<usize>>> {
    if pre_cfg.function_results.is_empty() {
        return Ok(None);
    }

    let mut indices = Vec::with_capacity(pre_cfg.function_results.len());
    for result in &pre_cfg.function_results {
        let name = result.output_name.as_str();
        let resolved_idx = schema
            .fields()
            .iter()
            .enumerate()
            .find(|(_, field)| field.name() == name)
            .map(|(idx, _)| idx)
            .with_context(|| format!("functionResult {} not found in Arrow schema", name))?;
        if let Some(output_type) = &result.output_type {
            let expected = base_type(output_type);
            let actual = arrow_base_type(schema.field(resolved_idx).data_type());
            if expected != actual {
                eprintln!(
                    "functionResult type mismatch for {}: config {}, schema {}",
                    name, expected, actual
                );
            }
        }
        indices.push(resolved_idx);
    }

    Ok(Some(indices))
}

struct WorkItem {
    seq: usize,
    rows: Vec<Vec<Value>>,
}

struct WorkResult {
    seq: usize,
    result: Result<Vec<Vec<Value>>>,
}

fn read_msgpack_row<R: Read>(reader: &mut R) -> Result<Option<Vec<Value>>> {
    match msgpack_decode::read_value(reader) {
        Ok(value) => match value {
            Value::Array(values) => Ok(Some(values)),
            other => bail!("expected msgpack array row, got {:?}", other),
        },
        Err(err) => {
            if is_msgpack_eof(&err) {
                Ok(None)
            } else {
                Err(err.into())
            }
        }
    }
}

fn is_msgpack_eof(err: &msgpack_decode::Error) -> bool {
    match err {
        msgpack_decode::Error::InvalidMarkerRead(inner)
        | msgpack_decode::Error::InvalidDataRead(inner) => inner.kind() == ErrorKind::UnexpectedEof,
        _ => false,
    }
}

fn write_msgpack_rows<W: Write>(writer: &mut W, rows: &[Vec<Value>]) -> Result<()> {
    for row in rows {
        write_msgpack_row_values(writer, row)?;
    }
    Ok(())
}

fn write_msgpack_row_values<W: Write>(writer: &mut W, row: &[Value]) -> Result<()> {
    rmp::encode::write_array_len(writer, row.len() as u32)
        .context("write msgpack row header")?;
    for value in row {
        msgpack_encode::write_value(writer, value).context("write msgpack value")?;
    }
    Ok(())
}

fn write_output_rows_streaming<W: Write>(
    writer: &mut W,
    rows: &[Vec<Value>],
    session: &SessionConfig,
    output_columns: &[InputColumn],
) -> Result<()> {
    let row_len = session.output_row_len as u32;
    for (row_idx, source_row) in rows.iter().enumerate() {
        rmp::encode::write_array_len(writer, row_len)
            .context("write msgpack row header")?;
        for slot in &session.output_slots {
            match slot {
                OutputSlotKind::Op => {
                    let value = source_row.get(0).unwrap_or(&Value::Nil);
                    msgpack_encode::write_value(writer, value)?;
                }
                OutputSlotKind::RowId => {
                    let value = source_row.get(1).unwrap_or(&Value::Nil);
                    msgpack_encode::write_value(writer, value)?;
                }
                OutputSlotKind::InputPos(pre_pos) => {
                    let value = source_row.get(*pre_pos).unwrap_or(&Value::Nil);
                    msgpack_encode::write_value(writer, value)?;
                }
                OutputSlotKind::Output(output_idx) => {
                    let value = output_column_to_value(
                        &output_columns[*output_idx],
                        row_idx,
                        session
                            .output_types
                            .get(*output_idx)
                            .unwrap_or(&FieldType::String),
                    )?;
                    msgpack_encode::write_value(writer, &value)?;
                }
                OutputSlotKind::Nil => {
                    msgpack_encode::write_value(writer, &Value::Nil)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_row_len(row: &[Value], expected_min: usize) -> Result<()> {
    if row.len() < expected_min {
        bail!(
            "msgpack row has {} values but expected at least {}",
            row.len(),
            expected_min
        );
    }
    Ok(())
}

fn apply_udf_to_rows(
    rows: &[Vec<Value>],
    session: &SessionConfig,
    udf: &UdfHandle,
    method: &str,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) -> Result<Vec<Vec<Value>>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    let input_columns = build_input_columns(rows, &session.arg_positions, &session.arg_types)?;
    let output_columns = call_udf_to_columns(udf, method, &input_columns, &session.output_names)?;

    let mut out_rows = Vec::with_capacity(rows.len());
    for row_idx in 0..rows.len() {
        let source_row = rows
            .get(row_idx)
            .context("missing source row")?;

        let mut out_row = if session.passthrough_identity
            && source_row.len() == session.output_row_len
        {
            source_row.clone()
        } else {
            let mut row = vec![Value::Nil; session.output_row_len];
            if let Some(op) = source_row.get(0) {
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
                            let row_id = row.get(1).cloned().unwrap_or(Value::Nil);
                            row[source.pos] = row_id;
                        }
                    }
                    PostFieldSourceKind::InputPos(pre_pos) => {
                        if source.pos < row.len() {
                            row[source.pos] = source_row
                                .get(pre_pos)
                                .cloned()
                                .unwrap_or(Value::Nil);
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
            let value = output_column_to_value(
                &output_columns[idx],
                row_idx,
                session
                    .output_types
                    .get(idx)
                    .unwrap_or(&FieldType::String),
            )?;
            out_row[*pos] = value;
        }
        out_rows.push(out_row);
    }

    maybe_print_debug_rows(
        rows,
        session,
        &out_rows,
        debug_sample_rows,
        debug_batches_remaining,
    );

    Ok(out_rows)
}

fn apply_udf_to_rows_stream<W: Write>(
    writer: &mut W,
    rows: &[Vec<Value>],
    session: &SessionConfig,
    udf: &UdfHandle,
    method: &str,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }

    if debug_sample_rows > 0 && *debug_batches_remaining > 0 {
        let out_rows = apply_udf_to_rows(
            rows,
            session,
            udf,
            method,
            debug_sample_rows,
            debug_batches_remaining,
        )?;
        write_msgpack_rows(writer, &out_rows)?;
        return Ok(());
    }

    let input_columns = build_input_columns(rows, &session.arg_positions, &session.arg_types)?;
    let output_columns = call_udf_to_columns(udf, method, &input_columns, &session.output_names)?;

    if output_columns.len() < session.output_positions.len() {
        bail!(
            "UDF returned {} columns but functionResults resolved {} targets",
            output_columns.len(),
            session.output_positions.len()
        );
    }

    write_output_rows_streaming(writer, rows, session, &output_columns)?;
    Ok(())
}

fn build_input_columns(
    rows: &[Vec<Value>],
    arg_positions: &[usize],
    arg_types: &[FieldType],
) -> Result<Vec<InputColumn>> {
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

fn build_input_column(
    rows: &[Vec<Value>],
    index: usize,
    target_type: &FieldType,
) -> Result<InputColumn> {
    match target_type {
        FieldType::String | FieldType::Unknown(_) => {
            let mut values = Vec::with_capacity(rows.len());
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                if matches!(value, Value::Nil) {
                    values.push(None);
                } else {
                    values.push(Some(value_to_string(value)?));
                }
            }
            Ok(InputColumn::String(values))
        }
        FieldType::Boolean => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                let prev_len = values.len();
                if matches!(value, Value::Nil) {
                    values.push(false);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(value_to_bool(value)?);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
            Ok(InputColumn::Bool { values, is_null: nulls })
        }
        FieldType::Int64 | FieldType::Timestamp { .. } | FieldType::Date => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                let prev_len = values.len();
                if matches!(value, Value::Nil) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    let parsed = match target_type {
                        FieldType::Timestamp { unit } => value_to_timestamp_millis(value, unit)?,
                        FieldType::Date => value_to_date_millis(value)?,
                        _ => value_to_i64(value)?,
                    };
                    values.push(parsed);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
            Ok(InputColumn::I64 { values, is_null: nulls })
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                let prev_len = values.len();
                if matches!(value, Value::Nil) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    let parsed = value_to_i64(value)?;
                    let casted = i32::try_from(parsed)
                        .map_err(|_| anyhow::anyhow!("value {} overflows i32", parsed))?;
                    values.push(casted);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
            Ok(InputColumn::I32 { values, is_null: nulls })
        }
        FieldType::Float64 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                let prev_len = values.len();
                if matches!(value, Value::Nil) {
                    values.push(0.0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(value_to_f64(value)?);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
            Ok(InputColumn::F64 { values, is_null: nulls })
        }
        FieldType::Float32 => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                let prev_len = values.len();
                if matches!(value, Value::Nil) {
                    values.push(0.0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(value_to_f64(value)? as f32);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
            Ok(InputColumn::F32 { values, is_null: nulls })
        }
        FieldType::Decimal { scale, .. } => {
            let mut values = Vec::with_capacity(rows.len());
            let mut nulls: Option<Vec<bool>> = None;
            for row in rows {
                let value = row.get(index).unwrap_or(&Value::Nil);
                let prev_len = values.len();
                if matches!(value, Value::Nil) {
                    values.push(0);
                    push_null(&mut nulls, prev_len, true);
                } else {
                    values.push(value_to_decimal_i128(value, *scale)?);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
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

fn input_column_to_value(
    column: &InputColumn,
    row: usize,
    source_type: &FieldType,
    target_type: &FieldType,
) -> Result<Value> {
    if column_is_null(column, row) {
        return Ok(Value::Nil);
    }

    match target_type {
        FieldType::String | FieldType::Unknown(_) => {
            let s = input_column_to_string(column, row, source_type)?;
            Ok(Value::String(Utf8String::from(s)))
        }
        FieldType::Boolean => Ok(Value::Boolean(input_column_to_bool(
            column, row, source_type,
        )?)),
        FieldType::Int64 => {
            let v = input_column_to_i64(column, row, source_type)?;
            Ok(Value::Integer(v.into()))
        }
        FieldType::Int32 | FieldType::Int16 | FieldType::Int8 => {
            let v = input_column_to_i64(column, row, source_type)?;
            let casted = match target_type {
                FieldType::Int32 => i32::try_from(v).map(|v| v as i64),
                FieldType::Int16 => i16::try_from(v).map(|v| v as i64),
                FieldType::Int8 => i8::try_from(v).map(|v| v as i64),
                _ => Ok(v),
            }
            .map_err(|_| anyhow::anyhow!("value {} overflows target int", v))?;
            Ok(Value::Integer(casted.into()))
        }
        FieldType::Float64 => {
            let v = input_column_to_f64(column, row, source_type)?;
            Ok(Value::F64(v))
        }
        FieldType::Float32 => {
            let v = input_column_to_f64(column, row, source_type)?;
            Ok(Value::F32(v as f32))
        }
        FieldType::Decimal { scale, .. } => {
            let mut value = input_column_to_decimal(column, row, source_type)?;
            let source_scale = source_type.decimal_scale().unwrap_or(0);
            if source_scale != *scale {
                value = convert_decimal_scale(value, source_scale, *scale)?;
            }
            let s = decimal_to_string(value, *scale);
            Ok(Value::String(Utf8String::from(s)))
        }
        FieldType::Timestamp { unit } => {
            let millis = input_column_to_i64(column, row, source_type)?;
            let value = timestamp_from_millis(millis, unit)?;
            Ok(Value::Integer(value.into()))
        }
        FieldType::Date => {
            let millis = input_column_to_i64(column, row, source_type)?;
            Ok(Value::Integer(millis.into()))
        }
    }
}

fn output_column_to_value(
    column: &InputColumn,
    row: usize,
    output_type: &FieldType,
) -> Result<Value> {
    let source_type = infer_source_type(column, output_type);
    input_column_to_value(column, row, &source_type, output_type)
}

fn infer_source_type(column: &InputColumn, output_type: &FieldType) -> FieldType {
    match column {
        InputColumn::String(_) => FieldType::String,
        InputColumn::Bool { .. } => FieldType::Boolean,
        InputColumn::I64 { .. } => FieldType::Int64,
        InputColumn::I32 { .. } => FieldType::Int32,
        InputColumn::F64 { .. } => FieldType::Float64,
        InputColumn::F32 { .. } => FieldType::Float32,
        InputColumn::Decimal128 { .. } => {
            let scale = output_type.decimal_scale().unwrap_or(0);
            FieldType::Decimal {
                precision: None,
                scale,
            }
        }
    }
}

fn column_is_null(column: &InputColumn, row: usize) -> bool {
    match column {
        InputColumn::String(values) => values
            .get(row)
            .map(|v| v.is_none())
            .unwrap_or(true),
        InputColumn::I64 { is_null, .. }
        | InputColumn::I32 { is_null, .. }
        | InputColumn::F64 { is_null, .. }
        | InputColumn::F32 { is_null, .. }
        | InputColumn::Bool { is_null, .. }
        | InputColumn::Decimal128 { is_null, .. } => is_null_at(is_null.as_deref(), row),
    }
}

fn input_column_to_string(
    column: &InputColumn,
    row: usize,
    source_type: &FieldType,
) -> Result<String> {
    match column {
        InputColumn::String(values) => values
            .get(row)
            .and_then(|v| v.clone())
            .ok_or_else(|| anyhow::anyhow!("null string value")),
        InputColumn::Bool { values, .. } => Ok(values
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default()),
        InputColumn::I64 { values, .. } => Ok(values
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default()),
        InputColumn::I32 { values, .. } => Ok(values
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default()),
        InputColumn::F64 { values, .. } => Ok(values
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default()),
        InputColumn::F32 { values, .. } => Ok(values
            .get(row)
            .map(|v| v.to_string())
            .unwrap_or_default()),
        InputColumn::Decimal128 { values, .. } => {
            let scale = source_type.decimal_scale().unwrap_or(0);
            let value = values.get(row).copied().unwrap_or(0);
            Ok(decimal_to_string(value, scale))
        }
    }
}

fn input_column_to_bool(
    column: &InputColumn,
    row: usize,
    source_type: &FieldType,
) -> Result<bool> {
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
            .map(|v| parse_bool_string(v))
            .unwrap_or(Ok(false)),
    }
}

fn input_column_to_i64(
    column: &InputColumn,
    row: usize,
    source_type: &FieldType,
) -> Result<i64> {
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

fn input_column_to_f64(
    column: &InputColumn,
    row: usize,
    source_type: &FieldType,
) -> Result<f64> {
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

fn input_column_to_decimal(
    column: &InputColumn,
    row: usize,
    source_type: &FieldType,
) -> Result<i128> {
    match column {
        InputColumn::Decimal128 { values, .. } => Ok(values.get(row).copied().unwrap_or(0)),
        InputColumn::I64 { values, .. } => {
            let factor = pow10_i128(source_type.decimal_scale().unwrap_or(0))?;
            let base = *values.get(row).unwrap_or(&0) as i128;
            base.checked_mul(factor)
                .context("decimal scale overflow")
        }
        InputColumn::I32 { values, .. } => {
            let factor = pow10_i128(source_type.decimal_scale().unwrap_or(0))?;
            let base = *values.get(row).unwrap_or(&0) as i128;
            base.checked_mul(factor)
                .context("decimal scale overflow")
        }
        InputColumn::F64 { values, .. } => {
            let value = values.get(row).copied().unwrap_or(0.0);
            parse_decimal_to_i128(&value.to_string(), source_type.decimal_scale().unwrap_or(0))
        }
        InputColumn::F32 { values, .. } => {
            let value = values.get(row).copied().unwrap_or(0.0);
            parse_decimal_to_i128(&value.to_string(), source_type.decimal_scale().unwrap_or(0))
        }
        InputColumn::Bool { values, .. } => {
            let value = if *values.get(row).unwrap_or(&false) { 1 } else { 0 };
            let factor = pow10_i128(source_type.decimal_scale().unwrap_or(0))?;
            (value as i128)
                .checked_mul(factor)
                .context("decimal scale overflow")
        }
        InputColumn::String(values) => {
            let value = values
                .get(row)
                .and_then(|v| v.as_deref())
                .ok_or_else(|| anyhow::anyhow!("null string value"))?;
            parse_decimal_to_i128(value, source_type.decimal_scale().unwrap_or(0))
        }
    }
}

fn maybe_print_debug_rows(
    input_rows: &[Vec<Value>],
    session: &SessionConfig,
    output_rows: &[Vec<Value>],
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) {
    if debug_sample_rows == 0 || *debug_batches_remaining == 0 || input_rows.is_empty() {
        return;
    }

    let sample_rows = debug_sample_rows.min(input_rows.len());
    println!(
        "Debug sample ({} of {} rows):",
        sample_rows,
        input_rows.len()
    );
    for row_idx in 0..sample_rows {
        let row = &input_rows[row_idx];
        let mut input_parts = Vec::with_capacity(session.arg_names.len());
        for (arg_pos, name) in session.arg_names.iter().enumerate() {
            let idx = session.arg_positions.get(arg_pos).copied().unwrap_or(0);
            let value = row.get(idx).unwrap_or(&Value::Nil);
            input_parts.push(format!("{}={}", name, debug_value(value)));
        }

        let mut output_parts = Vec::new();
        if !session.output_positions.is_empty() {
            for (idx, name) in session.output_names.iter().enumerate() {
                let pos = session.output_positions.get(idx).copied().unwrap_or(0);
                let value = output_rows
                    .get(row_idx)
                    .and_then(|row| row.get(pos))
                    .unwrap_or(&Value::Nil);
                output_parts.push(format!("{}={}", name, debug_value(value)));
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

fn debug_value(value: &Value) -> String {
    match value {
        Value::Nil => "<null>".to_string(),
        Value::Boolean(v) => v.to_string(),
        Value::Integer(v) => v.to_string(),
        Value::F32(v) => v.to_string(),
        Value::F64(v) => v.to_string(),
        Value::String(s) => s.as_str().unwrap_or("").to_string(),
        Value::Binary(b) => format!("{:?}", b),
        Value::Array(_) => "<array>".to_string(),
        Value::Map(_) => "<map>".to_string(),
        Value::Ext(_, _) => "<ext>".to_string(),
    }
}

fn value_to_string(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.as_str().unwrap_or("").to_string()),
        Value::Binary(b) => Ok(String::from_utf8_lossy(b).to_string()),
        Value::Integer(v) => Ok(v.to_string()),
        Value::Boolean(v) => Ok(v.to_string()),
        Value::F32(v) => Ok(v.to_string()),
        Value::F64(v) => Ok(v.to_string()),
        Value::Nil => bail!("value is null"),
        other => Ok(format!("{other:?}")),
    }
}

fn value_to_bool(value: &Value) -> Result<bool> {
    match value {
        Value::Boolean(v) => Ok(*v),
        Value::Integer(v) => Ok(v.as_i64().unwrap_or(0) != 0),
        Value::F32(v) => Ok(*v != 0.0),
        Value::F64(v) => Ok(*v != 0.0),
        Value::String(s) => parse_bool_string(s.as_str().unwrap_or("")),
        Value::Binary(b) => parse_bool_string(&String::from_utf8_lossy(b)),
        Value::Nil => bail!("value is null"),
        _ => bail!("unsupported boolean value {:?}", value),
    }
}

fn parse_bool_string(value: &str) -> Result<bool> {
    let normalized = value.trim().to_lowercase();
    match normalized.as_str() {
        "true" | "t" | "1" | "yes" | "y" => Ok(true),
        "false" | "f" | "0" | "no" | "n" => Ok(false),
        _ => bail!("failed to parse boolean from {}", value),
    }
}

fn value_to_i64(value: &Value) -> Result<i64> {
    match value {
        Value::Integer(v) => {
            if let Some(signed) = v.as_i64() {
                return Ok(signed);
            }
            if let Some(unsigned) = v.as_u64() {
                if unsigned > i64::MAX as u64 {
                    bail!("integer out of range for i64");
                }
                return Ok(unsigned as i64);
            }
            bail!("integer out of range for i64")
        }
        Value::Boolean(v) => Ok(if *v { 1 } else { 0 }),
        Value::F32(v) => Ok(*v as i64),
        Value::F64(v) => Ok(*v as i64),
        Value::String(s) => parse_with(s.as_str().unwrap_or(""), "i64"),
        Value::Binary(b) => parse_with(&String::from_utf8_lossy(b), "i64"),
        Value::Nil => bail!("value is null"),
        _ => bail!("unsupported integer value {:?}", value),
    }
}

fn value_to_f64(value: &Value) -> Result<f64> {
    match value {
        Value::Integer(v) => v
            .as_i64()
            .map(|v| v as f64)
            .or_else(|| v.as_u64().map(|v| v as f64))
            .context("integer out of range for f64"),
        Value::Boolean(v) => Ok(if *v { 1.0 } else { 0.0 }),
        Value::F32(v) => Ok(*v as f64),
        Value::F64(v) => Ok(*v),
        Value::String(s) => parse_with(s.as_str().unwrap_or(""), "f64"),
        Value::Binary(b) => parse_with(&String::from_utf8_lossy(b), "f64"),
        Value::Nil => bail!("value is null"),
        _ => bail!("unsupported float value {:?}", value),
    }
}

fn value_to_decimal_i128(value: &Value, scale: i8) -> Result<i128> {
    match value {
        Value::String(s) => parse_decimal_to_i128(s.as_str().unwrap_or(""), scale),
        Value::Binary(b) => parse_decimal_to_i128(&String::from_utf8_lossy(b), scale),
        Value::Nil => bail!("value is null"),
        _ => bail!("decimal must be encoded as string in msgpack"),
    }
}

fn value_to_timestamp_millis(value: &Value, unit: &TimeUnit) -> Result<i64> {
    match value {
        Value::Integer(_) | Value::Boolean(_) | Value::F32(_) | Value::F64(_) => {
            let raw = value_to_i64(value)?;
            Ok(timestamp_to_millis(raw, unit)?)
        }
        Value::String(s) => {
            let raw = timestamp_string_to_value(s.as_str().unwrap_or(""), unit)?;
            Ok(timestamp_to_millis(raw, unit)?)
        }
        Value::Binary(b) => {
            let raw = timestamp_string_to_value(&String::from_utf8_lossy(b), unit)?;
            Ok(timestamp_to_millis(raw, unit)?)
        }
        Value::Nil => bail!("value is null"),
        _ => bail!("unsupported timestamp value {:?}", value),
    }
}

fn value_to_date_millis(value: &Value) -> Result<i64> {
    match value {
        Value::Integer(_) | Value::Boolean(_) | Value::F32(_) | Value::F64(_) => value_to_i64(value),
        Value::String(s) => parse_date_string(s.as_str().unwrap_or("")),
        Value::Binary(b) => parse_date_string(&String::from_utf8_lossy(b)),
        Value::Nil => bail!("value is null"),
        _ => bail!("unsupported date value {:?}", value),
    }
}

fn parse_date_string(value: &str) -> Result<i64> {
    let date = NaiveDate::parse_from_str(value, "%Y-%m-%d")
        .with_context(|| format!("failed to parse date from {value}"))?;
    let dt = date
        .and_hms_opt(0, 0, 0)
        .context("invalid date")?;
    Ok(DateTime::<Utc>::from_utc(dt, Utc).timestamp_millis())
}

fn timestamp_to_millis(value: i64, unit: &TimeUnit) -> Result<i64> {
    match unit {
        TimeUnit::Second => value
            .checked_mul(1_000)
            .context("timestamp second overflow"),
        TimeUnit::Millisecond => Ok(value),
        TimeUnit::Microsecond => Ok(value / 1_000),
        TimeUnit::Nanosecond => Ok(value / 1_000_000),
    }
}

fn timestamp_from_millis(value: i64, unit: &TimeUnit) -> Result<i64> {
    match unit {
        TimeUnit::Second => Ok(value / 1_000),
        TimeUnit::Millisecond => Ok(value),
        TimeUnit::Microsecond => value
            .checked_mul(1_000)
            .context("timestamp microsecond overflow"),
        TimeUnit::Nanosecond => value
            .checked_mul(1_000_000)
            .context("timestamp nanosecond overflow"),
    }
}

fn pow10_i128(scale: i8) -> Result<i128> {
    if scale <= 0 {
        return Ok(1);
    }
    let mut value: i128 = 1;
    for _ in 0..scale {
        value = value
            .checked_mul(10)
            .context("decimal scale overflow")?;
    }
    Ok(value)
}

fn convert_decimal_scale(value: i128, source_scale: i8, target_scale: i8) -> Result<i128> {
    if source_scale == target_scale {
        return Ok(value);
    }
    if source_scale < target_scale {
        let factor = pow10_i128(target_scale - source_scale)?;
        return value
            .checked_mul(factor)
            .context("decimal scale overflow");
    }
    let factor = pow10_i128(source_scale - target_scale)?;
    Ok(value / factor)
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
#[derive(Clone, Copy, Debug)]
struct OutputTarget {
    target_idx: usize,
    source_idx: usize,
}

fn apply_udf_to_batch(
    batch: &arrow_array::RecordBatch,
    schema: &arrow_schema::Schema,
    arg_indices: &[usize],
    arg_names: &[String],
    result_targets: Option<&[usize]>,
    result_names: Option<&[String]>,
    udf: &UdfHandle,
    method: &str,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) -> Result<arrow_array::RecordBatch> {
    let row_count = batch.num_rows();
    let mut input_columns = Vec::with_capacity(arg_indices.len());
    let mut debug_arg_columns: Option<Vec<Vec<Option<String>>>> =
        if debug_sample_rows > 0 && *debug_batches_remaining > 0 {
            Some(Vec::with_capacity(arg_indices.len()))
        } else {
            None
        };
    for &idx in arg_indices {
        let array = batch.column(idx);
        input_columns.push(column_to_input(array)?);
        if let Some(debug_cols) = debug_arg_columns.as_mut() {
            debug_cols.push(column_to_strings(array)?);
        }
    }

    let output_names = result_names.unwrap_or(&[]);
    let output_columns = call_udf_to_columns(udf, method, &input_columns, output_names)?;
    let output_targets = resolve_output_targets(
        batch,
        arg_indices,
        result_targets,
        result_names,
        arg_names,
        output_columns.len(),
    )?;

    if let Some(debug_args) = debug_arg_columns.as_ref() {
        let mut debug_outputs = Vec::with_capacity(output_targets.len());
        for target in &output_targets {
            if target.source_idx >= output_columns.len() {
                debug_outputs.push(Vec::new());
                continue;
            }
            let field = schema.field(target.target_idx);
            debug_outputs.push(output_column_to_debug_strings(
                &output_columns[target.source_idx],
                field.data_type(),
            )?);
        }
        maybe_print_debug_sample(
            schema,
            arg_names,
            debug_args,
            &output_targets,
            &debug_outputs,
            row_count,
            debug_sample_rows,
            debug_batches_remaining,
        );
    }

    let new_columns = build_output_columns(batch, schema, &output_targets, &output_columns)?;
    arrow_array::RecordBatch::try_new(batch.schema(), new_columns)
        .context("build output record batch")
}

fn call_udf_to_columns(
    udf: &UdfHandle,
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

fn resolve_output_targets(
    batch: &arrow_array::RecordBatch,
    arg_indices: &[usize],
    result_targets: Option<&[usize]>,
    result_names: Option<&[String]>,
    arg_names: &[String],
    output_arity: usize,
) -> Result<Vec<OutputTarget>> {
    if let Some(targets) = result_targets {
        if targets.is_empty() {
            bail!("functionResults was provided but no targets were resolved");
        }
        if output_arity == targets.len() {
            let mapped = targets
                .iter()
                .enumerate()
                .map(|(source_idx, &target_idx)| OutputTarget {
                    target_idx,
                    source_idx,
                })
                .collect();
            return Ok(mapped);
        }
        let field_count = batch.num_columns();
        if output_arity == field_count {
            let mapped = targets
                .iter()
                .map(|&target_idx| OutputTarget {
                    target_idx,
                    source_idx: target_idx,
                })
                .collect();
            return Ok(mapped);
        }
        if output_arity == arg_indices.len() {
            let names = result_names
                .filter(|names| names.len() == targets.len())
                .context("functionResults names missing for name-based output mapping")?;
            let mut mapped = Vec::with_capacity(targets.len());
            for (idx, &target_idx) in targets.iter().enumerate() {
                let name = names
                    .get(idx)
                    .map(|value| value.as_str())
                    .unwrap_or("<unknown>");
                let source_idx = arg_names
                    .iter()
                    .position(|arg| arg == name)
                    .with_context(|| {
                        format!(
                            "functionResult {} not found in functionArgs for name-based mapping",
                            name
                        )
                    })?;
                mapped.push(OutputTarget {
                    target_idx,
                    source_idx,
                });
            }
            return Ok(mapped);
        }
        bail!(
            "UDF returned {} columns but functionResults resolved {} targets",
            output_arity,
            targets.len()
        );
    }

    let field_count = batch.num_columns();
    if output_arity == field_count {
        return Ok((0..field_count)
            .map(|idx| OutputTarget {
                target_idx: idx,
                source_idx: idx,
            })
            .collect());
    }
    if output_arity == arg_indices.len() {
        return Ok(arg_indices
            .iter()
            .enumerate()
            .map(|(source_idx, &target_idx)| OutputTarget {
                target_idx,
                source_idx,
            })
            .collect());
    }
    if output_arity == 1 {
        return Ok(vec![OutputTarget {
            target_idx: arg_indices[0],
            source_idx: 0,
        }]);
    }
    bail!(
        "UDF returned {} columns but expected 1, {} (args), or {} (schema)",
        output_arity,
        arg_indices.len(),
        field_count
    );
}

fn build_output_columns(
    batch: &arrow_array::RecordBatch,
    schema: &arrow_schema::Schema,
    output_targets: &[OutputTarget],
    output_columns: &[InputColumn],
) -> Result<Vec<ArrayRef>> {
    let row_count = batch.num_rows();
    let mut new_columns: Vec<ArrayRef> = batch.columns().to_vec();

    for target in output_targets {
        if target.source_idx >= output_columns.len() {
            bail!(
                "output source index {} is out of range for {} columns",
                target.source_idx,
                output_columns.len()
            );
        }
        let field = schema.field(target.target_idx);
        let column = &output_columns[target.source_idx];
        if column.len() != row_count {
            bail!(
                "output column {} has {} rows but batch has {}",
                target.source_idx,
                column.len(),
                row_count
            );
        }
        let array = output_column_to_array(column, field.data_type())
            .with_context(|| format!("convert output for field {}", field.name()))?;
        new_columns[target.target_idx] = array;
    }

    Ok(new_columns)
}

fn maybe_print_debug_sample(
    schema: &arrow_schema::Schema,
    arg_names: &[String],
    arg_columns: &[Vec<Option<String>>],
    output_targets: &[OutputTarget],
    output_columns: &[Vec<Option<String>>],
    row_count: usize,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) {
    if debug_sample_rows == 0
        || *debug_batches_remaining == 0
        || row_count == 0
        || arg_columns.is_empty()
        || output_columns.is_empty()
    {
        return;
    }

    let sample_rows = debug_sample_rows.min(row_count);
    println!("Debug sample ({} of {} rows):", sample_rows, row_count);
    for row_idx in 0..sample_rows {
        let mut input_parts = Vec::with_capacity(arg_columns.len());
        for (arg_idx, column) in arg_columns.iter().enumerate() {
            let name = arg_names
                .get(arg_idx)
                .map(|value| value.as_str())
                .unwrap_or("<arg>");
            let value = column
                .get(row_idx)
                .and_then(|v| v.as_deref())
                .unwrap_or("<null>");
            input_parts.push(format!("{name}={value}"));
        }

        let mut output_parts = Vec::with_capacity(output_targets.len());
        for target in output_targets {
            if target.source_idx >= output_columns.len() {
                continue;
            }
            let column = &output_columns[target.source_idx];
            let name = schema.field(target.target_idx).name();
            let value = column
                .get(row_idx)
                .and_then(|v| v.as_deref())
                .unwrap_or("<null>");
            output_parts.push(format!("{name}={value}"));
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

fn column_to_input(array: &ArrayRef) -> Result<InputColumn> {
    match array.data_type() {
        DataType::Int64 => column_to_i64(array),
        DataType::Int32 => column_to_i32(array),
        DataType::Int16 => column_to_i32(array),
        DataType::Int8 => column_to_i32(array),
        DataType::Float64 => column_to_f64(array),
        DataType::Float32 => column_to_f32(array),
        DataType::Boolean => column_to_bool(array),
        DataType::Timestamp(unit, _) => column_to_timestamp_millis(array, unit),
        DataType::Decimal128(_, _) => column_to_decimal128(array),
        DataType::Utf8 | DataType::LargeUtf8 => Ok(InputColumn::String(column_to_strings(array)?)),
        _ => Ok(InputColumn::String(column_to_strings(array)?)),
    }
}

fn column_to_i64(array: &ArrayRef) -> Result<InputColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<Int64Array>()
        .context("downcast Int64")?;
    let len = arr.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if arr.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };
    for i in 0..len {
        if arr.is_null(i) {
            values.push(0);
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(true);
            }
        } else {
            values.push(arr.value(i));
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(false);
            }
        }
    }
    Ok(InputColumn::I64 {
        values,
        is_null: nulls,
    })
}

fn column_to_i32(array: &ArrayRef) -> Result<InputColumn> {
    let len = array.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if array.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };

    match array.data_type() {
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("downcast Int32")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i));
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
        DataType::Int16 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int16Array>()
                .context("downcast Int16")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i) as i32);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
        DataType::Int8 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int8Array>()
                .context("downcast Int8")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i) as i32);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
        _ => {
            bail!("column_to_i32 called with unsupported type {}", array.data_type());
        }
    }

    Ok(InputColumn::I32 {
        values,
        is_null: nulls,
    })
}

fn column_to_f64(array: &ArrayRef) -> Result<InputColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .context("downcast Float64")?;
    let len = arr.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if arr.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };
    for i in 0..len {
        if arr.is_null(i) {
            values.push(0.0);
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(true);
            }
        } else {
            values.push(arr.value(i));
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(false);
            }
        }
    }
    Ok(InputColumn::F64 {
        values,
        is_null: nulls,
    })
}

fn column_to_f32(array: &ArrayRef) -> Result<InputColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<Float32Array>()
        .context("downcast Float32")?;
    let len = arr.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if arr.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };
    for i in 0..len {
        if arr.is_null(i) {
            values.push(0.0);
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(true);
            }
        } else {
            values.push(arr.value(i));
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(false);
            }
        }
    }
    Ok(InputColumn::F32 {
        values,
        is_null: nulls,
    })
}

fn column_to_bool(array: &ArrayRef) -> Result<InputColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .context("downcast Boolean")?;
    let len = arr.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if arr.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };
    for i in 0..len {
        if arr.is_null(i) {
            values.push(false);
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(true);
            }
        } else {
            values.push(arr.value(i));
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(false);
            }
        }
    }
    Ok(InputColumn::Bool {
        values,
        is_null: nulls,
    })
}

fn column_to_timestamp_millis(array: &ArrayRef, unit: &TimeUnit) -> Result<InputColumn> {
    let len = array.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if array.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };

    match unit {
        TimeUnit::Second => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampSecondArray>()
                .context("downcast TimestampSecond")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i) * 1_000);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
        TimeUnit::Millisecond => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .context("downcast TimestampMillisecond")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i));
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
        TimeUnit::Microsecond => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .context("downcast TimestampMicrosecond")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i) / 1_000);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
        TimeUnit::Nanosecond => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .context("downcast TimestampNanosecond")?;
            for i in 0..len {
                if arr.is_null(i) {
                    values.push(0);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(true);
                    }
                } else {
                    values.push(arr.value(i) / 1_000_000);
                    if let Some(nulls) = nulls.as_mut() {
                        nulls.push(false);
                    }
                }
            }
        }
    }

    Ok(InputColumn::I64 {
        values,
        is_null: nulls,
    })
}

fn column_to_decimal128(array: &ArrayRef) -> Result<InputColumn> {
    let arr = array
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .context("downcast Decimal128")?;
    let len = arr.len();
    let mut values = Vec::with_capacity(len);
    let mut nulls = if arr.null_count() > 0 {
        Some(Vec::with_capacity(len))
    } else {
        None
    };
    for i in 0..len {
        if arr.is_null(i) {
            values.push(0);
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(true);
            }
        } else {
            values.push(arr.value(i));
            if let Some(nulls) = nulls.as_mut() {
                nulls.push(false);
            }
        }
    }
    Ok(InputColumn::Decimal128 {
        values,
        is_null: nulls,
    })
}

fn output_column_to_array(column: &InputColumn, data_type: &DataType) -> Result<ArrayRef> {
    match column {
        InputColumn::String(values) => strings_to_array(values, data_type),
        InputColumn::Bool { values, is_null } => match data_type {
            DataType::Boolean => build_bool_array(values, is_null.as_deref()),
            _ => strings_to_array(&bools_to_strings(values, is_null.as_deref()), data_type),
        },
        InputColumn::Decimal128 { values, is_null } => match data_type {
            DataType::Decimal128(precision, scale) => {
                build_decimal_array(values, is_null.as_deref(), *precision, *scale)
            }
            _ => strings_to_array(
                &decimal_to_strings(values, is_null.as_deref(), 0),
                data_type,
            ),
        },
        InputColumn::F64 { values, is_null } => match data_type {
            DataType::Float64 => build_f64_array(values, is_null.as_deref()),
            DataType::Float32 => build_f32_from_f64(values, is_null.as_deref()),
            _ => strings_to_array(&f64_to_strings(values, is_null.as_deref()), data_type),
        },
        InputColumn::F32 { values, is_null } => match data_type {
            DataType::Float32 => build_f32_array(values, is_null.as_deref()),
            DataType::Float64 => build_f64_from_f32(values, is_null.as_deref()),
            _ => strings_to_array(&f32_to_strings(values, is_null.as_deref()), data_type),
        },
        InputColumn::I32 { values, is_null } => match data_type {
            DataType::Int32 => build_i32_array(values, is_null.as_deref()),
            DataType::Int64 => build_i64_from_i32(values, is_null.as_deref()),
            DataType::Int16 => build_i16_from_i32(values, is_null.as_deref()),
            DataType::Int8 => build_i8_from_i32(values, is_null.as_deref()),
            _ => strings_to_array(&i32_to_strings(values, is_null.as_deref()), data_type),
        },
        InputColumn::I64 { values, is_null } => match data_type {
            DataType::Int64 => build_i64_array(values, is_null.as_deref()),
            DataType::Int32 => build_i32_from_i64(values, is_null.as_deref()),
            DataType::Int16 => build_i16_from_i64(values, is_null.as_deref()),
            DataType::Int8 => build_i8_from_i64(values, is_null.as_deref()),
            DataType::Timestamp(unit, _) => {
                build_timestamp_array(values, is_null.as_deref(), unit)
            }
            DataType::Date64 => build_date64_array(values, is_null.as_deref()),
            DataType::Date32 => build_date32_array(values, is_null.as_deref()),
            _ => strings_to_array(&i64_to_strings(values, is_null.as_deref()), data_type),
        },
    }
}

fn output_column_to_debug_strings(
    column: &InputColumn,
    data_type: &DataType,
) -> Result<Vec<Option<String>>> {
    match column {
        InputColumn::String(values) => Ok(values.clone()),
        InputColumn::Bool { values, is_null } => Ok(bools_to_strings(values, is_null.as_deref())),
        InputColumn::Decimal128 { values, is_null } => match data_type {
            DataType::Decimal128(_, scale) => Ok(decimal_to_strings(values, is_null.as_deref(), *scale)),
            _ => Ok(decimal_to_strings(values, is_null.as_deref(), 0)),
        },
        InputColumn::F64 { values, is_null } => Ok(f64_to_strings(values, is_null.as_deref())),
        InputColumn::F32 { values, is_null } => Ok(f32_to_strings(values, is_null.as_deref())),
        InputColumn::I32 { values, is_null } => Ok(i32_to_strings(values, is_null.as_deref())),
        InputColumn::I64 { values, is_null } => match data_type {
            DataType::Timestamp(unit, _) => {
                i64_timestamp_to_strings(values, is_null.as_deref(), unit)
            }
            _ => Ok(i64_to_strings(values, is_null.as_deref())),
        },
    }
}

fn build_i64_array(values: &[i64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Int64Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i32_array(values: &[i32], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Int32Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i16_from_i32(values: &[i32], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::Int16Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as i16);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i8_from_i32(values: &[i32], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::Int8Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as i8);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i64_from_i32(values: &[i32], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Int64Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as i64);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i32_from_i64(values: &[i64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Int32Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as i32);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i16_from_i64(values: &[i64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::Int16Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as i16);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_i8_from_i64(values: &[i64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::Int8Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as i8);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_f64_array(values: &[f64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Float64Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_f32_array(values: &[f32], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Float32Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_f32_from_f64(values: &[f64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Float32Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as f32);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_f64_from_f32(values: &[f32], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = Float64Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value as f64);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_bool_array(values: &[bool], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::BooleanBuilder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_timestamp_array(
    values: &[i64],
    nulls: Option<&[bool]>,
    unit: &TimeUnit,
) -> Result<ArrayRef> {
    match unit {
        TimeUnit::Second => {
            let mut builder = TimestampSecondBuilder::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                if is_null_at(nulls, idx) {
                    builder.append_null();
                } else {
                    builder.append_value(value / 1_000);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        TimeUnit::Millisecond => {
            let mut builder = TimestampMillisecondBuilder::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                if is_null_at(nulls, idx) {
                    builder.append_null();
                } else {
                    builder.append_value(*value);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        TimeUnit::Microsecond => {
            let mut builder = TimestampMicrosecondBuilder::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                if is_null_at(nulls, idx) {
                    builder.append_null();
                } else {
                    let converted = value
                        .checked_mul(1_000)
                        .context("timestamp microsecond overflow")?;
                    builder.append_value(converted);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        TimeUnit::Nanosecond => {
            let mut builder = TimestampNanosecondBuilder::with_capacity(values.len());
            for (idx, value) in values.iter().enumerate() {
                if is_null_at(nulls, idx) {
                    builder.append_null();
                } else {
                    let converted = value
                        .checked_mul(1_000_000)
                        .context("timestamp nanosecond overflow")?;
                    builder.append_value(converted);
                }
            }
            Ok(Arc::new(builder.finish()))
        }
    }
}

fn build_date64_array(values: &[i64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::Date64Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_date32_array(values: &[i64], nulls: Option<&[bool]>) -> Result<ArrayRef> {
    let mut builder = arrow_array::builder::Date32Builder::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value((value / 86_400_000) as i32);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn build_decimal_array(
    values: &[i128],
    nulls: Option<&[bool]>,
    precision: u8,
    scale: i8,
) -> Result<ArrayRef> {
    let mut builder = Decimal128Builder::with_capacity(values.len())
        .with_precision_and_scale(precision, scale)
        .context("configure decimal builder")?;
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            builder.append_null();
        } else {
            builder.append_value(*value);
        }
    }
    Ok(Arc::new(builder.finish()))
}

fn is_null_at(nulls: Option<&[bool]>, idx: usize) -> bool {
    nulls.map(|vals| vals.get(idx).copied().unwrap_or(false))
        .unwrap_or(false)
}

fn i64_to_strings(values: &[i64], nulls: Option<&[bool]>) -> Vec<Option<String>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if is_null_at(nulls, idx) {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect()
}

fn i32_to_strings(values: &[i32], nulls: Option<&[bool]>) -> Vec<Option<String>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if is_null_at(nulls, idx) {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect()
}

fn f64_to_strings(values: &[f64], nulls: Option<&[bool]>) -> Vec<Option<String>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if is_null_at(nulls, idx) {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect()
}

fn f32_to_strings(values: &[f32], nulls: Option<&[bool]>) -> Vec<Option<String>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if is_null_at(nulls, idx) {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect()
}

fn bools_to_strings(values: &[bool], nulls: Option<&[bool]>) -> Vec<Option<String>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if is_null_at(nulls, idx) {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect()
}

fn i64_timestamp_to_strings(
    values: &[i64],
    nulls: Option<&[bool]>,
    unit: &TimeUnit,
) -> Result<Vec<Option<String>>> {
    let mut out = Vec::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        if is_null_at(nulls, idx) {
            out.push(None);
        } else {
            let converted = match unit {
                TimeUnit::Second => value / 1_000,
                TimeUnit::Millisecond => *value,
                TimeUnit::Microsecond => value
                    .checked_mul(1_000)
                    .context("timestamp microsecond overflow")?,
                TimeUnit::Nanosecond => value
                    .checked_mul(1_000_000)
                    .context("timestamp nanosecond overflow")?,
            };
            out.push(Some(timestamp_to_string(converted, unit)?));
        }
    }
    Ok(out)
}

fn decimal_to_strings(
    values: &[i128],
    nulls: Option<&[bool]>,
    scale: i8,
) -> Vec<Option<String>> {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            if is_null_at(nulls, idx) {
                None
            } else {
                Some(decimal_to_string(*value, scale))
            }
        })
        .collect()
}

fn column_to_strings(array: &ArrayRef) -> Result<Vec<Option<String>>> {
    let len = array.len();
    let mut out = Vec::with_capacity(len);
    match array.data_type() {
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<StringArray>()
                .context("downcast Utf8")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::LargeUtf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .context("downcast LargeUtf8")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .context("downcast Int32")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .context("downcast Int64")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::UInt32 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("downcast UInt32")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::UInt64 => {
            let arr = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("downcast UInt64")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("downcast Float32")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .context("downcast Float64")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    out.push(Some(arr.value(i).to_string()));
                }
            }
        }
        DataType::Decimal128(_, scale) => {
            let arr = array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .context("downcast Decimal128")?;
            for i in 0..len {
                if arr.is_null(i) {
                    out.push(None);
                } else {
                    let value = arr.value(i);
                    out.push(Some(decimal_to_string(value, *scale)));
                }
            }
        }
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .context("downcast TimestampSecond")?;
                for i in 0..len {
                    if arr.is_null(i) {
                        out.push(None);
                    } else {
                        out.push(Some(timestamp_to_string(arr.value(i), unit)?));
                    }
                }
            }
            TimeUnit::Millisecond => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .context("downcast TimestampMillisecond")?;
                for i in 0..len {
                    if arr.is_null(i) {
                        out.push(None);
                    } else {
                        out.push(Some(timestamp_to_string(arr.value(i), unit)?));
                    }
                }
            }
            TimeUnit::Microsecond => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .context("downcast TimestampMicrosecond")?;
                for i in 0..len {
                    if arr.is_null(i) {
                        out.push(None);
                    } else {
                        out.push(Some(timestamp_to_string(arr.value(i), unit)?));
                    }
                }
            }
            TimeUnit::Nanosecond => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .context("downcast TimestampNanosecond")?;
                for i in 0..len {
                    if arr.is_null(i) {
                        out.push(None);
                    } else {
                        out.push(Some(timestamp_to_string(arr.value(i), unit)?));
                    }
                }
            }
        },
        other => bail!("unsupported Arrow type for UDF: {}", other),
    }
    Ok(out)
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

fn timestamp_to_string(value: i64, unit: &TimeUnit) -> Result<String> {
    let (scale, frac_digits) = match unit {
        TimeUnit::Second => (1_i64, 0_usize),
        TimeUnit::Millisecond => (1_000_i64, 3_usize),
        TimeUnit::Microsecond => (1_000_000_i64, 6_usize),
        TimeUnit::Nanosecond => (1_000_000_000_i64, 9_usize),
    };
    let seconds = value.div_euclid(scale);
    let remainder = value.rem_euclid(scale);
    let nanos = match unit {
        TimeUnit::Second => 0_u32,
        TimeUnit::Millisecond => (remainder * 1_000_000) as u32,
        TimeUnit::Microsecond => (remainder * 1_000) as u32,
        TimeUnit::Nanosecond => remainder as u32,
    };

    let dt = DateTime::<Utc>::from_timestamp(seconds, nanos)
        .with_context(|| format!("invalid timestamp value {} for unit {:?}", value, unit))?;
    let naive = dt.naive_utc();
    let formatted = match frac_digits {
        0 => naive.format("%Y-%m-%d %H:%M:%S").to_string(),
        3 => naive.format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
        6 => naive.format("%Y-%m-%d %H:%M:%S%.6f").to_string(),
        9 => naive.format("%Y-%m-%d %H:%M:%S%.9f").to_string(),
        _ => naive.format("%Y-%m-%d %H:%M:%S%.9f").to_string(),
    };
    Ok(formatted)
}

fn strings_to_array(values: &[Option<String>], data_type: &DataType) -> Result<ArrayRef> {
    match data_type {
        DataType::Utf8 => {
            let mut builder = StringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::LargeUtf8 => {
            let mut builder =
                LargeStringBuilder::with_capacity(values.len(), values.len() * 8);
            for value in values {
                match value {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Int32 => {
            let mut builder = Int32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_with(v, "i32")?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Int64 => {
            let mut builder = Int64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_with(v, "i64")?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::UInt32 => {
            let mut builder = UInt32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_with(v, "u32")?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::UInt64 => {
            let mut builder = UInt64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_with(v, "u64")?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float32 => {
            let mut builder = Float32Builder::with_capacity(values.len());
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_with(v, "f32")?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Float64 => {
            let mut builder = Float64Builder::with_capacity(values.len());
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_with(v, "f64")?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Decimal128(precision, scale) => {
            let mut builder = Decimal128Builder::with_capacity(values.len())
                .with_precision_and_scale(*precision, *scale)
                .context("configure decimal builder")?;
            for value in values {
                match value {
                    Some(v) => builder.append_value(parse_decimal_to_i128(v, *scale)?),
                    None => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        DataType::Timestamp(unit, _) => match unit {
            TimeUnit::Second => {
                let mut builder = TimestampSecondBuilder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(timestamp_string_to_value(v, unit)?),
                        None => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            TimeUnit::Millisecond => {
                let mut builder = TimestampMillisecondBuilder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(timestamp_string_to_value(v, unit)?),
                        None => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            TimeUnit::Microsecond => {
                let mut builder = TimestampMicrosecondBuilder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(timestamp_string_to_value(v, unit)?),
                        None => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
            TimeUnit::Nanosecond => {
                let mut builder = TimestampNanosecondBuilder::with_capacity(values.len());
                for value in values {
                    match value {
                        Some(v) => builder.append_value(timestamp_string_to_value(v, unit)?),
                        None => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            }
        },
        other => bail!("unsupported Arrow output type for UDF: {}", other),
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

    let mut value = digits
        .parse::<i128>()
        .map_err(|err| anyhow::anyhow!("failed to parse decimal from {value}: {err}"))?;
    if negative {
        value = -value;
    }
    Ok(value)
}

fn timestamp_string_to_value(value: &str, unit: &TimeUnit) -> Result<i64> {
    let dt = parse_timestamp(value)?;
    let utc = dt.and_utc();
    let out = match unit {
        TimeUnit::Second => utc.timestamp(),
        TimeUnit::Millisecond => utc.timestamp_millis(),
        TimeUnit::Microsecond => utc.timestamp_micros(),
        TimeUnit::Nanosecond => utc
            .timestamp_nanos_opt()
            .context("timestamp out of range for nanoseconds")?,
    };
    Ok(out)
}

fn parse_timestamp(value: &str) -> Result<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S"))
        .with_context(|| format!("failed to parse timestamp from {value}"))
}

fn parse_field_type(type_str: &str) -> FieldType {
    let base = base_type(type_str);
    match base.as_str() {
        "STRING" | "VARCHAR" | "CHAR" | "TEXT" => FieldType::String,
        "BOOLEAN" | "BOOL" => FieldType::Boolean,
        "BIGINT" | "LONG" => FieldType::Int64,
        "INT64" => FieldType::Int64,
        "INT" | "INTEGER" => FieldType::Int32,
        "INT32" => FieldType::Int32,
        "SMALLINT" => FieldType::Int16,
        "INT16" => FieldType::Int16,
        "TINYINT" => FieldType::Int8,
        "INT8" => FieldType::Int8,
        "DOUBLE" => FieldType::Float64,
        "FLOAT64" => FieldType::Float64,
        "FLOAT" | "REAL" => FieldType::Float32,
        "FLOAT32" => FieldType::Float32,
        "DECIMAL" | "NUMERIC" => {
            let (precision, scale) = parse_decimal_precision_scale(type_str);
            FieldType::Decimal { precision, scale }
        }
        "TIMESTAMP" => {
            let unit = parse_timestamp_unit(type_str);
            FieldType::Timestamp { unit }
        }
        "DATE" => FieldType::Date,
        "BIN" | "BINARY" | "VARBINARY" => FieldType::Unknown("BIN".to_string()),
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
        let precision = parts
            .next()
            .and_then(|p| p.trim().parse::<u8>().ok());
        let scale = parts
            .next()
            .and_then(|s| s.trim().parse::<i8>().ok())
            .unwrap_or(0);
        (precision, scale)
    } else {
        (None, 0)
    }
}

fn parse_timestamp_unit(type_str: &str) -> TimeUnit {
    let trimmed = type_str.trim();
    let start = trimmed.find('(');
    let end = trimmed.find(')');
    let scale = if let (Some(start), Some(end)) = (start, end) {
        trimmed[start + 1..end].trim().parse::<u8>().ok()
    } else {
        None
    };
    match scale.unwrap_or(3) {
        0 => TimeUnit::Second,
        3 => TimeUnit::Millisecond,
        6 => TimeUnit::Microsecond,
        9 => TimeUnit::Nanosecond,
        _ => TimeUnit::Millisecond,
    }
}

fn base_type(type_str: &str) -> String {
    let trimmed = type_str.trim();
    let base = trimmed.split('(').next().unwrap_or(trimmed);
    base.trim().to_uppercase()
}

fn arrow_base_type(data_type: &DataType) -> String {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 => "STRING".to_string(),
        DataType::Int32 => "INTEGER".to_string(),
        DataType::Int64 => "BIGINT".to_string(),
        DataType::UInt32 => "UINT32".to_string(),
        DataType::UInt64 => "UINT64".to_string(),
        DataType::Float32 => "FLOAT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Decimal128(_, _) => "DECIMAL".to_string(),
        DataType::Timestamp(_, _) => "TIMESTAMP".to_string(),
        other => other.to_string().to_uppercase(),
    }
}
