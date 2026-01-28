use anyhow::{bail, Context, Result};
use arrow_array::array::{
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt32Array, UInt64Array,
};
use arrow_array::builder::{
    Decimal128Builder, Float32Builder, Float64Builder, Int32Builder, Int64Builder,
    LargeStringBuilder, StringBuilder, TimestampMicrosecondBuilder, TimestampMillisecondBuilder,
    TimestampNanosecondBuilder, TimestampSecondBuilder, UInt32Builder, UInt64Builder,
};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, TimeUnit};
use clap::Parser;
use chrono::{DateTime, NaiveDateTime, Utc};
use serde::Deserialize;
use std::io::{BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use crate::java_udf::{JavaArg, UdfHandle};

mod java_udf;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    listen_host: String,
    #[arg(long)]
    in_port: u16,

    #[arg(long, default_value = "262144")]
    buf_size: usize,

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
    #[serde(rename = "type")]
    arg_type: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
struct FunctionResult {
    output_name: String,
    #[serde(default)]
    output_type: Option<String>,
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

    let desired_udf_class = pre_cfg
        .function_class
        .as_deref()
        .context("functionClass missing from PRE config")?
        .to_string();
    let desired_udf_types: Vec<String> = pre_cfg
        .function_args
        .iter()
        .map(|arg| arg.arg_type.clone())
        .collect();

    let class_changed = current_udf_class
        .as_deref()
        .map(|current| current != desired_udf_class)
        .unwrap_or(true);
    let types_changed = current_udf_types
        .as_ref()
        .map(|current| current != &desired_udf_types)
        .unwrap_or(true);
    if class_changed || types_changed || udf.is_none() {
        let current_class_display = current_udf_class.as_deref().unwrap_or("<unset>");
        let current_types_display = current_udf_types
            .as_ref()
            .map(|v| format!("{v:?}"))
            .unwrap_or_else(|| "<unset>".to_string());
        println!(
            "Switching UDF config: class {} -> {}, types {} -> {:?}",
            current_class_display, desired_udf_class, current_types_display, desired_udf_types
        );
        *udf = Some(UdfHandle::new_with_args(
            &args.udf_jars,
            &args.udf_adapter_class,
            "(Ljava/lang/String;[Ljava/lang/String;)V",
            &[
                JavaArg::String(desired_udf_class.clone()),
                JavaArg::StringArray(desired_udf_types.clone()),
            ],
        )?);
        *current_udf_class = Some(desired_udf_class);
        *current_udf_types = Some(desired_udf_types);
    }

    let udf_handle = udf.as_mut().context("UDF handle not initialized")?;
    if udf_handle.reload_if_changed()? {
        println!("Reloaded UDF classes after jar change");
    }

    let mut out = BufWriter::with_capacity(args.buf_size, post);
    let reader = StreamReader::try_new(pre, None).context("create Arrow IPC reader")?;
    let schema = reader.schema();
    let arg_indices = resolve_function_arg_indices(schema.as_ref(), &pre_cfg)?;
    println!("Resolved {} UDF args at indices {:?}", arg_indices.len(), arg_indices);
    let result_indices = resolve_function_result_indices(schema.as_ref(), &pre_cfg)?;
    if let Some(indices) = &result_indices {
        println!(
            "Resolved {} UDF results at indices {:?}",
            indices.len(),
            indices
        );
    }
    let result_names: Option<Vec<&str>> = if pre_cfg.function_results.is_empty() {
        None
    } else {
        Some(
            pre_cfg
                .function_results
                .iter()
                .map(|result| result.output_name.as_str())
                .collect(),
        )
    };
    let arg_names: Vec<&str> = pre_cfg
        .function_args
        .iter()
        .map(|arg| arg.name.as_str())
        .collect();
    let mut debug_batches_remaining = args.debug_sample_batches;
    let mut writer =
        StreamWriter::try_new(&mut out, schema.as_ref()).context("create Arrow IPC writer")?;

    for maybe_batch in reader {
        let batch = maybe_batch.context("read Arrow record batch")?;
        let new_batch = apply_udf_to_batch(
            &batch,
            schema.as_ref(),
            &arg_indices,
            &arg_names,
            result_indices.as_deref(),
            result_names.as_deref(),
            udf_handle,
            &args.udf_method,
            &args.udf_sig,
            args.debug_sample_rows,
            &mut debug_batches_remaining,
        )?;
        writer.write(&new_batch).context("write Arrow record batch")?;
    }

    writer.finish().context("finish Arrow IPC writer")?;
    drop(writer);
    out.flush().ok();
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

#[derive(Clone, Copy, Debug)]
struct OutputTarget {
    target_idx: usize,
    source_idx: usize,
}

fn apply_udf_to_batch(
    batch: &arrow_array::RecordBatch,
    schema: &arrow_schema::Schema,
    arg_indices: &[usize],
    arg_names: &[&str],
    result_targets: Option<&[usize]>,
    result_names: Option<&[&str]>,
    udf: &UdfHandle,
    method: &str,
    method_sig: &str,
    debug_sample_rows: usize,
    debug_batches_remaining: &mut usize,
) -> Result<arrow_array::RecordBatch> {
    let row_count = batch.num_rows();
    let mut arg_columns = Vec::with_capacity(arg_indices.len());
    for &idx in arg_indices {
        arg_columns.push(column_to_strings(batch.column(idx))?);
    }

    let output_columns = call_udf_to_columns(udf, method, method_sig, &arg_columns)?;
    let output_targets = resolve_output_targets(
        batch,
        arg_indices,
        result_targets,
        result_names,
        arg_names,
        output_columns.len(),
    )?;

    maybe_print_debug_sample(
        schema,
        arg_names,
        &arg_columns,
        &output_targets,
        &output_columns,
        row_count,
        debug_sample_rows,
        debug_batches_remaining,
    );

    let new_columns = build_output_columns(batch, schema, &output_targets, &output_columns)?;
    arrow_array::RecordBatch::try_new(batch.schema(), new_columns)
        .context("build output record batch")
}

fn call_udf_to_columns(
    udf: &UdfHandle,
    method: &str,
    method_sig: &str,
    arg_columns: &[Vec<Option<String>>],
) -> Result<Vec<Vec<Option<String>>>> {
    let columns_method = if method.ends_with("ToColumns") {
        method.to_string()
    } else {
        format!("{method}ToColumns")
    };
    let columns_sig = to_string_matrix_signature(method_sig)?;

    if method_sig.contains("[[Ljava/lang/String;") {
        return udf.call_string_matrix_to_columns(&columns_method, &columns_sig, arg_columns);
    }
    if arg_columns.len() != 1 {
        bail!(
            "UDF method signature {} expects 1 argument but config resolved {}",
            method_sig,
            arg_columns.len()
        );
    }
    udf.call_string_array_to_columns(&columns_method, &columns_sig, &arg_columns[0])
}

fn resolve_output_targets(
    batch: &arrow_array::RecordBatch,
    arg_indices: &[usize],
    result_targets: Option<&[usize]>,
    result_names: Option<&[&str]>,
    arg_names: &[&str],
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
                let name = names.get(idx).copied().unwrap_or("<unknown>");
                let source_idx = arg_names
                    .iter()
                    .position(|arg| *arg == name)
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
    output_columns: &[Vec<Option<String>>],
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
        let values = &output_columns[target.source_idx];
        if values.len() != row_count {
            bail!(
                "output column {} has {} rows but batch has {}",
                target.source_idx,
                values.len(),
                row_count
            );
        }
        let field = schema.field(target.target_idx);
        let array = strings_to_array(values, field.data_type())
            .with_context(|| format!("convert output for field {}", field.name()))?;
        new_columns[target.target_idx] = array;
    }

    Ok(new_columns)
}

fn maybe_print_debug_sample(
    schema: &arrow_schema::Schema,
    arg_names: &[&str],
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
            let name = arg_names.get(arg_idx).copied().unwrap_or("<arg>");
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

fn to_string_matrix_signature(method_sig: &str) -> Result<String> {
    let prefix = method_sig
        .strip_suffix(")V")
        .context("method signature must end with )V")?;
    Ok(format!("{prefix})[[Ljava/lang/String;"))
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
