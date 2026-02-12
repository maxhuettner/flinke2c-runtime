use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::udf::UdfLanguage;

pub fn default_rust_udf_lib() -> PathBuf {
    PathBuf::from("lib")
}

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(long, default_value = "0.0.0.0")]
    pub listen_host: String,
    #[arg(long)]
    pub in_port: u16,

    #[arg(long, default_value_t = num_cpus::get())]
    pub workers: usize,
    #[arg(long, default_value_t = 0)]
    pub max_in_flight: usize,

    #[arg(long, default_value_t = true)]
    pub udf_reload_watch: bool,

    #[arg(long, value_delimiter = ',', default_value = "jar/flinke2c.jar")]
    pub udf_jars: Vec<PathBuf>,
    #[arg(long, default_value = "org.example.flinke2c.runtime.ScalarFunctionAdapter")]
    pub udf_adapter_class: String,
    #[arg(long, default_value = "evalBatch")]
    pub udf_method: String,
    #[arg(long, default_value = "([[Ljava/lang/String;)V")]
    pub udf_sig: String,
    #[arg(long, value_enum, default_value_t = UdfLanguage::Java)]
    pub udf_lang: UdfLanguage,
    #[arg(long, default_value_os_t = default_rust_udf_lib())]
    pub rust_udf_lib: PathBuf,

    #[arg(long = "udf-batch-size", alias = "batch-size")]
    pub udf_batch_size: Option<usize>,

    #[arg(long, env = "JVM_OPTS", value_delimiter = ' ', default_value = "-Xms256m -Xmx512m -XX:+UseG1GC -XX:+AlwaysPreTouch")]
    pub jvm_opts: Vec<String>,

    #[arg(long, default_value_t = 0)]
    pub debug_sample_rows: usize,
    #[arg(long, default_value_t = 1)]
    pub debug_sample_batches: usize,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionArg {
    pub name: String,
    #[serde(rename = "type", alias = "wireType")]
    pub arg_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FunctionResult {
    pub output_name: String,
    #[serde(default, rename = "outputType", alias = "outputWireType", alias = "wireType")]
    pub output_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FieldSpec {
    pub name: String,
    #[serde(default, rename = "wireType", alias = "type")]
    pub field_type: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ConfigMessage {
    pub role: String,
    #[serde(default)]
    pub function_class: Option<String>,
    #[serde(default)]
    pub function_kind: Option<String>,
    #[serde(default)]
    pub function_args: Vec<FunctionArg>,
    #[serde(default)]
    pub function_results: Vec<FunctionResult>,
    #[serde(default)]
    pub reorder_responses: bool,
    #[serde(default, rename = "batchSize", alias = "commBatchSize")]
    pub comm_batch_size: Option<usize>,
    #[serde(default)]
    pub pre_fields: Vec<FieldSpec>,
    #[serde(default)]
    pub post_fields: Vec<FieldSpec>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldType {
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
    pub fn decimal_scale(&self) -> Option<i8> {
        match self {
            FieldType::Decimal { scale, .. } => Some(*scale),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionKind {
    Scalar,
    Filter,
}

impl FunctionKind {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let Some(raw) = value else {
            return Ok(FunctionKind::Scalar);
        };
        let normalized = raw.trim().to_lowercase();
        if normalized.is_empty() {
            return Ok(FunctionKind::Scalar);
        }
        match normalized.as_str() {
            "scalar" => Ok(FunctionKind::Scalar),
            "filter" => Ok(FunctionKind::Filter),
            other => bail!("unsupported functionKind {}", other),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub function_kind: FunctionKind,
    pub expected_input_len: usize,
    pub output_row_len: usize,
    pub post_field_sources: Vec<PostFieldSource>,
    pub passthrough_identity: bool,

    pub arg_positions: Vec<usize>,
    pub arg_names: Vec<String>,
    pub arg_types: Vec<FieldType>,

    pub output_positions: Vec<usize>,
    pub output_names: Vec<String>,
    pub output_types: Vec<FieldType>,

    // payload fields (excluding __op/__rowId), in config order, mapped to row positions
    pub pre_payload_positions: Vec<usize>,
    pub pre_payload_types: Vec<FieldType>,
    /// Whether a pre-payload slot is required for UDF args or output passthrough.
    pub pre_payload_needed: Vec<bool>,
    pub post_payload_positions: Vec<usize>,
    pub post_payload_types: Vec<FieldType>,
    /// Direct columnar source for each post-payload field (parallel to post_payload_positions/types).
    pub post_payload_sources: Vec<PayloadSource>,
    /// Reverse map: row position -> payload slot index (None for __op/__rowId or unused).
    pub pre_pos_to_payload_slot: Vec<Option<usize>>,
}

#[derive(Clone, Debug)]
pub enum PostFieldSourceKind {
    Op,
    RowId,
    InputPos(usize),
    Output,
}

#[derive(Clone, Debug)]
pub struct PostFieldSource {
    pub pos: usize,
    pub kind: PostFieldSourceKind,
}

#[derive(Clone, Debug)]
pub enum PayloadSource {
    InputAt(usize),
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

pub fn build_session_config(pre_cfg: &ConfigMessage) -> Result<SessionConfig> {
    let function_kind = FunctionKind::parse(pre_cfg.function_kind.as_deref())
        .context("parse functionKind")?;
    if function_kind == FunctionKind::Filter && !pre_cfg.function_results.is_empty() {
        bail!("functionKind=filter requires empty functionResults");
    }

    if pre_cfg.pre_fields.is_empty() {
        bail!("preFields missing from PRE config");
    }
    if pre_cfg.post_fields.is_empty() {
        bail!("postFields missing from PRE config");
    }
    // __op + __rowId
    let base_offset = 2;

    let mut pre_name_to_pos = HashMap::new();
    let mut pre_payload_positions = Vec::new();
    let mut pre_payload_types = Vec::new();
    let mut pre_payload_idx = 0usize;
    let mut saw_pre_row_id = false;

    for field in &pre_cfg.pre_fields {
        match field.name.as_str() {
            "__op" => {
                if pre_name_to_pos.insert(field.name.clone(), 0).is_some() {
                    bail!("duplicate preField name {}", field.name);
                }
            }
            "__rowId" => {
                saw_pre_row_id = true;
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
    let mut saw_post_row_id = false;

    for field in &pre_cfg.post_fields {
        let pos = match field.name.as_str() {
            "__op" => 0,
            "__rowId" => {
                saw_post_row_id = true;
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

    let mut post_name_to_pos = HashMap::new();
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
    if output_row_len > 1 {
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
            if pos != 1 {
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
                if pos != 1 {
                    passthrough_identity = false;
                }
            }
            OutputSlotKind::Output => {}
        }
    }

    // Build per-payload-slot source map for direct columnar encode.
    let mut output_name_to_idx = HashMap::new();
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
        } else if let Some(&pre_pos) = pre_name_to_pos.get(field.name.as_str()) {
            post_payload_sources.push(PayloadSource::InputAt(pre_pos));
        } else {
            bail!("postField {} has no source", field.name);
        }
    }

    let pre_payload_needed = if passthrough_identity {
        vec![true; pre_payload_positions.len()]
    } else {
        let mut needed_positions = HashSet::new();
        for pos in &arg_positions {
            needed_positions.insert(*pos);
        }
        for source in &post_field_sources {
            if let PostFieldSourceKind::InputPos(pre_pos) = &source.kind {
                needed_positions.insert(*pre_pos);
            }
        }
        for source in &post_payload_sources {
            if let PayloadSource::InputAt(pre_pos) = source {
                needed_positions.insert(*pre_pos);
            }
        }

        pre_payload_positions
            .iter()
            .map(|pos| needed_positions.contains(pos))
            .collect()
    };

    let pre_pos_to_payload_slot = {
        let mut map = vec![None; expected_input_len];
        for (slot, &pos) in pre_payload_positions.iter().enumerate() {
            map[pos] = Some(slot);
        }
        map
    };

    if !saw_pre_row_id {
        bail!("preFields must include __rowId");
    }
    if !saw_post_row_id {
        bail!("postFields must include __rowId");
    }

    Ok(SessionConfig {
        function_kind,
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
        pre_payload_needed,
        post_payload_positions,
        post_payload_types,
        post_payload_sources,
        pre_pos_to_payload_slot,
    })
}

pub fn parse_field_type(type_str: &str) -> FieldType {
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
        let scale = parts
            .next()
            .and_then(|s| s.trim().parse::<i8>().ok())
            .unwrap_or(0);
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
