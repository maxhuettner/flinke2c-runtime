use anyhow::{bail, Context, Result};
use arrow_array::array::{
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    LargeStringArray, StringArray, UInt32Array, UInt64Array,
};
use arrow_ipc::reader::StreamReader;
use arrow_schema::DataType;
use clap::Parser;
use serde::Deserialize;
use std::io::{BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};
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

    // Flush after this many bytes (not frames)
    #[arg(long, default_value = "16384")]
    flush_every: usize,

    // Flush at least every N ms even if buffer not full
    #[arg(long, default_value = "2")]
    flush_max_ms: u64,

    #[arg(long, value_delimiter = ',', default_value = "jar/flinke2c.jar")]
    udf_jars: Vec<PathBuf>,
    #[arg(long, default_value = "org.example.flinke2c.CurrencyConversionFunction")]
    udf_class: String,
    #[arg(long, default_value = "org.example.proxy.ScalarFunctionAdapter")]
    udf_adapter_class: String,
    #[arg(long, default_value = "evalBatch")]
    udf_method: String,
    #[arg(long, default_value = "([Ljava/lang/String;)V")]
    udf_sig: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct ConfigMessage {
    role: String,
    tm_host: String,
    job_id: String,
    calc_field_name: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let listener = TcpListener::bind((args.listen_host.as_str(), args.in_port))
        .context("bind in-port")?;
    let mut udf = UdfHandle::new_with_args(
        &args.udf_jars,
        &args.udf_adapter_class,
        "(Ljava/lang/String;)V",
        &[JavaArg::String(args.udf_class.clone())],
    )?;

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

        if let Err(err) = run_session(session, &mut udf, &args) {
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
    udf: &mut UdfHandle,
    args: &Args,
) -> Result<()> {
    println!(
        "PRE config: jobId={}, tmHost={}, calcFieldName={}",
        pre_cfg.job_id, pre_cfg.tm_host, pre_cfg.calc_field_name
    );
    println!(
        "POST config: jobId={}, tmHost={}, calcFieldName={}",
        post_cfg.job_id, post_cfg.tm_host, post_cfg.calc_field_name
    );
    if pre_cfg.calc_field_name != post_cfg.calc_field_name {
        eprintln!(
            "PRE/POST config mismatch: pre {}, post {}",
            pre_cfg.calc_field_name, post_cfg.calc_field_name
        );
    }

    if udf.reload_if_changed()? {
        println!("Reloaded UDF classes after jar change");
    }

    let mut out = BufWriter::with_capacity(args.buf_size, post);
    let max_latency = Duration::from_millis(args.flush_max_ms);
    let tee = TeeReader::new(pre, &mut out, args.flush_every, max_latency);
    {
        let reader = StreamReader::try_new(tee, None).context("create Arrow IPC reader")?;
        let schema = reader.schema();
        let field_index = resolve_field_index(schema.as_ref(), &pre_cfg)?;

        for maybe_batch in reader {
            let batch = maybe_batch.context("read Arrow record batch")?;
            apply_udf_to_batch(
                &batch,
                field_index,
                udf,
                &args.udf_method,
                &args.udf_sig,
            )?;
        }
    }
    out.flush().ok();
    Ok(())
}

fn resolve_field_index(schema: &arrow_schema::Schema, cfg: &ConfigMessage) -> Result<usize> {
    if let Some((idx, _)) = schema
        .fields()
        .iter()
        .enumerate()
        .find(|(_, field)| field.name() == &cfg.calc_field_name)
    {
        return Ok(idx);
    }
    bail!(
        "calcFieldName {} not found in Arrow schema",
        cfg.calc_field_name
    );
}

fn apply_udf_to_batch(
    batch: &arrow_array::RecordBatch,
    field_index: usize,
    udf: &UdfHandle,
    method: &str,
    method_sig: &str,
) -> Result<()> {
    let values = column_to_strings(batch.column(field_index))?;
    udf.call_string_array(method, method_sig, &values)?;
    Ok(())
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

struct TeeReader<'a> {
    inner: TcpStream,
    out: &'a mut BufWriter<TcpStream>,
    flush_every: usize,
    max_latency: Duration,
    since_flush: usize,
    last_flush: Instant,
}

impl<'a> TeeReader<'a> {
    fn new(
        inner: TcpStream,
        out: &'a mut BufWriter<TcpStream>,
        flush_every: usize,
        max_latency: Duration,
    ) -> Self {
        Self {
            inner,
            out,
            flush_every,
            max_latency,
            since_flush: 0,
            last_flush: Instant::now(),
        }
    }

    fn maybe_flush(&mut self) -> std::io::Result<()> {
        if (self.flush_every > 0 && self.since_flush >= self.flush_every)
            || self.last_flush.elapsed() >= self.max_latency
        {
            self.out.flush()?;
            self.since_flush = 0;
            self.last_flush = Instant::now();
        }
        Ok(())
    }
}

impl Read for TeeReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n == 0 {
            return Ok(0);
        }
        self.out.write_all(&buf[..n])?;
        self.since_flush += n;
        self.maybe_flush()?;
        Ok(n)
    }
}
