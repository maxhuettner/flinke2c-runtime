use anyhow::{bail, Context, Result};
use std::collections::BTreeMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use crate::codec::{
    decode_payload_to_row, read_framed_payload, read_framed_row, read_i32_be_stream,
    validate_row_len, write_output_blocks, OutputBlock,
};
use crate::config::{build_session_config, Args, ConfigMessage};
use crate::constants::DEFAULT_BUF_SIZE;
use crate::reload::{
    maybe_reload_udf, resolve_rust_udf_lib, udf_reload_watch_paths, ReloadSignal, ReloadWatcher,
};
use crate::udf::UdfHandle;
use crate::udf_exec::{apply_udf_to_rows, apply_udf_to_rows_stream, UdfConfig};
use crate::values::V;

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

pub fn run_server(args: Args) -> Result<()> {
    crate::java_udf::set_jvm_opts(args.jvm_opts.clone());
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
        if let Err(err) = run_session(TcpSessionConfig {
            pre,
            pre_cfg,
            post,
            post_cfg,
            udf: &mut udf,
            current_udf_class: &mut current_udf_class,
            current_udf_types: &mut current_udf_types,
            args: &args,
        }) {
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

fn run_session(config: TcpSessionConfig) -> Result<()> {
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

    let class_changed = config
        .current_udf_class
        .as_deref()
        .map(|current| current != desired_udf_class)
        .unwrap_or(true);
    let types_changed = config
        .current_udf_types
        .as_ref()
        .map(|current| current != &desired_udf_types)
        .unwrap_or(true);

    if class_changed || types_changed {
        let current_class_display = config.current_udf_class.as_deref().unwrap_or("<unset>");
        let current_types_display = config
            .current_udf_types
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

    let mut reload_signal: Option<ReloadSignal> = None;
    let _reload_watcher = if config.args.udf_reload_watch {
        let udf_class = config.current_udf_class.as_ref().context("missing UDF class")?;
        let watch_paths = udf_reload_watch_paths(config.args, udf_class);
        match ReloadWatcher::new(watch_paths) {
            Ok(watcher) => {
                reload_signal = Some(watcher.signal());
                Some(watcher)
            }
            Err(err) => {
                eprintln!("Failed to initialize UDF watcher: {err:#}");
                None
            }
        }
    } else {
        None
    };

    let worker_count = config.args.workers.max(1);

    if worker_count == 1 {
        if class_changed || types_changed || config.udf.is_none() {
            let rust_udf_lib = resolve_rust_udf_lib(
                &config.args.rust_udf_lib,
                config.current_udf_class.as_ref().context("missing UDF class")?,
            );
            *config.udf = Some(UdfHandle::new(
                config.args.udf_lang,
                &config.args.udf_jars,
                &config.args.udf_adapter_class,
                config.current_udf_class.as_ref().context("missing UDF class")?,
                config.current_udf_types.as_ref().context("missing UDF types")?,
                &rust_udf_lib,
            )?);
        }

        let udf_handle = config.udf.as_mut().context("UDF handle not initialized")?;
        if let Err(err) = udf_handle.reload_if_changed() {
            eprintln!("UDF reload failed (keeping current): {err:#}");
        }

        let mut debug_batches_remaining = config.args.debug_sample_batches;
        let mut reader = BufReader::with_capacity(DEFAULT_BUF_SIZE, pre);
        let mut writer = BufWriter::with_capacity(DEFAULT_BUF_SIZE, post);

        let batch_size = config.args.batch_size.max(1);
        let mut last_reload_version = reload_signal.as_ref().map(|s| s.current()).unwrap_or(0);
        let udf_class = config
            .current_udf_class
            .as_ref()
            .context("missing UDF class")?
            .clone();
        let mut batch_rows: Vec<Vec<V>> = Vec::with_capacity(batch_size);

        let mut scratch = Vec::<u8>::new();

        while let Some(row) = read_framed_row(&mut reader, &session_cfg, &mut scratch)? {
            validate_row_len(&row, session_cfg.expected_input_len)?;
            batch_rows.push(row);
            if batch_rows.len() >= batch_size {
                maybe_reload_udf(
                    udf_handle,
                    &udf_class,
                    reload_signal.as_ref(),
                    &mut last_reload_version,
                )?;
                let mut udf_config = UdfConfig {
                    writer: &mut writer,
                    rows: &batch_rows,
                    session: &session_cfg,
                    udf: udf_handle,
                    method: &config.args.udf_method,
                    debug_sample_rows: config.args.debug_sample_rows,
                    debug_batches_remaining: &mut debug_batches_remaining,
                };
                apply_udf_to_rows_stream(&mut udf_config)?;
                batch_rows.clear();
            }
        }

        if !batch_rows.is_empty() {
            maybe_reload_udf(
                udf_handle,
                &udf_class,
                reload_signal.as_ref(),
                &mut last_reload_version,
            )?;
            let mut udf_config = UdfConfig {
                writer: &mut writer,
                rows: &batch_rows,
                session: &session_cfg,
                udf: udf_handle,
                method: &config.args.udf_method,
                debug_sample_rows: config.args.debug_sample_rows,
                debug_batches_remaining: &mut debug_batches_remaining,
            };
            apply_udf_to_rows_stream(&mut udf_config)?;
        }

        writer.flush().ok();
        return Ok(());
    }

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
    let inflight = Arc::new((Mutex::new(0usize), Condvar::new()));

    for _ in 0..worker_count {
        let (tx, rx) = mpsc::channel::<Option<WorkItem>>();
        senders.push(tx);

        let result_tx = result_tx.clone();
        let session_cfg = session_cfg.clone();
        let udf_jars = config.args.udf_jars.clone();
        let udf_adapter = config.args.udf_adapter_class.clone();
        let udf_method = config.args.udf_method.clone();
        let udf_lang = config.args.udf_lang;
        let udf_class = config.current_udf_class.as_ref().context("missing UDF class")?.clone();
        let udf_types = config.current_udf_types.as_ref().context("missing UDF types")?.clone();
        let rust_udf_lib = resolve_rust_udf_lib(&config.args.rust_udf_lib, &udf_class);
        let reload_signal = reload_signal.clone();

        let handle = thread::spawn(move || {
            let udf_handle = UdfHandle::new(
                udf_lang,
                &udf_jars,
                &udf_adapter,
                &udf_class,
                &udf_types,
                &rust_udf_lib,
            );
            let mut udf_handle = match udf_handle {
                Ok(handle) => handle,
                Err(err) => {
                    let _ = result_tx.send(WorkResult { seq: 0, result: Err(err) });
                    return;
                }
            };

            let mut debug_batches_remaining = 0usize;
            let mut last_reload_version = reload_signal.as_ref().map(|s| s.current()).unwrap_or(0);
            for msg in rx {
                let Some(work) = msg else { break };
                if let Err(err) = maybe_reload_udf(
                    &mut udf_handle,
                    &udf_class,
                    reload_signal.as_ref(),
                    &mut last_reload_version,
                ) {
                    let _ = result_tx.send(WorkResult { seq: work.seq, result: Err(err) });
                    continue;
                }
                let result = (|| {
                    let mut rows = Vec::with_capacity(work.payloads.len());
                    for payload in &work.payloads {
                        let row = decode_payload_to_row(payload, &session_cfg)?;
                        validate_row_len(&row, session_cfg.expected_input_len)?;
                        rows.push(row);
                    }
                    apply_udf_to_rows(
                        &rows,
                        &session_cfg,
                        &mut udf_handle,
                        &udf_method,
                        0,
                        &mut debug_batches_remaining,
                    )
                })();
                if result_tx.send(WorkResult { seq: work.seq, result }).is_err() {
                    break;
                }
            }
        });
        worker_handles.push(handle);
    }
    drop(result_tx);

    let inflight_writer = Arc::clone(&inflight);
    let writer_buf_size = DEFAULT_BUF_SIZE;
    let writer_session = session_cfg.clone();
    let writer_handle = thread::spawn(move || -> Result<()> {
        let mut writer = BufWriter::with_capacity(writer_buf_size, post);
        let mut pending: BTreeMap<usize, Vec<OutputBlock>> = BTreeMap::new();
        let mut next_seq = 0usize;
        let mut payload_buf = Vec::with_capacity(256);

        while let Ok(work) = result_rx.recv() {
            let WorkResult { seq, result } = work;
            let rows = result?;
            if seq == next_seq {
                write_output_blocks(&mut writer, &rows, &writer_session, &mut payload_buf)?;
                writer.flush().ok();
                next_seq += 1;
                while let Some(next_rows) = pending.remove(&next_seq) {
                    write_output_blocks(&mut writer, &next_rows, &writer_session, &mut payload_buf)?;
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
    let mut reader = BufReader::with_capacity(DEFAULT_BUF_SIZE, pre);

    let batch_size = config.args.batch_size.max(1);
    let mut batch_payloads: Vec<Vec<u8>> = Vec::with_capacity(batch_size);

    loop {
        match read_framed_payload(&mut reader)? {
            Some(payload) => {
                batch_payloads.push(payload);
                if batch_payloads.len() < batch_size {
                    continue;
                }
            }
            None => {
                if batch_payloads.is_empty() {
                    break;
                }
            }
        }

        let payloads = std::mem::take(&mut batch_payloads);
        let (lock, cvar) = &*inflight;
        let mut count = lock.lock().expect("lock inflight");
        while *count >= max_in_flight {
            count = cvar.wait(count).expect("wait inflight");
        }
        *count += 1;
        drop(count);

        let sender = &senders[send_index % senders.len()];
        sender
            .send(Some(WorkItem { seq: dispatched, payloads }))
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

struct WorkItem {
    seq: usize,
    payloads: Vec<Vec<u8>>,
}

struct WorkResult {
    seq: usize,
    result: Result<Vec<OutputBlock>>,
}
