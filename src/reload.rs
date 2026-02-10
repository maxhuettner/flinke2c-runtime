use anyhow::Result;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::config::Args;
use crate::udf::{UdfHandle, UdfLanguage};

#[derive(Clone, Debug)]
pub struct ReloadSignal {
    version: Arc<AtomicU64>,
    last_paths: Arc<Mutex<Vec<PathBuf>>>,
}

impl ReloadSignal {
    fn new() -> Self {
        Self {
            version: Arc::new(AtomicU64::new(0)),
            last_paths: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn bump(&self, paths: Vec<PathBuf>) {
        if let Ok(mut guard) = self.last_paths.lock() {
            *guard = paths;
        }
        self.version.fetch_add(1, Ordering::AcqRel);
    }

    pub fn current(&self) -> u64 {
        self.version.load(Ordering::Acquire)
    }

    pub fn take_paths(&self) -> Vec<PathBuf> {
        if let Ok(mut guard) = self.last_paths.lock() {
            return std::mem::take(&mut *guard);
        }
        Vec::new()
    }
}

pub struct ReloadWatcher {
    _watcher: RecommendedWatcher,
    signal: ReloadSignal,
}

impl ReloadWatcher {
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let signal = ReloadSignal::new();
        let signal_cb = signal.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                signal_cb.bump(event.paths);
            }
        })?;

        for path in &paths {
            println!("UDF watcher: watching {}", path.display());
        }
        for path in paths {
            if let Err(err) = watcher.watch(&path, RecursiveMode::NonRecursive) {
                eprintln!("Failed to watch {}: {}", path.display(), err);
            }
        }

        Ok(Self {
            _watcher: watcher,
            signal,
        })
    }

    pub fn signal(&self) -> ReloadSignal {
        self.signal.clone()
    }
}

fn rust_udf_lib_name_class(udf_class: &str) -> String {
    let stem = udf_class.replace('.', "_");
    rust_udf_lib_name_from_stem(&stem)
}

fn rust_udf_lib_name_snake(udf_class: &str) -> String {
    let mut out = String::with_capacity(udf_class.len());
    let mut prev_is_lower = false;
    for ch in udf_class.chars() {
        if ch == '.' {
            out.push('_');
            prev_is_lower = false;
            continue;
        }
        if ch.is_ascii_uppercase() {
            if prev_is_lower {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
            prev_is_lower = false;
        } else {
            out.push(ch);
            prev_is_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        }
    }
    rust_udf_lib_name_from_stem(&out)
}

fn rust_udf_lib_name_from_stem(stem: &str) -> String {
    if cfg!(target_os = "windows") {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

fn rust_udf_default_filename() -> String {
    if cfg!(target_os = "windows") {
        "rust_udf.dll".to_string()
    } else if cfg!(target_os = "macos") {
        "librust_udf.dylib".to_string()
    } else {
        "librust_udf.so".to_string()
    }
}

pub fn resolve_rust_udf_lib(base: &Path, udf_class: &str) -> PathBuf {
    let treat_as_dir = base.is_dir() || base.extension().is_none();
    if treat_as_dir {
        let class_path = base.join(rust_udf_lib_name_class(udf_class));
        if class_path.exists() {
            return class_path;
        }
        let snake_path = base.join(rust_udf_lib_name_snake(udf_class));
        if snake_path.exists() {
            return snake_path;
        }
        base.join(rust_udf_default_filename())
    } else {
        base.to_path_buf()
    }
}

pub fn udf_reload_watch_paths(args: &Args, udf_class: &str) -> Vec<PathBuf> {
    let mut paths = HashSet::new();

    match args.udf_lang {
        UdfLanguage::Java => {
            for jar in &args.udf_jars {
                if let Some(parent) = jar.parent() {
                    paths.insert(parent.to_path_buf());
                } else {
                    paths.insert(jar.to_path_buf());
                }
            }
        }
        UdfLanguage::Rust => {
            let rust_base = &args.rust_udf_lib;
            let rust_watch = if rust_base.is_dir() || rust_base.extension().is_none() {
                rust_base.clone()
            } else {
                rust_base.parent().unwrap_or(Path::new(".")).to_path_buf()
            };
            paths.insert(rust_watch);

            // Also watch the resolved class-specific path if it has a parent dir.
            let class_path = resolve_rust_udf_lib(&args.rust_udf_lib, udf_class);
            if let Some(parent) = class_path.parent() {
                paths.insert(parent.to_path_buf());
            }
        }
    }

    paths.into_iter().collect()
}

pub fn maybe_reload_udf(
    udf: &mut UdfHandle,
    udf_class: &str,
    reload_signal: Option<&ReloadSignal>,
    last_version: &mut u64,
) -> Result<()> {
    if let Some(signal) = reload_signal {
        let version = signal.current();
        if version != *last_version {
            let paths = signal.take_paths();
            if !paths.is_empty() {
                let list = paths
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("UDF change detected for {}: {}", udf_class, list);
            } else {
                println!("UDF change detected for {}", udf_class);
            }

            match udf.reload_if_changed() {
                Ok(true) => {
                    println!("UDF reloaded for {} (updated version running)", udf_class);
                    *last_version = version;
                }
                Ok(false) => {
                    *last_version = version;
                }
                Err(err) => {
                    eprintln!(
                        "UDF reload failed for {} (keeping current): {:#}",
                        udf_class, err
                    );
                }
            }
        }
        return Ok(());
    }

    match udf.reload_if_changed() {
        Ok(true) => println!("Reloaded UDF after change"),
        Ok(false) => {}
        Err(err) => eprintln!("UDF reload failed (keeping current): {:#}", err),
    }
    Ok(())
}
