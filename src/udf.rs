use anyhow::{Result, bail};
use clap::ValueEnum;
use std::path::{Path, PathBuf};

pub use crate::java_udf::InputColumn;
use crate::java_udf::{JavaArg, JavaUdfHandle};
use crate::rust_udf::RustUdfHandle;

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum UdfLanguage {
    #[value(alias = "jvm")]
    Java,
    #[value(alias = "native")]
    Rust,
}

#[derive(Debug)]
pub enum UdfHandle {
    Java(JavaUdfHandle),
    Rust(RustUdfHandle),
}

impl UdfHandle {
    pub fn new(
        lang: UdfLanguage,
        classpath_jars: &[PathBuf],
        adapter_class: &str,
        udf_class: &str,
        arg_types: &[String],
        rust_udf_lib: &Path,
    ) -> Result<Self> {
        match lang {
            UdfLanguage::Java => Ok(Self::Java(JavaUdfHandle::new_with_args(
                classpath_jars,
                adapter_class,
                "(Ljava/lang/String;[Ljava/lang/String;)V",
                &[
                    JavaArg::String(udf_class.to_string()),
                    JavaArg::StringArray(arg_types.to_vec()),
                ],
            )?)),
            UdfLanguage::Rust => {
                if rust_udf_lib.as_os_str().is_empty() {
                    bail!("rust udf lib path is empty");
                }
                Ok(Self::Rust(RustUdfHandle::new(rust_udf_lib, udf_class)?))
            }
        }
    }

    pub fn reload_if_changed(&mut self) -> Result<bool> {
        match self {
            UdfHandle::Java(handle) => handle.reload_if_changed(),
            UdfHandle::Rust(handle) => handle.reload_if_changed(),
        }
    }

    pub fn call_typed_columns_to_typed_results(
        &mut self,
        method: &str,
        columns: &[InputColumn],
    ) -> Result<Vec<InputColumn>> {
        match self {
            UdfHandle::Java(handle) => handle.call_typed_columns_to_typed_results(method, columns),
            UdfHandle::Rust(handle) => handle.call_typed_columns_to_typed_results(method, columns),
        }
    }

    pub fn call_typed_columns_to_named_results(
        &mut self,
        method: &str,
        columns: &[InputColumn],
        output_names: &[String],
    ) -> Result<Vec<InputColumn>> {
        match self {
            UdfHandle::Java(handle) => handle.call_typed_columns_to_named_results(method, columns, output_names),
            UdfHandle::Rust(handle) => handle.call_typed_columns_to_named_results(method, columns, output_names),
        }
    }
}
