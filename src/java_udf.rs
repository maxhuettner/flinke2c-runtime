use anyhow::{bail, Context, Result};
use jni::objects::{GlobalRef, JClass, JObject, JObjectArray, JString, JValue};
use jni::sys::{jboolean, jbyte};
use jni::{InitArgsBuilder, JNIVersion, JNIEnv, JavaVM};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

static JVM: OnceLock<JavaVM> = OnceLock::new();
static JVM_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone)]
pub enum JavaArg {
    Null,
    String(String),
    StringArray(Vec<String>),
    BigDecimal(String),
    Boolean(bool),
    Byte(i8),
    Short(i16),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Char(u16),
}

#[derive(Debug)]
pub enum InputColumn {
    String(Vec<Option<String>>),
    I64 { values: Vec<i64>, is_null: Option<Vec<bool>> },
    I32 { values: Vec<i32>, is_null: Option<Vec<bool>> },
    F64 { values: Vec<f64>, is_null: Option<Vec<bool>> },
    F32 { values: Vec<f32>, is_null: Option<Vec<bool>> },
    Bool { values: Vec<bool>, is_null: Option<Vec<bool>> },
    Decimal128 { values: Vec<i128>, is_null: Option<Vec<bool>> },
}

impl InputColumn {
    pub fn len(&self) -> usize {
        match self {
            Self::String(values) => values.len(),
            Self::I64 { values, .. } => values.len(),
            Self::I32 { values, .. } => values.len(),
            Self::F64 { values, .. } => values.len(),
            Self::F32 { values, .. } => values.len(),
            Self::Bool { values, .. } => values.len(),
            Self::Decimal128 { values, .. } => values.len(),
        }
    }
}

#[derive(Debug)]
pub struct JavaUdfHandle {
    classpath_jars: Vec<PathBuf>,
    class_name: String,
    ctor_sig: String,
    ctor_args: Vec<JavaArg>,
    jar_state: Vec<JarState>,
    class_loader: GlobalRef,
    udf_obj: GlobalRef,
    /// True once setContextClassLoader has been called on the owning thread.
    /// Avoids Thread.currentThread() + setContextClassLoader() on every batch.
    context_loader_set: bool,
    /// Cached Java String[] for the output-column names; rebuilt only when
    /// the names vector changes between sessions.
    cached_output_names: Option<(Vec<String>, GlobalRef)>,
    snapshot_paths: Vec<PathBuf>,
}

impl JavaUdfHandle {
    /// Load the jar(s) once with constructor arguments and cache the instance.
    pub fn new_with_args(
        classpath_jars: &[PathBuf],
        class_name: &str,
        ctor_sig: &str,
        ctor_args: &[JavaArg],
    ) -> Result<Self> {
        let classpath = normalize_classpath(classpath_jars)?;
        let jar_state = collect_jar_state(&classpath)?;
        let (udf_obj, class_loader, snapshot_paths) =
            create_udf_instance_snapshot(&classpath, class_name, ctor_sig, ctor_args)?;

        Ok(Self {
            classpath_jars: classpath,
            class_name: class_name.to_string(),
            ctor_sig: ctor_sig.to_string(),
            ctor_args: ctor_args.to_vec(),
            jar_state,
            class_loader,
            udf_obj,
            context_loader_set: false,
            cached_output_names: None,
            snapshot_paths,
        })
    }

    /// Call a method that takes Object[] columns + boolean[][] nulls and returns ColumnarResult.
    pub fn call_typed_columns_to_typed_results(
        &mut self,
        method: &str,
        columns: &[InputColumn],
    ) -> Result<Vec<InputColumn>> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }
        let row_count = columns[0].len();
        if columns.iter().any(|col| col.len() != row_count) {
            bail!("typed columns have mismatched lengths");
        }

        let jvm = get_or_create_jvm()?;
        let mut env = jvm
            .attach_current_thread()
            .context("attach JVM thread")?;
        if !self.context_loader_set {
            set_context_class_loader(&mut env, self.class_loader.as_obj())?;
            self.context_loader_set = true;
        }

        let (columns_obj, nulls_obj) = new_typed_columns(&mut env, columns)?;
        let ret = env.call_method(
            self.udf_obj.as_obj(),
            method,
            "([Ljava/lang/Object;[[Z)Lorg/example/flinke2c/runtime/ScalarFunctionAdapter$ColumnarResult;",
            &[
                JValue::Object(&columns_obj),
                JValue::Object(&nulls_obj),
            ],
        )?;
        check_exception(&mut env, "invoke method")?;
        let result_obj = ret.l()?;
        if result_obj.is_null() {
            return Ok(Vec::new());
        }

        let columns_val =
            env.call_method(&result_obj, "columns", "()[Ljava/lang/Object;", &[])?;
        check_exception(&mut env, "ColumnarResult.columns")?;
        let columns_obj = columns_val.l()?;
        let columns_array = JObjectArray::from(columns_obj);

        let nulls_val = env.call_method(&result_obj, "nulls", "()[[Z", &[])?;
        check_exception(&mut env, "ColumnarResult.nulls")?;
        let nulls_obj = nulls_val.l()?;
        let nulls_array = if nulls_obj.is_null() {
            None
        } else {
            Some(JObjectArray::from(nulls_obj))
        };

        typed_columns_to_vec(&mut env, columns_array, nulls_array)
    }

    /// Call a method that takes Object[] columns + boolean[][] nulls + String[] outputNames
    /// and returns ColumnarResult.
    pub fn call_typed_columns_to_named_results(
        &mut self,
        method: &str,
        columns: &[InputColumn],
        output_names: &[String],
    ) -> Result<Vec<InputColumn>> {
        if columns.is_empty() {
            return Ok(Vec::new());
        }
        let row_count = columns[0].len();
        if columns.iter().any(|col| col.len() != row_count) {
            bail!("typed columns have mismatched lengths");
        }

        let jvm = get_or_create_jvm()?;
        let mut env = jvm
            .attach_current_thread()
            .context("attach JVM thread")?;
        if !self.context_loader_set {
            set_context_class_loader(&mut env, self.class_loader.as_obj())?;
            self.context_loader_set = true;
        }

        let (columns_obj, nulls_obj) = new_typed_columns(&mut env, columns)?;
        // Cache the output-names String[] across batches; rebuild only when names change.
        if self.cached_output_names.as_ref().is_none_or(|(names, _)| names != output_names) {
            let new_obj = new_string_array_from_strings(&mut env, output_names)?;
            let global = env.new_global_ref(&new_obj)?;
            self.cached_output_names = Some((output_names.to_vec(), global));
        }
        let names_ref = self.cached_output_names.as_ref().unwrap().1.as_obj();
        let ret = env.call_method(
            self.udf_obj.as_obj(),
            method,
            "([Ljava/lang/Object;[[Z[Ljava/lang/String;)Lorg/example/flinke2c/runtime/ScalarFunctionAdapter$ColumnarResult;",
            &[
                JValue::Object(&columns_obj),
                JValue::Object(&nulls_obj),
                JValue::Object(names_ref),
            ],
        )?;
        check_exception(&mut env, "invoke method")?;
        let result_obj = ret.l()?;
        if result_obj.is_null() {
            return Ok(Vec::new());
        }

        let columns_val =
            env.call_method(&result_obj, "columns", "()[Ljava/lang/Object;", &[])?;
        check_exception(&mut env, "ColumnarResult.columns")?;
        let columns_obj = columns_val.l()?;
        let columns_array = JObjectArray::from(columns_obj);

        let nulls_val = env.call_method(&result_obj, "nulls", "()[[Z", &[])?;
        check_exception(&mut env, "ColumnarResult.nulls")?;
        let nulls_obj = nulls_val.l()?;
        let nulls_array = if nulls_obj.is_null() {
            None
        } else {
            Some(JObjectArray::from(nulls_obj))
        };

        typed_columns_to_vec(&mut env, columns_array, nulls_array)
    }

    /// Reload only if any jar file has changed.
    pub fn reload_if_changed(&mut self) -> Result<bool> {
        let classpath = self.classpath_jars.clone();
        let new_state = collect_jar_state(&classpath)?;
        if new_state != self.jar_state {
            self.reload_with_classpath(&classpath)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Swap to a new classpath and rebuild the cached instance.
    pub fn reload_with_classpath(&mut self, classpath_jars: &[PathBuf]) -> Result<()> {
        let classpath = normalize_classpath(classpath_jars)?;
        let jar_state = collect_jar_state(&classpath)?;
        let (udf_obj, class_loader, snapshot_paths) = create_udf_instance_snapshot(
            &classpath,
            &self.class_name,
            &self.ctor_sig,
            &self.ctor_args,
        )?;

        let old_loader = std::mem::replace(&mut self.class_loader, class_loader);
        let old_snapshot = std::mem::replace(&mut self.snapshot_paths, snapshot_paths);
        let _old_udf = std::mem::replace(&mut self.udf_obj, udf_obj);

        self.close_class_loader_ref(&old_loader);
        cleanup_snapshot_paths(&old_snapshot);

        self.classpath_jars = classpath;
        self.jar_state = jar_state;
        self.context_loader_set = false;
        self.cached_output_names = None;
        Ok(())
    }

    fn close_class_loader_ref(&self, class_loader: &GlobalRef) {
        if let Ok(jvm) = get_or_create_jvm() {
            if let Ok(mut env) = jvm.attach_current_thread() {
                let _ = env.call_method(class_loader.as_obj(), "close", "()V", &[]);
            }
        }
    }

    fn close_class_loader(&self) {
        self.close_class_loader_ref(&self.class_loader);
    }
}

impl Drop for JavaUdfHandle {
    fn drop(&mut self) {
        self.close_class_loader();
        cleanup_snapshot_paths(&self.snapshot_paths);
    }
}

fn get_or_create_jvm() -> Result<&'static JavaVM> {
    if let Some(jvm) = JVM.get() {
        return Ok(jvm);
    }

    let lock = JVM_LOCK.get_or_init(|| Mutex::new(()));
    let _guard = lock.lock().expect("lock JVM init mutex");
    if let Some(jvm) = JVM.get() {
        return Ok(jvm);
    }

    let args = InitArgsBuilder::new()
        .version(JNIVersion::V8)
        .build()
        .context("build JVM args")?;
    let jvm = JavaVM::new(args).context("create JVM")?;
    let _ = JVM.set(jvm);
    Ok(JVM.get().expect("JVM initialized"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct JarState {
    modified: Option<SystemTime>,
    len: u64,
}

fn collect_jar_state(classpath_jars: &[PathBuf]) -> Result<Vec<JarState>> {
    classpath_jars
        .iter()
        .map(|path| jar_state(path))
        .collect()
}

fn jar_state(path: &Path) -> Result<JarState> {
    let meta = fs::metadata(path).with_context(|| format!("metadata {}", path.display()))?;
    Ok(JarState {
        modified: meta.modified().ok(),
        len: meta.len(),
    })
}

fn normalize_classpath(classpath_jars: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for jar in augment_classpath(classpath_jars) {
        let jar = jar
            .canonicalize()
            .with_context(|| format!("canonicalize {}", jar.display()))?;
        if !out.iter().any(|existing| existing == &jar) {
            out.push(jar);
        }
    }
    if out.is_empty() {
        bail!("classpath_jars is empty");
    }
    Ok(out)
}

fn create_udf_instance_snapshot(
    classpath_jars: &[PathBuf],
    class_name: &str,
    ctor_sig: &str,
    ctor_args: &[JavaArg],
) -> Result<(GlobalRef, GlobalRef, Vec<PathBuf>)> {
    let snapshot_paths = snapshot_classpath(classpath_jars)?;
    match create_udf_instance(&snapshot_paths, class_name, ctor_sig, ctor_args) {
        Ok((udf_obj, class_loader)) => Ok((udf_obj, class_loader, snapshot_paths)),
        Err(err) => {
            cleanup_snapshot_paths(&snapshot_paths);
            Err(err)
        }
    }
}

fn snapshot_classpath(classpath_jars: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let base_dir = env::temp_dir().join("flinke2c-runtime-jars");
    fs::create_dir_all(&base_dir)
        .with_context(|| format!("create jar snapshot dir {}", base_dir.display()))?;

    let mut out = Vec::with_capacity(classpath_jars.len());
    for jar in classpath_jars {
        let dest = snapshot_path(&base_dir, jar)?;
        fs::copy(jar, &dest)
            .with_context(|| format!("copy {} -> {}", jar.display(), dest.display()))?;
        out.push(dest);
    }
    Ok(out)
}

fn snapshot_path(base_dir: &Path, jar: &Path) -> Result<PathBuf> {
    let stem = jar
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("udf");
    let ext = jar
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("jar");
    let pid = std::process::id();
    let counter = SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let filename = format!("{}-{}-{}-{}.{}", stem, pid, ts, counter, ext);
    Ok(base_dir.join(filename))
}

fn cleanup_snapshot_paths(paths: &[PathBuf]) {
    for path in paths {
        let _ = fs::remove_file(path);
    }
}

fn create_udf_instance(
    classpath_jars: &[PathBuf],
    class_name: &str,
    ctor_sig: &str,
    ctor_args: &[JavaArg],
) -> Result<(GlobalRef, GlobalRef)> {
    let jvm = get_or_create_jvm()?;
    let mut env = jvm
        .attach_current_thread()
        .context("attach JVM thread")?;

    let (udf_obj, class_loader) =
        load_udf_instance_with_args(&mut env, classpath_jars, class_name, ctor_sig, ctor_args)?;
    let udf_obj = env.new_global_ref(udf_obj)?;
    let class_loader = env.new_global_ref(class_loader)?;

    Ok((udf_obj, class_loader))
}

fn load_udf_instance_with_args<'local>(
    env: &mut JNIEnv<'local>,
    classpath_jars: &[PathBuf],
    class_name: &str,
    ctor_sig: &str,
    ctor_args: &[JavaArg],
) -> Result<(JObject<'local>, JObject<'local>)> {
    let (class_obj, class_loader) = load_udf_class(env, classpath_jars, class_name)?;
    let mut keepalive = Vec::new();
    let jargs = build_jargs(env, ctor_args, &mut keepalive)?;
    let udf_obj = env.new_object(JClass::from(class_obj), ctor_sig, &jargs)?;
    check_exception(env, "Class.<init>")?;
    Ok((udf_obj, class_loader))
}

fn load_udf_class<'local>(
    env: &mut JNIEnv<'local>,
    classpath_jars: &[PathBuf],
    class_name: &str,
) -> Result<(JObject<'local>, JObject<'local>)> {
    if classpath_jars.is_empty() {
        bail!("classpath_jars is empty");
    }

    let url_array = build_url_array(env, classpath_jars)?;
    let sys_cl = env.call_static_method(
        "java/lang/ClassLoader",
        "getSystemClassLoader",
        "()Ljava/lang/ClassLoader;",
        &[],
    )?;
    check_exception(env, "getSystemClassLoader")?;

    let class_loader = env.new_object(
        "java/net/URLClassLoader",
        "([Ljava/net/URL;Ljava/lang/ClassLoader;)V",
        &[
            JValue::Object(&url_array),
            JValue::Object(&sys_cl.l()?),
        ],
    )?;
    check_exception(env, "URLClassLoader.<init>")?;
    set_context_class_loader(env, &class_loader)?;

    let class_name = env.new_string(class_name)?;
    let class_obj = env.call_method(
        &class_loader,
        "loadClass",
        "(Ljava/lang/String;)Ljava/lang/Class;",
        &[JValue::Object(&JObject::from(class_name))],
    )?;
    check_exception(env, "ClassLoader.loadClass")?;

    Ok((class_obj.l()?, class_loader))
}

fn build_url_array<'local>(
    env: &mut JNIEnv<'local>,
    jar_paths: &[PathBuf],
) -> Result<JObject<'local>> {
    let url_class = env.find_class("java/net/URL")?;
    let url_array = env.new_object_array(jar_paths.len() as i32, url_class, JObject::null())?;

    for (idx, jar_path) in jar_paths.iter().enumerate() {
        let url = jar_path_to_url(env, jar_path)?;
        env.set_object_array_element(&url_array, idx as i32, url)?;
    }

    Ok(JObject::from(url_array))
}

fn jar_path_to_url<'local>(
    env: &mut JNIEnv<'local>,
    jar_path: &Path,
) -> Result<JObject<'local>> {
    let jar_path = jar_path
        .canonicalize()
        .with_context(|| format!("canonicalize {}", jar_path.display()))?;
    let path_str = jar_path.to_string_lossy();
    let jpath = env.new_string(path_str.as_ref())?;
    let file = env.new_object(
        "java/io/File",
        "(Ljava/lang/String;)V",
        &[JValue::Object(&JObject::from(jpath))],
    )?;

    let uri = env.call_method(file, "toURI", "()Ljava/net/URI;", &[])?;
    check_exception(env, "File.toURI")?;
    let url = env.call_method(uri.l()?, "toURL", "()Ljava/net/URL;", &[])?;
    check_exception(env, "URI.toURL")?;
    Ok(url.l()?)
}

fn new_big_decimal<'local>(
    env: &mut JNIEnv<'local>,
    value: &str,
) -> Result<JObject<'local>> {
    let jvalue = env.new_string(value)?;
    let obj = env.new_object(
        "java/math/BigDecimal",
        "(Ljava/lang/String;)V",
        &[JValue::Object(&JObject::from(jvalue))],
    )?;
    check_exception(env, "BigDecimal.<init>")?;
    Ok(obj)
}

fn check_exception(env: &mut JNIEnv<'_>, context: &str) -> Result<()> {
    if env.exception_check()? {
        env.exception_describe()?;
        env.exception_clear()?;
        bail!("Java exception during {}", context);
    }
    Ok(())
}

fn set_context_class_loader(env: &mut JNIEnv<'_>, class_loader: &JObject<'_>) -> Result<()> {
    let thread = env.call_static_method(
        "java/lang/Thread",
        "currentThread",
        "()Ljava/lang/Thread;",
        &[],
    )?;
    check_exception(env, "Thread.currentThread")?;
    env.call_method(
        thread.l()?,
        "setContextClassLoader",
        "(Ljava/lang/ClassLoader;)V",
        &[JValue::Object(class_loader)],
    )?;
    check_exception(env, "Thread.setContextClassLoader")?;
    Ok(())
}

fn build_jargs<'local, 'b>(
    env: &mut JNIEnv<'local>,
    args: &[JavaArg],
    keepalive: &'b mut Vec<JObject<'local>>,
) -> Result<Vec<JValue<'local, 'b>>> {
    enum PreparedArg {
        Obj(usize),
        Boolean(bool),
        Byte(i8),
        Short(i16),
        Int(i32),
        Long(i64),
        Float(f32),
        Double(f64),
        Char(u16),
    }

    let mut prepared = Vec::with_capacity(args.len());
    for arg in args {
        match arg {
            JavaArg::Null => {
                keepalive.push(JObject::null());
                prepared.push(PreparedArg::Obj(keepalive.len() - 1));
            }
            JavaArg::String(v) => {
                let jvalue = env.new_string(v)?;
                keepalive.push(JObject::from(jvalue));
                prepared.push(PreparedArg::Obj(keepalive.len() - 1));
            }
            JavaArg::StringArray(values) => {
                let obj = new_string_array_from_strings(env, values)?;
                keepalive.push(obj);
                prepared.push(PreparedArg::Obj(keepalive.len() - 1));
            }
            JavaArg::BigDecimal(v) => {
                let obj = new_big_decimal(env, v)?;
                keepalive.push(obj);
                prepared.push(PreparedArg::Obj(keepalive.len() - 1));
            }
            JavaArg::Boolean(v) => prepared.push(PreparedArg::Boolean(*v)),
            JavaArg::Byte(v) => prepared.push(PreparedArg::Byte(*v)),
            JavaArg::Short(v) => prepared.push(PreparedArg::Short(*v)),
            JavaArg::Int(v) => prepared.push(PreparedArg::Int(*v)),
            JavaArg::Long(v) => prepared.push(PreparedArg::Long(*v)),
            JavaArg::Float(v) => prepared.push(PreparedArg::Float(*v)),
            JavaArg::Double(v) => prepared.push(PreparedArg::Double(*v)),
            JavaArg::Char(v) => prepared.push(PreparedArg::Char(*v)),
        }
    }

    let mut jargs = Vec::with_capacity(prepared.len());
    for arg in prepared {
        match arg {
            PreparedArg::Obj(idx) => jargs.push(JValue::Object(&keepalive[idx])),
            PreparedArg::Boolean(v) => jargs.push(JValue::Bool(if v { 1 } else { 0 })),
            PreparedArg::Byte(v) => jargs.push(JValue::Byte(v)),
            PreparedArg::Short(v) => jargs.push(JValue::Short(v)),
            PreparedArg::Int(v) => jargs.push(JValue::Int(v)),
            PreparedArg::Long(v) => jargs.push(JValue::Long(v)),
            PreparedArg::Float(v) => jargs.push(JValue::Float(v)),
            PreparedArg::Double(v) => jargs.push(JValue::Double(v)),
            PreparedArg::Char(v) => jargs.push(JValue::Char(v)),
        }
    }

    Ok(jargs)
}

fn new_string_array_from_strings<'local>(
    env: &mut JNIEnv<'local>,
    values: &[String],
) -> Result<JObject<'local>> {
    let string_class = env.find_class("java/lang/String")?;
    let array: JObjectArray =
        env.new_object_array(values.len() as i32, string_class, JObject::null())?;

    for (idx, value) in values.iter().enumerate() {
        let jstr = env.new_string(value)?;
        env.set_object_array_element(&array, idx as i32, JObject::from(jstr))?;
    }

    Ok(JObject::from(array))
}

fn typed_columns_to_vec(
    env: &mut JNIEnv<'_>,
    columns_obj: JObjectArray<'_>,
    nulls_obj: Option<JObjectArray<'_>>,
) -> Result<Vec<InputColumn>> {
    let long_array_class = env.find_class("[J")?;
    let int_array_class = env.find_class("[I")?;
    let double_array_class = env.find_class("[D")?;
    let float_array_class = env.find_class("[F")?;
    let boolean_array_class = env.find_class("[Z")?;
    let byte_array_class = env.find_class("[B")?;
    let string_array_class = env.find_class("[Ljava/lang/String;")?;

    let col_len = env.get_array_length(&columns_obj)? as usize;
    let nulls = nulls_matrix_to_vec(env, nulls_obj, col_len)?;
    let mut columns = Vec::with_capacity(col_len);

    for idx in 0..col_len {
        let col_obj = env.get_object_array_element(&columns_obj, idx as i32)?;
        if col_obj.is_null() {
            columns.push(InputColumn::String(Vec::new()));
            continue;
        }

        let nulls_col = nulls.get(idx).cloned().unwrap_or(None);

        if env.is_instance_of(&col_obj, &long_array_class)? {
            let values = long_array_to_vec(env, col_obj)?;
            columns.push(InputColumn::I64 {
                values,
                is_null: nulls_col,
            });
            continue;
        }
        if env.is_instance_of(&col_obj, &int_array_class)? {
            let values = int_array_to_vec(env, col_obj)?;
            columns.push(InputColumn::I32 {
                values,
                is_null: nulls_col,
            });
            continue;
        }
        if env.is_instance_of(&col_obj, &double_array_class)? {
            let values = double_array_to_vec(env, col_obj)?;
            columns.push(InputColumn::F64 {
                values,
                is_null: nulls_col,
            });
            continue;
        }
        if env.is_instance_of(&col_obj, &float_array_class)? {
            let values = float_array_to_vec(env, col_obj)?;
            columns.push(InputColumn::F32 {
                values,
                is_null: nulls_col,
            });
            continue;
        }
        if env.is_instance_of(&col_obj, &boolean_array_class)? {
            let values = boolean_array_to_vec(env, col_obj)?;
            columns.push(InputColumn::Bool {
                values,
                is_null: nulls_col,
            });
            continue;
        }
        if env.is_instance_of(&col_obj, &byte_array_class)? {
            let values = byte_array_to_i128_vec(env, col_obj)?;
            columns.push(InputColumn::Decimal128 {
                values,
                is_null: nulls_col,
            });
            continue;
        }
        if env.is_instance_of(&col_obj, &string_array_class)? {
            let values = string_array_to_vec(env, col_obj)?;
            columns.push(InputColumn::String(values));
            continue;
        }

        bail!("Unsupported output column type from Java");
    }

    Ok(columns)
}

fn nulls_matrix_to_vec(
    env: &mut JNIEnv<'_>,
    nulls_obj: Option<JObjectArray<'_>>,
    col_len: usize,
) -> Result<Vec<Option<Vec<bool>>>> {
    let mut out = vec![None; col_len];
    let Some(nulls_obj) = nulls_obj else {
        return Ok(out);
    };

    let outer_len = env.get_array_length(&nulls_obj)? as usize;
    let num_cols = outer_len.min(col_len);
    for (idx, out) in out.iter_mut().take(num_cols).enumerate() {
        let inner_obj = env.get_object_array_element(&nulls_obj, idx as i32)?;
        if inner_obj.is_null() {
            continue;
        }
        let values = boolean_array_to_vec(env, inner_obj)?;
        *out = Some(values);
    }
    Ok(out)
}

fn long_array_to_vec(env: &mut JNIEnv<'_>, obj: JObject<'_>) -> Result<Vec<i64>> {
    let array = jni::objects::JLongArray::from(obj);
    let len = env.get_array_length(&array)?;
    let mut out = vec![0_i64; len as usize];
    env.get_long_array_region(&array, 0, &mut out)?;
    Ok(out)
}

fn int_array_to_vec(env: &mut JNIEnv<'_>, obj: JObject<'_>) -> Result<Vec<i32>> {
    let array = jni::objects::JIntArray::from(obj);
    let len = env.get_array_length(&array)?;
    let mut out = vec![0_i32; len as usize];
    env.get_int_array_region(&array, 0, &mut out)?;
    Ok(out)
}

fn double_array_to_vec(env: &mut JNIEnv<'_>, obj: JObject<'_>) -> Result<Vec<f64>> {
    let array = jni::objects::JDoubleArray::from(obj);
    let len = env.get_array_length(&array)?;
    let mut out = vec![0_f64; len as usize];
    env.get_double_array_region(&array, 0, &mut out)?;
    Ok(out)
}

fn float_array_to_vec(env: &mut JNIEnv<'_>, obj: JObject<'_>) -> Result<Vec<f32>> {
    let array = jni::objects::JFloatArray::from(obj);
    let len = env.get_array_length(&array)?;
    let mut out = vec![0_f32; len as usize];
    env.get_float_array_region(&array, 0, &mut out)?;
    Ok(out)
}

fn boolean_array_to_vec(env: &mut JNIEnv<'_>, obj: JObject<'_>) -> Result<Vec<bool>> {
    let array = jni::objects::JBooleanArray::from(obj);
    let len = env.get_array_length(&array)?;
    let mut raw = vec![0_u8; len as usize];
    env.get_boolean_array_region(&array, 0, &mut raw)?;
    Ok(raw.iter().map(|v| *v != 0).collect())
}

fn byte_array_to_i128_vec(env: &mut JNIEnv<'_>, obj: JObject<'_>) -> Result<Vec<i128>> {
    let array = jni::objects::JByteArray::from(obj);
    let len = env.get_array_length(&array)? as usize;
    if !len.is_multiple_of(16) {
        bail!("decimal byte array length {} is not divisible by 16", len);
    }
    let mut raw = vec![0_i8; len];
    env.get_byte_array_region(&array, 0, &mut raw)?;
    let mut out = Vec::with_capacity(len / 16);
    for chunk in raw.chunks_exact(16) {
        let mut bytes = [0_u8; 16];
        for (idx, b) in chunk.iter().enumerate() {
            bytes[idx] = *b as u8;
        }
        out.push(i128::from_be_bytes(bytes));
    }
    Ok(out)
}

fn string_array_to_vec(env: &mut JNIEnv<'_>, array_obj: JObject<'_>) -> Result<Vec<Option<String>>> {
    if array_obj.is_null() {
        return Ok(Vec::new());
    }

    let array = JObjectArray::from(array_obj);
    let len = env.get_array_length(&array)?;
    let mut out = Vec::with_capacity(len as usize);
    for idx in 0..len {
        let elem = env.get_object_array_element(&array, idx)?;
        if elem.is_null() {
            out.push(None);
            continue;
        }
        let s: String = env.get_string(&JString::from(elem))?.into();
        out.push(Some(s));
    }
    Ok(out)
}

fn new_string_array<'local>(
    env: &mut JNIEnv<'local>,
    values: &[Option<String>],
) -> Result<JObject<'local>> {
    let string_class = env.find_class("java/lang/String")?;
    let array: JObjectArray = env.new_object_array(values.len() as i32, string_class, JObject::null())?;

    for (idx, value) in values.iter().enumerate() {
        if let Some(v) = value {
            let jstr = env.new_string(v)?;
            env.set_object_array_element(&array, idx as i32, JObject::from(jstr))?;
        }
    }

    Ok(JObject::from(array))
}

fn new_typed_columns<'local>(
    env: &mut JNIEnv<'local>,
    columns: &[InputColumn],
) -> Result<(JObjectArray<'local>, JObjectArray<'local>)> {
    let object_class = env.find_class("java/lang/Object")?;
    let boolean_array_class = env.find_class("[Z")?;
    let col_array =
        env.new_object_array(columns.len() as i32, object_class, JObject::null())?;
    let nulls_array =
        env.new_object_array(columns.len() as i32, boolean_array_class, JObject::null())?;

    for (idx, column) in columns.iter().enumerate() {
        let (col_obj, nulls_obj) = input_column_to_java(env, column)?;
        env.set_object_array_element(&col_array, idx as i32, col_obj)?;
        if let Some(nulls_obj) = nulls_obj {
            env.set_object_array_element(&nulls_array, idx as i32, nulls_obj)?;
        }
    }

    Ok((col_array, nulls_array))
}

fn input_column_to_java<'local>(
    env: &mut JNIEnv<'local>,
    column: &InputColumn,
) -> Result<(JObject<'local>, Option<JObject<'local>>)> {
    match column {
        InputColumn::String(values) => {
            let array = new_string_array(env, values)?;
            Ok((array, None))
        }
        InputColumn::I64 { values, is_null } => {
            let array = env.new_long_array(values.len() as i32)?;
            env.set_long_array_region(&array, 0, values)?;
            let nulls = build_nulls_array(env, is_null.as_deref())?;
            Ok((JObject::from(array), nulls))
        }
        InputColumn::I32 { values, is_null } => {
            let array = env.new_int_array(values.len() as i32)?;
            env.set_int_array_region(&array, 0, values)?;
            let nulls = build_nulls_array(env, is_null.as_deref())?;
            Ok((JObject::from(array), nulls))
        }
        InputColumn::F64 { values, is_null } => {
            let array = env.new_double_array(values.len() as i32)?;
            env.set_double_array_region(&array, 0, values)?;
            let nulls = build_nulls_array(env, is_null.as_deref())?;
            Ok((JObject::from(array), nulls))
        }
        InputColumn::F32 { values, is_null } => {
            let array = env.new_float_array(values.len() as i32)?;
            env.set_float_array_region(&array, 0, values)?;
            let nulls = build_nulls_array(env, is_null.as_deref())?;
            Ok((JObject::from(array), nulls))
        }
        InputColumn::Bool { values, is_null } => {
            let array = env.new_boolean_array(values.len() as i32)?;
            let raw: Vec<jboolean> = values
                .iter()
                .map(|v| if *v { 1_u8 } else { 0_u8 })
                .collect();
            env.set_boolean_array_region(&array, 0, &raw)?;
            let nulls = build_nulls_array(env, is_null.as_deref())?;
            Ok((JObject::from(array), nulls))
        }
        InputColumn::Decimal128 { values, is_null } => {
            let mut raw: Vec<jbyte> = Vec::with_capacity(values.len() * 16);
            for value in values {
                let bytes = value.to_be_bytes();
                raw.extend(bytes.iter().map(|b| *b as i8));
            }
            let array = env.new_byte_array(raw.len() as i32)?;
            env.set_byte_array_region(&array, 0, &raw)?;
            let nulls = build_nulls_array(env, is_null.as_deref())?;
            Ok((JObject::from(array), nulls))
        }
    }
}

fn build_nulls_array<'local>(
    env: &mut JNIEnv<'local>,
    nulls: Option<&[bool]>,
) -> Result<Option<JObject<'local>>> {
    let Some(nulls) = nulls else {
        return Ok(None);
    };
    let array = env.new_boolean_array(nulls.len() as i32)?;
    let raw: Vec<jboolean> = nulls
        .iter()
        .map(|v| if *v { 1_u8 } else { 0_u8 })
        .collect();
    env.set_boolean_array_region(&array, 0, &raw)?;
    Ok(Some(JObject::from(array)))
}

fn augment_classpath(classpath_jars: &[PathBuf]) -> Vec<PathBuf> {
    let mut out = classpath_jars.to_vec();
    append_if_exists(&mut out, PathBuf::from("jar/flink-stubs.jar"));
    append_if_exists(&mut out, PathBuf::from("jar/udf-adapter.jar"));
    out
}

fn append_if_exists(out: &mut Vec<PathBuf>, path: PathBuf) {
    if path.exists() && !out.iter().any(|p| p == &path) {
        out.push(path);
    }
}
