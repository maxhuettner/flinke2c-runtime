use anyhow::{bail, Context, Result};
use jni::objects::{GlobalRef, JClass, JObject, JObjectArray, JString, JValue, JValueOwned};
use jni::{InitArgsBuilder, JNIVersion, JNIEnv, JavaVM};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

static JVM: OnceLock<JavaVM> = OnceLock::new();

#[derive(Clone, Debug)]
pub enum JavaArg {
    Null,
    String(String),
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

pub struct UdfHandle {
    classpath_jars: Vec<PathBuf>,
    class_name: String,
    ctor_sig: String,
    ctor_args: Vec<JavaArg>,
    jar_state: Vec<JarState>,
    class_loader: GlobalRef,
    udf_obj: GlobalRef,
}

impl UdfHandle {
    /// Load the jar(s) once and keep the UDF instance cached for fast calls.
    pub fn new(classpath_jars: &[PathBuf], class_name: &str) -> Result<Self> {
        Self::new_with_args(classpath_jars, class_name, "()V", &[])
    }

    /// Load the jar(s) once with constructor arguments and cache the instance.
    pub fn new_with_args(
        classpath_jars: &[PathBuf],
        class_name: &str,
        ctor_sig: &str,
        ctor_args: &[JavaArg],
    ) -> Result<Self> {
        let classpath = normalize_classpath(classpath_jars)?;
        let jar_state = collect_jar_state(&classpath)?;
        let (udf_obj, class_loader) =
            create_udf_instance(&classpath, class_name, ctor_sig, ctor_args)?;

        Ok(Self {
            classpath_jars: classpath,
            class_name: class_name.to_string(),
            ctor_sig: ctor_sig.to_string(),
            ctor_args: ctor_args.to_vec(),
            jar_state,
            class_loader,
            udf_obj,
        })
    }

    /// Call a method on the cached UDF and return its result via `toString`.
    pub fn call_to_string(
        &self,
        method: &str,
        method_sig: &str,
        args: &[JavaArg],
    ) -> Result<Option<String>> {
        let jvm = get_or_create_jvm()?;
        let mut env = jvm
            .attach_current_thread()
            .context("attach JVM thread")?;
        set_context_class_loader(&mut env, self.class_loader.as_obj())?;

        let mut keepalive = Vec::new();
        let jargs = build_jargs(&mut env, args, &mut keepalive)?;
        let ret = env.call_method(self.udf_obj.as_obj(), method, method_sig, &jargs)?;
        check_exception(&mut env, "invoke method")?;
        jvalue_to_string(&mut env, ret)
    }

    /// Call a method that takes a single String[] argument and returns void.
    pub fn call_string_array(
        &self,
        method: &str,
        method_sig: &str,
        values: &[Option<String>],
    ) -> Result<()> {
        let jvm = get_or_create_jvm()?;
        let mut env = jvm
            .attach_current_thread()
            .context("attach JVM thread")?;
        set_context_class_loader(&mut env, self.class_loader.as_obj())?;

        let array = new_string_array(&mut env, values)?;
        let ret = env.call_method(
            self.udf_obj.as_obj(),
            method,
            method_sig,
            &[JValue::Object(&array)],
        )?;
        check_exception(&mut env, "invoke method")?;
        if !matches!(ret, JValueOwned::Void) {
            let _ = jvalue_to_string(&mut env, ret)?;
        }
        Ok(())
    }

    /// Force a reload of the jar(s) and rebuild the cached instance.
    pub fn reload(&mut self) -> Result<()> {
        let classpath = self.classpath_jars.clone();
        self.reload_with_classpath(&classpath)
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
        let (udf_obj, class_loader) = create_udf_instance(
            &classpath,
            &self.class_name,
            &self.ctor_sig,
            &self.ctor_args,
        )?;

        self.close_class_loader();
        self.classpath_jars = classpath;
        self.jar_state = jar_state;
        self.udf_obj = udf_obj;
        self.class_loader = class_loader;
        Ok(())
    }

    pub fn classpath(&self) -> &[PathBuf] {
        &self.classpath_jars
    }

    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    fn close_class_loader(&self) {
        if let Ok(jvm) = get_or_create_jvm() {
            if let Ok(mut env) = jvm.attach_current_thread() {
                let _ = env.call_method(self.class_loader.as_obj(), "close", "()V", &[]);
            }
        }
    }
}

impl Drop for UdfHandle {
    fn drop(&mut self) {
        self.close_class_loader();
    }
}

/// Call any Java method and return the result as a string via `toString`.
/// This creates a new classloader each call; use `UdfHandle` for hot code paths.
/// Provide the exact JNI method signature (e.g. "(Ljava/lang/String;)I").
pub fn call_udf_to_string(
    classpath_jars: &[PathBuf],
    class_name: &str,
    method: &str,
    method_sig: &str,
    args: &[JavaArg],
) -> Result<Option<String>> {
    let jvm = get_or_create_jvm()?;
    let mut env = jvm
        .attach_current_thread()
        .context("attach JVM thread")?;

    let classpath = normalize_classpath(classpath_jars)?;

    // Create a fresh classloader each call so updated jars can be picked up.
    let (udf_obj, class_loader) = load_udf_instance(&mut env, &classpath, class_name)?;
    let mut keepalive = Vec::new();
    let jargs = build_jargs(&mut env, args, &mut keepalive)?;

    let ret = env.call_method(udf_obj, method, method_sig, &jargs)?;
    check_exception(&mut env, "invoke method")?;
    let out = jvalue_to_string(&mut env, ret)?;

    let _ = env.call_method(&class_loader, "close", "()V", &[]);
    Ok(out)
}

/// Evaluate the Flink UDF inside the jar and return the result as a string.
/// Pass decimals as strings to avoid float precision loss.
pub fn eval_currency_conversion(
    jar_path: impl AsRef<Path>,
    price: Option<&str>,
) -> Result<Option<String>> {
    let jars = vec![jar_path.as_ref().to_path_buf()];
    call_decimal_udf(
        &jars,
        "org.example.flinke2c.CurrencyConversionFunction",
        "eval",
        price,
    )
}

/// Invoke a Java method that takes/returns BigDecimal.
/// `classpath_jars` can include extra jars if your UDF depends on them.
pub fn call_decimal_udf(
    classpath_jars: &[PathBuf],
    class_name: &str,
    method: &str,
    price: Option<&str>,
) -> Result<Option<String>> {
    let args = match price {
        Some(v) => vec![JavaArg::BigDecimal(v.to_string())],
        None => vec![JavaArg::Null],
    };
    call_udf_to_string(
        classpath_jars,
        class_name,
        method,
        "(Ljava/math/BigDecimal;)Ljava/math/BigDecimal;",
        &args,
    )
}

fn get_or_create_jvm() -> Result<&'static JavaVM> {
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

fn load_udf_instance<'local>(
    env: &mut JNIEnv<'local>,
    classpath_jars: &[PathBuf],
    class_name: &str,
) -> Result<(JObject<'local>, JObject<'local>)> {
    let (class_obj, class_loader) = load_udf_class(env, classpath_jars, class_name)?;
    let udf_obj = env.call_method(class_obj, "newInstance", "()Ljava/lang/Object;", &[])?;
    check_exception(env, "Class.newInstance")?;

    Ok((udf_obj.l()?, class_loader))
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

fn jvalue_to_string(env: &mut JNIEnv<'_>, value: JValueOwned<'_>) -> Result<Option<String>> {
    match value {
        JValueOwned::Void => Ok(None),
        JValueOwned::Object(obj) => {
            if obj.is_null() {
                return Ok(None);
            }
            let s = env.call_method(obj, "toString", "()Ljava/lang/String;", &[])?;
            check_exception(env, "Object.toString")?;
            let s: String = env.get_string(&JString::from(s.l()?))?.into();
            Ok(Some(s))
        }
        JValueOwned::Bool(v) => Ok(Some((v != 0).to_string())),
        JValueOwned::Byte(v) => Ok(Some(v.to_string())),
        JValueOwned::Char(v) => Ok(Some((v as u32).to_string())),
        JValueOwned::Short(v) => Ok(Some(v.to_string())),
        JValueOwned::Int(v) => Ok(Some(v.to_string())),
        JValueOwned::Long(v) => Ok(Some(v.to_string())),
        JValueOwned::Float(v) => Ok(Some(v.to_string())),
        JValueOwned::Double(v) => Ok(Some(v.to_string())),
    }
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
