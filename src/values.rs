use anyhow::{bail, Context, Result};

#[derive(Clone, Debug)]
pub enum V {
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

// V conversions

pub fn v_to_string(v: &V) -> Result<String> {
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

pub fn v_to_bool(v: &V) -> Result<bool> {
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

pub fn v_to_i64(v: &V) -> Result<i64> {
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

pub fn v_to_f64(v: &V) -> Result<f64> {
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

pub fn v_to_decimal_i128(v: &V) -> Result<i128> {
    match v {
        V::DecimalI128(x) => Ok(*x),
        V::I64(x) => Ok(*x as i128),
        V::I32(x) => Ok(*x as i128),
        V::String(s) => parse_decimal_to_i128(s, 0),
        V::Bytes(b) => parse_decimal_to_i128(&String::from_utf8_lossy(b), 0),
        _ => bail!("cannot convert {:?} to decimal i128", v),
    }
}

// parsing helpers

pub fn parse_bool_string(value: &str) -> Result<bool> {
    let normalized = value.trim().to_lowercase();
    match normalized.as_str() {
        "true" | "t" | "1" | "yes" | "y" => Ok(true),
        "false" | "f" | "0" | "no" | "n" => Ok(false),
        _ => bail!("failed to parse boolean from {}", value),
    }
}

pub fn parse_with<T>(value: &str, type_name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("failed to parse {type_name} from {value}: {err}"))
}

pub fn parse_decimal_to_i128(value: &str, scale: i8) -> Result<i128> {
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

pub fn convert_decimal_scale(value: i128, source_scale: i8, target_scale: i8) -> Result<i128> {
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

pub fn pow10_i128(scale: i8) -> Result<i128> {
    if scale <= 0 {
        return Ok(1);
    }
    let mut value: i128 = 1;
    for _ in 0..scale {
        value = value.checked_mul(10).context("decimal scale overflow")?;
    }
    Ok(value)
}

pub fn decimal_to_i64(value: i128, scale: i8) -> Result<i64> {
    let scaled = convert_decimal_scale(value, scale, 0)?;
    i64::try_from(scaled).map_err(|_| anyhow::anyhow!("decimal overflow for i64"))
}

pub fn decimal_to_f64(value: i128, scale: i8) -> f64 {
    if scale == 0 {
        return value as f64;
    }
    let factor = 10f64.powi(scale as i32);
    (value as f64) / factor
}

pub fn decimal_to_string(value: i128, scale: i8) -> String {
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
