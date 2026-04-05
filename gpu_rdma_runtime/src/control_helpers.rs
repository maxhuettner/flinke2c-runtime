use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt::Display;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::str::FromStr;

use anyhow::{Context, Result};
use sideway::ibverbs::device_context::Mtu;

pub fn send_json<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let bytes = serde_json::to_vec(value)?;
    let len = bytes.len() as u32;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(&bytes)?;
    Ok(())
}

pub fn recv_json<T: DeserializeOwned>(stream: &mut TcpStream) -> Result<T> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    let val = serde_json::from_slice(&buf)?;
    Ok(val)
}

pub fn accept_one(addr: &str) -> Result<TcpStream> {
    let listener = TcpListener::bind(addr)?;
    let (stream, _) = listener.accept()?;
    Ok(stream)
}

pub fn connect(addr: &str) -> Result<TcpStream> {
    TcpStream::connect(addr).with_context(|| format!("connect to {addr}"))
}

#[derive(Debug, Clone)]
pub struct CliMtu(pub Mtu);

impl CliMtu {
    pub fn mtu(&self) -> Mtu {
        self.0
    }
}

impl FromStr for CliMtu {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "256" => Ok(CliMtu(Mtu::Mtu256)),
            "512" => Ok(CliMtu(Mtu::Mtu512)),
            "1024" => Ok(CliMtu(Mtu::Mtu1024)),
            "2048" => Ok(CliMtu(Mtu::Mtu2048)),
            "4096" => Ok(CliMtu(Mtu::Mtu4096)),
            _ => Err(anyhow::anyhow!("invalid MTU value: {s}")),
        }
    }
}

impl Display for CliMtu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mtu_str = match self.0 {
            Mtu::Mtu256 => "256",
            Mtu::Mtu512 => "512",
            Mtu::Mtu1024 => "1024",
            Mtu::Mtu2048 => "2048",
            Mtu::Mtu4096 => "4096",
        };
        write!(f, "{mtu_str}")
    }
}
