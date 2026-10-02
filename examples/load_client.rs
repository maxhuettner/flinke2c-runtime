//! Synthetic PRE/POST client for measuring the runtime's TCP path in isolation.
//! Usage: load_client <port> [rows] [extra_len] [batch_size] [filter|imputation]
//! Sends `rows` bid-shaped rows as PRE (one frame per row, like the Flink operator)
//! and reads the counted responses as POST. Prints end-to-end rows/s.

use std::io::{BufReader, BufWriter, Read, Write};
use std::net::TcpStream;
use std::time::Instant;

const PRE_CONFIG: &str = r#"{"role":"pre","functionClass":"org.example.flinke2c.PriceGreaterThan","functionKind":"filter","externalOnly":true,"reorderResponses":false,"countedResponses":true,"batchSize":__BATCH__,
"functionArgs":[{"name":"price","type":"DECIMAL(23,3)"}],
"preFields":[{"name":"__op","wireType":"INT32"},{"name":"__rowId","wireType":"INT64"},{"name":"auction","wireType":"BIGINT"},{"name":"bidder","wireType":"BIGINT"},{"name":"price","wireType":"DECIMAL(23,3)"},{"name":"dateTime","wireType":"TIMESTAMP"},{"name":"extra","wireType":"STRING"},{"name":"latency_ts","wireType":"BIGINT"}],
"postFields":[{"name":"__op","wireType":"INT32"},{"name":"__rowId","wireType":"INT64"},{"name":"auction","wireType":"BIGINT"},{"name":"bidder","wireType":"BIGINT"},{"name":"price","wireType":"DECIMAL(23,3)"},{"name":"dateTime","wireType":"TIMESTAMP"},{"name":"extra","wireType":"STRING"},{"name":"latency_ts","wireType":"BIGINT"}]}"#;
const POST_CONFIG: &str = r#"{"role":"post"}"#;

/// Config copied from what the Flink PRE operator sends for the ImputationFunction query.
const IMPUTATION_PRE_CONFIG: &str = r#"{"role":"pre","functionClass":"org.example.flinke2c.ImputationFunction","functionKind":"scalar","externalOnly":true,"reorderResponses":false,"countedResponses":true,"batchSize":__BATCH__,
"functionArgs":[{"name":"price","type":"DECIMAL(23, 3)"},{"name":"auction","type":"BIGINT"},{"name":"bidder","type":"BIGINT"},{"name":"channel","type":"VARCHAR(2147483647)"},{"name":"url","type":"VARCHAR(2147483647)"},{"name":"dateTime","type":"TIMESTAMP(3)"},{"name":"extra","type":"VARCHAR(2147483647)"}],
"functionResults":[{"outputName":"price","outputType":"DECIMAL(23, 3)"}],
"preFields":[{"name":"__op","wireType":"INT32"},{"name":"__rowId","wireType":"INT64"},{"name":"auction","wireType":"INT64"},{"name":"bidder","wireType":"INT64"},{"name":"price","wireType":"DECIMAL_UNSCALED_BYTES"},{"name":"channel","wireType":"STRING"},{"name":"url","wireType":"STRING"},{"name":"dateTime","wireType":"TIMESTAMP_MILLIS"},{"name":"extra","wireType":"STRING"},{"name":"latency_ts","wireType":"INT64"}],
"postFields":[{"name":"__op","wireType":"INT32"},{"name":"__rowId","wireType":"INT64"},{"name":"auction","wireType":"INT64"},{"name":"bidder","wireType":"INT64"},{"name":"price","wireType":"DECIMAL_UNSCALED_BYTES"},{"name":"channel","wireType":"STRING"},{"name":"url","wireType":"STRING"},{"name":"dateTime","wireType":"TIMESTAMP_MILLIS"},{"name":"extra","wireType":"STRING"},{"name":"latency_ts","wireType":"INT64"}]}"#;

fn send_config(stream: &mut TcpStream, json: &str) {
    stream.write_all(&(json.len() as i32).to_be_bytes()).unwrap();
    stream.write_all(json.as_bytes()).unwrap();
    stream.flush().unwrap();
}

/// Minimal big-endian two's complement encoding of a non-negative value.
fn twos_complement_be(v: u128, out: &mut Vec<u8>) {
    let bytes = v.to_be_bytes();
    let first = bytes.iter().position(|b| *b != 0).unwrap_or(15);
    if bytes[first] & 0x80 != 0 {
        out.push(0);
    }
    out.extend_from_slice(&bytes[first..]);
}

fn main() {
    let mut args = std::env::args().skip(1);
    let port: u16 = args.next().expect("port").parse().unwrap();
    let rows: u64 = args.next().map(|v| v.parse().unwrap()).unwrap_or(5_000_000);
    let extra_len: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(100);
    let batch_size: usize = args.next().map(|v| v.parse().unwrap()).unwrap_or(2048);
    let imputation = args.next().as_deref() == Some("imputation");
    let addr = ("127.0.0.1", port);

    let mut pre = TcpStream::connect(addr).expect("connect PRE");
    pre.set_nodelay(true).unwrap();
    send_config(
        &mut pre,
        &(if imputation { IMPUTATION_PRE_CONFIG } else { PRE_CONFIG }).replace("__BATCH__", &batch_size.to_string()),
    );
    let mut post = TcpStream::connect(addr).expect("connect POST");
    post.set_nodelay(true).unwrap();
    send_config(&mut post, POST_CONFIG);

    let reader = std::thread::spawn(move || {
        let mut r = BufReader::with_capacity(512 * 1024, post);
        let mut buf = vec![0u8; 1 << 16];
        let mut bytes = 0u64;
        let mut passed = 0u64;
        for _ in 0..rows {
            let mut len = [0u8; 4];
            r.read_exact(&mut len).unwrap();
            let len = i32::from_be_bytes(len) as usize;
            if buf.len() < len {
                buf.resize(len, 0);
            }
            r.read_exact(&mut buf[..len]).unwrap();
            // op(4) rowId(8) countNullBitmap(1) count(4)
            if i32::from_be_bytes(buf[13..17].try_into().unwrap()) > 0 {
                passed += 1;
            }
            bytes += 4 + len as u64;
        }
        (bytes, passed)
    });

    let extra = vec![b'x'; extra_len];
    let mut w = BufWriter::with_capacity(512 * 1024, pre);
    let mut payload = Vec::with_capacity(256);
    let mut rng: u64 = 0x2545F4914F6CDD1D;
    let start = Instant::now();
    for row_id in 0..rows {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // price in [0.001, ~100000] dollars, unscaled with scale 3
        let price_unscaled = ((rng >> 33) % 100_000_000) as u128 + 1;

        payload.clear();
        payload.extend_from_slice(&0i32.to_be_bytes()); // op = INSERT
        payload.extend_from_slice(&(row_id as i64).to_be_bytes());
        if imputation {
            // fields: auction, bidder, price, channel, url, dateTime, extra, latency_ts
            payload.push(if rng % 10 == 0 { 1 << 2 } else { 0 }); // ~10% null price (bit 2)
            payload.extend_from_slice(&(row_id as i64 % 1000).to_be_bytes());
            payload.extend_from_slice(&(rng as i64 & 0xffff).to_be_bytes());
            if rng % 10 != 0 {
                let mut dec = Vec::with_capacity(16);
                twos_complement_be(price_unscaled, &mut dec);
                payload.extend_from_slice(&(dec.len() as i32).to_be_bytes());
                payload.extend_from_slice(&dec);
            }
            for text in [&b"Google"[..], &b"https://www.nexmark.com/item.htm?query=1&id=12345"[..]] {
                payload.extend_from_slice(&(text.len() as i32).to_be_bytes());
                payload.extend_from_slice(text);
            }
            payload.extend_from_slice(&(1_700_000_000_000i64 + row_id as i64).to_be_bytes());
            payload.extend_from_slice(&(extra.len() as i32).to_be_bytes());
            payload.extend_from_slice(&extra);
            payload.extend_from_slice(&0i64.to_be_bytes());
        } else {
            payload.push(0); // null bitmap (6 payload fields)
            payload.extend_from_slice(&(row_id as i64 % 1000).to_be_bytes()); // auction
            payload.extend_from_slice(&(rng as i64 & 0xffff).to_be_bytes()); // bidder
            let mut dec = Vec::with_capacity(16);
            twos_complement_be(price_unscaled, &mut dec);
            payload.extend_from_slice(&(dec.len() as i32).to_be_bytes());
            payload.extend_from_slice(&dec);
            payload.extend_from_slice(&1_700_000_000_000i64.to_be_bytes()); // dateTime
            payload.extend_from_slice(&(extra.len() as i32).to_be_bytes());
            payload.extend_from_slice(&extra);
            payload.extend_from_slice(&0i64.to_be_bytes()); // latency_ts
        }

        w.write_all(&(payload.len() as i32).to_be_bytes()).unwrap();
        w.write_all(&payload).unwrap();
    }
    w.flush().unwrap();
    let sent = start.elapsed();
    let (bytes, passed) = reader.join().unwrap();
    let total = start.elapsed();

    println!(
        "rows={rows} sent_in={:.2}s total={:.2}s  => {:.0} rows/s end-to-end, {:.1} MB/s back, passed={:.1}%",
        sent.as_secs_f64(),
        total.as_secs_f64(),
        rows as f64 / total.as_secs_f64(),
        bytes as f64 / total.as_secs_f64() / 1e6,
        100.0 * passed as f64 / rows as f64
    );
}
