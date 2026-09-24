mod cuda;
mod field;
mod pattern;
mod walk;

use base64::{Engine, engine::general_purpose::STANDARD};
use bpaf::Bpaf;
use pattern::Matcher;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use walk::Ctx;

/// Search for WireGuard keypairs whose base64 public key starts with a prefix.
#[derive(Bpaf, Debug)]
#[bpaf(options)]
struct Args {
    /// Match prefixes case-insensitively
    #[bpaf(short, long)]
    ignore_case: bool,
    /// Number of keypairs to find
    #[bpaf(short('n'), long, fallback(1))]
    count: usize,
    /// Base64 public key prefixes (up to 10 characters)
    #[bpaf(positional("PREFIX"), some("at least one prefix is required"))]
    prefixes: Vec<String>,
}

fn main() {
    let args = args().run();
    let matcher = Matcher::new(&args.prefixes, args.ignore_case).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2);
    });
    let expected = 1.0 / matcher.probability();
    let (tx, rx) = mpsc::channel();
    let ctx = Ctx {
        matcher,
        keys: 0.into(),
        stop: false.into(),
        tx,
    };

    std::thread::scope(|s| {
        s.spawn(|| cuda::worker(&ctx));
        let tick = Duration::from_secs(1);
        let (start, mut found) = (Instant::now(), 0);
        let mut last = (start, 0);
        while found < args.count {
            if let Ok(hit) = rx.recv_timeout(tick.saturating_sub(last.0.elapsed())) {
                found += 1;
                let private = zeroize::Zeroizing::new(STANDARD.encode(*hit.private));
                eprint!("\r\x1b[K");
                println!(
                    "private: {}\npublic:  {}",
                    *private,
                    STANDARD.encode(hit.public)
                );
            }
            if last.0.elapsed() < tick {
                continue;
            }
            let (now, keys) = (Instant::now(), ctx.keys.load(Relaxed));
            let rate = (keys - last.1) as f64 / (now - last.0).as_secs_f64();
            last = (now, keys);
            if keys == 0 {
                continue;
            }
            // Expected time for all hits at the average rate so far.
            let elapsed = (now - start).as_secs_f64();
            let total = expected * args.count as f64 / (keys as f64 / elapsed);
            let progress = elapsed / total;
            eprint!(
                "\r\x1b[K[{}] {:>3.0}% {}/{} found  {} keys/s  {} keys  {} / ~{}",
                bar(progress, 20),
                progress * 100.0,
                found,
                args.count,
                si(rate),
                si(keys as f64),
                time(elapsed),
                time(total),
            );
        }
        ctx.stop.store(true, Relaxed);
        let secs = start.elapsed().as_secs_f64();
        let keys = ctx.keys.load(Relaxed) as f64;
        eprintln!(
            "\r\x1b[K{} keys at {} keys/s in {}",
            si(keys),
            si(keys / secs),
            time(secs)
        );
    });
}

fn bar(progress: f64, width: usize) -> String {
    let filled = (progress.clamp(0.0, 1.0) * width as f64).round() as usize;
    "█".repeat(filled) + &"░".repeat(width - filled)
}

/// Formats with a k, M, G, ... suffix.
fn si(x: f64) -> String {
    const UNITS: [&str; 7] = ["", "k", "M", "G", "T", "P", "E"];
    let i = ((x.max(1.0).log10() / 3.0) as usize).min(UNITS.len() - 1);
    format!("{:.2}{}", x / 1e3f64.powi(i as i32), UNITS[i])
}

fn time(secs: f64) -> String {
    let s = secs.round() as u64;
    match secs {
        ..60.0 => format!("{s}s"),
        ..3600.0 => format!("{}m{:02}s", s / 60, s % 60),
        ..86400.0 => format!("{}h{:02}m", s / 3600, s / 60 % 60),
        ..31_557_600.0 => format!("{:.1}d", secs / 86400.0),
        _ => format!("{}y", si(secs / 31_557_600.0)),
    }
}
