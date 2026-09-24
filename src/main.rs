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
        let (start, mut found) = (Instant::now(), 0);
        let mut last = (start, 0);
        while found < args.count {
            match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(hit) => {
                    found += 1;
                    let private = zeroize::Zeroizing::new(STANDARD.encode(*hit.private));
                    eprint!("\r\x1b[K");
                    println!(
                        "private: {}\npublic:  {}",
                        *private,
                        STANDARD.encode(hit.public)
                    );
                }
                Err(_) => {
                    let (now, keys) = (Instant::now(), ctx.keys.load(Relaxed));
                    let rate = (keys - last.1) as f64 / (now - last.0).as_secs_f64();
                    last = (now, keys);
                    eprint!(
                        "\r\x1b[K{:.2} Mkeys/s, {:.0} keys per hit (~{:.1}s)",
                        rate / 1e6,
                        expected,
                        expected / rate
                    );
                }
            }
        }
        ctx.stop.store(true, Relaxed);
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "\r\x1b[K{:.2} Mkeys/s over {secs:.1}s",
            ctx.keys.load(Relaxed) as f64 / secs / 1e6
        );
    });
}
