mod cuda;
mod field;
mod pattern;
mod walk;

use base64::{Engine, engine::general_purpose::STANDARD};
use bpaf::Bpaf;
use color_eyre::eyre::{Result, eyre};
use human_format::Formatter;
use indicatif::{HumanDuration, ProgressBar, ProgressStyle};
use pattern::Matcher;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use walk::Ctx;
use zeroize::Zeroizing;

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

fn main() -> Result<()> {
    color_eyre::config::HookBuilder::default()
        .display_location_section(false)
        .display_env_section(false)
        .install()?;
    let args = args().run();
    let matcher = Matcher::new(&args.prefixes, args.ignore_case)?;
    let expected = 1.0 / matcher.probability();
    let (tx, rx) = mpsc::channel();
    let ctx = Ctx {
        matcher,
        keys: 0.into(),
        stop: false.into(),
        tx,
    };
    let bar = ProgressBar::new(1000).with_style(
        ProgressStyle::with_template("[{bar:20}] {percent:>3}% {msg}")?.progress_chars("█░"),
    );
    let num = |x: f64| Formatter::new().format(x);

    std::thread::scope(|s| {
        let worker = s.spawn(|| cuda::run(&ctx));
        let tick = Duration::from_secs(1);
        let (start, mut found) = (Instant::now(), 0);
        let mut last = (start, 0);
        while found < args.count && !worker.is_finished() {
            if let Ok(hit) = rx.recv_timeout(tick.saturating_sub(last.0.elapsed())) {
                let private = Zeroizing::new(STANDARD.encode(*hit.private));
                bar.suspend(|| {
                    if found > 0 {
                        println!("{}", "-".repeat(53));
                    }
                    println!("private: {}\npublic:  {}", *private, STANDARD.encode(hit.public));
                });
                found += 1;
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
            let elapsed = now - start;
            let total = expected * args.count as f64 * elapsed.as_secs_f64() / keys as f64;
            let total = Duration::try_from_secs_f64(total).unwrap_or(Duration::MAX);
            let msg = format!(
                "{found}/{} found  {}keys/s  {}keys  {} / ~{}",
                args.count,
                num(rate),
                num(keys as f64),
                HumanDuration(elapsed),
                HumanDuration(total),
            );
            let progress = elapsed.as_secs_f64() / total.as_secs_f64();
            bar.set_position((1000.0 * progress) as u64);
            if bar.is_hidden() {
                eprintln!("{:.0}% {msg}", 100.0 * progress);
            }
            bar.set_message(msg);
        }
        ctx.stop.store(true, Relaxed);
        bar.finish_and_clear();
        worker.join().map_err(|_| eyre!("CUDA worker panicked"))??;
        let (secs, keys) = (start.elapsed(), ctx.keys.load(Relaxed) as f64);
        eprintln!(
            "{}keys at {}keys/s in {}",
            num(keys),
            num(keys / secs.as_secs_f64()),
            HumanDuration(secs)
        );
        Ok(())
    })
}
