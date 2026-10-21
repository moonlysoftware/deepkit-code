//! Deepkit Code process wiring; see `v1.md` §2.4.

use std::io::Write;

macro_rules! outln {
    ($($args:tt)*) => {
        crate::write_stdout(format_args!($($args)*), true)?
    };
}

macro_rules! out {
    ($($args:tt)*) => {
        crate::write_stdout(format_args!($($args)*), false)?
    };
}

mod cli;
#[expect(
    dead_code,
    reason = "T0.14 wires configuration loading into CLI commands and indexing"
)]
mod config;
mod models;
mod queue;
mod search;
mod serve;
#[expect(
    dead_code,
    reason = "T0.14 wires the verified store API into CLI commands and indexing"
)]
mod store;
#[expect(
    dead_code,
    reason = "T0.14 wires filesystem synchronization into dkc index and status"
)]
mod sync;
mod treat;

fn main() {
    let arguments: Vec<_> = std::env::args_os().collect();
    if arguments
        .get(1)
        .is_some_and(|argument| argument == "config")
    {
        if let Err(error) = config::run_command(&arguments[2..]) {
            if is_broken_pipe(&error) {
                return;
            }
            eprintln!("{error}");
            std::process::exit(2);
        }
        return;
    }
    if let Err(error) = cli::run(arguments) {
        if is_broken_pipe(error.as_ref()) {
            return;
        }
        eprintln!("{error:#}");
        std::process::exit(1);
    }
}

fn write_stdout(args: std::fmt::Arguments<'_>, newline: bool) -> std::io::Result<()> {
    let stdout = std::io::stdout();
    let mut locked = stdout.lock();
    locked.write_fmt(args)?;
    if newline {
        locked.write_all(b"\n")?;
    }
    Ok(())
}

fn is_broken_pipe(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::BrokenPipe)
        {
            return true;
        }
        match error.source() {
            Some(source) => error = source,
            None => return false,
        }
    }
}
