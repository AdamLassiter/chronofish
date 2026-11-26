use chronofish_notation::{position_from_uci, replay_checked_uci};
use std::{
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use chronofish_5d_rules_reference::core::{action::Action, state::State, variants::startpos};
use rand::{SeedableRng, rngs::StdRng};

fn output(message: &str) {
    let _ = writeln!(io::stdout().lock(), "{message}");
    let _ = io::stdout().flush();
}

fn position(input: &str) -> Result<State, String> {
    let (setup, moves) = input
        .split_once(" moves ")
        .unwrap_or((input.trim_end_matches(" moves"), ""));
    let candidate = if setup == "startpos" {
        startpos()
    } else {
        position_from_uci(setup)?
    };
    replay_checked_uci(&candidate, moves)
}

fn search_limits(args: &str) -> (u32, Option<Duration>) {
    let mut nodes = None;
    let mut depth = None;
    let mut time = None;
    let mut tokens = args.split_whitespace().peekable();
    while let Some(key) = tokens.next() {
        if let Some(value) = tokens.peek().and_then(|token| token.parse::<u64>().ok()) {
            match key {
                "nodes" => nodes = Some(value),
                "depth" => depth = Some(value.saturating_mul(10)),
                "movetime" => time = Some(Duration::from_millis(value)),
                _ => continue,
            }
            tokens.next();
        }
    }
    let budget = nodes.or(depth).unwrap_or(if time.is_some() {
        u64::from(u32::MAX)
    } else {
        128
    });
    (u32::try_from(budget).unwrap_or(u32::MAX), time)
}

pub fn run(mode: &str, seed: Option<&String>) -> Result<(), String> {
    let seed = seed
        .map_or(Ok(42), |value| value.parse::<u64>())
        .map_err(|err| err.to_string())?;
    let mut state = None;
    let mut worker: Option<(Arc<AtomicBool>, Arc<AtomicBool>, JoinHandle<()>)> = None;
    for input in io::stdin().lock().lines() {
        let input = input.map_err(|err| err.to_string())?;
        let (command, args) = input
            .trim()
            .split_once(char::is_whitespace)
            .unwrap_or((input.trim(), ""));
        let args = args.trim();
        if worker
            .as_ref()
            .is_some_and(|(_, done, task)| done.load(Ordering::Acquire) || task.is_finished())
        {
            let (_, _, finished) = worker.take().unwrap();
            finished.join().map_err(|_| "search worker panicked")?;
        }
        match command {
            "5duci" => {
                output("id name Chronofish ftxi");
                output("id author ftxi and Chronofish contributors");
                output("5duciok");
            }
            "isready" => output("readyok"),
            "stop" | "quit" => {
                if let Some((stop, _, task)) = worker.take() {
                    stop.store(true, Ordering::Relaxed);
                    task.join().map_err(|_| "search worker panicked")?;
                }
                if command == "quit" {
                    return Ok(());
                }
            }
            "position" | "go" | "5ducinewgame" if worker.is_some() => {
                output("info string search is running");
            }
            "5ducinewgame" => state = None,
            "position" => match position(args) {
                Ok(candidate) => state = Some(candidate),
                Err(err) => output(&format!("info string {err}")),
            },
            "go" => {
                let Some(position) = state.clone() else {
                    output("info string set a position before searching");
                    output("nobestmove");
                    continue;
                };
                let (simulations, time) = search_limits(args);
                let stop = Arc::new(AtomicBool::new(false));
                let cancelled = Arc::clone(&stop);
                let done = Arc::new(AtomicBool::new(false));
                let finished = Arc::clone(&done);
                let mode = mode.to_owned();
                let task = thread::spawn(move || {
                    let started = Instant::now();
                    let mut rng = StdRng::seed_from_u64(seed);
                    let result = super::choose(&position, simulations, &mode, &mut rng, || {
                        cancelled.load(Ordering::Relaxed)
                            || time.is_some_and(|limit| started.elapsed() >= limit)
                    });
                    finished.store(true, Ordering::Release);
                    match result {
                        Ok(Some(turn)) => {
                            let action = Action::from_seq(&turn, &position);
                            output(&format!("bestmove {}", action.to_string().trim_end()));
                        }
                        Ok(None) => output("nobestmove"),
                        Err(err) => {
                            output(&format!("info string {err}"));
                            output("nobestmove");
                        }
                    }
                });
                worker = Some((stop, done, task));
            }
            _ => {}
        }
    }
    if let Some((_, _, task)) = worker {
        task.join().map_err(|_| "search worker panicked")?;
    }
    Ok(())
}
