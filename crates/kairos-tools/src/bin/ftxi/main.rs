use chronofish_notation::replay_checked_uci;
mod protocol;

use std::process::ExitCode;

use chronofish_5d_turngen::core::{
    action::{Action, MoveSeq},
    referee::play_action,
    state::State,
    variants::{startpos, state_from_setup},
};
use chronofish_ftxi::{LeafEvaluation, LinearProfile, flat_ucb_until, monkey_until, search_until};
use chronofish_notation::pgn::{PgnOptions, action_to_pgn_opts, parse_5dpgn};
use rand::{SeedableRng, rngs::StdRng};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|arg| arg == "5duci") {
        return protocol::run(args.get(1).map_or("0", String::as_str), args.get(2))
            .map_err(Into::into);
    }
    let [simulations, rollout_turns, seed, format, input @ ..] = args.as_slice() else {
        return Err(
            "usage: chronofish-ftxi SIMULATIONS ROLLOUT_TURNS|weighted:N|zero-capture|zero-capture-check-pw|flat:N|flat-weighted:N|linear:N|linear-trained:N|linear-weighted:N|monkey SEED pgn FILE | lan [MOVE ...]"
                .into(),
        );
    };
    let position = match (format.as_str(), input) {
        ("pgn", [path]) => {
            let game = parse_5dpgn(&std::fs::read_to_string(path)?, None)?;
            let mut state = state_from_setup(&game.setup)?;
            for action in game.root.mainline() {
                state = play_action(&state, action.moves())?;
            }
            state
        }
        ("lan", moves) => replay_checked_uci(&startpos(), &moves.join(" "))?,
        _ => return Err("expected pgn FILE or lan [MOVE ...]".into()),
    };
    let mut rng = StdRng::seed_from_u64(seed.parse()?);
    let simulations = simulations.parse()?;
    let turn = choose(&position, simulations, rollout_turns, &mut rng, || false)?;
    if let Some(sequence) = turn {
        let action = Action::from_seq(&sequence, &position);
        let notation = action_to_pgn_opts(
            &position,
            &action,
            None,
            PgnOptions {
                short: true,
                mate: false,
            },
        );
        play_action(&position, action.moves())?;
        println!("bestmove {action}submit");
        println!("5dpgn {notation}");
    } else {
        println!("nobestmove");
    }
    Ok(())
}

fn choose(
    position: &State,
    simulations: u32,
    mode: &str,
    rng: &mut StdRng,
    stop: impl Fn() -> bool,
) -> Result<Option<MoveSeq>, Box<dyn std::error::Error>> {
    let turn = match mode {
        "monkey" => monkey_until(position, rng, stop),
        value if value.starts_with("flat:") => {
            let evaluation = LeafEvaluation::Rollout {
                max_turns: value["flat:".len()..].parse()?,
            };
            flat_ucb_until(position, simulations, evaluation, rng, stop)
        }
        value if value.starts_with("flat-weighted:") => {
            let evaluation = LeafEvaluation::WeightedRollout {
                max_turns: value["flat-weighted:".len()..].parse()?,
            };
            flat_ucb_until(position, simulations, evaluation, rng, stop)
        }
        value => {
            let evaluation = match value {
                value if value.starts_with("linear") => {
                    let (mode, limit) = value
                        .split_once(':')
                        .ok_or("expected linear mode followed by :N")?;
                    let (profile, weighted) = match mode {
                        "linear" => (LinearProfile::Default, false),
                        "linear-trained" => (LinearProfile::Trained, false),
                        "linear-weighted" => (LinearProfile::Default, true),
                        _ => return Err("unknown linear mode".into()),
                    };
                    LeafEvaluation::Linear {
                        max_turns: limit.parse()?,
                        profile,
                        weighted,
                    }
                }
                "zero-capture" => LeafEvaluation::ZeroCapture,
                "zero-capture-check-pw" => LeafEvaluation::ZeroCaptureCheckWidening,
                value if value.starts_with("weighted:") => LeafEvaluation::WeightedRollout {
                    max_turns: value["weighted:".len()..].parse()?,
                },
                value => match value.parse()? {
                    0 => LeafEvaluation::Zero,
                    max_turns => LeafEvaluation::Rollout { max_turns },
                },
            };
            search_until(position, simulations, evaluation, rng, stop)
        }
    };
    Ok(turn)
}
