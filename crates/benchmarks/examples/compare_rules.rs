//! Compare both generators and 48-simulation searches on the same move history.
use chronofish_alphazero::{Mcts, PolicyValueNetwork, SearchConfig};
use chronofish_core::{Game, Ruleset};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .ok_or("expected a checkpoint path and optional LAN history")?;
    let history = args
        .next()
        .unwrap_or_else(|| "(0T1)g1f3 submit (0T1)g8f6 submit (0T2)f3(0T1)d3 submit".into());
    let model = PolicyValueNetwork::load(checkpoint)?;
    for (name, initial) in [
        ("move-first", Game::new(Ruleset::Multiverse)),
        ("reference", Game::new_reference(Ruleset::Multiverse)),
    ] {
        let game = chronofish_notation::replay_lan(&initial, &history)?;
        let start = Instant::now();
        let count = game.legal_actions().len();
        let choices = start.elapsed();
        let search = Mcts::new(
            &model,
            SearchConfig {
                simulations: 48,
                ..SearchConfig::default()
            },
        );
        let mut times = Vec::new();
        for _ in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            let start = Instant::now();
            std::hint::black_box(search.search(&game, false, &mut rng));
            times.push(start.elapsed());
        }
        times.sort_unstable();
        println!(
            "{name} actions={count} movegen={choices:?} median_search={:?}",
            times[2]
        );
    }
    Ok(())
}
