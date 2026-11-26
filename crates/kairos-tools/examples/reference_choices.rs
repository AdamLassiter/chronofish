use chronofish_5d_rules_reference::core::referee::draft_choices;
use std::time::{Duration, Instant};
fn main() {
    println!("fixture active elapsed complete moves");
    for (i, line) in include_str!("../data/bench_timeline.g")
        .lines()
        .filter(|l| l.starts_with("G "))
        .enumerate()
    {
        let s = chronofish_notation::parse_g_line(line).unwrap();
        let start = Instant::now();
        let r = draft_choices(&s, &[], Duration::from_secs(2)).unwrap();
        println!(
            "{i} {} {:?} {} {}",
            s.active_timelines(),
            start.elapsed(),
            r.complete,
            r.moves.len()
        );
    }
}
