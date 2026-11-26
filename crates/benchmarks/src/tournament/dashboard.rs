//! Live tournament dashboard: progress, standings with a running Elo fit, the games in
//! play and recent results. The tournament runs on worker threads and reports here
//! through [`Progress`] events; the final report prints once the terminal is restored.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Gauge, Paragraph, Row, Table};
use ratatui::{DefaultTerminal, Frame};

use super::setup::{ACCENT, BAD, CHANGED, GOOD, MUTED, duration, footer, header, panel};
use super::{Pairing, Progress, elo, format_score};

/// How the dashboard was left.
pub(crate) enum Exit {
    /// The tournament finished; its report, or the error that ended it.
    Finished(Result<String, String>),
    /// Stopped early by the user; the standings so far.
    Stopped(String),
}

const RECENT: usize = 200;

/// A stage's fitted ratings and, when bootstrapped, their 95% intervals.
type FinalRatings = (Vec<f64>, Option<Vec<(f64, f64)>>);

struct Stage {
    title: String,
    labels: Vec<String>,
    anchors: Vec<Option<f64>>,
    games: usize,
    results: Vec<(usize, usize, f64)>,
    /// Wins, draws and losses per player.
    record: Vec<[usize; 3]>,
    thinking: Vec<(Duration, usize)>,
    adjudicated: usize,
    live: Vec<f64>,
    final_ratings: Option<FinalRatings>,
    started: Instant,
    finished_at: Option<Instant>,
}

impl Stage {
    fn new(title: String, labels: Vec<String>, anchors: Vec<Option<f64>>, games: usize) -> Self {
        let players = labels.len();
        let live = anchors.iter().map(|anchor| anchor.unwrap_or(0.0)).collect();
        Self {
            title,
            labels,
            anchors,
            games,
            results: Vec::new(),
            record: vec![[0; 3]; players],
            thinking: vec![(Duration::ZERO, 0); players],
            adjudicated: 0,
            live,
            final_ratings: None,
            started: Instant::now(),
            finished_at: None,
        }
    }

    fn anchored(&self) -> bool {
        self.anchors.iter().any(Option::is_some)
    }

    fn rating(&self, player: usize) -> f64 {
        self.final_ratings
            .as_ref()
            .map_or(self.live[player], |(values, _)| values[player])
    }

    fn order(&self) -> Vec<usize> {
        let mut order = (0..self.labels.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| self.rating(b).total_cmp(&self.rating(a)));
        order
    }

    fn rating_cell(&self, player: usize) -> String {
        if self.anchors[player].is_none() && self.record[player].iter().sum::<usize>() == 0 {
            return "—".to_owned();
        }
        let rating = self.rating(player);
        let shown = if self.anchored() {
            format!("{rating:.0}")
        } else {
            format!("{rating:+.0}")
        };
        if self.anchors[player].is_some() {
            return format!("{shown} anchor");
        }
        match &self.final_ratings {
            Some((_, Some(intervals))) => {
                let (low, high) = intervals[player];
                format!("{shown} ±{:.0}", (high - low) / 2.0)
            }
            _ => shown,
        }
    }

    fn record(
        &mut self,
        pairing: Pairing,
        white_score: f64,
        reason: &str,
        thinking: [Duration; 2],
        moves: [usize; 2],
    ) {
        let (white, black) = (pairing.white, pairing.black);
        self.results.push((white, black, white_score));
        let outcome = |score: f64| {
            if score > 0.5 {
                0
            } else if score < 0.5 {
                2
            } else {
                1
            }
        };
        self.record[white][outcome(white_score)] += 1;
        self.record[black][outcome(1.0 - white_score)] += 1;
        for (player, side) in [(white, 0), (black, 1)] {
            self.thinking[player].0 += thinking[side];
            self.thinking[player].1 += moves[side];
        }
        self.adjudicated += usize::from(reason.starts_with("adjudicated"));
        if self.results.len() == self.games {
            self.finished_at = Some(Instant::now());
        }
    }

    fn refit(&mut self) {
        self.live = elo(self.labels.len(), &self.results, &self.anchors);
    }

    fn text(&self) -> String {
        let width = self
            .labels
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(4)
            .max(4);
        let mut text = format!(
            "\n{} · {} of {} games\n",
            self.title,
            self.results.len(),
            self.games
        );
        let _ = writeln!(
            text,
            "  {:<width$}  {:>5}  {:>4} {:>4} {:>4}  {:>6}  {:>9}",
            "Bot", "Games", "W", "D", "L", "Score", "Elo"
        );
        for player in self.order() {
            let [wins, draws, losses] = self.record[player];
            let games = wins + draws + losses;
            #[allow(clippy::cast_precision_loss)]
            let score = (wins as f64 + draws as f64 * 0.5) / games.max(1) as f64;
            let _ = writeln!(
                text,
                "  {:<width$}  {games:>5}  {wins:>4} {draws:>4} {losses:>4}  {:>5.1}%  {:>9}",
                self.labels[player],
                score * 100.0,
                self.rating_cell(player)
            );
        }
        text
    }
}

struct App {
    summary: String,
    stages: Vec<Stage>,
    viewing: usize,
    follow: bool,
    running: BTreeMap<(usize, usize), (Pairing, Instant)>,
    recent: VecDeque<Line<'static>>,
    started: Instant,
    done: Option<Result<String, String>>,
    confirm_quit: bool,
    dirty: bool,
}

impl App {
    fn new(summary: &str) -> Self {
        Self {
            summary: summary.to_owned(),
            stages: Vec::new(),
            viewing: 0,
            follow: true,
            running: BTreeMap::new(),
            recent: VecDeque::new(),
            started: Instant::now(),
            done: None,
            confirm_quit: false,
            dirty: false,
        }
    }

    fn current(&self) -> usize {
        self.stages.len().saturating_sub(1)
    }

    fn apply(&mut self, event: Progress) {
        match event {
            Progress::Stage {
                title,
                labels,
                anchors,
                games,
            } => {
                self.stages.push(Stage::new(title, labels, anchors, games));
                self.running.clear();
                if self.follow {
                    self.viewing = self.current();
                }
            }
            Progress::Started { index, pairing } => {
                let stage = self.current();
                self.running
                    .insert((stage, index), (pairing, Instant::now()));
            }
            Progress::Finished {
                index,
                pairing,
                white_score,
                actions,
                reason,
                thinking,
                moves,
            } => {
                let stage_index = self.current();
                let took = self
                    .running
                    .remove(&(stage_index, index))
                    .map(|(_, since)| since.elapsed());
                let Some(stage) = self.stages.last_mut() else {
                    return;
                };
                stage.record(pairing, white_score, &reason, thinking, moves);
                self.dirty = true;
                let (white, black) = (&stage.labels[pairing.white], &stage.labels[pairing.black]);
                let colour = if white_score > 0.5 {
                    GOOD
                } else if white_score < 0.5 {
                    BAD
                } else {
                    CHANGED
                };
                let line = Line::from(vec![
                    Span::styled(
                        format!("{:>4} ", stage.results.len()),
                        Style::new().fg(MUTED),
                    ),
                    Span::raw(format!("{white} – {black} ")),
                    Span::styled(
                        format!(
                            "{}-{}",
                            format_score(white_score),
                            format_score(1.0 - white_score)
                        ),
                        Style::new().fg(colour).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(
                            "  {} · {} · {actions} actions{}",
                            short_ruleset(pairing),
                            reason,
                            took.map_or_else(String::new, |took| format!(
                                " · {}",
                                duration(took.as_secs_f64())
                            ))
                        ),
                        Style::new().fg(MUTED),
                    ),
                ]);
                self.recent.push_front(line);
                self.recent.truncate(RECENT);
            }
            Progress::Rated { values, intervals } => {
                if let Some(stage) = self.stages.last_mut() {
                    stage.final_ratings = Some((values, intervals));
                }
            }
            Progress::Done(result) => {
                self.running.clear();
                self.done = Some(result);
            }
        }
    }

    fn partial(&self) -> String {
        self.stages.iter().map(Stage::text).collect()
    }
}

fn short_ruleset(pairing: Pairing) -> &'static str {
    match pairing.ruleset {
        chronofish_core::Ruleset::Standard => "standard",
        chronofish_core::Ruleset::Multiverse => "5D",
        chronofish_core::Ruleset::MultiverseVariant => "5D variant",
    }
}

/// Runs the dashboard until the tournament finishes and the user leaves, or the user
/// stops it early.
pub(crate) fn run(events: &Receiver<Progress>, summary: &str) -> std::io::Result<Exit> {
    let mut terminal = ratatui::try_init()?;
    terminal.clear()?;
    let result = event_loop(&mut terminal, events, &mut App::new(summary));
    ratatui::restore();
    result
}

fn event_loop(
    terminal: &mut DefaultTerminal,
    events: &Receiver<Progress>,
    app: &mut App,
) -> std::io::Result<Exit> {
    let mut last_fit = Instant::now();
    loop {
        loop {
            match events.try_recv() {
                Ok(event) => app.apply(event),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    if app.done.is_none() {
                        app.done = Some(Err("the tournament stopped unexpectedly".to_owned()));
                    }
                    break;
                }
            }
        }
        // The fit is cheap but runs to convergence; refresh it a few times a second.
        if app.dirty && last_fit.elapsed() >= Duration::from_millis(300) {
            if let Some(stage) = app.stages.last_mut() {
                stage.refit();
            }
            app.dirty = false;
            last_fit = Instant::now();
        }
        terminal.draw(|frame| draw(frame, app))?;
        if !event::poll(Duration::from_millis(200))? {
            continue;
        }
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
            || (control && key.code == KeyCode::Char('c'));
        if quit {
            if let Some(result) = app.done.take() {
                return Ok(Exit::Finished(result));
            }
            if app.confirm_quit || control {
                if let Some(stage) = app.stages.last_mut() {
                    stage.refit();
                }
                return Ok(Exit::Stopped(app.partial()));
            }
            app.confirm_quit = true;
            continue;
        }
        app.confirm_quit = false;
        match key.code {
            KeyCode::Left | KeyCode::Char('h') => {
                app.viewing = app.viewing.saturating_sub(1);
                app.follow = false;
            }
            KeyCode::Right | KeyCode::Char('l') => {
                app.viewing = (app.viewing + 1).min(app.current());
                app.follow = app.viewing == app.current();
            }
            KeyCode::Enter if app.done.is_some() => {
                return Ok(Exit::Finished(app.done.take().unwrap_or(Ok(String::new()))));
            }
            _ => {}
        }
    }
}

fn draw(frame: &mut Frame, app: &App) {
    let [top, progress, body, status, bottom] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Min(8),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    header(frame, top, &app.summary);
    draw_progress(frame, progress, app);
    let (left, right) = if body.width >= 120 {
        let [left, right] =
            Layout::horizontal([Constraint::Percentage(55), Constraint::Percentage(45)])
                .areas(body);
        (left, right)
    } else {
        let [left, right] =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(body);
        (left, right)
    };
    draw_standings(frame, left, app);
    let running_height = u16::try_from(app.running.len().clamp(1, 16) + 2).unwrap_or(18);
    let [running, recent] =
        Layout::vertical([Constraint::Length(running_height), Constraint::Min(4)]).areas(right);
    draw_running(frame, running, app);
    draw_recent(frame, recent, app);
    draw_status(frame, status, app);
    let keys: &[(&str, &str)] = if app.done.is_some() {
        &[("←→", "stage"), ("q/Enter", "leave and print the report")]
    } else {
        &[("←→", "stage"), ("q", "stop")]
    };
    footer(frame, bottom, keys);
}

fn draw_progress(frame: &mut Frame, area: Rect, app: &App) {
    let Some(stage) = app.stages.get(app.viewing) else {
        frame.render_widget(
            Paragraph::new("Starting engines…").block(panel("Progress")),
            area,
        );
        return;
    };
    let played = stage.results.len();
    let elapsed = stage
        .finished_at
        .unwrap_or_else(Instant::now)
        .duration_since(stage.started)
        .as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let eta = if played == 0 || played == stage.games {
        String::new()
    } else {
        let remaining = elapsed / played as f64 * (stage.games - played) as f64;
        format!(" · ≈ {} left", duration(remaining))
    };
    #[allow(clippy::cast_precision_loss)]
    let ratio = if stage.games == 0 {
        1.0
    } else {
        played as f64 / stage.games as f64
    };
    let label = format!(
        "{played}/{} games · {}{eta}",
        stage.games,
        duration(elapsed)
    );
    let title = if app.stages.len() > 1 {
        format!(
            "{} ({} of {})",
            stage.title,
            app.viewing + 1,
            app.stages.len()
        )
    } else {
        stage.title.clone()
    };
    let gauge = Gauge::default()
        .block(panel(&title))
        .gauge_style(Style::new().fg(ACCENT).bg(Color::Indexed(236)))
        .ratio(ratio.clamp(0.0, 1.0))
        .label(Span::styled(
            label,
            Style::new().add_modifier(Modifier::BOLD),
        ));
    frame.render_widget(gauge, area);
}

fn draw_standings(frame: &mut Frame, area: Rect, app: &App) {
    let Some(stage) = app.stages.get(app.viewing) else {
        frame.render_widget(panel("Standings"), area);
        return;
    };
    let rows = stage.order().into_iter().enumerate().map(|(rank, player)| {
        let [wins, draws, losses] = stage.record[player];
        let games = wins + draws + losses;
        #[allow(clippy::cast_precision_loss)]
        let score = (wins as f64 + draws as f64 * 0.5) / games.max(1) as f64;
        let (time, moves) = stage.thinking[player];
        #[allow(clippy::cast_precision_loss)]
        let per_action = time.as_secs_f64() * 1_000.0 / moves.max(1) as f64;
        let rating_style = if stage.anchors[player].is_some() {
            Style::new().fg(MUTED)
        } else if stage.final_ratings.is_some() {
            Style::new().fg(GOOD).add_modifier(Modifier::BOLD)
        } else {
            Style::new().add_modifier(Modifier::BOLD)
        };
        Row::new(vec![
            Cell::from(Span::styled(
                format!("{:>2}", rank + 1),
                Style::new().fg(MUTED),
            )),
            Cell::from(stage.labels[player].clone()),
            Cell::from(Span::styled(stage.rating_cell(player), rating_style)),
            Cell::from(format!("{games}")),
            Cell::from(format!("{wins}-{draws}-{losses}")),
            Cell::from(format!("{:.1}%", score * 100.0)),
            Cell::from(Span::styled(
                if moves == 0 {
                    "—".to_owned()
                } else {
                    format!("{per_action:.1}")
                },
                Style::new().fg(MUTED),
            )),
        ])
    });
    let caption = match (&stage.final_ratings, stage.anchored()) {
        (Some((_, Some(_))), _) => "final · ± is half the 95% interval",
        (Some(_), _) => "final",
        (None, true) => "live fit · anchors held",
        (None, false) => "live fit · centred on zero",
    };
    let title = format!("Standings · {caption}");
    let table = Table::new(
        rows,
        [
            Constraint::Length(3),
            Constraint::Min(16),
            Constraint::Length(11),
            Constraint::Length(5),
            Constraint::Length(10),
            Constraint::Length(7),
            Constraint::Length(9),
        ],
    )
    .header(
        Row::new(["#", "Bot", "Elo", "Games", "W-D-L", "Score", "ms/act"])
            .style(Style::new().fg(MUTED).add_modifier(Modifier::BOLD)),
    )
    .block(panel(&title));
    frame.render_widget(table, area);
}

fn draw_running(frame: &mut Frame, area: Rect, app: &App) {
    let Some(stage) = app.stages.last() else {
        frame.render_widget(panel("In play"), area);
        return;
    };
    let mut games = app.running.values().collect::<Vec<_>>();
    games.sort_by_key(|(_, since)| *since);
    let lines = if games.is_empty() {
        vec![Line::from(Span::styled(
            if app.done.is_some() {
                "Finished."
            } else {
                "Waiting for games…"
            },
            Style::new().fg(MUTED),
        ))]
    } else {
        games
            .into_iter()
            .map(|(pairing, since)| {
                let seconds = since.elapsed().as_secs_f64();
                let colour = if seconds > 300.0 { CHANGED } else { MUTED };
                Line::from(vec![
                    Span::raw(format!(
                        "{} – {}",
                        stage.labels[pairing.white], stage.labels[pairing.black]
                    )),
                    Span::styled(
                        format!(
                            "  {} · opening {} · {}",
                            short_ruleset(*pairing),
                            pairing.opening + 1,
                            duration(seconds)
                        ),
                        Style::new().fg(colour),
                    ),
                ])
            })
            .collect()
    };
    let title = format!("In play · {}", app.running.len());
    frame.render_widget(Paragraph::new(lines).block(panel(&title)), area);
}

fn draw_recent(frame: &mut Frame, area: Rect, app: &App) {
    let lines = app.recent.iter().cloned().collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines).block(panel("Recent results")), area);
}

fn draw_status(frame: &mut Frame, area: Rect, app: &App) {
    let line = match &app.done {
        Some(Ok(_)) => Line::from(Span::styled(
            format!(
                " Finished in {}. Press q to leave; the full report with the crosstable prints next.",
                duration(app.started.elapsed().as_secs_f64())
            ),
            Style::new().fg(GOOD),
        )),
        Some(Err(error)) => Line::from(Span::styled(format!(" ✗ {error}"), Style::new().fg(BAD))),
        None if app.confirm_quit => Line::from(Span::styled(
            " Press q again to stop. Finished games are kept in the CSV; games in play are lost.",
            Style::new().fg(CHANGED),
        )),
        None => {
            let adjudicated = app
                .stages
                .get(app.viewing)
                .map_or(0, |stage| stage.adjudicated);
            Line::from(Span::styled(
                format!(
                    " {} elapsed · {adjudicated} adjudicated at the action limit",
                    duration(app.started.elapsed().as_secs_f64())
                ),
                Style::new().fg(MUTED),
            ))
        }
    };
    frame.render_widget(Paragraph::new(line), area);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chronofish_core::Ruleset;

    fn pairing(white: usize, black: usize) -> Pairing {
        Pairing {
            ruleset: Ruleset::Multiverse,
            opening: 0,
            white,
            black,
        }
    }

    #[test]
    fn events_build_standings_and_a_live_fit() {
        let mut app = App::new("test");
        app.apply(Progress::Stage {
            title: "Round robin".to_owned(),
            labels: vec!["A".to_owned(), "B".to_owned()],
            anchors: vec![None, None],
            games: 2,
        });
        for (index, score) in [(0, 1.0), (1, 0.0)] {
            let pairing = if index == 0 {
                pairing(0, 1)
            } else {
                pairing(1, 0)
            };
            app.apply(Progress::Started { index, pairing });
            app.apply(Progress::Finished {
                index,
                pairing,
                white_score: score,
                actions: 10,
                reason: "Checkmate".to_owned(),
                thinking: [Duration::from_millis(10); 2],
                moves: [5, 5],
            });
        }
        let stage = &mut app.stages[0];
        stage.refit();
        assert_eq!(stage.record[0], [2, 0, 0]);
        assert!(stage.live[0] > stage.live[1]);
        assert!(stage.finished_at.is_some());
        assert!(app.running.is_empty());
        assert!(app.partial().contains("2 of 2 games"));
        app.apply(Progress::Done(Ok("report".to_owned())));
        assert!(app.done.is_some());
    }
}
