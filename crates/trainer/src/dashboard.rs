//! Live terminal dashboard for the trainer, built on ratatui.
//!
//! The dashboard is a process-wide singleton. Every reporting function in this
//! module is a no-op while it is inactive, so callers can report unconditionally
//! and the plain line-oriented output remains the fallback for pipes and logs.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::VecDeque;
use std::io::{self, IsTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use chronofish_core::{GameOutcome, Ruleset, Side};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Cell, Gauge, Paragraph, Row, Sparkline, Table, TableState, Wrap,
};
use ratatui::{DefaultTerminal, Frame};

const TICK: Duration = Duration::from_millis(100);
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);
const SAMPLE_HISTORY: usize = 240;
const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const BAD: Color = Color::Red;
const NEUTRAL: Color = Color::Gray;
const ACTIVE: Color = Color::Yellow;
const MUTED: Color = Color::DarkGray;

static DASHBOARD: OnceLock<Dashboard> = OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Setup,
    SelfPlay,
    Fitting,
    Arena,
    Done,
}

impl Phase {
    const fn label(self) -> &'static str {
        match self {
            Self::Setup => "SETUP",
            Self::SelfPlay => "SELF-PLAY",
            Self::Fitting => "FITTING",
            Self::Arena => "ARENA",
            Self::Done => "DONE",
        }
    }

    const fn color(self) -> Color {
        match self {
            Self::Setup => Color::Blue,
            Self::SelfPlay => Color::Magenta,
            Self::Fitting => Color::LightBlue,
            Self::Arena => ACTIVE,
            Self::Done => GOOD,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameKind {
    SelfPlay,
    Arena,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Queued,
    Running,
    Finished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    CandidateWin,
    IncumbentWin,
    WhiteWin,
    BlackWin,
    Draw,
}

impl Verdict {
    const fn color(self) -> Color {
        match self {
            Self::CandidateWin => GOOD,
            Self::IncumbentWin => BAD,
            Self::WhiteWin => Color::White,
            Self::BlackWin => Color::LightMagenta,
            Self::Draw => NEUTRAL,
        }
    }
}

struct GameRow {
    ruleset: Ruleset,
    candidate_side: Option<Side>,
    status: Status,
    actions: usize,
    started: Option<Instant>,
    elapsed: Option<Duration>,
    verdict: Option<Verdict>,
    result: String,
}

impl GameRow {
    fn elapsed(&self) -> Duration {
        self.elapsed
            .or_else(|| self.started.map(|started| started.elapsed()))
            .unwrap_or_default()
    }
}

#[derive(Clone, Copy, Default)]
struct Inference {
    batches: usize,
    positions: usize,
    target: usize,
    wait: Duration,
    compute: Duration,
    requests: usize,
}

struct IterationRecord {
    iteration: usize,
    score: Option<f32>,
    promoted: Option<bool>,
    policy_loss: Option<f32>,
    value_loss: Option<f32>,
}

#[derive(Clone, Copy)]
enum LogKind {
    Section,
    Field,
    Plain,
    Warning,
}

struct LogLine {
    kind: LogKind,
    label: String,
    text: String,
}

impl LogLine {
    fn plain_text(&self) -> String {
        match self.kind {
            LogKind::Section => {
                format!("\n{}\n{}", self.text, "-".repeat(self.text.chars().count()))
            }
            LogKind::Field => format!("  {:<20} {}", self.label, self.text),
            LogKind::Plain => self.text.clone(),
            LogKind::Warning => format!("warning: {}", self.text),
        }
    }
}

struct State {
    started: Instant,
    phase: Phase,
    phase_started: Instant,
    iteration: Option<(usize, usize)>,
    summary: Vec<(String, String)>,
    kind: GameKind,
    games: Vec<GameRow>,
    max_actions: usize,
    promotion_score: f32,
    total_actions: u64,
    sampled_actions: u64,
    last_sample: Instant,
    rates: VecDeque<u64>,
    inference: Inference,
    inference_started: Instant,
    history: Vec<IterationRecord>,
    logs: Vec<LogLine>,
    log_scroll: usize,
    table: TableState,
    follow_table: bool,
}

impl State {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            started: now,
            phase: Phase::Setup,
            phase_started: now,
            iteration: None,
            summary: Vec::new(),
            kind: GameKind::SelfPlay,
            games: Vec::new(),
            max_actions: 1,
            promotion_score: 0.5,
            total_actions: 0,
            sampled_actions: 0,
            last_sample: now,
            rates: VecDeque::with_capacity(SAMPLE_HISTORY),
            inference: Inference::default(),
            inference_started: now,
            history: Vec::new(),
            logs: Vec::new(),
            log_scroll: 0,
            table: TableState::default(),
            follow_table: true,
        }
    }

    fn push_log(&mut self, kind: LogKind, label: &str, text: String) {
        self.logs.push(LogLine {
            kind,
            label: label.to_owned(),
            text,
        });
    }

    fn sample(&mut self) {
        if self.last_sample.elapsed() < SAMPLE_PERIOD {
            return;
        }
        let seconds = self.last_sample.elapsed().as_secs_f64();
        let delta = self.total_actions - self.sampled_actions;
        self.sampled_actions = self.total_actions;
        self.last_sample = Instant::now();
        if self.rates.len() == SAMPLE_HISTORY {
            self.rates.pop_front();
        }
        self.rates
            .push_back((delta as f64 / seconds).round() as u64);
    }

    fn current_record(&mut self) -> &mut IterationRecord {
        let iteration = self.iteration.map_or(0, |(current, _)| current);
        if self
            .history
            .last()
            .is_none_or(|record| record.iteration != iteration)
        {
            self.history.push(IterationRecord {
                iteration,
                score: None,
                promoted: None,
                policy_loss: None,
                value_loss: None,
            });
        }
        self.history.last_mut().expect("record was just ensured")
    }

    fn tally(&self) -> (usize, usize, usize) {
        let count = |wanted: &[Verdict]| {
            self.games
                .iter()
                .filter(|game| {
                    game.verdict
                        .is_some_and(|verdict| wanted.contains(&verdict))
                })
                .count()
        };
        match self.kind {
            GameKind::Arena => (
                count(&[Verdict::CandidateWin]),
                count(&[Verdict::IncumbentWin]),
                count(&[Verdict::Draw]),
            ),
            GameKind::SelfPlay => (
                count(&[Verdict::WhiteWin]),
                count(&[Verdict::BlackWin]),
                count(&[Verdict::Draw]),
            ),
        }
    }
}

struct Dashboard {
    state: Mutex<State>,
    stop: AtomicBool,
    renderer: Mutex<Option<JoinHandle<()>>>,
}

impl Dashboard {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

fn with_state(update: impl FnOnce(&mut State)) {
    if let Some(dashboard) = DASHBOARD.get() {
        update(&mut dashboard.lock());
    }
}

/// Starts the dashboard when stdout is an interactive terminal. Returns whether it is active.
pub fn start() -> bool {
    if !io::stdout().is_terminal() || DASHBOARD.get().is_some() {
        return DASHBOARD.get().is_some();
    }
    let mut terminal = match ratatui::try_init() {
        Ok(terminal) => terminal,
        Err(error) => {
            eprintln!("warning: live dashboard unavailable ({error}); using plain output");
            return false;
        }
    };
    // The setup screen may have just used the same terminal.
    let _ = terminal.clear();
    let dashboard = DASHBOARD.get_or_init(|| Dashboard {
        state: Mutex::new(State::new()),
        stop: AtomicBool::new(false),
        renderer: Mutex::new(None),
    });
    let handle = std::thread::Builder::new()
        .name("chronofish-dashboard".to_owned())
        .spawn(move || render_loop(dashboard, terminal))
        .expect("the dashboard thread can be spawned");
    *dashboard
        .renderer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(handle);
    true
}

/// Whether reports are going to the dashboard rather than stdout.
pub fn is_active() -> bool {
    DASHBOARD.get().is_some()
}

/// Shows the final state until a key is pressed, restores the terminal, and replays the log.
pub fn shutdown(wait_for_key: bool) {
    let Some(dashboard) = DASHBOARD.get() else {
        return;
    };
    set_phase(Phase::Done);
    if wait_for_key {
        dashboard.lock().push_log(
            LogKind::Plain,
            "",
            "Finished. Press any key to exit.".to_owned(),
        );
        while !dashboard.stop.load(Ordering::Relaxed) {
            match event::read() {
                Ok(Event::Key(key)) if key.kind == KeyEventKind::Press => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
    }
    stop_renderer(dashboard);
    print_log(&dashboard.lock().logs);
}

fn stop_renderer(dashboard: &Dashboard) {
    dashboard.stop.store(true, Ordering::Relaxed);
    let handle = dashboard
        .renderer
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take();
    if let Some(handle) = handle {
        // The renderer restores the terminal before it returns.
        let _ = handle.join();
    }
}

fn print_log(logs: &[LogLine]) {
    for line in logs {
        match line.kind {
            LogKind::Warning => eprintln!("{}", line.plain_text()),
            _ => println!("{}", line.plain_text()),
        }
    }
}

pub fn section(title: &str) {
    with_state(|state| state.push_log(LogKind::Section, "", title.to_owned()));
}

pub fn field(label: &str, value: &str) {
    with_state(|state| {
        state.push_log(LogKind::Field, label, value.to_owned());
        if state.phase == Phase::Setup
            && matches!(label, "Rules" | "Model" | "Compute" | "Workers" | "GPU")
        {
            state.summary.push((label.to_owned(), value.to_owned()));
        }
    });
}

pub fn line(text: String) {
    with_state(|state| state.push_log(LogKind::Plain, "", text));
}

pub fn warning(text: String) {
    with_state(|state| state.push_log(LogKind::Warning, "", text));
}

pub fn set_iteration(current: usize, total: usize) {
    with_state(|state| state.iteration = Some((current, total)));
}

pub fn set_phase(phase: Phase) {
    with_state(|state| {
        if state.phase != phase {
            state.phase = phase;
            state.phase_started = Instant::now();
        }
    });
}

pub fn set_promotion_score(score: f32) {
    with_state(|state| state.promotion_score = score);
}

/// Replaces the game table with a freshly scheduled set of games.
pub fn schedule(kind: GameKind, games: &[(Ruleset, Option<Side>)], max_actions: usize) {
    with_state(|state| {
        state.kind = kind;
        state.max_actions = max_actions.max(1);
        state.inference = Inference::default();
        state.inference_started = Instant::now();
        state.table = TableState::default();
        state.follow_table = true;
        state.games = games
            .iter()
            .map(|&(ruleset, candidate_side)| GameRow {
                ruleset,
                candidate_side,
                status: Status::Queued,
                actions: 0,
                started: None,
                elapsed: None,
                verdict: None,
                result: String::new(),
            })
            .collect();
    });
}

pub fn game_started(index: usize) {
    with_state(|state| {
        if let Some(game) = state.games.get_mut(index) {
            game.status = Status::Running;
            game.started = Some(Instant::now());
        }
    });
}

pub fn game_progress(index: usize, actions: usize) {
    with_state(|state| {
        if let Some(game) = state.games.get_mut(index) {
            let delta = actions.saturating_sub(game.actions);
            game.actions = actions;
            state.total_actions += delta as u64;
        }
    });
}

pub fn game_finished(index: usize, actions: usize, outcome: Option<GameOutcome>) {
    game_progress(index, actions);
    with_state(|state| {
        let kind = state.kind;
        let Some(game) = state.games.get_mut(index) else {
            return;
        };
        let verdict = match (outcome.and_then(|outcome| outcome.winner), kind) {
            (None, _) => Verdict::Draw,
            (Some(winner), GameKind::Arena) if Some(winner) == game.candidate_side => {
                Verdict::CandidateWin
            }
            (Some(_), GameKind::Arena) => Verdict::IncumbentWin,
            (Some(Side::White), GameKind::SelfPlay) => Verdict::WhiteWin,
            (Some(Side::Black), GameKind::SelfPlay) => Verdict::BlackWin,
        };
        let winner = match verdict {
            Verdict::CandidateWin => "Candidate",
            Verdict::IncumbentWin => "Incumbent",
            Verdict::WhiteWin => "White",
            Verdict::BlackWin => "Black",
            Verdict::Draw => "Draw",
        };
        let reason = outcome.map_or_else(
            || "action limit".to_owned(),
            |outcome| humanize(&format!("{:?}", outcome.reason)),
        );
        game.status = Status::Finished;
        game.elapsed = Some(game.elapsed());
        game.verdict = Some(verdict);
        game.result = if verdict == Verdict::Draw {
            format!("Draw · {reason}")
        } else {
            format!("{winner} · {reason}")
        };
    });
}

/// Turns a `CamelCase` debug name into lowercase words.
fn humanize(name: &str) -> String {
    let mut words = String::with_capacity(name.len() + 4);
    for (index, character) in name.chars().enumerate() {
        if character.is_uppercase() && index > 0 {
            words.push(' ');
        }
        words.extend(character.to_lowercase());
    }
    words
}

pub fn inference_batch(
    positions: usize,
    requests: usize,
    target: usize,
    wait: Duration,
    compute: Duration,
) {
    with_state(|state| {
        let inference = &mut state.inference;
        inference.batches += 1;
        inference.positions += positions;
        inference.requests += requests;
        inference.target = target;
        inference.wait += wait;
        inference.compute += compute;
    });
}

pub fn fit_result(policy_loss: f32, value_loss: f32) {
    with_state(|state| {
        let record = state.current_record();
        record.policy_loss = Some(policy_loss);
        record.value_loss = Some(value_loss);
    });
}

pub fn arena_result(score: f32, promoted: bool) {
    with_state(|state| {
        let record = state.current_record();
        record.score = Some(score);
        record.promoted = Some(promoted);
    });
}

fn render_loop(dashboard: &'static Dashboard, mut terminal: DefaultTerminal) {
    let mut frame_index = 0_usize;
    while !dashboard.stop.load(Ordering::Relaxed) {
        {
            let mut state = dashboard.lock();
            state.sample();
            let _ = terminal.draw(|frame| draw(frame, &mut state, frame_index));
        }
        frame_index = frame_index.wrapping_add(1);
        if event::poll(TICK).unwrap_or(false)
            && let Ok(Event::Key(key)) = event::read()
            && key.kind == KeyEventKind::Press
        {
            let quit = matches!(key.code, KeyCode::Char('q') | KeyCode::Esc)
                || (key.code == KeyCode::Char('c')
                    && key.modifiers.contains(KeyModifiers::CONTROL));
            if quit {
                if dashboard.lock().phase == Phase::Done {
                    break;
                }
                ratatui::restore();
                print_log(&dashboard.lock().logs);
                eprintln!("interrupted; work saved before the current phase is kept");
                std::process::exit(130);
            }
            handle_scroll(&mut dashboard.lock(), key.code);
        }
    }
    {
        let mut state = dashboard.lock();
        let _ = terminal.draw(|frame| draw(frame, &mut state, 0));
    }
    ratatui::restore();
}

fn handle_scroll(state: &mut State, code: KeyCode) {
    let rows = state.games.len();
    match code {
        KeyCode::Down | KeyCode::Char('j') if rows > 0 => {
            state.follow_table = false;
            let next = state
                .table
                .selected()
                .map_or(0, |index| (index + 1).min(rows - 1));
            state.table.select(Some(next));
        }
        KeyCode::Up | KeyCode::Char('k') if rows > 0 => {
            state.follow_table = false;
            let next = state
                .table
                .selected()
                .map_or(0, |index| index.saturating_sub(1));
            state.table.select(Some(next));
        }
        KeyCode::Char('f') => {
            state.follow_table = true;
            state.table.select(None);
        }
        KeyCode::PageUp => state.log_scroll = state.log_scroll.saturating_add(5),
        KeyCode::PageDown => state.log_scroll = state.log_scroll.saturating_sub(5),
        KeyCode::End => state.log_scroll = 0,
        _ => {}
    }
}

fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(MUTED))
        .title(Line::from(format!(" {title} ")).fg(ACCENT).bold())
}

fn draw(frame: &mut Frame, state: &mut State, frame_index: usize) {
    let [header, body, log, footer] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Min(10),
        Constraint::Length(9),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    let [games, side] =
        Layout::horizontal([Constraint::Percentage(66), Constraint::Percentage(34)]).areas(body);
    let [score, throughput, inference, history] = Layout::vertical([
        Constraint::Length(9),
        Constraint::Length(6),
        Constraint::Length(7),
        Constraint::Min(4),
    ])
    .areas(side);

    draw_header(frame, header, state, frame_index);
    draw_games(frame, games, state, frame_index);
    draw_score(frame, score, state);
    draw_throughput(frame, throughput, state);
    draw_inference(frame, inference, state);
    draw_history(frame, history, state);
    draw_log(frame, log, state);
    draw_footer(frame, footer, state);
}

fn draw_header(frame: &mut Frame, area: Rect, state: &State, frame_index: usize) {
    let spinner = if state.phase == Phase::Done {
        "✔"
    } else {
        SPINNER[frame_index % SPINNER.len()]
    };
    let iteration = state
        .iteration
        .map_or_else(String::new, |(current, total)| {
            format!("Iteration {current}/{total}")
        });
    let title = Line::from(vec![
        Span::raw(" ♞ ").fg(ACCENT).bold(),
        Span::raw("CHRONOFISH TRAINER").bold(),
        Span::raw("  "),
        Span::raw(format!(" {spinner} {} ", state.phase.label()))
            .fg(Color::Black)
            .bg(state.phase.color())
            .bold(),
        Span::raw("  "),
        Span::raw(iteration).bold(),
    ]);
    let timing = Line::from(vec![
        Span::raw(" elapsed ").fg(MUTED),
        Span::raw(super::format_duration(state.started.elapsed())),
        Span::raw("   phase ").fg(MUTED),
        Span::raw(super::format_duration(state.phase_started.elapsed())),
        Span::raw("   actions ").fg(MUTED),
        Span::raw(super::format_count(state.total_actions as usize)),
    ]);
    let mut summary = vec![Span::raw(" ")];
    for (index, (label, value)) in state.summary.iter().enumerate() {
        if index > 0 {
            summary.push(Span::raw("  │  ").fg(MUTED));
        }
        summary.push(Span::raw(format!("{label} ")).fg(MUTED));
        summary.push(Span::raw(value.clone()));
    }
    let block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(state.phase.color()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [top, bottom] =
        Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(inner);
    let [left, right] =
        Layout::horizontal([Constraint::Min(30), Constraint::Length(48)]).areas(top);
    frame.render_widget(Paragraph::new(title), left);
    frame.render_widget(Paragraph::new(timing).alignment(Alignment::Right), right);
    frame.render_widget(Paragraph::new(Line::from(summary)), bottom);
}

const fn short_ruleset(ruleset: Ruleset) -> &'static str {
    match ruleset {
        Ruleset::Standard => "Standard",
        Ruleset::Multiverse => "5D",
        Ruleset::MultiverseVariant => "5D Variant",
    }
}

fn bar(ratio: f64, width: usize) -> (String, String) {
    const PARTIAL: [&str; 8] = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"];
    let ratio = ratio.clamp(0.0, 1.0);
    let eighths = (ratio * width as f64 * 8.0).round() as usize;
    let full = eighths / 8;
    let mut filled = "█".repeat(full);
    let mut used = full;
    if full < width && !eighths.is_multiple_of(8) {
        filled.push_str(PARTIAL[eighths % 8]);
        used += 1;
    }
    (filled, "·".repeat(width.saturating_sub(used)))
}

#[allow(clippy::too_many_lines)]
fn draw_games(frame: &mut Frame, area: Rect, state: &mut State, frame_index: usize) {
    let finished = state
        .games
        .iter()
        .filter(|game| game.status == Status::Finished)
        .count();
    let running = state
        .games
        .iter()
        .filter(|game| game.status == Status::Running)
        .count();
    let kind = match state.kind {
        GameKind::Arena => "Arena games",
        GameKind::SelfPlay => "Self-play games",
    };
    let title = format!(
        "{kind} · {finished}/{} done · {running} live",
        state.games.len()
    );
    let bar_width = 14;
    let rows = state.games.iter().enumerate().map(|(index, game)| {
        let (status, status_style) = match game.status {
            Status::Queued => ("○ queued".to_owned(), Style::new().fg(MUTED)),
            Status::Running => (
                format!("{} live", SPINNER[(frame_index + index) % SPINNER.len()]),
                Style::new().fg(ACTIVE),
            ),
            Status::Finished => (
                "● done".to_owned(),
                Style::new().fg(game.verdict.map_or(NEUTRAL, Verdict::color)),
            ),
        };
        let ratio = if game.status == Status::Finished {
            1.0
        } else {
            game.actions as f64 / state.max_actions as f64
        };
        let (filled, empty) = bar(ratio, bar_width);
        let bar_color = match game.status {
            Status::Finished => game.verdict.map_or(NEUTRAL, Verdict::color),
            Status::Running => ACCENT,
            Status::Queued => MUTED,
        };
        let progress = Line::from(vec![
            Span::raw(filled).fg(bar_color),
            Span::raw(empty).fg(MUTED),
            Span::raw(format!(" {:>4}", game.actions)),
        ]);
        let elapsed = game.elapsed();
        let rate = if game.status == Status::Queued || elapsed.is_zero() {
            "—".to_owned()
        } else {
            format!("{:.1}", game.actions as f64 / elapsed.as_secs_f64())
        };
        let side = game
            .candidate_side
            .map_or_else(|| "—".to_owned(), |side| format!("{side:?}"));
        let time = if game.status == Status::Queued {
            "—".to_owned()
        } else {
            super::format_duration(elapsed)
        };
        Row::new(vec![
            Cell::from(format!("{:>2}", index + 1)).fg(MUTED),
            Cell::from(short_ruleset(game.ruleset)),
            Cell::from(side),
            Cell::from(status).style(status_style),
            Cell::from(progress),
            Cell::from(rate),
            Cell::from(time),
            Cell::from(game.result.clone()).fg(game.verdict.map_or(NEUTRAL, Verdict::color)),
        ])
    });
    let side_header = if state.kind == GameKind::Arena {
        "Cand."
    } else {
        "Side"
    };
    let header = Row::new(vec![
        "#",
        "Rules",
        side_header,
        "Status",
        "Progress",
        "act/s",
        "Time",
        "Result",
    ])
    .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD))
    .bottom_margin(1);
    let widths = [
        Constraint::Length(3),
        Constraint::Length(10),
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Length(bar_width as u16 + 5),
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Min(12),
    ];
    if state.follow_table {
        // Keep the earliest live game in view while following.
        let focus = state
            .games
            .iter()
            .position(|game| game.status != Status::Finished)
            .unwrap_or(state.games.len());
        let visible = usize::from(area.height.saturating_sub(4)).max(1);
        *state.table.offset_mut() = focus
            .saturating_sub(2)
            .min(state.games.len().saturating_sub(visible));
    }
    let table = Table::new(rows, widths)
        .header(header)
        .column_spacing(1)
        .row_highlight_style(Style::new().bg(Color::Rgb(40, 44, 52)).bold())
        .block(panel(&title));
    if state.games.is_empty() {
        let waiting = Paragraph::new(Line::from("Waiting for the first schedule…").fg(MUTED))
            .alignment(Alignment::Center)
            .block(panel(kind));
        frame.render_widget(waiting, area);
    } else {
        frame.render_stateful_widget(table, area, &mut state.table);
    }
}

#[allow(clippy::too_many_lines)]
fn draw_score(frame: &mut Frame, area: Rect, state: &State) {
    let (first, second, draws) = state.tally();
    let played = first + second + draws;
    let total = state.games.len();
    let block = panel(if state.kind == GameKind::Arena {
        "Scoreboard"
    } else {
        "Self-play results"
    });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let [rows_area, status_area, gauge_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(2),
        Constraint::Length(1),
    ])
    .areas(inner);
    let labels = match state.kind {
        GameKind::Arena => [("Candidate", GOOD), ("Incumbent", BAD), ("Draws", NEUTRAL)],
        GameKind::SelfPlay => [
            ("White", Color::White),
            ("Black", Color::LightMagenta),
            ("Draws", NEUTRAL),
        ],
    };
    let width = usize::from(rows_area.width).saturating_sub(18).max(4);
    let lines = labels
        .iter()
        .zip([first, second, draws])
        .map(|(&(label, color), count)| {
            let (filled, empty) = bar(
                if total == 0 {
                    0.0
                } else {
                    count as f64 / total as f64
                },
                width,
            );
            Line::from(vec![
                Span::raw(format!(" {label:<10}")).fg(color).bold(),
                Span::raw(filled).fg(color),
                Span::raw(empty).fg(MUTED),
                Span::raw(format!(" {count:>3}")).bold(),
            ])
        })
        .collect::<Vec<_>>();
    frame.render_widget(Paragraph::new(lines), rows_area);

    if state.kind == GameKind::SelfPlay {
        let positions: usize = state.games.iter().map(|game| game.actions).sum();
        let status = vec![
            Line::from(vec![
                Span::raw(" Played ").fg(MUTED),
                Span::raw(format!("{played}/{total}")),
                Span::raw("   positions ").fg(MUTED),
                Span::raw(super::format_count(positions)),
            ]),
            Line::default(),
        ];
        frame.render_widget(Paragraph::new(status), status_area);
        let ratio = if total == 0 {
            0.0
        } else {
            played as f64 / total as f64
        };
        frame.render_widget(
            Gauge::default()
                .ratio(ratio)
                .label(format!("{:.0}% complete", ratio * 100.0))
                .gauge_style(Style::new().fg(Color::Magenta).bg(Color::Rgb(30, 30, 36))),
            gauge_area,
        );
        return;
    }

    let threshold = f64::from(state.promotion_score);
    let remaining = total - played;
    let score = |wins: usize, draws: usize| {
        if total == 0 {
            0.0
        } else {
            (wins as f64 + draws as f64 * 0.5) / total as f64
        }
    };
    let current = if played == 0 {
        0.0
    } else {
        (first as f64 + draws as f64 * 0.5) / played as f64
    };
    let floor = score(first, draws);
    let ceiling = score(first + remaining, draws);
    let (verdict, verdict_color) = if total == 0 {
        ("no games scheduled", MUTED)
    } else if floor >= threshold {
        ("clinched · promote", GOOD)
    } else if ceiling < threshold {
        ("clinched · retain", BAD)
    } else if current >= threshold {
        ("on track to promote", GOOD)
    } else {
        ("below the bar", ACTIVE)
    };
    let status = vec![
        Line::from(vec![
            Span::raw(" Played ").fg(MUTED),
            Span::raw(format!("{played}/{total}")),
            Span::raw("   range ").fg(MUTED),
            Span::raw(format!("{floor:.3}–{ceiling:.3}")),
        ]),
        Line::from(vec![
            Span::raw(" Outlook ").fg(MUTED),
            Span::raw(verdict).fg(verdict_color).bold(),
        ]),
    ];
    frame.render_widget(Paragraph::new(status), status_area);
    let color = if current >= threshold { GOOD } else { BAD };
    frame.render_widget(
        Gauge::default()
            .ratio(current.clamp(0.0, 1.0))
            .label(format!("score {current:.3} · need {threshold:.3}"))
            .gauge_style(Style::new().fg(color).bg(Color::Rgb(30, 30, 36))),
        gauge_area,
    );
}

fn draw_throughput(frame: &mut Frame, area: Rect, state: &State) {
    let current = state.rates.back().copied().unwrap_or(0);
    let peak = state.rates.iter().copied().max().unwrap_or(0);
    let title = format!("Throughput · {current} act/s · peak {peak}");
    let block = panel(&title);
    let width = usize::from(block.inner(area).width);
    let skip = state.rates.len().saturating_sub(width);
    let data = state.rates.iter().skip(skip).copied().collect::<Vec<_>>();
    frame.render_widget(
        Sparkline::default()
            .block(block)
            .data(&data)
            .style(Style::new().fg(ACCENT)),
        area,
    );
}

fn draw_inference(frame: &mut Frame, area: Rect, state: &State) {
    let inference = state.inference;
    let lines = if inference.batches == 0 {
        vec![Line::from(" no inference batches yet").fg(MUTED)]
    } else {
        let mean = inference.positions as f64 / inference.batches as f64;
        let fill = if inference.target == 0 {
            0.0
        } else {
            mean / inference.target as f64
        };
        let wait = inference.wait / u32::try_from(inference.requests.max(1)).unwrap_or(u32::MAX);
        let window = state
            .inference_started
            .elapsed()
            .as_secs_f64()
            .max(f64::EPSILON);
        // Candidate and incumbent coordinators both report here, so this can exceed 100%.
        let compute_share = inference.compute.as_secs_f64() / window;
        let busy = compute_share.min(1.0);
        let (fill_bar, fill_rest) = bar(fill, 12);
        let (busy_bar, busy_rest) = bar(busy, 12);
        vec![
            Line::from(vec![
                Span::raw(" Batches   ").fg(MUTED),
                Span::raw(super::format_count(inference.batches)),
                Span::raw("  positions ").fg(MUTED),
                Span::raw(super::format_count(inference.positions)),
            ]),
            Line::from(vec![
                Span::raw(" Fill      ").fg(MUTED),
                Span::raw(fill_bar).fg(ACCENT),
                Span::raw(fill_rest).fg(MUTED),
                Span::raw(format!(" {mean:.1}/{}", inference.target)),
            ]),
            Line::from(vec![
                Span::raw(" Compute   ").fg(MUTED),
                Span::raw(busy_bar).fg(Color::LightBlue),
                Span::raw(busy_rest).fg(MUTED),
                Span::raw(format!(
                    " {} · {:.0}% of wall",
                    super::format_duration(inference.compute),
                    compute_share * 100.0
                )),
            ]),
            Line::from(vec![
                Span::raw(" Queue wait ").fg(MUTED),
                Span::raw(format!("{wait:.2?}")),
            ]),
        ]
    };
    frame.render_widget(Paragraph::new(lines).block(panel("Inference")), area);
}

fn draw_history(frame: &mut Frame, area: Rect, state: &State) {
    let block = panel("Iterations");
    if state.history.is_empty() {
        frame.render_widget(
            Paragraph::new(Line::from(" no completed fits yet").fg(MUTED)).block(block),
            area,
        );
        return;
    }
    let rows = state.history.iter().rev().map(|record| {
        let decision = match record.promoted {
            Some(true) => Cell::from("▲ promoted").fg(GOOD),
            Some(false) => Cell::from("▼ retained").fg(BAD),
            None => Cell::from("…").fg(MUTED),
        };
        let number = |value: Option<f32>, digits: usize| {
            value.map_or_else(|| "—".to_owned(), |value| format!("{value:.digits$}"))
        };
        Row::new(vec![
            Cell::from(format!("#{}", record.iteration)).fg(MUTED),
            Cell::from(number(record.score, 3)).bold(),
            decision,
            Cell::from(number(record.policy_loss, 4)),
            Cell::from(number(record.value_loss, 4)),
        ])
    });
    let header = Row::new(vec!["It", "Score", "Decision", "Policy", "Value"])
        .style(Style::new().fg(ACCENT).add_modifier(Modifier::BOLD));
    let table = Table::new(
        rows,
        [
            Constraint::Length(4),
            Constraint::Length(6),
            Constraint::Length(11),
            Constraint::Length(7),
            Constraint::Length(7),
        ],
    )
    .header(header)
    .block(block);
    frame.render_widget(table, area);
}

fn draw_log(frame: &mut Frame, area: Rect, state: &State) {
    let block = panel("Log");
    let height = usize::from(block.inner(area).height);
    let lines = state
        .logs
        .iter()
        .map(|line| match line.kind {
            LogKind::Section => Line::from(vec![
                Span::raw("▌ ").fg(ACCENT),
                Span::raw(line.text.clone()).fg(ACCENT).bold(),
            ]),
            LogKind::Field => Line::from(vec![
                Span::raw(format!("  {:<20} ", line.label)).fg(MUTED),
                Span::raw(line.text.clone()),
            ]),
            LogKind::Plain => Line::from(line.text.clone()),
            LogKind::Warning => Line::from(format!("⚠ {}", line.text)).fg(ACTIVE),
        })
        .collect::<Vec<_>>();
    let max_scroll = lines.len().saturating_sub(height);
    let scroll = max_scroll.saturating_sub(state.log_scroll.min(max_scroll));
    frame.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: false })
            .scroll((u16::try_from(scroll).unwrap_or(u16::MAX), 0)),
        area,
    );
}

fn draw_footer(frame: &mut Frame, area: Rect, state: &State) {
    let key = |key: &'static str| Span::raw(format!(" {key} ")).fg(Color::Black).bg(MUTED);
    let quit = if state.phase == Phase::Done {
        " exit   "
    } else {
        " quit   "
    };
    let footer = Line::from(vec![
        key("q"),
        Span::raw(quit),
        key("↑↓"),
        Span::raw(" select game   "),
        key("f"),
        Span::raw(if state.follow_table {
            " following   "
        } else {
            " follow live   "
        }),
        key("PgUp/PgDn"),
        Span::raw(" scroll log"),
    ])
    .fg(MUTED);
    frame.render_widget(Paragraph::new(footer), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bars_fill_to_the_requested_width() {
        let (filled, empty) = bar(0.5, 10);
        assert_eq!(filled.chars().count() + empty.chars().count(), 10);
        let (filled, empty) = bar(1.0, 10);
        assert_eq!((filled.chars().count(), empty.chars().count()), (10, 0));
        let (filled, empty) = bar(0.0, 10);
        assert_eq!((filled.chars().count(), empty.chars().count()), (0, 10));
    }

    #[test]
    fn outcome_reasons_read_as_words() {
        assert_eq!(humanize("ThreefoldRepetition"), "threefold repetition");
        assert_eq!(humanize("Checkmate"), "checkmate");
    }

    #[test]
    fn dashboard_renders_an_arena_in_progress() {
        use ratatui::Terminal;
        use ratatui::backend::TestBackend;

        let mut state = State::new();
        state.kind = GameKind::Arena;
        state.max_actions = 100;
        state.iteration = Some((1, 3));
        state.games = (0..3)
            .map(|index| GameRow {
                ruleset: Ruleset::Standard,
                candidate_side: Some(Side::White),
                status: if index == 0 {
                    Status::Finished
                } else {
                    Status::Running
                },
                actions: 40,
                started: Some(Instant::now()),
                elapsed: None,
                verdict: (index == 0).then_some(Verdict::CandidateWin),
                result: String::new(),
            })
            .collect();
        let mut terminal = Terminal::new(TestBackend::new(140, 40)).unwrap();
        terminal.draw(|frame| draw(frame, &mut state, 0)).unwrap();
        let text = format!("{}", terminal.backend());
        assert!(text.contains("Scoreboard"));
        assert!(text.contains("Arena games · 1/3 done · 2 live"));
    }
}
