//! Native training orchestration shared by the CPU and GPU command-line tools.

use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fs::{self, File, OpenOptions},
    io::{self, IsTerminal, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
        Mutex,
        OnceLock,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, List, ListItem, Paragraph, Wrap},
    Frame,
};
use serde::{Deserialize, Serialize};

pub struct TerminalGuard {
    active: bool,
}

impl TerminalGuard {
    pub fn enter() -> Result<Self, String> {
        install_terminal_panic_hook();
        crossterm::terminal::enable_raw_mode()
            .map_err(|error| format!("failed to enable terminal raw mode: {error}"))?;
        if let Err(error) = crossterm::execute!(
            io::stdout(),
            crossterm::terminal::EnterAlternateScreen,
            crossterm::event::EnableMouseCapture,
            crossterm::cursor::Hide
        ) {
            let _ = crossterm::terminal::disable_raw_mode();
            return Err(format!("failed to enter training terminal: {error}"));
        }
        Ok(Self { active: true })
    }

    pub fn restore(&mut self) {
        if self.active {
            restore_terminal();
            self.active = false;
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore();
    }
}

fn restore_terminal() {
    let _ = crossterm::execute!(
        io::stdout(),
        crossterm::cursor::Show,
        crossterm::event::DisableMouseCapture,
        crossterm::terminal::LeaveAlternateScreen
    );
    let _ = crossterm::terminal::disable_raw_mode();
}

fn install_terminal_panic_hook() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore_terminal();
            previous(info);
        }));
    });
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum UiMode {
    #[default]
    Auto,
    Tui,
    Plain,
    Json,
}

static GLOBAL_UI_MODE: AtomicU8 = AtomicU8::new(0);

pub fn set_global_ui_mode(mode: UiMode) {
    let value = match mode.resolve() {
        UiMode::Auto => 0,
        UiMode::Tui => 1,
        UiMode::Plain => 2,
        UiMode::Json => 3,
    };
    GLOBAL_UI_MODE.store(value, Ordering::Release);
}

pub fn global_ui_mode() -> UiMode {
    match GLOBAL_UI_MODE.load(Ordering::Acquire) {
        1 => UiMode::Tui,
        2 => UiMode::Plain,
        3 => UiMode::Json,
        _ => UiMode::Auto,
    }
}

pub fn render_structured_event(event: &TrainingEvent) {
    if let Ok(sender) = event_sender().lock() {
        if let Some(sender) = sender.as_ref() {
            let _ = sender.send(event.clone());
            return;
        }
    }
    if GLOBAL_UI_MODE.load(Ordering::Acquire) == 0 {
        return;
    }
    let mode = global_ui_mode();
    if mode != UiMode::Tui {
        render_event(mode, event);
    }
}

pub fn log(level: LogLevel, scope: impl Into<String>, message: impl Into<String>) {
    render_structured_event(&TrainingEvent::Log {
        level,
        scope: scope.into(),
        message: message.into(),
    });
}

fn event_sender() -> &'static Mutex<Option<Sender<TrainingEvent>>> {
    static SENDER: OnceLock<Mutex<Option<Sender<TrainingEvent>>>> = OnceLock::new();
    SENDER.get_or_init(|| Mutex::new(None))
}

struct EventSessionGuard;

impl Drop for EventSessionGuard {
    fn drop(&mut self) {
        if let Ok(mut sender) = event_sender().lock() {
            *sender = None;
        }
    }
}

fn global_control() -> &'static ControlToken {
    static CONTROL: OnceLock<ControlToken> = OnceLock::new();
    CONTROL.get_or_init(ControlToken::default)
}

fn global_deadline() -> &'static Mutex<Option<Instant>> {
    static DEADLINE: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();
    DEADLINE.get_or_init(|| Mutex::new(None))
}

pub fn set_cooperative_deadline(deadline: Option<Instant>) {
    if let Ok(mut current) = global_deadline().lock() {
        *current = deadline;
    }
}

fn cooperative_deadline() -> Option<Instant> {
    global_deadline().lock().ok().and_then(|deadline| *deadline)
}

pub fn cooperative_checkpoint() -> Checkpoint {
    global_control().wait_at_checkpoint(cooperative_deadline())
}

pub fn cooperative_cancelled() -> bool {
    matches!(
        global_control().checkpoint(cooperative_deadline()),
        Checkpoint::Cancelled
    )
}

pub fn cooperative_timed_out() -> bool {
    matches!(
        global_control().checkpoint(cooperative_deadline()),
        Checkpoint::TimedOut
    )
}

pub fn run_interactive<T, F>(work: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let (sender, receiver) = event_channel();
    *event_sender()
        .lock()
        .map_err(|_| "training event sender lock poisoned")? = Some(sender);
    let _event_session = EventSessionGuard;
    global_control().reset();
    let mut guard = match TerminalGuard::enter() {
        Ok(guard) => guard,
        Err(error) => {
            if let Ok(mut sender) = event_sender().lock() {
                *sender = None;
            }
            return Err(error);
        }
    };
    let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
    let mut terminal = match ratatui::Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(error) => {
            if let Ok(mut sender) = event_sender().lock() {
                *sender = None;
            }
            return Err(format!("failed to create training terminal: {error}"));
        }
    };
    let join = thread::spawn(work);
    let mut reducer = EventReducer::default();
    let mut viewport = Viewport::default();
    while !join.is_finished() {
        while let Ok(event) = receiver.try_recv() {
            reducer.apply(&event);
        }
        let rows = reducer.jobs().count();
        let geometry = terminal
            .size()
            .map(|size| ui_geometry(Rect::new(0, 0, size.width, size.height)))
            .unwrap_or_default();
        viewport.select(viewport.selected, rows, geometry.visible_jobs);
        terminal
            .draw(|frame| draw_training_ui(frame, &reducer, viewport, cooperative_deadline()))
            .map_err(|error| format!("failed to draw training terminal: {error}"))?;
        if crossterm::event::poll(Duration::from_millis(50))
            .map_err(|error| format!("failed to poll terminal: {error}"))?
        {
            let event = crossterm::event::read()
                .map_err(|error| format!("failed to read terminal input: {error}"))?;
            match handle_ui_event(
                event,
                &mut viewport,
                rows,
                geometry.visible_jobs,
                geometry.first_job_row,
            ) {
                UiAction::PauseResume => match global_control().checkpoint(None) {
                    Checkpoint::Paused => {
                        global_control().resume();
                        render_structured_event(&TrainingEvent::Control {
                            state: JobState::Running,
                            message: "resume requested".into(),
                        });
                    }
                    _ => {
                        global_control().request_pause();
                        render_structured_event(&TrainingEvent::Control {
                            state: JobState::PauseRequested,
                            message: "pause requested; waiting for a safe checkpoint".into(),
                        });
                    }
                },
                UiAction::Cancel | UiAction::Quit => {
                    global_control().cancel();
                    render_structured_event(&TrainingEvent::Control {
                        state: JobState::CancelRequested,
                        message: "cancellation requested; waiting for a safe checkpoint".into(),
                    });
                }
                UiAction::Restart => {
                    global_control().reset();
                    if let Some(job) = reducer.jobs().nth(viewport.selected) {
                        render_structured_event(&TrainingEvent::Restarted {
                            job_id: job.metadata.id.clone(),
                        });
                    }
                }
                UiAction::None => {}
            }
        }
    }
    while let Ok(event) = receiver.try_recv() {
        reducer.apply(&event);
    }
    guard.restore();
    join.join()
        .map_err(|_| "training worker panicked".to_string())
}

impl UiMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "tui" => Ok(Self::Tui),
            "plain" => Ok(Self::Plain),
            "json" => Ok(Self::Json),
            other => Err(format!(
                "unknown UI mode `{other}`; expected auto, tui, plain, or json"
            )),
        }
    }

    pub fn resolve(self) -> Self {
        match self {
            Self::Auto if io::stdin().is_terminal() && io::stdout().is_terminal() => Self::Tui,
            Self::Auto => Self::Plain,
            mode => mode,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JobState {
    Pending,
    Running,
    PauseRequested,
    Paused,
    CancelRequested,
    Cancelled,
    Completed,
    Failed,
    TimedOut,
    DependencyBlocked,
}

impl JobState {
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Cancelled | Self::Completed | Self::Failed | Self::TimedOut
        )
    }

    pub fn can_transition_to(self, next: Self) -> bool {
        use JobState::*;
        matches!(
            (self, next),
            (
                Pending,
                Running
                    | PauseRequested
                    | CancelRequested
                    | Cancelled
                    | DependencyBlocked
                    | TimedOut
            ) | (
                Running,
                PauseRequested | CancelRequested | Completed | Failed | TimedOut
            ) | (
                PauseRequested,
                Running | Paused | CancelRequested | Completed | Failed | TimedOut
            ) | (Paused, Running | CancelRequested | TimedOut)
                | (CancelRequested, Cancelled | Completed | Failed | TimedOut)
                | (DependencyBlocked, Pending | Cancelled)
                | (Cancelled | Failed | TimedOut, Pending)
        ) || self == next
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobProgress {
    pub current: u64,
    pub total: u64,
    pub current_metric: Option<f64>,
    pub best_metric: Option<f64>,
    pub games_or_samples: u64,
    pub detail: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobMetadata {
    pub id: String,
    pub label: String,
    pub kind: String,
    pub seed: u64,
    pub dependencies: Vec<String>,
    pub persistence_path: Option<PathBuf>,
    pub detail: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSnapshot {
    pub metadata: JobMetadata,
    pub state: JobState,
    pub progress: JobProgress,
    pub error: Option<String>,
    pub latest_journal_event: Option<String>,
    pub restart_count: u32,
}

impl JobSnapshot {
    pub fn new(metadata: JobMetadata) -> Self {
        Self {
            metadata,
            state: JobState::Pending,
            progress: JobProgress::default(),
            error: None,
            latest_journal_event: None,
            restart_count: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum TrainingEvent {
    Added {
        job: JobSnapshot,
    },
    State {
        job_id: String,
        state: JobState,
        error: Option<String>,
    },
    Progress {
        job_id: String,
        progress: JobProgress,
    },
    Persisted {
        job_id: String,
        path: PathBuf,
        summary: String,
    },
    Restarted {
        job_id: String,
    },
    Control {
        state: JobState,
        message: String,
    },
    Log {
        level: LogLevel,
        scope: String,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Debug,
    Info,
    Warn,
    Error,
    Success,
}

#[derive(Default)]
pub struct EventReducer {
    jobs: BTreeMap<String, JobSnapshot>,
    logs: VecDeque<(LogLevel, String, String)>,
}

impl EventReducer {
    pub fn apply(&mut self, event: &TrainingEvent) {
        match event {
            TrainingEvent::Added { job } => {
                if let Some(existing) = self.jobs.get_mut(&job.metadata.id) {
                    existing.metadata = job.metadata.clone();
                } else {
                    self.jobs.insert(job.metadata.id.clone(), job.clone());
                }
                if job.metadata.kind != "paired-match" {
                    self.record_log(
                        LogLevel::Debug,
                        job.metadata.id.clone(),
                        format!(
                            "registered kind={} dependencies={}",
                            job.metadata.kind,
                            dependency_summary(&job.metadata.dependencies)
                        ),
                    );
                }
            }
            TrainingEvent::State {
                job_id,
                state,
                error,
            } => {
                if let Some(job) = self.jobs.get_mut(job_id) {
                    if job.state.can_transition_to(*state) {
                        job.state = *state;
                        job.error = error.clone();
                    }
                }
                if !job_id.starts_with("cpu-match-") {
                    self.record_log(
                        if *state == JobState::Failed {
                            LogLevel::Error
                        } else if matches!(state, JobState::TimedOut | JobState::Cancelled) {
                            LogLevel::Warn
                        } else if *state == JobState::Completed {
                            LogLevel::Success
                        } else {
                            LogLevel::Info
                        },
                        job_id.clone(),
                        error.as_ref().map_or_else(
                            || format!("state={state:?}"),
                            |error| format!("state={state:?} error={error}"),
                        ),
                    );
                }
            }
            TrainingEvent::Progress { job_id, progress } => {
                let job = self.jobs.entry(job_id.clone()).or_insert_with(|| {
                    JobSnapshot::new(JobMetadata {
                        id: job_id.clone(),
                        label: job_id.clone(),
                        kind: "training".into(),
                        seed: 0,
                        dependencies: Vec::new(),
                        persistence_path: None,
                        detail: BTreeMap::new(),
                    })
                });
                if job.state == JobState::Pending {
                    job.state = JobState::Running;
                }
                job.progress = progress.clone();
            }
            TrainingEvent::Persisted {
                job_id,
                path,
                summary,
            } => {
                let job = self.jobs.entry(job_id.clone()).or_insert_with(|| {
                    JobSnapshot::new(JobMetadata {
                        id: job_id.clone(),
                        label: job_id.clone(),
                        kind: "persistence".into(),
                        seed: 0,
                        dependencies: Vec::new(),
                        persistence_path: None,
                        detail: BTreeMap::new(),
                    })
                });
                job.metadata.persistence_path = Some(path.clone());
                job.latest_journal_event = Some(summary.clone());
                self.record_log(
                    LogLevel::Success,
                    job_id.clone(),
                    format!("persisted {}: {summary}", path.display()),
                );
            }
            TrainingEvent::Restarted { job_id } => {
                if let Some(job) = self.jobs.get_mut(job_id) {
                    job.state = JobState::Pending;
                    job.progress = JobProgress::default();
                    job.error = None;
                    job.restart_count += 1;
                }
                self.record_log(LogLevel::Info, job_id.clone(), "restarted".into());
            }
            TrainingEvent::Control { state, message } => {
                for job in self.jobs.values_mut().filter(|job| !job.state.terminal()) {
                    if job.state.can_transition_to(*state) {
                        job.state = *state;
                    }
                }
                self.record_log(
                    LogLevel::Info,
                    "controls".into(),
                    format!("state={state:?} {message}"),
                );
            }
            TrainingEvent::Log {
                level,
                scope,
                message,
            } => {
                self.record_log(*level, scope.clone(), message.clone());
            }
        }
    }

    fn record_log(&mut self, level: LogLevel, scope: String, message: String) {
        self.logs.push_back((level, scope, message));
        while self.logs.len() > 200 {
            self.logs.pop_front();
        }
    }

    pub fn jobs(&self) -> impl Iterator<Item = &JobSnapshot> {
        self.jobs.values()
    }

    pub fn logs(&self) -> std::collections::vec_deque::Iter<'_, (LogLevel, String, String)> {
        self.logs.iter()
    }

    pub fn invalidate_downstream(&mut self, upstream: &str) -> Vec<String> {
        let mut invalidated = Vec::new();
        let mut frontier = vec![upstream.to_string()];
        while let Some(id) = frontier.pop() {
            let dependants = self
                .jobs
                .values()
                .filter(|job| job.metadata.dependencies.contains(&id))
                .map(|job| job.metadata.id.clone())
                .collect::<Vec<_>>();
            for dependant in dependants {
                if invalidated.contains(&dependant) {
                    continue;
                }
                if let Some(job) = self.jobs.get_mut(&dependant) {
                    job.state = JobState::DependencyBlocked;
                    job.progress = JobProgress::default();
                    job.error = None;
                }
                invalidated.push(dependant.clone());
                frontier.push(dependant);
            }
        }
        invalidated
    }
}

fn dependency_summary(dependencies: &[String]) -> String {
    if dependencies.is_empty() {
        "none".into()
    } else {
        dependencies.join(", ")
    }
}

const RUN: u8 = 0;
const PAUSE: u8 = 1;
const CANCEL: u8 = 2;

#[derive(Clone, Default)]
pub struct ControlToken {
    state: Arc<AtomicU8>,
    pause_announced: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checkpoint {
    Continue,
    Paused,
    Cancelled,
    TimedOut,
}

impl ControlToken {
    pub fn reset(&self) {
        self.state.store(RUN, Ordering::Release);
        self.pause_announced.store(false, Ordering::Release);
    }

    pub fn request_pause(&self) {
        self.pause_announced.store(false, Ordering::Release);
        self.state.store(PAUSE, Ordering::Release);
    }

    pub fn resume(&self) {
        self.state
            .compare_exchange(PAUSE, RUN, Ordering::AcqRel, Ordering::Acquire)
            .ok();
        self.pause_announced.store(false, Ordering::Release);
    }

    pub fn cancel(&self) {
        self.state.store(CANCEL, Ordering::Release);
    }

    pub fn checkpoint(&self, deadline: Option<Instant>) -> Checkpoint {
        if deadline.is_some_and(|limit| Instant::now() >= limit) {
            return Checkpoint::TimedOut;
        }
        match self.state.load(Ordering::Acquire) {
            CANCEL => Checkpoint::Cancelled,
            PAUSE => Checkpoint::Paused,
            _ => Checkpoint::Continue,
        }
    }

    pub fn wait_at_checkpoint(&self, deadline: Option<Instant>) -> Checkpoint {
        let mut announced_pause = false;
        loop {
            match self.checkpoint(deadline) {
                Checkpoint::Paused => {
                    if self
                        .pause_announced
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        render_structured_event(&TrainingEvent::Control {
                            state: JobState::Paused,
                            message: "paused at a safe checkpoint".into(),
                        });
                        announced_pause = true;
                    }
                    thread::sleep(Duration::from_millis(25));
                }
                result => {
                    if announced_pause && result == Checkpoint::Continue {
                        render_structured_event(&TrainingEvent::Control {
                            state: JobState::Running,
                            message: "training resumed".into(),
                        });
                    }
                    return result;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Viewport {
    pub selected: usize,
    pub offset: usize,
}

impl Viewport {
    pub fn select(&mut self, index: usize, rows: usize, visible: usize) {
        if rows == 0 {
            self.selected = 0;
            self.offset = 0;
            return;
        }
        self.selected = index.min(rows - 1);
        let visible = visible.max(1);
        if self.selected < self.offset {
            self.offset = self.selected;
        }
        if self.selected >= self.offset + visible {
            self.offset = self.selected + 1 - visible;
        }
        self.offset = self.offset.min(rows.saturating_sub(visible));
    }

    pub fn move_by(&mut self, delta: isize, rows: usize, visible: usize) {
        self.select(self.selected.saturating_add_signed(delta), rows, visible);
    }

    pub fn hit_test(
        &self,
        mouse_y: u16,
        first_row_y: u16,
        visible: usize,
        rows: usize,
    ) -> Option<usize> {
        let relative = mouse_y.checked_sub(first_row_y)? as usize;
        (relative < visible)
            .then_some(self.offset + relative)
            .filter(|index| *index < rows)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiAction {
    None,
    PauseResume,
    Cancel,
    Restart,
    Quit,
}

pub fn handle_ui_event(
    event: crossterm::event::Event,
    viewport: &mut Viewport,
    rows: usize,
    visible: usize,
    first_row_y: u16,
) -> UiAction {
    use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind};
    match event {
        Event::Key(key) if key.kind == KeyEventKind::Release => UiAction::None,
        Event::Key(key)
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') =>
        {
            UiAction::Quit
        }
        Event::Key(key) => match key.code {
            KeyCode::Char('q') => UiAction::Quit,
            KeyCode::Char(' ') => UiAction::PauseResume,
            KeyCode::Char('c') => UiAction::Cancel,
            KeyCode::Char('r') => UiAction::Restart,
            KeyCode::Up | KeyCode::Char('k') => {
                viewport.move_by(-1, rows, visible);
                UiAction::None
            }
            KeyCode::Down | KeyCode::Char('j') => {
                viewport.move_by(1, rows, visible);
                UiAction::None
            }
            KeyCode::PageUp => {
                viewport.move_by(-(visible as isize), rows, visible);
                UiAction::None
            }
            KeyCode::PageDown => {
                viewport.move_by(visible as isize, rows, visible);
                UiAction::None
            }
            KeyCode::Home => {
                viewport.select(0, rows, visible);
                UiAction::None
            }
            KeyCode::End => {
                viewport.select(rows.saturating_sub(1), rows, visible);
                UiAction::None
            }
            _ => UiAction::None,
        },
        Event::Mouse(mouse) => match mouse.kind {
            MouseEventKind::Moved => {
                if let Some(index) = viewport.hit_test(mouse.row, first_row_y, visible, rows) {
                    viewport.select(index, rows, visible);
                }
                UiAction::None
            }
            MouseEventKind::ScrollUp => {
                viewport.move_by(-3, rows, visible);
                UiAction::None
            }
            MouseEventKind::ScrollDown => {
                viewport.move_by(3, rows, visible);
                UiAction::None
            }
            _ => UiAction::None,
        },
        _ => UiAction::None,
    }
}

pub fn draw_training_ui(
    frame: &mut Frame<'_>,
    reducer: &EventReducer,
    viewport: Viewport,
    deadline: Option<Instant>,
) {
    let area = frame.area();
    let geometry = ui_geometry(area);
    let chunks = geometry.chunks;
    let panes = geometry.panes;
    let jobs = reducer.jobs().collect::<Vec<_>>();
    let visible = geometry.visible_jobs;
    let items = jobs
        .iter()
        .skip(viewport.offset)
        .take(visible)
        .enumerate()
        .map(|(offset, job)| {
            let marker = if viewport.offset + offset == viewport.selected {
                "▶"
            } else {
                " "
            };
            let progress = if job.progress.total == 0 {
                "-".into()
            } else {
                format!("{}/{}", job.progress.current, job.progress.total)
            };
            let line = format!(
                "{marker} {:<20} {:<17} {:>9} {}",
                job.metadata.label,
                format!("{:?}", job.state),
                progress,
                job.progress.detail
            );
            let style = if viewport.offset + offset == viewport.selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            ListItem::new(Line::from(line)).style(style)
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        List::new(items).block(
            Block::default()
                .title(" Training jobs ")
                .borders(Borders::ALL),
        ),
        panes[0],
    );
    let detail = jobs
        .get(viewport.selected)
        .map(|job| {
            let mut lines = vec![
                format!("id: {}", job.metadata.id),
                format!("state: {:?}", job.state),
                format!("seed: {}", job.metadata.seed),
                format!(
                    "dependencies: {}",
                    dependency_summary(&job.metadata.dependencies)
                ),
                format!(
                    "artifact: {}",
                    job.metadata
                        .persistence_path
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "-".into())
                ),
                format!(
                    "current/best: {:?} / {:?}",
                    job.progress.current_metric, job.progress.best_metric
                ),
                format!("games/samples: {}", job.progress.games_or_samples),
            ];
            for (key, value) in &job.metadata.detail {
                lines.push(format!("{key}: {value}"));
            }
            if let Some(error) = &job.error {
                lines.push(format!("error: {error}"));
            }
            if let Some(event) = &job.latest_journal_event {
                lines.push(format!("journal: {event}"));
            }
            lines.join("\n")
        })
        .unwrap_or_else(|| "No job selected".into());
    frame.render_widget(
        Paragraph::new(detail)
            .wrap(Wrap { trim: false })
            .block(Block::default().title(" Details ").borders(Borders::ALL)),
        panes[1],
    );
    let log_lines = reducer
        .logs()
        .rev()
        .take(chunks[1].height.saturating_sub(2) as usize)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .map(|(level, scope, message)| {
            Line::styled(
                format!("[{level:?}] {scope}: {message}"),
                Style::default().fg(match level {
                    LogLevel::Debug => Color::DarkGray,
                    LogLevel::Info => Color::White,
                    LogLevel::Warn => Color::Yellow,
                    LogLevel::Error => Color::Red,
                    LogLevel::Success => Color::Green,
                }),
            )
        })
        .collect::<Vec<_>>();
    frame.render_widget(
        Paragraph::new(log_lines)
            .wrap(Wrap { trim: false })
            .block(Block::default().title(" Log ").borders(Borders::ALL)),
        chunks[1],
    );
    let remaining = deadline
        .map(|limit| {
            limit
                .saturating_duration_since(Instant::now())
                .as_secs()
                .to_string()
        })
        .unwrap_or_else(|| "unlimited".into());
    frame.render_widget(Paragraph::new(format!("↑/↓ j/k PgUp/PgDn Home/End  Space pause  c cancel  r restart  q quit    remaining: {remaining}s")), Rect { x: chunks[2].x, y: chunks[2].y, width: chunks[2].width, height: chunks[2].height });
}

pub fn event_channel() -> (Sender<TrainingEvent>, Receiver<TrainingEvent>) {
    mpsc::channel()
}

pub fn render_event(mode: UiMode, event: &TrainingEvent) {
    match mode.resolve() {
        UiMode::Json => println!(
            "{}",
            serde_json::to_string(event).expect("training event should serialize")
        ),
        UiMode::Plain | UiMode::Tui | UiMode::Auto => match event {
            TrainingEvent::Log {
                level,
                scope,
                message,
            } => {
                if *level != LogLevel::Debug {
                    println!(
                        "[{}] {scope}: {message}",
                        format!("{level:?}").to_ascii_lowercase()
                    )
                }
            }
            TrainingEvent::State {
                job_id,
                state,
                error,
            } => println!(
                "training job={job_id} state={state:?}{}",
                error
                    .as_deref()
                    .map(|e| format!(" error={e}"))
                    .unwrap_or_default()
            ),
            TrainingEvent::Progress { job_id, progress } => println!(
                "training job={job_id} progress={}/{} detail={}",
                progress.current, progress.total, progress.detail
            ),
            TrainingEvent::Persisted {
                job_id,
                path,
                summary,
            } => println!(
                "training job={job_id} persisted={} {summary}",
                path.display()
            ),
            TrainingEvent::Added { job } => println!(
                "training job={} state={:?} label={}",
                job.metadata.id, job.state, job.metadata.label
            ),
            TrainingEvent::Restarted { job_id } => {
                println!("training job={job_id} state=restarted")
            }
            TrainingEvent::Control { state, message } => {
                println!("training control state={state:?} message={message}")
            }
        },
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct UiGeometry {
    chunks: [Rect; 3],
    panes: [Rect; 2],
    visible_jobs: usize,
    first_job_row: u16,
}

fn ui_geometry(area: Rect) -> UiGeometry {
    let chunk_rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(68),
            Constraint::Percentage(32),
            Constraint::Length(2),
        ])
        .split(area);
    let pane_rects = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunk_rects[0]);
    let chunks = [chunk_rects[0], chunk_rects[1], chunk_rects[2]];
    let panes = [pane_rects[0], pane_rects[1]];
    UiGeometry {
        visible_jobs: panes[0].height.saturating_sub(2).max(1) as usize,
        first_job_row: panes[0].y.saturating_add(1),
        chunks,
        panes,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImprovementRecord {
    pub run_id: String,
    pub job_id: String,
    pub timestamp_ms: u128,
    pub seed: u64,
    pub baseline_metric: f64,
    pub candidate_metric: f64,
    pub reason: String,
    pub artifact_path: PathBuf,
    pub outcome: String,
}

impl ImprovementRecord {
    pub fn now(
        run_id: impl Into<String>,
        job_id: impl Into<String>,
        seed: u64,
        baseline_metric: f64,
        candidate_metric: f64,
        reason: impl Into<String>,
        artifact_path: PathBuf,
    ) -> Self {
        Self {
            run_id: run_id.into(),
            job_id: job_id.into(),
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            seed,
            baseline_metric,
            candidate_metric,
            reason: reason.into(),
            artifact_path,
            outcome: "candidate".into(),
        }
    }
}

pub enum PersistenceRequest {
    Candidate {
        path: PathBuf,
        bytes: Vec<u8>,
        record: ImprovementRecord,
    },
    Journal {
        record: ImprovementRecord,
    },
    Shutdown,
}

pub struct PersistenceWorker {
    sender: Sender<PersistenceRequest>,
    join: Option<thread::JoinHandle<Result<(), String>>>,
}

impl PersistenceWorker {
    pub fn start(journal_path: PathBuf) -> Self {
        let (sender, receiver) = mpsc::channel();
        let join = thread::spawn(move || persistence_loop(receiver, &journal_path));
        Self {
            sender,
            join: Some(join),
        }
    }

    pub fn sender(&self) -> Sender<PersistenceRequest> {
        self.sender.clone()
    }

    pub fn shutdown(mut self) -> Result<(), String> {
        let _ = self.sender.send(PersistenceRequest::Shutdown);
        self.join
            .take()
            .expect("persistence worker join handle")
            .join()
            .map_err(|_| "persistence worker panicked".to_string())?
    }
}

fn persistence_loop(
    receiver: Receiver<PersistenceRequest>,
    journal_path: &Path,
) -> Result<(), String> {
    for request in receiver {
        match request {
            PersistenceRequest::Candidate {
                path,
                bytes,
                record,
            } => {
                atomic_replace(&path, &bytes)?;
                append_journal(journal_path, &record)?;
                render_structured_event(&TrainingEvent::Persisted {
                    job_id: record.job_id.clone(),
                    path,
                    summary: format!(
                        "candidate_metric={} baseline_metric={} outcome={}",
                        record.candidate_metric, record.baseline_metric, record.outcome
                    ),
                });
            }
            PersistenceRequest::Journal { record } => {
                append_journal(journal_path, &record)?;
                log(
                    LogLevel::Info,
                    "persistence",
                    format!(
                        "journaled job={} outcome={} path={}",
                        record.job_id,
                        record.outcome,
                        journal_path.display()
                    ),
                );
            }
            PersistenceRequest::Shutdown => break,
        }
    }
    Ok(())
}

pub fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("candidate");
    let temporary = parent.join(format!(".{name}.{nonce}.tmp"));
    let result = (|| -> io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(|e| format!("failed to atomically replace {}: {e}", path.display()))
}

fn append_journal(path: &Path, record: &ImprovementRecord) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    serde_json::to_writer(&mut file, record)
        .map_err(|e| format!("failed to encode journal record: {e}"))?;
    file.write_all(b"\n")
        .and_then(|_| file.flush())
        .and_then(|_| file.sync_data())
        .map_err(|e| format!("failed to flush {}: {e}", path.display()))
}

pub fn stable_run_id(seed: u64) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{millis:x}-{seed:016x}")
}

pub fn validate_dependencies(jobs: &[JobMetadata]) -> Result<(), String> {
    let ids = jobs
        .iter()
        .map(|job| job.id.as_str())
        .collect::<HashSet<_>>();
    for job in jobs {
        for dependency in &job.dependencies {
            if !ids.contains(dependency.as_str()) {
                return Err(format!(
                    "job {} has unknown dependency {dependency}",
                    job.id
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(id: &str, dependencies: &[&str]) -> JobMetadata {
        JobMetadata {
            id: id.into(),
            label: id.into(),
            kind: "test".into(),
            seed: 7,
            dependencies: dependencies.iter().map(|s| (*s).into()).collect(),
            persistence_path: None,
            detail: BTreeMap::new(),
        }
    }

    #[test]
    fn reducer_restarts_and_invalidates_dependencies() {
        let mut reducer = EventReducer::default();
        for job in [
            metadata("sample", &[]),
            metadata("project", &["sample"]),
            metadata("train", &["project"]),
        ] {
            reducer.apply(&TrainingEvent::Added {
                job: JobSnapshot::new(job),
            });
        }
        assert_eq!(
            reducer.invalidate_downstream("sample"),
            vec!["project", "train"]
        );
        reducer.apply(&TrainingEvent::Restarted {
            job_id: "sample".into(),
        });
        assert_eq!(
            reducer
                .jobs()
                .find(|job| job.metadata.id == "sample")
                .unwrap()
                .restart_count,
            1
        );
    }

    #[test]
    fn deadline_expires_while_paused() {
        let token = ControlToken::default();
        token.request_pause();
        assert_eq!(
            token.wait_at_checkpoint(Some(Instant::now() + Duration::from_millis(10))),
            Checkpoint::TimedOut
        );
    }

    #[test]
    fn viewport_scrolls_and_hit_tests() {
        let mut view = Viewport::default();
        view.select(8, 10, 3);
        assert_eq!(view.offset, 6);
        assert_eq!(view.hit_test(11, 10, 3, 10), Some(7));
        view.move_by(-20, 10, 3);
        assert_eq!(view.selected, 0);
    }

    #[test]
    fn keyboard_controls_update_selection_and_emit_actions() {
        use crossterm::event::{
            Event,
            KeyCode,
            KeyEvent,
            KeyEventKind,
            KeyEventState,
            KeyModifiers,
        };
        let mut view = Viewport::default();
        assert_eq!(
            handle_ui_event(
                Event::Key(KeyEvent::new(KeyCode::Char('j'), KeyModifiers::NONE)),
                &mut view,
                10,
                3,
                1,
            ),
            UiAction::None
        );
        assert_eq!(view.selected, 1);
        assert_eq!(
            handle_ui_event(
                Event::Key(KeyEvent::new(KeyCode::Char(' '), KeyModifiers::NONE)),
                &mut view,
                10,
                3,
                1,
            ),
            UiAction::PauseResume
        );
        assert_eq!(
            handle_ui_event(
                Event::Key(KeyEvent {
                    code: KeyCode::Char(' '),
                    modifiers: KeyModifiers::NONE,
                    kind: KeyEventKind::Release,
                    state: KeyEventState::NONE,
                }),
                &mut view,
                10,
                3,
                1,
            ),
            UiAction::None
        );
    }

    #[test]
    fn navigation_uses_the_rendered_jobs_pane_height() {
        let geometry = ui_geometry(Rect::new(0, 0, 100, 30));
        assert!(geometry.visible_jobs < 26);
        assert_eq!(geometry.first_job_row, geometry.panes[0].y + 1);
        let mut viewport = Viewport::default();
        for _ in 0..geometry.visible_jobs {
            viewport.move_by(1, 50, geometry.visible_jobs);
        }
        assert_eq!(viewport.offset, 1);
    }

    #[test]
    fn late_metadata_preserves_progress_and_control_states_are_visible() {
        let mut reducer = EventReducer::default();
        reducer.apply(&TrainingEvent::Progress {
            job_id: "worker".into(),
            progress: JobProgress {
                current: 4,
                total: 10,
                ..Default::default()
            },
        });
        reducer.apply(&TrainingEvent::Added {
            job: JobSnapshot::new(metadata("worker", &["baseline"])),
        });
        reducer.apply(&TrainingEvent::Control {
            state: JobState::PauseRequested,
            message: "pause requested".into(),
        });
        let worker = reducer.jobs().next().unwrap();
        assert_eq!(worker.progress.current, 4);
        assert_eq!(worker.metadata.dependencies, ["baseline"]);
        assert_eq!(worker.state, JobState::PauseRequested);
    }

    #[test]
    fn reducer_keeps_a_bounded_ordered_log() {
        let mut reducer = EventReducer::default();
        for index in 0..205 {
            reducer.apply(&TrainingEvent::Log {
                level: LogLevel::Info,
                scope: "test".into(),
                message: format!("event {index}"),
            });
        }
        assert_eq!(reducer.logs().count(), 200);
        assert_eq!(reducer.logs().next().unwrap().2, "event 5");
        assert_eq!(reducer.logs().next_back().unwrap().2, "event 204");
    }

    #[test]
    fn json_log_event_has_stable_structure() {
        let json = serde_json::to_string(&TrainingEvent::Log {
            level: LogLevel::Warn,
            scope: "cpu/sweep".into(),
            message: "deadline reached".into(),
        })
        .unwrap();
        assert_eq!(
            json,
            r#"{"type":"log","level":"warn","scope":"cpu/sweep","message":"deadline reached"}"#
        );
    }

    #[test]
    fn tui_draws_jobs_and_logs_without_writing_stdout() {
        let backend = ratatui::backend::TestBackend::new(100, 30);
        let mut terminal = ratatui::Terminal::new(backend).unwrap();
        let mut reducer = EventReducer::default();
        reducer.apply(&TrainingEvent::Progress {
            job_id: "cpu-sweep".into(),
            progress: JobProgress {
                current: 2,
                total: 5,
                detail: "rook=180".into(),
                ..Default::default()
            },
        });
        reducer.apply(&TrainingEvent::Log {
            level: LogLevel::Success,
            scope: "cpu/sweep".into(),
            message: "rook improved".into(),
        });
        terminal
            .draw(|frame| draw_training_ui(frame, &reducer, Viewport::default(), None))
            .unwrap();
        let rendered = terminal.backend().to_string();
        assert!(rendered.contains("cpu-sweep"));
        assert!(rendered.contains("rook improved"));
    }

    #[test]
    fn atomic_candidate_and_journal_are_durable() {
        let root = std::env::temp_dir().join(format!("chronofish-runtime-{}", stable_run_id(1)));
        let artifact = root.join("candidate.bin");
        let journal = root.join("improvements.jsonl");
        let worker = PersistenceWorker::start(journal.clone());
        worker
            .sender()
            .send(PersistenceRequest::Candidate {
                path: artifact.clone(),
                bytes: b"new model".to_vec(),
                record: ImprovementRecord::now(
                    "run",
                    "job",
                    1,
                    2.0,
                    1.0,
                    "improved",
                    artifact.clone(),
                ),
            })
            .unwrap();
        worker.shutdown().unwrap();
        assert_eq!(fs::read(&artifact).unwrap(), b"new model");
        let line = fs::read_to_string(&journal).unwrap();
        assert_eq!(line.lines().count(), 1);
        let _ = fs::remove_dir_all(root);
    }
}
