//! Interactive setup screen: review and edit every trainer parameter before a run.
//!
//! The form is driven by clap's own argument definitions, so parsing, defaults and
//! choice lists stay in one place. Starting a run rebuilds a command line from the
//! form and parses it exactly as if it had been typed.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, Parser};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::{DefaultTerminal, Frame};

use crate::Arguments;

/// Last-used settings, stored in the work directory.
const SETTINGS_FILE: &str = "trainer-settings.txt";
/// Typical self-play game length, used only for the outlook estimates.
const ASSUMED_GAME_ACTIONS: f64 = 160.0;

const ACCENT: Color = Color::Cyan;
const GOOD: Color = Color::Green;
const BAD: Color = Color::Red;
const CHANGED: Color = Color::Yellow;
const MUTED: Color = Color::DarkGray;

struct Spec {
    id: &'static str,
    group: &'static str,
    about: &'static str,
    suggested: Option<(&'static str, &'static str)>,
    /// Remembered between runs; one-shot modes are not.
    persist: bool,
}

const fn field(id: &'static str, group: &'static str, about: &'static str) -> Spec {
    Spec {
        id,
        group,
        about,
        suggested: None,
        persist: true,
    }
}

impl Spec {
    const fn suggest(mut self, value: &'static str, why: &'static str) -> Self {
        self.suggested = Some((value, why));
        self
    }

    const fn once(mut self) -> Self {
        self.persist = false;
        self
    }
}

const RUN: &str = "Run";
const SELF_PLAY: &str = "Self-play";
const TRAINING: &str = "Training";
const ARENA: &str = "Arena";
const HARDWARE: &str = "Hardware";
const MODE: &str = "Mode";

const SPECS: &[Spec] = &[
    field(
        "work_dir",
        RUN,
        "Directory holding the best model, the candidate, the replay buffer and these saved settings. Use a separate directory per experiment. Pass --work-dir on the command line to load another directory's saved settings.",
    )
    .once(),
    field(
        "iterations",
        RUN,
        "Self-play → fit → arena cycles to run before exiting. Each iteration saves its checkpoints, so a long run can be stopped and resumed.",
    ),
    field(
        "ruleset",
        RUN,
        "Rulesets used for self-play and the arena. A network only learns the rulesets it plays: train on 'all' if the bot should play 5D chess.",
    ),
    field(
        "model_size",
        RUN,
        "Network preset for a new model. Larger networks learn more but evaluate more slowly. Ignored when a checkpoint already exists in the work directory.",
    ),
    field(
        "seed",
        RUN,
        "Base seed for self-play and arena randomness. Change it to get a different but reproducible run.",
    ),
    field(
        "games",
        SELF_PLAY,
        "Self-play games generated per iteration. This is the training data: with too few, the fit replays the same games until the value head memorises their results instead of learning to judge positions.",
    )
    .suggest(
        "64",
        "About 10k fresh positions per iteration, so each one is fitted only a few times before it leaves the replay",
    ),
    field(
        "simulations",
        SELF_PLAY,
        "MCTS simulations per self-play action. The visit counts become the policy target, and too few give flat, uninformative targets. Cost grows linearly.",
    )
    .suggest("128", "Sharper policy targets than 48, at about 2.7 times the self-play cost"),
    field(
        "max_actions",
        SELF_PLAY,
        "Action limit for a self-play game. Unfinished games are recorded as draws, which teach the value head nothing.",
    ),
    field(
        "epochs",
        TRAINING,
        "Passes over the sampled examples in each fit. Extra passes over a small replay mostly memorise it.",
    )
    .suggest("1", "One pass per iteration, with fresh games doing the rest"),
    field(
        "training_examples",
        TRAINING,
        "Replay positions sampled for each fit. Together with epochs and games, this sets how often each position is fitted (see Outlook).",
    ),
    field(
        "replay_capacity",
        TRAINING,
        "Positions kept in the replay window. Older games are dropped first. A bigger window holds more diverse data but keeps stale games longer.",
    ),
    field(
        "batch_size",
        TRAINING,
        "Examples per optimiser step, before the token budget splits oversized batches.",
    ),
    field(
        "learning_rate",
        TRAINING,
        "Adam learning rate. Lower it if losses oscillate, and raise it carefully if fits barely move.",
    ),
    field(
        "persistent_adam",
        TRAINING,
        "Carry Adam's moment estimates in the candidate checkpoint between fits, instead of restarting the optimiser each iteration.",
    ),
    field(
        "reset_candidate",
        TRAINING,
        "Discard the accumulated candidate and fit from the incumbent best model this run. Useful after a candidate has overfitted.",
    )
    .once(),
    field(
        "arena_games",
        ARENA,
        "Promotion games between candidate and best. Six covers each ruleset once per colour. More games make promotion less of a coin flip.",
    )
    .suggest("24", "Four colour-reversed pairs per ruleset, so luck rarely decides a promotion"),
    field(
        "arena_simulations",
        ARENA,
        "MCTS simulations per arena action, set independently of self-play.",
    )
    .suggest("32", "Enough search for the arena to reflect playing strength"),
    field(
        "arena_max_actions",
        ARENA,
        "Action limit for an arena game. Unfinished games count as draws.",
    ),
    field(
        "arena_opening_actions",
        ARENA,
        "Random opening actions shared by each colour-reversed arena pair. Arena search is deterministic, so with zero every pair replays the same game.",
    ),
    field(
        "promotion_score",
        ARENA,
        "Minimum candidate score needed for promotion. 0.5 accepts ties, which lets overfitted candidates through.",
    )
    .suggest("0.55", "Promote only on a measurable improvement"),
    field(
        "verbose",
        ARENA,
        "Log arena game starts, periodic action counts and low-level device identifiers.",
    ),
    field(
        "arena_progress_actions",
        ARENA,
        "Actions between arena progress log lines when verbose is on. Zero disables them.",
    ),
    field(
        "threads",
        HARDWARE,
        "Worker threads for independent self-play and arena games. Defaults to every available core.",
    ),
    field(
        "training_device",
        HARDWARE,
        "Backend used to fit the network. Vulkan needs a discrete GPU.",
    ),
    field(
        "inference_device",
        HARDWARE,
        "Backend for self-play and arena inference. Auto matches the fitting device.",
    ),
    field(
        "gpu_index",
        HARDWARE,
        "Zero-based index among discrete Vulkan GPUs.",
    ),
    field(
        "mcts_batch_size",
        HARDWARE,
        "Leaves each MCTS search selects before sending one inference request. Higher batches better but searches slightly less sharply.",
    ),
    field(
        "inference_batch_size",
        HARDWARE,
        "Positions merged from concurrent games into one network inference batch.",
    ),
    field(
        "batch_token_budget",
        HARDWARE,
        "Approximate padded board and action elements allowed in one fitting batch. Large 5D positions get split to stay under it.",
    ),
    field(
        "inference_memory_budget_mib",
        HARDWARE,
        "Conservative dynamic-tensor memory budget for one inference batch, in MiB.",
    ),
    field(
        "training_memory_budget_mib",
        HARDWARE,
        "Conservative dynamic-tensor memory budget for one fitting batch, in MiB.",
    ),
    field(
        "reconstruction_memory_budget_mib",
        HARDWARE,
        "Host memory budget for replay positions expanded for a fit, in MiB.",
    ),
    field(
        "arena_only",
        MODE,
        "Only evaluate the saved candidate against the best model. No self-play or fitting.",
    )
    .once(),
    field(
        "fit_only",
        MODE,
        "Only fit the saved replay. No self-play or arena.",
    )
    .once(),
    field(
        "check_training_device",
        MODE,
        "Initialise and verify the selected fitting device, then exit.",
    )
    .once(),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Default,
    Saved,
    CommandLine,
    Edited,
}

enum Kind {
    Flag,
    Choice(Vec<String>),
    Text,
}

struct Field {
    spec: &'static Spec,
    long: String,
    kind: Kind,
    default: String,
    value: String,
    source: Source,
}

impl Field {
    fn push_args(&self, argv: &mut Vec<String>) {
        if matches!(self.kind, Kind::Flag) {
            if self.value == "true" {
                argv.push(format!("--{}", self.long));
            }
        } else {
            argv.push(format!("--{}", self.long));
            argv.push(self.value.clone());
        }
    }

    /// Parses this field alone, so a bad value is flagged next to the field itself.
    fn check(&self) -> Result<(), String> {
        let mut argv = vec!["chronofish-trainer".to_owned()];
        self.push_args(&mut argv);
        Arguments::try_parse_from(argv)
            .map(drop)
            .map_err(|error| clap_message(&error))
    }

    fn display(value: &str, kind: &Kind) -> String {
        match (kind, value) {
            (Kind::Flag, "true") => "on".to_owned(),
            (Kind::Flag, _) => "off".to_owned(),
            _ => value.to_owned(),
        }
    }

    fn cycle(&self, step: isize) -> Option<String> {
        match &self.kind {
            Kind::Flag => Some(
                if self.value == "true" {
                    "false"
                } else {
                    "true"
                }
                .to_owned(),
            ),
            Kind::Choice(choices) => {
                let current = choices.iter().position(|choice| *choice == self.value);
                let len = choices.len().cast_signed();
                let next = current.map_or(0, |index| (index.cast_signed() + step).rem_euclid(len));
                Some(choices[next.cast_unsigned()].clone())
            }
            Kind::Text => None,
        }
    }
}

fn clap_message(error: &clap::Error) -> String {
    let rendered = error.render().to_string();
    let line = rendered.lines().next().unwrap_or_default();
    line.strip_prefix("error: ").unwrap_or(line).to_owned()
}

fn build_fields(matches: &ArgMatches, saved: &HashMap<String, String>) -> Vec<Field> {
    let command = Arguments::command();
    SPECS
        .iter()
        .map(|spec| {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == spec.id)
                .unwrap_or_else(|| panic!("setup lists unknown trainer argument {}", spec.id));
            let long = arg
                .get_long()
                .expect("trainer arguments have long names")
                .to_owned();
            let kind = if !arg.get_action().takes_values() {
                Kind::Flag
            } else if arg.get_possible_values().is_empty() {
                Kind::Text
            } else {
                Kind::Choice(
                    arg.get_possible_values()
                        .iter()
                        .filter(|value| !value.is_hide_set())
                        .map(|value| value.get_name().to_owned())
                        .collect(),
                )
            };
            let default = match kind {
                Kind::Flag => "false".to_owned(),
                _ => arg
                    .get_default_values()
                    .first()
                    .map_or_else(String::new, |value| value.to_string_lossy().into_owned()),
            };
            let current = match kind {
                Kind::Flag => matches.get_flag(spec.id).to_string(),
                _ => matches
                    .get_raw(spec.id)
                    .and_then(|mut values| values.next())
                    .map_or_else(
                        || default.clone(),
                        |value| value.to_string_lossy().into_owned(),
                    ),
            };
            let mut field = Field {
                spec,
                long,
                kind,
                default,
                value: current,
                source: Source::Default,
            };
            if matches.value_source(spec.id) == Some(ValueSource::CommandLine) {
                field.source = Source::CommandLine;
            } else if let Some(saved) = saved.get(spec.id).filter(|_| spec.persist) {
                let previous = std::mem::replace(&mut field.value, saved.clone());
                if field.check().is_ok() {
                    field.source = Source::Saved;
                } else {
                    field.value = previous;
                }
            }
            field
        })
        .collect()
}

fn load_saved(work_dir: &Path) -> HashMap<String, String> {
    fs::read_to_string(work_dir.join(SETTINGS_FILE))
        .unwrap_or_default()
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect()
}

fn save(work_dir: &Path, fields: &[Field]) -> std::io::Result<()> {
    fs::create_dir_all(work_dir)?;
    let mut text =
        String::from("# Last trainer setup; values equal to the defaults are omitted.\n");
    for field in fields {
        if field.spec.persist && field.value != field.default {
            let _ = writeln!(text, "{}={}", field.spec.id, field.value);
        }
    }
    fs::write(work_dir.join(SETTINGS_FILE), text)
}

enum RowKind {
    Group(&'static str),
    Field(usize),
    Start,
}

enum Outcome {
    Continue,
    Quit,
    Start,
}

struct App {
    fields: Vec<Field>,
    rows: Vec<RowKind>,
    table: TableState,
    editing: Option<String>,
    status: Option<(bool, String)>,
}

impl App {
    fn new(fields: Vec<Field>) -> Self {
        let mut rows = Vec::new();
        let mut group = "";
        for (index, field) in fields.iter().enumerate() {
            if field.spec.group != group {
                group = field.spec.group;
                rows.push(RowKind::Group(group));
            }
            rows.push(RowKind::Field(index));
        }
        rows.push(RowKind::Start);
        let mut app = Self {
            fields,
            rows,
            table: TableState::default(),
            editing: None,
            status: None,
        };
        app.table.select(Some(1));
        app
    }

    fn selected_row(&self) -> &RowKind {
        &self.rows[self.table.selected().unwrap_or(0)]
    }

    fn selected_field(&self) -> Option<usize> {
        match self.selected_row() {
            RowKind::Field(index) => Some(*index),
            _ => None,
        }
    }

    fn move_by(&mut self, step: isize) {
        let len = self.rows.len().cast_signed();
        let mut row = self.table.selected().unwrap_or(0).cast_signed();
        loop {
            row = (row + step).clamp(0, len - 1);
            if !matches!(self.rows[row.cast_unsigned()], RowKind::Group(_)) {
                break;
            }
            if row == 0 {
                row = 1;
                break;
            }
            if row == len - 1 {
                break;
            }
        }
        self.table.select(Some(row.cast_unsigned()));
    }

    fn set(&mut self, index: usize, value: String) {
        let field = &mut self.fields[index];
        if field.value != value {
            field.value = value;
            field.source = Source::Edited;
        }
        self.status = field.check().err().map(|error| (true, error));
    }

    fn apply_suggestions(&mut self) -> usize {
        let mut applied = 0;
        for index in 0..self.fields.len() {
            if let Some((value, _)) = self.fields[index].spec.suggested
                && self.fields[index].value != value
            {
                self.set(index, value.to_owned());
                applied += 1;
            }
        }
        applied
    }

    fn reset_all(&mut self) {
        for index in 0..self.fields.len() {
            let default = self.fields[index].default.clone();
            self.set(index, default);
        }
    }

    fn handle_edit(&mut self, key: KeyEvent, control: bool) {
        let Some(buffer) = &mut self.editing else {
            return;
        };
        match key.code {
            KeyCode::Esc => {
                self.editing = None;
                self.status = None;
            }
            KeyCode::Enter => {
                let value = buffer.trim().to_owned();
                self.editing = None;
                if let Some(index) = self.selected_field() {
                    self.set(index, value);
                }
            }
            KeyCode::Backspace => {
                buffer.pop();
            }
            KeyCode::Char('u') if control => buffer.clear(),
            KeyCode::Char(character) if !control => buffer.push(character),
            _ => {}
        }
    }

    fn handle(&mut self, key: KeyEvent) -> Outcome {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        if control && key.code == KeyCode::Char('c') {
            return Outcome::Quit;
        }
        if self.editing.is_some() {
            self.handle_edit(key, control);
            return Outcome::Continue;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => return Outcome::Quit,
            KeyCode::F(5) => return Outcome::Start,
            KeyCode::Char('s') if control => return Outcome::Start,
            KeyCode::Up | KeyCode::Char('k') => self.move_by(-1),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => self.move_by(1),
            KeyCode::PageUp => self.move_by(-10),
            KeyCode::PageDown => self.move_by(10),
            KeyCode::Home | KeyCode::Char('g') => {
                self.table.select(Some(0));
                self.move_by(1);
            }
            KeyCode::End | KeyCode::Char('G') => self.table.select(Some(self.rows.len() - 1)),
            KeyCode::Enter | KeyCode::Char(' ') => match self.selected_row() {
                RowKind::Start => return Outcome::Start,
                RowKind::Group(_) => {}
                RowKind::Field(index) => {
                    let index = *index;
                    match self.fields[index].cycle(1) {
                        Some(value) => self.set(index, value),
                        None => self.editing = Some(self.fields[index].value.clone()),
                    }
                }
            },
            KeyCode::Left | KeyCode::Right | KeyCode::Char('h' | 'l') => {
                let step = if matches!(key.code, KeyCode::Left | KeyCode::Char('h')) {
                    -1
                } else {
                    1
                };
                if let Some(index) = self.selected_field()
                    && let Some(value) = self.fields[index].cycle(step)
                {
                    self.set(index, value);
                }
            }
            KeyCode::Char('s') => {
                if let Some(index) = self.selected_field() {
                    match self.fields[index].spec.suggested {
                        Some((value, _)) => self.set(index, value.to_owned()),
                        None => {
                            self.status = Some((
                                false,
                                "No suggestion; the default is fine here.".to_owned(),
                            ));
                        }
                    }
                }
            }
            KeyCode::Char('S') => {
                let applied = self.apply_suggestions();
                self.status = Some((false, format!("Applied {applied} suggested value(s).")));
            }
            KeyCode::Char('d') => {
                if let Some(index) = self.selected_field() {
                    let default = self.fields[index].default.clone();
                    self.set(index, default);
                }
            }
            KeyCode::Char('D') => {
                self.reset_all();
                self.status = Some((false, "Every parameter reset to its default.".to_owned()));
            }
            KeyCode::Char(character)
                if character.is_ascii_digit() || character == '.' || character == '/' =>
            {
                if let Some(index) = self.selected_field()
                    && matches!(self.fields[index].kind, Kind::Text)
                {
                    self.editing = Some(character.to_string());
                }
            }
            _ => {}
        }
        Outcome::Continue
    }

    fn build(&self) -> Result<Arguments, String> {
        let mut argv = vec!["chronofish-trainer".to_owned(), "--skip-setup".to_owned()];
        for field in &self.fields {
            field.push_args(&mut argv);
        }
        let arguments = Arguments::try_parse_from(argv).map_err(|error| clap_message(&error))?;
        crate::validate(&arguments)?;
        Ok(arguments)
    }

    fn number(&self, id: &str) -> Option<f64> {
        self.fields
            .iter()
            .find(|field| field.spec.id == id)
            .and_then(|field| field.value.parse().ok())
    }

    fn flag(&self, id: &str) -> bool {
        self.fields
            .iter()
            .any(|field| field.spec.id == id && field.value == "true")
    }
}

/// Shows the setup form. Returns the chosen arguments, or `None` when the user quits.
pub fn run(matches: &ArgMatches) -> Result<Option<Arguments>, Box<dyn Error>> {
    let work_dir = matches
        .get_one::<PathBuf>("work_dir")
        .cloned()
        .unwrap_or_default();
    let mut app = App::new(build_fields(matches, &load_saved(&work_dir)));
    let mut terminal = ratatui::try_init()?;
    let result = event_loop(&mut terminal, &mut app);
    ratatui::restore();
    let arguments = result?;
    if let Some(arguments) = &arguments
        && let Err(error) = save(&arguments.work_dir, &app.fields)
    {
        eprintln!("warning: could not save trainer settings: {error}");
    }
    Ok(arguments)
}

fn event_loop(terminal: &mut DefaultTerminal, app: &mut App) -> std::io::Result<Option<Arguments>> {
    loop {
        terminal.draw(|frame| draw(frame, app))?;
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match app.handle(key) {
            Outcome::Continue => {}
            Outcome::Quit => return Ok(None),
            Outcome::Start => match app.build() {
                Ok(arguments) => return Ok(Some(arguments)),
                Err(error) => app.status = Some((true, format!("Cannot start: {error}"))),
            },
        }
    }
}

fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(MUTED))
        .title(Span::styled(
            format!(" {title} "),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
}

fn draw(frame: &mut Frame, app: &mut App) {
    let [header, body, status, footer] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(8),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    draw_header(frame, header);
    let (list, side) = if body.width >= 110 {
        let [list, side] =
            Layout::horizontal([Constraint::Percentage(58), Constraint::Percentage(42)])
                .areas(body);
        (list, side)
    } else {
        let [list, side] =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(body);
        (list, side)
    };
    let [detail, outlook] =
        Layout::vertical([Constraint::Min(6), Constraint::Length(9)]).areas(side);
    draw_table(frame, list, app);
    draw_detail(frame, detail, app);
    draw_outlook(frame, outlook, app);
    draw_status(frame, status, app);
    draw_footer(frame, footer, app);
}

fn draw_header(frame: &mut Frame, area: Rect) {
    let line = Line::from(vec![
        Span::styled(
            " CHRONOFISH TRAINER ",
            Style::new()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            "  setup · review the parameters, then start training",
            Style::new().fg(MUTED),
        ),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_table(frame: &mut Frame, area: Rect, app: &mut App) {
    let selected = app.table.selected();
    let rows = app
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| match row {
            RowKind::Group(group) => Row::new(vec![Cell::from(Span::styled(
                group.to_uppercase(),
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ))]),
            RowKind::Start => Row::new(vec![Cell::from(Span::styled(
                "▶ Start training",
                Style::new().fg(GOOD).add_modifier(Modifier::BOLD),
            ))])
            .top_margin(1),
            RowKind::Field(index) => {
                let field = &app.fields[*index];
                let value = match &app.editing {
                    Some(buffer) if selected == Some(row_index) => Span::styled(
                        format!("{buffer}▏"),
                        Style::new().fg(Color::Black).bg(CHANGED),
                    ),
                    _ => {
                        let style = if field.check().is_err() {
                            Style::new().fg(BAD).add_modifier(Modifier::BOLD)
                        } else if field.value == field.default {
                            Style::new()
                        } else {
                            Style::new().fg(CHANGED).add_modifier(Modifier::BOLD)
                        };
                        Span::styled(Field::display(&field.value, &field.kind), style)
                    }
                };
                let suggested = match field.spec.suggested {
                    Some((value, _)) if value == field.value => {
                        Span::styled(format!("✓ {value}"), Style::new().fg(GOOD))
                    }
                    Some((value, _)) => Span::styled(value, Style::new().fg(ACCENT)),
                    None => Span::styled("", Style::new()),
                };
                let marker = match field.source {
                    Source::CommandLine => "⌘",
                    Source::Saved => "↺",
                    Source::Edited | Source::Default => " ",
                };
                Row::new(vec![
                    Cell::from(format!("{marker} {}", field.long)),
                    Cell::from(value),
                    Cell::from(Span::styled(
                        Field::display(&field.default, &field.kind),
                        Style::new().fg(MUTED),
                    )),
                    Cell::from(suggested),
                ])
            }
        });
    let table = Table::new(
        rows,
        [
            Constraint::Length(34),
            Constraint::Min(10),
            Constraint::Length(16),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new(["  Parameter", "Value", "Default", "Suggested"])
            .style(Style::new().fg(MUTED).add_modifier(Modifier::BOLD)),
    )
    .row_highlight_style(Style::new().bg(Color::Indexed(236)))
    .highlight_symbol("▸ ")
    .block(panel("Parameters"));
    frame.render_stateful_widget(table, area, &mut app.table);
}

fn draw_detail(frame: &mut Frame, area: Rect, app: &App) {
    let Some(index) = app.selected_field() else {
        let text = match app.selected_row() {
            RowKind::Start => vec![
                Line::from("Start training with these settings."),
                Line::from(""),
                Line::from(Span::styled(
                    "The live dashboard opens next. Values that differ from the defaults are saved in the work directory and pre-filled next time.",
                    Style::new().fg(MUTED),
                )),
            ],
            _ => Vec::new(),
        };
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: true })
                .block(panel("Start")),
            area,
        );
        return;
    };
    let field = &app.fields[index];
    let label = |text: &'static str| Span::styled(text, Style::new().fg(MUTED));
    let mut lines = vec![
        Line::from(Span::styled(
            format!(
                "--{} {}",
                field.long,
                Field::display(&field.value, &field.kind)
            ),
            Style::new().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(field.spec.about),
        Line::from(""),
        Line::from(vec![
            label("Default    "),
            Span::raw(Field::display(&field.default, &field.kind)),
        ]),
    ];
    if let Some((value, why)) = field.spec.suggested {
        lines.push(Line::from(vec![
            label("Suggested  "),
            Span::styled(value, Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" · {why}"), Style::new().fg(ACCENT)),
        ]));
    }
    if let Kind::Choice(choices) = &field.kind {
        lines.push(Line::from(vec![
            label("Choices    "),
            Span::raw(choices.join(" · ")),
        ]));
    }
    let source = match field.source {
        Source::Default => "default",
        Source::Saved => "saved from the last run",
        Source::CommandLine => "command line",
        Source::Edited => "edited here",
    };
    lines.push(Line::from(vec![label("Source     "), Span::raw(source)]));
    if let Err(error) = field.check() {
        lines.push(Line::from(Span::styled(error, Style::new().fg(BAD))));
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel(field.spec.group)),
        area,
    );
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn draw_outlook(frame: &mut Frame, area: Rect, app: &App) {
    let value = |id| app.number(id).unwrap_or(0.0);
    let games = value("games");
    let positions = games * ASSUMED_GAME_ACTIONS;
    let capacity = value("replay_capacity").max(1.0);
    let sampled = value("training_examples").min(capacity);
    let refits = value("epochs") * sampled / capacity * (capacity / positions.max(1.0)).max(1.0);
    let (refit_color, refit_note) = if refits <= 4.0 {
        (GOOD, "healthy")
    } else if refits <= 16.0 {
        (CHANGED, "risks memorising games")
    } else {
        (BAD, "value head will memorise games")
    };
    let evaluations = positions * value("simulations");
    let label = |text: &'static str| Span::styled(text, Style::new().fg(MUTED));
    let mode = if app.flag("check_training_device") {
        "device check only"
    } else if app.flag("arena_only") {
        "arena only"
    } else if app.flag("fit_only") {
        "fit only"
    } else {
        "self-play → fit → arena"
    };
    let lines = vec![
        Line::from(vec![
            label("Run        "),
            Span::raw(format!(
                "{} iteration(s) · {mode}",
                value("iterations") as u64
            )),
        ]),
        Line::from(vec![
            label("Self-play  "),
            Span::raw(format!(
                "≈ {} new positions per iteration",
                compact(positions)
            )),
        ]),
        Line::from(vec![
            label("Refitting  "),
            Span::styled(
                format!("each position fitted ≈ {refits:.1}×"),
                Style::new().fg(refit_color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" · {refit_note}"), Style::new().fg(refit_color)),
        ]),
        Line::from(vec![
            label("Search     "),
            Span::raw(format!(
                "≈ {} network evaluations per iteration",
                compact(evaluations)
            )),
        ]),
        Line::from(vec![
            label("Arena      "),
            Span::raw(format!(
                "{} games · {} sims · promote at ≥ {:.0}%",
                value("arena_games") as u64,
                value("arena_simulations") as u64,
                value("promotion_score") * 100.0
            )),
        ]),
        Line::from(Span::styled(
            format!("Estimates assume ~{ASSUMED_GAME_ACTIONS:.0} actions per game."),
            Style::new().fg(MUTED),
        )),
    ];
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel("Outlook")),
        area,
    );
}

fn compact(value: f64) -> String {
    if value >= 1e6 {
        format!("{:.1}M", value / 1e6)
    } else if value >= 1e3 {
        format!("{:.1}k", value / 1e3)
    } else {
        format!("{value:.0}")
    }
}

fn draw_status(frame: &mut Frame, area: Rect, app: &App) {
    let line = match &app.status {
        Some((true, message)) => {
            Line::from(Span::styled(format!(" ✗ {message}"), Style::new().fg(BAD)))
        }
        Some((false, message)) => {
            Line::from(Span::styled(format!(" {message}"), Style::new().fg(GOOD)))
        }
        None => Line::from(Span::styled(
            " ⌘ from command line   ↺ saved from last run   yellow = changed from default",
            Style::new().fg(MUTED),
        )),
    };
    frame.render_widget(Paragraph::new(line), area);
}

fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let keys: &[(&str, &str)] = if app.editing.is_some() {
        &[("Enter", "accept"), ("Esc", "cancel"), ("Ctrl+U", "clear")]
    } else {
        &[
            ("↑↓", "move"),
            ("Enter", "edit/toggle"),
            ("←→", "cycle"),
            ("s/S", "suggested (one/all)"),
            ("d/D", "default (one/all)"),
            ("F5", "start"),
            ("q", "quit"),
        ]
    };
    let mut spans = Vec::new();
    for (key, action) in keys {
        spans.push(Span::styled(
            format!(" {key} "),
            Style::new().fg(Color::Black).bg(Color::Gray),
        ));
        spans.push(Span::styled(format!(" {action}  "), Style::new().fg(MUTED)));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_from(args: &[&str]) -> App {
        let matches = Arguments::command().get_matches_from(args);
        App::new(build_fields(&matches, &HashMap::new()))
    }

    #[test]
    fn every_spec_names_a_trainer_argument_and_defaults_round_trip() {
        let app = app_from(&["chronofish-trainer"]);
        assert_eq!(app.fields.len(), SPECS.len());
        let arguments = app.build().expect("defaults start a run");
        assert_eq!(arguments.games, 8);
        assert!(arguments.skip_setup);
    }

    #[test]
    fn every_argument_except_output_switches_is_in_the_form() {
        let command = Arguments::command();
        for arg in command.get_arguments() {
            let id = arg.get_id().as_str();
            if matches!(id, "plain" | "skip_setup" | "help") {
                continue;
            }
            assert!(
                SPECS.iter().any(|spec| spec.id == id),
                "{id} is missing from the setup form"
            );
        }
    }

    #[test]
    fn command_line_values_prefill_and_suggestions_apply() {
        let mut app = app_from(&["chronofish-trainer", "--games", "12", "--persistent-adam"]);
        let games = app
            .fields
            .iter()
            .find(|field| field.spec.id == "games")
            .unwrap();
        assert_eq!(games.value, "12");
        assert!(games.source == Source::CommandLine);
        assert!(app.apply_suggestions() > 0);
        let arguments = app.build().unwrap();
        assert_eq!(arguments.games, 64);
        assert_eq!(arguments.epochs, 1);
        assert!(arguments.persistent_adam);
        assert!((arguments.promotion_score - 0.55).abs() < f32::EPSILON);
    }

    #[test]
    fn invalid_values_are_reported_and_block_the_start() {
        let mut app = app_from(&["chronofish-trainer"]);
        let index = app
            .fields
            .iter()
            .position(|field| field.spec.id == "games")
            .unwrap();
        app.set(index, "lots".to_owned());
        assert!(app.status.as_ref().is_some_and(|(error, _)| *error));
        assert!(app.build().is_err());
        let threads = app
            .fields
            .iter()
            .position(|field| field.spec.id == "threads")
            .unwrap();
        app.set(index, "8".to_owned());
        app.set(threads, "0".to_owned());
        assert!(app.build().unwrap_err().contains("threads"));
    }

    #[test]
    fn choices_cycle_and_settings_persist_only_changes() {
        let mut app = app_from(&["chronofish-trainer"]);
        let ruleset = app
            .fields
            .iter()
            .position(|field| field.spec.id == "ruleset")
            .unwrap();
        let next = app.fields[ruleset].cycle(1).unwrap();
        assert_ne!(next, app.fields[ruleset].value);
        app.set(ruleset, next.clone());
        let directory =
            std::env::temp_dir().join(format!("chronofish-setup-{}", std::process::id()));
        save(&directory, &app.fields).unwrap();
        let saved = load_saved(&directory);
        assert_eq!(saved.get("ruleset"), Some(&next));
        assert_eq!(saved.len(), 1);
        let matches = Arguments::command().get_matches_from(["chronofish-trainer"]);
        let restored = build_fields(&matches, &saved);
        let field = restored
            .iter()
            .find(|field| field.spec.id == "ruleset")
            .unwrap();
        assert!(field.value == next && field.source == Source::Saved);
        fs::remove_dir_all(directory).unwrap();
    }
}
