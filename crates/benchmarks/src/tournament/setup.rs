//! Interactive setup screen: choose the bots and review every tournament parameter.
//!
//! Like the trainer's setup, the form is driven by clap's argument definitions, so
//! parsing, defaults and choice lists stay in one place. Bots are toggled one by one
//! instead of typed as a list. Starting rebuilds a command line from the form and
//! parses it exactly as if it had been typed.

use std::collections::HashMap;
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use chronofish_core::Ruleset;
use clap::parser::ValueSource;
use clap::{ArgMatches, CommandFactory, Parser};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Cell, Paragraph, Row, Table, TableState, Wrap};
use ratatui::{DefaultTerminal, Frame};

use super::{Arguments, Bot, ROSTER};

const SETTINGS_FILE: &str = "tournament-settings.txt";
/// Typical decisive-or-adjudicated game length, used only for the time estimate.
const ASSUMED_GAME_ACTIONS: f64 = 90.0;

pub(super) const ACCENT: Color = Color::Cyan;
pub(super) const GOOD: Color = Color::Green;
pub(super) const BAD: Color = Color::Red;
pub(super) const CHANGED: Color = Color::Yellow;
pub(super) const MUTED: Color = Color::DarkGray;

/// Bots offered as toggles, with what each one is for in a rating run.
const BOT_CHOICES: &[(&str, &str)] = &[
    (
        "fissure",
        "Bobby Fissure: uniformly random legal actions. The floor of the scale.",
    ),
    (
        "grabablanca",
        "Grabablanca: greedy, the best handcrafted evaluation one action ahead.",
    ),
    (
        "blueshift-1k",
        "Deep Blueshift with a 1k-node budget (the server's easy level).",
    ),
    (
        "blueshift-10k",
        "Deep Blueshift with a 10k-node budget (the server's medium level).",
    ),
    (
        "blueshift",
        "Deep Blueshift with its default 80k-node budget (the server's hard level). Slow in 5D.",
    ),
    (
        "carlsen",
        "Monte Carlsen: UCT tree search with capture-biased rollouts. Slow in 5D.",
    ),
    (
        "alpha-now",
        "AlphaNow: the network's top policy action, no search. Uses the model below.",
    ),
    (
        "chronofish",
        "Chronofish: AlphaZero MCTS guided by the model below.",
    ),
    (
        "kairos",
        "Kairos: alpha-beta over complete 5D turns, 500 ms per turn. 5D only.",
    ),
    (
        "loki",
        "Loki: Monte Carlo tree search over complete 5D turns, 500 ms per turn. 5D only.",
    ),
    (
        "sf-1320",
        "Stockfish limited to 1320 Elo, an anchor for the scale. Standard chess only; needs Stockfish.",
    ),
    (
        "sf-1600",
        "Stockfish limited to 1600 Elo, an anchor for the scale. Standard chess only; needs Stockfish.",
    ),
    (
        "sf-2000",
        "Stockfish limited to 2000 Elo, an anchor for the scale. Standard chess only; needs Stockfish.",
    ),
];

struct Spec {
    id: &'static str,
    group: &'static str,
    about: &'static str,
    suggested: Option<(&'static str, &'static str)>,
}

const fn field(id: &'static str, group: &'static str, about: &'static str) -> Spec {
    Spec {
        id,
        group,
        about,
        suggested: None,
    }
}

impl Spec {
    const fn suggest(mut self, value: &'static str, why: &'static str) -> Self {
        self.suggested = Some((value, why));
        self
    }
}

const BOTS: &str = "Bots";
const GAMES: &str = "Games";
const ENGINES: &str = "Engines";
const RATINGS: &str = "Rating scale";
const OUTPUT: &str = "Output";
const HARDWARE: &str = "Hardware";

/// Pseudo-argument id for bots outside the toggle list.
const EXTRA_BOTS: &str = "extra_bots";

const SPECS: &[Spec] = &[
    field(
        EXTRA_BOTS,
        BOTS,
        "More bots, comma-separated, for anything not in the toggles above: blueshift-<nodes>[k] (for example blueshift-30k), sf-<elo> from 1320 to 3190, or sf-skill<0-20>.",
    ),
    field(
        "ruleset",
        GAMES,
        "Rulesets to play. Kairos and Loki play only 5D and Stockfish only standard chess; each sits out the rest. Ratings mix every ruleset played.",
    ),
    field(
        "openings",
        GAMES,
        "Random openings per ruleset. Every pair plays each opening from both sides, so games grow with openings × pairs. More openings shrink the error bars.",
    )
    .suggest("4", "Narrow enough intervals to separate neighbouring bots"),
    field(
        "opening_actions",
        GAMES,
        "Random actions played before the bots take over, shared by both games of a pair. They keep deterministic bots from replaying the same game.",
    ),
    field(
        "max_actions",
        GAMES,
        "Actions after the opening before a game is adjudicated or drawn.",
    ),
    field(
        "adjudicate_cp",
        GAMES,
        "At the action limit, the side ahead by this many centipawns on the handcrafted evaluation wins; otherwise it is a draw. Zero draws every unfinished game.",
    ),
    field(
        "seed",
        GAMES,
        "Seed for the openings and bot randomness. The same seed and settings replay the same tournament.",
    ),
    field(
        "model",
        ENGINES,
        "Network checkpoint for AlphaNow and Chronofish. Leave blank to rate an untrained network.",
    )
    .suggest("models/training/best-v2.json", "The trainer's current best network"),
    field(
        "simulations",
        ENGINES,
        "MCTS simulations per action for Chronofish. More search plays stronger and slower.",
    ),
    field(
        "stockfish",
        ENGINES,
        "Stockfish binary for the sf-* bots. Blank uses $STOCKFISH. On Debian and Ubuntu it is /usr/games/stockfish.",
    ),
    field(
        "uci_movetime",
        ENGINES,
        "Thinking time per move for Stockfish, in milliseconds. Its Elo levels were calibrated at longer time controls, so very short times play somewhat below the label.",
    ),
    field(
        "anchor",
        RATINGS,
        "Hold bots at known ratings: name=elo pairs, comma-separated. Stockfish Elo levels are anchored automatically. With no anchor the ratings are centred on zero.",
    ),
    field(
        "bootstrap",
        RATINGS,
        "Bootstrap resamples of opening pairs for the 95% intervals. Zero skips the intervals.",
    ),
    field(
        "calibrate",
        RATINGS,
        "Two-stage calibration: rate the ladder against Stockfish in standard chess, then rate every 5D-capable bot in 5D against the ladder held at those ratings. Needs Stockfish.",
    ),
    field(
        "ladder",
        RATINGS,
        "Deep Blueshift budgets that carry the standard-chess scale into 5D during calibration, weakest first.",
    ),
    field(
        "references",
        RATINGS,
        "Stockfish anchors for the first calibration stage.",
    ),
    field(
        "csv",
        OUTPUT,
        "Write one row per game to this CSV file as each game finishes, so an interrupted run keeps its games.",
    ),
    field(
        "save_ratings",
        OUTPUT,
        "Write the fitted ratings and intervals to this JSON file.",
    ),
    field(
        "threads",
        HARDWARE,
        "Games played at once, one thread each, capped at the machine's cores. Kairos, Loki and Stockfish think for a fixed time, so leave cores free for the rest of the system.",
    ),
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
    /// A toggle for one bot of `--bots`.
    Bot(&'static str),
}

struct Field {
    id: &'static str,
    group: &'static str,
    about: &'static str,
    suggested: Option<(&'static str, &'static str)>,
    label: String,
    kind: Kind,
    default: String,
    value: String,
    source: Source,
}

impl Field {
    fn push_args(&self, argv: &mut Vec<String>) {
        match self.kind {
            Kind::Flag => {
                if self.value == "true" {
                    argv.push(format!("--{}", self.label));
                }
            }
            Kind::Bot(_) => {}
            _ if self.id == EXTRA_BOTS || self.value.is_empty() => {}
            _ => {
                argv.push(format!("--{}", self.label));
                argv.push(self.value.clone());
            }
        }
    }

    /// Parses this field alone, so a bad value is flagged next to the field itself.
    fn check(&self) -> Result<(), String> {
        if self.id == EXTRA_BOTS {
            return extra_bots(&self.value).map(drop);
        }
        let mut argv = vec!["tournament".to_owned()];
        self.push_args(&mut argv);
        let arguments = Arguments::try_parse_from(argv).map_err(|error| clap_message(&error))?;
        match self.id {
            "anchor" => super::parse_anchors(&arguments.anchor).map(drop),
            "ladder" => super::parse_bots(&arguments.ladder).map(drop),
            "references" => super::parse_bots(&arguments.references).map(drop),
            "model" | "stockfish"
                if !self.value.is_empty() && !PathBuf::from(&self.value).is_file() =>
            {
                Err(format!("{} does not exist", self.value))
            }
            _ => Ok(()),
        }
    }

    fn display(&self, value: &str) -> String {
        match (&self.kind, value) {
            (Kind::Flag | Kind::Bot(_), "true") => "on".to_owned(),
            (Kind::Flag | Kind::Bot(_), _) => "off".to_owned(),
            (_, "") => "—".to_owned(),
            _ => value.to_owned(),
        }
    }

    fn cycle(&self, step: isize) -> Option<String> {
        match &self.kind {
            Kind::Flag | Kind::Bot(_) => Some(
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

fn extra_bots(value: &str) -> Result<Vec<Bot>, String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(Bot::parse)
        .collect()
}

fn clap_message(error: &clap::Error) -> String {
    let rendered = error.render().to_string();
    let line = rendered.lines().next().unwrap_or_default();
    line.strip_prefix("error: ").unwrap_or(line).to_owned()
}

fn raw_values(matches: &ArgMatches, id: &str) -> Option<String> {
    matches.get_raw(id).map(|values| {
        values
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(",")
    })
}

/// One toggle per offered bot, and the extra bots the toggles cannot express.
fn bot_fields(
    matches: &ArgMatches,
    saved: &HashMap<String, String>,
) -> (Vec<Field>, String, Source) {
    // The bots come from the command line, else the saved list, else the default roster.
    let (chosen, source) = if matches.value_source("bots") == Some(ValueSource::CommandLine) {
        (
            raw_values(matches, "bots").unwrap_or_default(),
            Source::CommandLine,
        )
    } else if let Some(saved) = saved.get("bots").filter(|list| extra_bots(list).is_ok()) {
        (saved.clone(), Source::Saved)
    } else {
        (ROSTER.join(","), Source::Default)
    };
    let key = |name: &str| Bot::parse(name).expect("toggle names parse").key();
    let chosen_keys = extra_bots(&chosen)
        .unwrap_or_default()
        .into_iter()
        .map(Bot::key)
        .collect::<Vec<_>>();
    let roster_keys = ROSTER.iter().map(|name| key(name)).collect::<Vec<_>>();
    let fields = BOT_CHOICES
        .iter()
        .map(|&(name, about)| Field {
            id: name,
            group: BOTS,
            about,
            suggested: None,
            label: name.to_owned(),
            kind: Kind::Bot(name),
            default: roster_keys.contains(&key(name)).to_string(),
            value: chosen_keys.contains(&key(name)).to_string(),
            source,
        })
        .collect();
    let toggled = BOT_CHOICES
        .iter()
        .map(|(name, _)| key(name))
        .collect::<Vec<_>>();
    let extras = chosen_keys
        .into_iter()
        .filter(|chosen| !toggled.contains(chosen))
        .collect::<Vec<_>>()
        .join(",");
    (fields, extras, source)
}

fn build_fields(matches: &ArgMatches, saved: &HashMap<String, String>) -> Vec<Field> {
    let command = Arguments::command();
    let from_command_line = |id: &str| matches.value_source(id) == Some(ValueSource::CommandLine);
    let (mut fields, extras, bots_source) = bot_fields(matches, saved);
    for spec in SPECS {
        if spec.id == EXTRA_BOTS {
            fields.push(Field {
                id: spec.id,
                group: spec.group,
                about: spec.about,
                suggested: spec.suggested,
                label: "more bots".to_owned(),
                kind: Kind::Text,
                default: String::new(),
                value: extras.clone(),
                source: if extras.is_empty() {
                    Source::Default
                } else {
                    bots_source
                },
            });
            continue;
        }
        let arg = command
            .get_arguments()
            .find(|arg| arg.get_id() == spec.id)
            .unwrap_or_else(|| panic!("setup lists unknown tournament argument {}", spec.id));
        let label = arg
            .get_long()
            .expect("tournament arguments have long names")
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
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(","),
        };
        let current = match kind {
            Kind::Flag => matches.get_flag(spec.id).to_string(),
            _ => raw_values(matches, spec.id).unwrap_or_else(|| default.clone()),
        };
        let mut field = Field {
            id: spec.id,
            group: spec.group,
            about: spec.about,
            suggested: spec.suggested,
            label,
            kind,
            default,
            value: current,
            source: Source::Default,
        };
        if from_command_line(spec.id) {
            field.source = Source::CommandLine;
        } else if let Some(saved) = saved.get(spec.id) {
            let previous = std::mem::replace(&mut field.value, saved.clone());
            if field.check().is_ok() {
                field.source = Source::Saved;
            } else {
                field.value = previous;
            }
        }
        fields.push(field);
    }
    fields
}

fn settings_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(base.join("chronofish").join(SETTINGS_FILE))
}

fn parse_saved(text: &str) -> HashMap<String, String> {
    text.lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect()
}

fn saved_text(fields: &[Field]) -> String {
    let mut text =
        String::from("# Last tournament setup; values equal to the defaults are omitted.\n");
    let bots = bots_list(fields);
    let roster = ROSTER.join(",");
    if bots != roster {
        let _ = writeln!(text, "bots={bots}");
    }
    for field in fields {
        if !matches!(field.kind, Kind::Bot(_))
            && field.id != EXTRA_BOTS
            && field.value != field.default
        {
            let _ = writeln!(text, "{}={}", field.id, field.value);
        }
    }
    text
}

/// The `--bots` list the form describes: toggled bots, then the extra ones.
fn bots_list(fields: &[Field]) -> String {
    let mut names = fields
        .iter()
        .filter_map(|field| match field.kind {
            Kind::Bot(name) if field.value == "true" => Some(name.to_owned()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if let Some(extra) = fields.iter().find(|field| field.id == EXTRA_BOTS) {
        names.extend(
            extra
                .value
                .split(',')
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(str::to_owned),
        );
    }
    names.join(",")
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
            if field.group != group {
                group = field.group;
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

    #[cfg(test)]
    fn index_of(&self, id: &str) -> Option<usize> {
        self.fields.iter().position(|field| field.id == id)
    }

    #[cfg(test)]
    fn value(&self, id: &str) -> &str {
        self.index_of(id)
            .map_or("", |index| self.fields[index].value.as_str())
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
            if let Some((value, _)) = self.fields[index].suggested
                && self.fields[index].value != value
            {
                // A suggested path that does not exist here would only block the start.
                let previous = std::mem::replace(&mut self.fields[index].value, value.to_owned());
                let usable = self.fields[index].check().is_ok();
                self.fields[index].value = previous;
                if usable {
                    self.set(index, value.to_owned());
                    applied += 1;
                }
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

    fn set_all_bots(&mut self, on: bool) {
        for index in 0..self.fields.len() {
            if matches!(self.fields[index].kind, Kind::Bot(_)) {
                self.set(index, on.to_string());
            }
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
            KeyCode::Char('a') => {
                self.set_all_bots(true);
                self.status = Some((false, "Every bot toggled on.".to_owned()));
            }
            KeyCode::Char('n') => {
                self.set_all_bots(false);
                self.status = Some((false, "Every bot toggled off.".to_owned()));
            }
            KeyCode::Char('s') => {
                if let Some(index) = self.selected_field() {
                    match self.fields[index].suggested {
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
                if character.is_ascii_alphanumeric() || "./-_~=,".contains(character) =>
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

    fn argv(&self) -> Vec<String> {
        let mut argv = vec![
            "tournament".to_owned(),
            "--skip-setup".to_owned(),
            "--bots".to_owned(),
            bots_list(&self.fields),
        ];
        for field in &self.fields {
            field.push_args(&mut argv);
        }
        argv
    }

    fn build(&self) -> Result<Arguments, String> {
        if let Some(field) = self.fields.iter().find(|field| field.check().is_err()) {
            return Err(format!("{}: {}", field.label, field.check().unwrap_err()));
        }
        let arguments =
            Arguments::try_parse_from(self.argv()).map_err(|error| clap_message(&error))?;
        let plan = Plan::of(&arguments)?;
        if plan.stages.iter().any(|stage| stage.players < 2) {
            return Err("choose at least two bots that share a ruleset".to_owned());
        }
        if plan.needs_stockfish
            && arguments.stockfish.is_none()
            && std::env::var_os("STOCKFISH").is_none()
        {
            return Err("the sf-* bots need a Stockfish binary (Engines › stockfish)".to_owned());
        }
        Ok(arguments)
    }
}

/// The games a set of arguments will schedule, for the outlook and the start checks.
pub(super) struct Plan {
    pub stages: Vec<StagePlan>,
    pub needs_stockfish: bool,
}

pub(super) struct StagePlan {
    pub title: &'static str,
    pub players: usize,
    pub games: usize,
    /// Rough wall-clock seconds on one thread.
    pub seconds: f64,
}

impl Plan {
    fn of(arguments: &Arguments) -> Result<Self, String> {
        let bots = super::dedupe(super::parse_bots(&arguments.bots)?);
        let stage = |title, players: &[Bot], rulesets: &[Ruleset]| {
            let players = players
                .iter()
                .copied()
                .filter(|bot| rulesets.iter().any(|&ruleset| bot.supports(ruleset)))
                .collect::<Vec<_>>();
            let mut games = 0;
            let mut seconds = 0.0;
            for &ruleset in rulesets {
                for (index, &first) in players.iter().enumerate() {
                    for &second in &players[index + 1..] {
                        if first.supports(ruleset) && second.supports(ruleset) {
                            let pair = 2 * arguments.openings;
                            games += pair;
                            let per_action = f64::midpoint(
                                nominal_ms(first, arguments),
                                nominal_ms(second, arguments),
                            );
                            #[allow(clippy::cast_precision_loss)]
                            let actions = ASSUMED_GAME_ACTIONS.min(arguments.max_actions as f64);
                            #[allow(clippy::cast_precision_loss)]
                            {
                                seconds += pair as f64 * actions * per_action / 1_000.0;
                            }
                        }
                    }
                }
            }
            StagePlan {
                title,
                players: players.len(),
                games,
                seconds,
            }
        };
        let (stages, everyone) = if arguments.calibrate {
            let ladder = super::dedupe(super::parse_bots(&arguments.ladder)?);
            let references = super::dedupe(super::parse_bots(&arguments.references)?);
            let rest = bots
                .iter()
                .copied()
                .filter(|bot| !ladder.contains(bot) && !references.contains(bot))
                .collect::<Vec<_>>();
            let first = super::dedupe(ladder.iter().chain(&references).chain(&rest).copied());
            let second = super::dedupe(
                ladder
                    .iter()
                    .chain(rest.iter().filter(|bot| !bot.standard_only()))
                    .copied(),
            );
            (
                vec![
                    stage(
                        "Stage 1 · standard vs Stockfish",
                        &first,
                        &[Ruleset::Standard],
                    ),
                    stage(
                        "Stage 2 · 5D vs the ladder",
                        &second,
                        &[Ruleset::Multiverse, Ruleset::MultiverseVariant],
                    ),
                ],
                first,
            )
        } else {
            (
                vec![stage("Round robin", &bots, &arguments.ruleset.rulesets())],
                bots.clone(),
            )
        };
        Ok(Self {
            stages,
            needs_stockfish: everyone.iter().any(|bot| bot.standard_only()),
        })
    }
}

/// Rough milliseconds per action in 5D, from the calibration runs, for the estimate only.
fn nominal_ms(bot: Bot, arguments: &Arguments) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    match bot {
        Bot::Fissure => 0.3,
        Bot::Grabablanca => 50.0,
        Bot::Blueshift(nodes) => 0.13 * nodes as f64 + 20.0,
        Bot::Carlsen => 6_600.0,
        Bot::AlphaNow => 15.0,
        Bot::Chronofish => 2.0 * arguments.simulations as f64,
        Bot::Kairos | Bot::Loki => 350.0,
        Bot::Stockfish(_) => arguments.uci_movetime as f64,
    }
}

/// Shows the setup form. Returns the chosen arguments, or `None` when the user quits.
pub(super) fn run(matches: &ArgMatches) -> Result<Option<Arguments>, Box<dyn Error>> {
    let path = settings_path();
    let saved = path
        .as_ref()
        .and_then(|path| fs::read_to_string(path).ok())
        .map(|text| parse_saved(&text))
        .unwrap_or_default();
    let mut app = App::new(build_fields(matches, &saved));
    let mut terminal = ratatui::try_init()?;
    terminal.clear()?;
    let result = event_loop(&mut terminal, &mut app);
    ratatui::restore();
    let arguments = result?;
    if arguments.is_some()
        && let Some(path) = path
    {
        let written = path
            .parent()
            .map_or(Ok(()), fs::create_dir_all)
            .and_then(|()| fs::write(&path, saved_text(&app.fields)));
        if let Err(error) = written {
            eprintln!("warning: could not save tournament settings: {error}");
        }
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

pub(super) fn panel(title: &str) -> Block<'_> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(MUTED))
        .title(Span::styled(
            format!(" {title} "),
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
        ))
}

pub(super) fn header(frame: &mut Frame, area: Rect, subtitle: &str) {
    let line = Line::from(vec![
        Span::styled(
            " CHRONOFISH TOURNAMENT ",
            Style::new()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(format!("  {subtitle}"), Style::new().fg(MUTED)),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

pub(super) fn footer(frame: &mut Frame, area: Rect, keys: &[(&str, &str)]) {
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

fn draw(frame: &mut Frame, app: &mut App) {
    let [top, body, status, bottom] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(8),
        Constraint::Length(1),
        Constraint::Length(1),
    ])
    .areas(frame.area());
    header(
        frame,
        top,
        "setup · choose the bots and settings, then start rating",
    );
    let (list, side) = if body.width >= 110 {
        let [list, side] =
            Layout::horizontal([Constraint::Percentage(56), Constraint::Percentage(44)])
                .areas(body);
        (list, side)
    } else {
        let [list, side] =
            Layout::vertical([Constraint::Percentage(55), Constraint::Percentage(45)]).areas(body);
        (list, side)
    };
    let [detail, outlook] =
        Layout::vertical([Constraint::Min(6), Constraint::Length(10)]).areas(side);
    draw_table(frame, list, app);
    draw_detail(frame, detail, app);
    draw_outlook(frame, outlook, app);
    draw_status(frame, status, app);
    let keys: &[(&str, &str)] = if app.editing.is_some() {
        &[("Enter", "accept"), ("Esc", "cancel"), ("Ctrl+U", "clear")]
    } else {
        &[
            ("↑↓", "move"),
            ("Enter", "toggle/edit"),
            ("←→", "cycle"),
            ("a/n", "all/no bots"),
            ("s/S", "suggested"),
            ("d/D", "default"),
            ("F5", "start"),
            ("q", "quit"),
        ]
    };
    footer(frame, bottom, keys);
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
                "▶ Start the tournament",
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
                        } else if matches!(field.kind, Kind::Bot(_)) && field.value == "true" {
                            Style::new().fg(GOOD).add_modifier(Modifier::BOLD)
                        } else if field.value == field.default {
                            Style::new()
                        } else {
                            Style::new().fg(CHANGED).add_modifier(Modifier::BOLD)
                        };
                        let shown = match field.kind {
                            Kind::Bot(_) if field.value == "true" => "● on".to_owned(),
                            Kind::Bot(_) => "○ off".to_owned(),
                            _ => field.display(&field.value),
                        };
                        Span::styled(shown, style)
                    }
                };
                let suggested = match field.suggested {
                    Some((value, _)) if value == field.value => {
                        Span::styled("✓", Style::new().fg(GOOD))
                    }
                    Some(_) => Span::styled("◆", Style::new().fg(ACCENT)),
                    None => Span::raw(""),
                };
                let marker = match field.source {
                    Source::CommandLine => "⌘",
                    Source::Saved => "↺",
                    Source::Edited | Source::Default => " ",
                };
                Row::new(vec![
                    Cell::from(format!("{marker} {}", field.label)),
                    Cell::from(value),
                    Cell::from(suggested),
                ])
            }
        });
    let table = Table::new(
        rows,
        [
            Constraint::Length(22),
            Constraint::Min(12),
            Constraint::Length(3),
        ],
    )
    .header(
        Row::new(["  Setting", "Value", ""])
            .style(Style::new().fg(MUTED).add_modifier(Modifier::BOLD)),
    )
    .row_highlight_style(Style::new().bg(Color::Indexed(236)))
    .highlight_symbol("▸ ")
    .block(panel("Tournament"));
    frame.render_stateful_widget(table, area, &mut app.table);
}

fn draw_detail(frame: &mut Frame, area: Rect, app: &App) {
    let Some(index) = app.selected_field() else {
        let text = vec![
            Line::from("Start the tournament with these settings."),
            Line::from(""),
            Line::from(Span::styled(
                "The live dashboard opens next and the final standings print when you leave it. Settings that differ from the defaults are saved and pre-filled next time.",
                Style::new().fg(MUTED),
            )),
        ];
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
            match field.kind {
                Kind::Bot(name) => format!("{name} · {}", field.display(&field.value)),
                _ => format!("--{} {}", field.label, field.display(&field.value)),
            },
            Style::new().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(field.about),
        Line::from(""),
        Line::from(vec![
            label("Default    "),
            Span::raw(field.display(&field.default)),
        ]),
    ];
    if let Some((value, why)) = field.suggested {
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
            .block(panel(field.group)),
        area,
    );
}

fn draw_outlook(frame: &mut Frame, area: Rect, app: &App) {
    let label = |text: &'static str| Span::styled(text, Style::new().fg(MUTED));
    let mut lines = Vec::new();
    match Arguments::try_parse_from(app.argv())
        .map_err(|error| clap_message(&error))
        .and_then(|arguments| Plan::of(&arguments).map(|plan| (arguments, plan)))
    {
        Ok((arguments, plan)) => {
            let threads = arguments.threads.clamp(
                1,
                std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            );
            for stage in &plan.stages {
                #[allow(clippy::cast_precision_loss)]
                let wall = stage.seconds / threads.min(stage.games.max(1)) as f64;
                lines.push(Line::from(vec![
                    label("Games      "),
                    Span::styled(
                        format!("{} ", stage.games),
                        Style::new().add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!(
                        "· {} bots · {} · ≈ {}",
                        stage.players,
                        stage.title,
                        duration(wall)
                    )),
                ]));
            }
            let anchored = !arguments.anchor.is_empty()
                || super::parse_bots(&arguments.bots)
                    .unwrap_or_default()
                    .iter()
                    .any(|bot| bot.calibrated_elo().is_some());
            let scale = if arguments.calibrate {
                "Stockfish scale, carried into 5D by the ladder"
            } else if anchored {
                "anchored: Stockfish levels and --anchor values hold"
            } else {
                "relative: centred on zero, comparable only within this run"
            };
            lines.push(Line::from(vec![label("Scale      "), Span::raw(scale)]));
            let intervals = if arguments.bootstrap == 0 {
                "none".to_owned()
            } else {
                format!(
                    "95% from {} resamples of opening pairs",
                    arguments.bootstrap
                )
            };
            lines.push(Line::from(vec![label("Intervals  "), Span::raw(intervals)]));
            if plan.needs_stockfish {
                let found = arguments
                    .stockfish
                    .clone()
                    .or_else(|| std::env::var_os("STOCKFISH").map(PathBuf::from));
                lines.push(Line::from(vec![
                    label("Stockfish  "),
                    match found {
                        Some(path) => Span::raw(path.display().to_string()),
                        None => Span::styled("not set", Style::new().fg(BAD)),
                    },
                ]));
            }
            lines.push(Line::from(Span::styled(
                format!(
                    "Times assume ~{ASSUMED_GAME_ACTIONS:.0} actions per game at typical 5D speeds."
                ),
                Style::new().fg(MUTED),
            )));
        }
        Err(error) => lines.push(Line::from(Span::styled(error, Style::new().fg(BAD)))),
    }
    frame.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: true })
            .block(panel("Outlook")),
        area,
    );
}

pub(super) fn duration(seconds: f64) -> String {
    if seconds >= 5_400.0 {
        format!("{:.1} h", seconds / 3_600.0)
    } else if seconds >= 90.0 {
        format!("{:.0} min", seconds / 60.0)
    } else {
        format!("{seconds:.0} s")
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
        None => {
            let bots = bots_list(&app.fields);
            let count = bots.split(',').filter(|name| !name.is_empty()).count();
            Line::from(Span::styled(
                format!(
                    " {count} bot(s) · ⌘ command line  ↺ saved  yellow = changed  ◆ suggestion available"
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

    fn app_from(args: &[&str]) -> App {
        let matches = Arguments::command().get_matches_from(args);
        App::new(build_fields(&matches, &HashMap::new()))
    }

    #[test]
    fn every_spec_names_a_tournament_argument_and_defaults_round_trip() {
        let app = app_from(&["tournament"]);
        let arguments = app.build().expect("defaults start a tournament");
        assert_eq!(arguments.bots.join(","), ROSTER.join(","));
        assert!(arguments.skip_setup);
        assert_eq!(arguments.openings, 2);
    }

    #[test]
    fn every_argument_except_output_switches_is_in_the_form() {
        for arg in Arguments::command().get_arguments() {
            let id = arg.get_id().as_str();
            if matches!(id, "bots" | "skip_setup" | "help") {
                continue;
            }
            assert!(
                SPECS.iter().any(|spec| spec.id == id),
                "{id} is missing from the setup form"
            );
        }
    }

    #[test]
    fn command_line_bots_set_the_toggles_and_extras() {
        let app = app_from(&["tournament", "--bots", "kairos,blueshift-30k,fissure"]);
        let on = |name: &str| app.value(name) == "true";
        assert!(on("kairos") && on("fissure"));
        assert!(!on("loki") && !on("grabablanca"));
        assert_eq!(app.value(EXTRA_BOTS), "blueshift-30k");
        let arguments = app.build().unwrap();
        assert_eq!(arguments.bots, ["fissure", "kairos", "blueshift-30k"]);
    }

    #[test]
    fn invalid_values_block_the_start() {
        let mut app = app_from(&["tournament"]);
        let extra = app.index_of(EXTRA_BOTS).unwrap();
        app.set(extra, "kasparov".to_owned());
        assert!(app.status.as_ref().is_some_and(|(error, _)| *error));
        assert!(app.build().is_err());
        app.set(extra, String::new());
        app.set_all_bots(false);
        let fissure = app.index_of("fissure").unwrap();
        app.set(fissure, "true".to_owned());
        assert!(app.build().unwrap_err().contains("two bots"));
    }

    #[test]
    fn plans_count_games_like_the_round_robin() {
        let app = app_from(&[
            "tournament",
            "--bots",
            "fissure,grabablanca,kairos",
            "--ruleset",
            "all",
            "--openings",
            "3",
        ]);
        let plan = Plan::of(&app.build().unwrap()).unwrap();
        // Standard: one pair (Kairos sits out); each 5D ruleset: three pairs.
        assert_eq!(plan.stages[0].games, (1 + 3 + 3) * 2 * 3);
    }

    #[test]
    fn settings_round_trip_only_changes() {
        let mut app = app_from(&["tournament"]);
        let openings = app.index_of("openings").unwrap();
        app.set(openings, "5".to_owned());
        let loki = app.index_of("loki").unwrap();
        app.set(loki, "false".to_owned());
        let saved = parse_saved(&saved_text(&app.fields));
        assert_eq!(saved.get("openings").map(String::as_str), Some("5"));
        assert!(saved.get("bots").is_some_and(|bots| !bots.contains("loki")));
        assert_eq!(saved.len(), 2);
        let matches = Arguments::command().get_matches_from(["tournament"]);
        let restored = App::new(build_fields(&matches, &saved));
        assert_eq!(restored.value("openings"), "5");
        assert_eq!(restored.value("loki"), "false");
    }
}
