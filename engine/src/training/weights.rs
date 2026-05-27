impl EvalWeights {
    fn mutate(self, rng: &mut Lcg) -> Self {
        // Keep royal values fixed so training cannot discover incentives that
        // trade away the king-shaped objective for short-term material.
        Self {
            king: self.king,
            common_king: mutate_weight(self.common_king, rng, 80, 0, 2_000),
            queen: mutate_weight(self.queen, rng, 160, 100, 3_000),
            royal_queen: self.royal_queen,
            princess: mutate_weight(self.princess, rng, 140, 100, 3_000),
            rook: mutate_weight(self.rook, rng, 100, 100, 2_000),
            bishop: mutate_weight(self.bishop, rng, 80, 50, 2_000),
            unicorn: mutate_weight(self.unicorn, rng, 90, 50, 2_000),
            dragon: mutate_weight(self.dragon, rng, 120, 50, 2_000),
            knight: mutate_weight(self.knight, rng, 80, 50, 2_000),
            pawn: mutate_weight(self.pawn, rng, 30, 10, 600),
            brawn: mutate_weight(self.brawn, rng, 30, 10, 600),
            check_penalty: mutate_weight(self.check_penalty, rng, 120, 0, 3_000),
            active_timeline: mutate_weight(self.active_timeline, rng, 20, -500, 500),
            inactive_timeline: mutate_weight(self.inactive_timeline, rng, 20, -500, 500),
            present_progress: mutate_weight(self.present_progress, rng, 10, 0, 200),
            mobility: mutate_weight(self.mobility, rng, 4, 0, 80),
            branch_penalty: mutate_weight(self.branch_penalty, rng, 10, 0, 300),
            advancement: mutate_weight(self.advancement, rng, 4, 0, 80),
            centrality: mutate_weight(self.centrality, rng, 4, 0, 80),
            defended_piece: mutate_weight(self.defended_piece, rng, 6, 0, 200),
            attacked_piece: mutate_weight(self.attacked_piece, rng, 8, 0, 300),
            hanging_piece: mutate_weight(self.hanging_piece, rng, 10, 0, 400),
            royal_threat: mutate_weight(self.royal_threat, rng, 30, 0, 1_000),
            temporal_threat: mutate_weight(self.temporal_threat, rng, 8, 0, 300),
            pincer_threat: mutate_weight(self.pincer_threat, rng, 8, 0, 300),
            timeline_pincer: mutate_weight(self.timeline_pincer, rng, 12, 0, 500),
            historical_pincer: mutate_weight(self.historical_pincer, rng, 10, 0, 500),
            frontier_tempo: mutate_weight(self.frontier_tempo, rng, 5, -100, 200),
            present_anchor: mutate_weight(self.present_anchor, rng, 5, -100, 200),
            development: mutate_weight(self.development, rng, 6, 0, 200),
            branch_attack: mutate_weight(self.branch_attack, rng, 12, 0, 600),
            check_bonus: mutate_weight(self.check_bonus, rng, 40, 0, 2_000),
            royal_capture_threat: mutate_weight(self.royal_capture_threat, rng, 60, 0, 3_000),
            royal_escape_pressure: mutate_weight(self.royal_escape_pressure, rng, 10, 0, 400),
            forcing_move_pressure: mutate_weight(self.forcing_move_pressure, rng, 10, 0, 500),
            own_royal_exposure: mutate_weight(self.own_royal_exposure, rng, 50, 0, 3_000),
            fork_pressure: mutate_weight(self.fork_pressure, rng, 20, 0, 1_000),
            board_control: mutate_weight(self.board_control, rng, 3, 0, 80),
            piece_activity: mutate_weight(self.piece_activity, rng, 4, 0, 120),
            pawn_structure: mutate_weight(self.pawn_structure, rng, 6, 0, 200),
            timeline_economy: mutate_weight(self.timeline_economy, rng, 8, 0, 400),
            present_tempo: mutate_weight(self.present_tempo, rng, 8, -100, 300),
            royal_shelter: mutate_weight(self.royal_shelter, rng, 10, 0, 500),
            space_advantage: mutate_weight(self.space_advantage, rng, 4, 0, 120),
        }
    }

    fn crossover(left: Self, right: Self, rng: &mut Lcg) -> Self {
        // Uniform crossover lets each parameter independently come from either
        // parent, which fits this compact, flat genome.
        macro_rules! pick {
            ($field:ident) => {
                if rng.next_bool() {
                    left.$field
                } else {
                    right.$field
                }
            };
        }
        Self {
            king: left.king,
            common_king: pick!(common_king),
            queen: pick!(queen),
            royal_queen: left.royal_queen,
            princess: pick!(princess),
            rook: pick!(rook),
            bishop: pick!(bishop),
            unicorn: pick!(unicorn),
            dragon: pick!(dragon),
            knight: pick!(knight),
            pawn: pick!(pawn),
            brawn: pick!(brawn),
            check_penalty: pick!(check_penalty),
            active_timeline: pick!(active_timeline),
            inactive_timeline: pick!(inactive_timeline),
            present_progress: pick!(present_progress),
            mobility: pick!(mobility),
            branch_penalty: pick!(branch_penalty),
            advancement: pick!(advancement),
            centrality: pick!(centrality),
            defended_piece: pick!(defended_piece),
            attacked_piece: pick!(attacked_piece),
            hanging_piece: pick!(hanging_piece),
            royal_threat: pick!(royal_threat),
            temporal_threat: pick!(temporal_threat),
            pincer_threat: pick!(pincer_threat),
            timeline_pincer: pick!(timeline_pincer),
            historical_pincer: pick!(historical_pincer),
            frontier_tempo: pick!(frontier_tempo),
            present_anchor: pick!(present_anchor),
            development: pick!(development),
            branch_attack: pick!(branch_attack),
            check_bonus: pick!(check_bonus),
            royal_capture_threat: pick!(royal_capture_threat),
            royal_escape_pressure: pick!(royal_escape_pressure),
            forcing_move_pressure: pick!(forcing_move_pressure),
            own_royal_exposure: pick!(own_royal_exposure),
            fork_pressure: pick!(fork_pressure),
            board_control: pick!(board_control),
            piece_activity: pick!(piece_activity),
            pawn_structure: pick!(pawn_structure),
            timeline_economy: pick!(timeline_economy),
            present_tempo: pick!(present_tempo),
            royal_shelter: pick!(royal_shelter),
            space_advantage: pick!(space_advantage),
        }
    }

    fn to_json(self) -> String {
        format!(
            "{{\"king\":{},\"commonKing\":{},\"queen\":{},\"royalQueen\":{},\"princess\":{},\"rook\":{},\"bishop\":{},\"unicorn\":{},\"dragon\":{},\"knight\":{},\"pawn\":{},\"brawn\":{},\"checkPenalty\":{},\"activeTimeline\":{},\"inactiveTimeline\":{},\"presentProgress\":{},\"mobility\":{},\"branchPenalty\":{},\"advancement\":{},\"centrality\":{},\"defendedPiece\":{},\"attackedPiece\":{},\"hangingPiece\":{},\"royalThreat\":{},\"temporalThreat\":{},\"pincerThreat\":{},\"timelinePincer\":{},\"historicalPincer\":{},\"frontierTempo\":{},\"presentAnchor\":{},\"development\":{},\"branchAttack\":{},\"checkBonus\":{},\"royalCaptureThreat\":{},\"royalEscapePressure\":{},\"forcingMovePressure\":{},\"ownRoyalExposure\":{},\"forkPressure\":{},\"boardControl\":{},\"pieceActivity\":{},\"pawnStructure\":{},\"timelineEconomy\":{},\"presentTempo\":{},\"royalShelter\":{},\"spaceAdvantage\":{}}}",
            self.king,
            self.common_king,
            self.queen,
            self.royal_queen,
            self.princess,
            self.rook,
            self.bishop,
            self.unicorn,
            self.dragon,
            self.knight,
            self.pawn,
            self.brawn,
            self.check_penalty,
            self.active_timeline,
            self.inactive_timeline,
            self.present_progress,
            self.mobility,
            self.branch_penalty,
            self.advancement,
            self.centrality,
            self.defended_piece,
            self.attacked_piece,
            self.hanging_piece,
            self.royal_threat,
            self.temporal_threat,
            self.pincer_threat,
            self.timeline_pincer,
            self.historical_pincer,
            self.frontier_tempo,
            self.present_anchor,
            self.development,
            self.branch_attack,
            self.check_bonus,
            self.royal_capture_threat,
            self.royal_escape_pressure,
            self.forcing_move_pressure,
            self.own_royal_exposure,
            self.fork_pressure,
            self.board_control,
            self.piece_activity,
            self.pawn_structure,
            self.timeline_economy,
            self.present_tempo,
            self.royal_shelter,
            self.space_advantage
        )
    }

    fn from_json(value: &str) -> Result<Self, String> {
        Ok(Self {
            king: json_i32(value, "king")?,
            common_king: json_i32(value, "commonKing")?,
            queen: json_i32(value, "queen")?,
            royal_queen: json_i32(value, "royalQueen")?,
            princess: json_i32(value, "princess")?,
            rook: json_i32(value, "rook")?,
            bishop: json_i32(value, "bishop")?,
            unicorn: json_i32(value, "unicorn")?,
            dragon: json_i32(value, "dragon")?,
            knight: json_i32(value, "knight")?,
            pawn: json_i32(value, "pawn")?,
            brawn: json_i32(value, "brawn")?,
            check_penalty: json_i32(value, "checkPenalty")?,
            active_timeline: json_i32(value, "activeTimeline")?,
            inactive_timeline: json_i32(value, "inactiveTimeline")?,
            present_progress: json_i32(value, "presentProgress")?,
            mobility: json_i32(value, "mobility")?,
            branch_penalty: json_i32(value, "branchPenalty")?,
            advancement: json_i32(value, "advancement")?,
            centrality: json_i32(value, "centrality")?,
            defended_piece: json_i32(value, "defendedPiece")?,
            attacked_piece: json_i32(value, "attackedPiece")?,
            hanging_piece: json_i32(value, "hangingPiece")?,
            royal_threat: json_i32(value, "royalThreat")?,
            temporal_threat: json_i32(value, "temporalThreat")?,
            pincer_threat: json_i32(value, "pincerThreat")?,
            timeline_pincer: json_i32(value, "timelinePincer")?,
            historical_pincer: json_i32(value, "historicalPincer")?,
            frontier_tempo: json_i32(value, "frontierTempo")?,
            present_anchor: json_i32(value, "presentAnchor")?,
            development: json_i32(value, "development")?,
            branch_attack: json_i32(value, "branchAttack")?,
            check_bonus: json_i32(value, "checkBonus")?,
            royal_capture_threat: json_i32(value, "royalCaptureThreat")?,
            royal_escape_pressure: json_i32(value, "royalEscapePressure")?,
            forcing_move_pressure: json_i32(value, "forcingMovePressure")?,
            own_royal_exposure: json_i32(value, "ownRoyalExposure")?,
            fork_pressure: json_i32(value, "forkPressure")?,
            board_control: json_i32(value, "boardControl")?,
            piece_activity: json_i32(value, "pieceActivity")?,
            pawn_structure: json_i32(value, "pawnStructure")?,
            timeline_economy: json_i32(value, "timelineEconomy")?,
            present_tempo: json_i32(value, "presentTempo")?,
            royal_shelter: json_i32(value, "royalShelter")?,
            space_advantage: json_i32(value, "spaceAdvantage")?,
        })
    }

    fn to_rust_parameters(self) -> String {
        format!(
            "Self {{\n    king: {},\n    common_king: {},\n    queen: {},\n    royal_queen: {},\n    princess: {},\n    rook: {},\n    bishop: {},\n    unicorn: {},\n    dragon: {},\n    knight: {},\n    pawn: {},\n    brawn: {},\n    check_penalty: {},\n    active_timeline: {},\n    inactive_timeline: {},\n    present_progress: {},\n    mobility: {},\n    branch_penalty: {},\n    advancement: {},\n    centrality: {},\n    defended_piece: {},\n    attacked_piece: {},\n    hanging_piece: {},\n    royal_threat: {},\n    temporal_threat: {},\n    pincer_threat: {},\n    timeline_pincer: {},\n    historical_pincer: {},\n    frontier_tempo: {},\n    present_anchor: {},\n    development: {},\n    branch_attack: {},\n    check_bonus: {},\n    royal_capture_threat: {},\n    royal_escape_pressure: {},\n    forcing_move_pressure: {},\n    own_royal_exposure: {},\n    fork_pressure: {},\n    board_control: {},\n    piece_activity: {},\n    pawn_structure: {},\n    timeline_economy: {},\n    present_tempo: {},\n    royal_shelter: {},\n    space_advantage: {},\n}}\n",
            self.king,
            self.common_king,
            self.queen,
            self.royal_queen,
            self.princess,
            self.rook,
            self.bishop,
            self.unicorn,
            self.dragon,
            self.knight,
            self.pawn,
            self.brawn,
            self.check_penalty,
            self.active_timeline,
            self.inactive_timeline,
            self.present_progress,
            self.mobility,
            self.branch_penalty,
            self.advancement,
            self.centrality,
            self.defended_piece,
            self.attacked_piece,
            self.hanging_piece,
            self.royal_threat,
            self.temporal_threat,
            self.pincer_threat,
            self.timeline_pincer,
            self.historical_pincer,
            self.frontier_tempo,
            self.present_anchor,
            self.development,
            self.branch_attack,
            self.check_bonus,
            self.royal_capture_threat,
            self.royal_escape_pressure,
            self.forcing_move_pressure,
            self.own_royal_exposure,
            self.fork_pressure,
            self.board_control,
            self.piece_activity,
            self.pawn_structure,
            self.timeline_economy,
            self.present_tempo,
            self.royal_shelter,
            self.space_advantage
        )
    }
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper.max(1)
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
