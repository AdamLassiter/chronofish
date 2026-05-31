impl EvalWeights {
    fn mutate(self, rng: &mut Lcg) -> Self {
        self.mutate_with_scale(rng, 1.0)
    }

    fn mutate_with_scale(self, rng: &mut Lcg, scale: f32) -> Self {
        // Keep royal values fixed so training cannot discover incentives that
        // trade away the king-shaped objective for short-term material.
        let spread = |value: i32| ((value as f32) * scale).round().max(1.0) as i32;
        Self {
            king: self.king,
            common_king: mutate_weight(self.common_king, rng, spread(80), 0, 2_000),
            queen: mutate_weight(self.queen, rng, spread(160), 100, 3_000),
            royal_queen: self.royal_queen,
            princess: mutate_weight(self.princess, rng, spread(140), 100, 3_000),
            rook: mutate_weight(self.rook, rng, spread(100), 100, 2_000),
            bishop: mutate_weight(self.bishop, rng, spread(80), 50, 2_000),
            unicorn: mutate_weight(self.unicorn, rng, spread(90), 50, 2_000),
            dragon: mutate_weight(self.dragon, rng, spread(120), 50, 2_000),
            knight: mutate_weight(self.knight, rng, spread(80), 50, 2_000),
            pawn: mutate_weight(self.pawn, rng, spread(30), 10, 600),
            brawn: mutate_weight(self.brawn, rng, spread(30), 10, 600),
            check_penalty: mutate_weight(self.check_penalty, rng, spread(80), 0, 3_000),
            active_timeline: mutate_weight(self.active_timeline, rng, spread(20), -500, 500),
            inactive_timeline: mutate_weight(self.inactive_timeline, rng, spread(20), -500, 500),
            present_progress: mutate_weight(self.present_progress, rng, spread(10), 0, 200),
            mobility: mutate_weight(self.mobility, rng, spread(4), 0, 80),
            branch_penalty: mutate_weight(self.branch_penalty, rng, spread(10), 0, 300),
            advancement: mutate_weight(self.advancement, rng, spread(4), 0, 80),
            centrality: mutate_weight(self.centrality, rng, spread(4), 0, 80),
            defended_piece: mutate_weight(self.defended_piece, rng, spread(6), 0, 200),
            attacked_piece: mutate_weight(self.attacked_piece, rng, spread(8), 0, 300),
            hanging_piece: mutate_weight(self.hanging_piece, rng, spread(10), 0, 400),
            royal_threat: mutate_weight(self.royal_threat, rng, spread(20), 0, 1_000),
            temporal_threat: mutate_weight(self.temporal_threat, rng, spread(8), 0, 300),
            pincer_threat: mutate_weight(self.pincer_threat, rng, spread(8), 0, 300),
            timeline_pincer: mutate_weight(self.timeline_pincer, rng, spread(12), 0, 500),
            historical_pincer: mutate_weight(self.historical_pincer, rng, spread(10), 0, 500),
            frontier_tempo: mutate_weight(self.frontier_tempo, rng, spread(5), -100, 200),
            present_anchor: mutate_weight(self.present_anchor, rng, spread(5), -100, 200),
            development: mutate_weight(self.development, rng, spread(6), 0, 200),
            branch_attack: mutate_weight(self.branch_attack, rng, spread(12), 0, 600),
            check_bonus: mutate_weight(self.check_bonus, rng, spread(24), 0, 2_000),
            royal_capture_threat: mutate_weight(
                self.royal_capture_threat,
                rng,
                spread(30),
                0,
                3_000,
            ),
            royal_capture_setup: mutate_weight(
                self.royal_capture_setup,
                rng,
                spread(80),
                0,
                6_000,
            ),
            royal_escape_pressure: mutate_weight(
                self.royal_escape_pressure,
                rng,
                spread(10),
                0,
                400,
            ),
            forcing_move_pressure: mutate_weight(
                self.forcing_move_pressure,
                rng,
                spread(10),
                0,
                500,
            ),
            own_royal_exposure: mutate_weight(self.own_royal_exposure, rng, spread(30), 0, 3_000),
            fork_pressure: mutate_weight(self.fork_pressure, rng, spread(20), 0, 1_000),
            board_control: mutate_weight(self.board_control, rng, spread(3), 0, 80),
            piece_activity: mutate_weight(self.piece_activity, rng, spread(4), 0, 120),
            pawn_structure: mutate_weight(self.pawn_structure, rng, spread(6), 0, 200),
            timeline_economy: mutate_weight(self.timeline_economy, rng, spread(8), 0, 400),
            present_tempo: mutate_weight(self.present_tempo, rng, spread(8), -100, 300),
            royal_shelter: mutate_weight(self.royal_shelter, rng, spread(10), 0, 500),
            space_advantage: mutate_weight(self.space_advantage, rng, spread(4), 0, 120),
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
            royal_capture_setup: pick!(royal_capture_setup),
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
        serde_json::to_string(&self).expect("EvalWeights should serialize")
    }

    fn from_json(value: &str) -> Result<Self, String> {
        serde_json::from_str(value).map_err(|error| error.to_string())
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
