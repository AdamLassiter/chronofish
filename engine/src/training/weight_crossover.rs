use super::*;
use crate::cpu::EvalWeights;

impl EvalWeights {
    pub(crate) fn crossover(left: Self, right: Self, rng: &mut Lcg) -> Self {
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
            mandatory_move_burden: pick!(mandatory_move_burden),
            turn_completion_safety: pick!(turn_completion_safety),
            present_zugzwang: pick!(present_zugzwang),
            weakest_royal_safety: pick!(weakest_royal_safety),
            royal_liability_count: pick!(royal_liability_count),
            multi_royal_attack: pick!(multi_royal_attack),
            defensive_bandwidth: pick!(defensive_bandwidth),
            threat_overload: pick!(threat_overload),
            active_branch_capacity: pick!(active_branch_capacity),
            latent_timeline_reactivation: pick!(latent_timeline_reactivation),
            inactive_material_quality: pick!(inactive_material_quality),
            branch_payload: pick!(branch_payload),
            branch_waste: pick!(branch_waste),
            timeline_compaction: pick!(timeline_compaction),
            frontier_material: pick!(frontier_material),
            historical_access: pick!(historical_access),
            temporal_lane_control: pick!(temporal_lane_control),
            temporal_pin: pick!(temporal_pin),
            temporal_skewer: pick!(temporal_skewer),
            causal_battery: pick!(causal_battery),
            arrival_square_safety: pick!(arrival_square_safety),
            source_board_abandonment: pick!(source_board_abandonment),
            piece_temporal_flexibility: pick!(piece_temporal_flexibility),
            dimension_coverage_balance: pick!(dimension_coverage_balance),
            promotion_timeline_choice: pick!(promotion_timeline_choice),
            promotion_with_check: pick!(promotion_with_check),
            past_royal_vulnerability: pick!(past_royal_vulnerability),
            safe_haven_boards: pick!(safe_haven_boards),
            escape_branch_potential: pick!(escape_branch_potential),
            mate_net_depth_1_2: pick!(mate_net_depth_1_2),
            anti_mate_resources: pick!(anti_mate_resources),
            checking_move_quality: pick!(checking_move_quality),
            search_volatility: pick!(search_volatility),
            timeline_repetition_risk: pick!(timeline_repetition_risk),
            phase_by_multiverse_size: pick!(phase_by_multiverse_size),
            royal_distance_in_4d: pick!(royal_distance_in_4d),
            board_importance_weight: pick!(board_importance_weight),
        }
    }
}
