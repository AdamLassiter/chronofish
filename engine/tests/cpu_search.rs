use chronofish_engine::cpu::search::{
    bot_training_moves_key,
    breed_cpu_population,
    cpu_candidate_worker_count,
    cpu_label_worker_count,
    cpu_match_turn_time_ms,
    cpu_parameters_key,
    cpu_reference_candidate_average,
    cpu_reference_score_delta,
    cpu_reference_worker_count,
    cpu_screening_training_config,
    cpu_search_label_weight,
    cpu_training_adjudication_score,
    cpu_training_budget_ms,
    cpu_training_candidate_count,
    cpu_training_elite_count,
    cpu_training_finalist_target,
    cpu_training_no_move_score,
    cpu_training_position_search_config,
    cpu_training_position_target,
    cpu_training_position_worker_count,
    cpu_training_winner_score,
    crossover_cpu_parameters,
    mode_label_target,
    move_agreement_bonus,
    mutate_cpu_parameters,
    search,
    unique_cpu_parameters,
    CpuParameters,
    CpuReferenceScoreDelta,
    CpuSearchRequest,
    CpuTrainingMove,
    CpuTrainingPositionSearchConfig,
    CPU_TRAINING_WIN_SCORE,
    MAX_CPU_TRAINING_CANDIDATES,
    MAX_CPU_TRAINING_ELITES,
};

#[test]
fn native_cpu_search_returns_web_worker_compatible_json() {
    let response = search(CpuSearchRequest {
        depth: 1,
        min_depth: Some(1),
        nodes: 64,
        time_ms: 1_000,
        ..CpuSearchRequest::default()
    })
    .expect("run native CPU search");

    assert_eq!(response.cpu_search, "heuristic");
    let value: serde_json::Value =
        serde_json::from_str(&response.result_json).expect("CPU search JSON should parse");
    assert_eq!(value["status"], "ok");
    assert!(value["moves"].as_array().is_some());
    assert!(value["principalVariation"].as_array().is_some());
    assert_eq!(value["depth"], 1);
}

#[test]
fn native_cpu_search_rejects_invalid_snapshot_json() {
    let error = search(CpuSearchRequest {
        snapshot_json: Some("{\"turn\":\"white\",\"timelines\":null}".to_string()),
        ..CpuSearchRequest::default()
    })
    .expect_err("invalid snapshot should fail");

    assert!(error.contains("Snapshot timelines must be an array"));
}

#[test]
fn cpu_parameter_identity_matches_browser_policy() {
    let first = parameters(&[("mobility", 2), ("queen", 10)]);
    let reordered = parameters(&[("queen", 10), ("mobility", 2)]);
    let different = parameters(&[("queen", 11), ("mobility", 2)]);

    assert_eq!(cpu_parameters_key(&reordered), "mobility:2|queen:10");
    assert_eq!(
        unique_cpu_parameters(&[first.clone(), reordered, different.clone()]),
        vec![first, different]
    );
}

#[test]
fn cpu_reference_worker_count_matches_browser_bounds() {
    assert_eq!(cpu_reference_worker_count(0, 8, 8), 0);
    assert_eq!(cpu_reference_worker_count(12, 8, 3), 3);
    assert_eq!(cpu_reference_worker_count(2, 8, 3), 2);
    assert_eq!(cpu_reference_worker_count(12, 0, 0), 1);

    assert_eq!(cpu_training_position_worker_count(0, 8), 0);
    assert_eq!(cpu_training_position_worker_count(12, 8), 8);
    assert_eq!(cpu_training_position_worker_count(2, 8), 2);
    assert_eq!(cpu_training_position_worker_count(12, 0), 1);
    assert_eq!(cpu_label_worker_count(0, 8), 0);
    assert_eq!(cpu_label_worker_count(12, 8), 8);
    assert_eq!(cpu_label_worker_count(2, 8), 2);
    assert_eq!(cpu_label_worker_count(12, 0), 1);
    assert_eq!(cpu_candidate_worker_count(0, 8, 8), 0);
    assert_eq!(cpu_candidate_worker_count(12, 8, 3), 3);
    assert_eq!(cpu_candidate_worker_count(2, 8, 3), 2);
    assert_eq!(cpu_candidate_worker_count(12, 0, 0), 1);
}

#[test]
fn cpu_search_label_weight_matches_browser_policy() {
    assert_eq!(cpu_search_label_weight(0), 1.0);
    assert_eq!(cpu_search_label_weight(1), 1.0);
    assert_eq!(cpu_search_label_weight(2), 1.1);
}

#[test]
fn cpu_parameter_crossover_and_mutation_match_browser_fixtures() {
    let baseline = parameters(&[
        ("mobility", 100),
        ("queen", 900),
        ("king", 10_000),
        ("royal_queen", 9_000),
    ]);
    let right = parameters(&[
        ("mobility", 200),
        ("queen", 600),
        ("king", -1),
        ("royal_queen", -1),
        ("extra", 7),
    ]);

    assert_eq!(
        crossover_cpu_parameters(&baseline, &right, 99),
        parameters(&[
            ("mobility", 156),
            ("queen", 629),
            ("king", 10_000),
            ("royal_queen", 9_000),
            ("extra", 7),
        ])
    );
    assert_eq!(
        mutate_cpu_parameters(&baseline, 1234, 1.0),
        parameters(&[
            ("mobility", 98),
            ("queen", 944),
            ("king", 10_000),
            ("royal_queen", 9_000),
        ])
    );
}

#[test]
fn cpu_population_breeding_matches_browser_fixture() {
    let baseline = parameters(&[
        ("mobility", 100),
        ("queen", 900),
        ("king", 10_000),
        ("royal_queen", 9_000),
    ]);
    let elite = parameters(&[
        ("mobility", 108),
        ("queen", 880),
        ("king", 10_000),
        ("royal_queen", 9_000),
    ]);

    assert_eq!(
        breed_cpu_population(&baseline, &[elite], 6, 1234, 2, 1),
        vec![
            parameters(&[
                ("mobility", 100),
                ("queen", 900),
                ("king", 10_000),
                ("royal_queen", 9_000),
            ]),
            parameters(&[
                ("mobility", 108),
                ("queen", 880),
                ("king", 10_000),
                ("royal_queen", 9_000),
            ]),
            parameters(&[
                ("mobility", 91),
                ("queen", 886),
                ("king", 10_000),
                ("royal_queen", 9_000),
            ]),
            parameters(&[
                ("mobility", 107),
                ("queen", 902),
                ("king", 10_000),
                ("royal_queen", 9_000),
            ]),
            parameters(&[
                ("mobility", 115),
                ("queen", 990),
                ("king", 10_000),
                ("royal_queen", 9_000),
            ]),
            parameters(&[
                ("mobility", 103),
                ("queen", 918),
                ("king", 10_000),
                ("royal_queen", 9_000),
            ]),
        ]
    );
}

#[test]
fn cpu_match_budgeting_matches_browser_training_policy() {
    assert_eq!(cpu_match_turn_time_ms(500, 1_000.0, 4_000.0, 3), 500);
    assert_eq!(cpu_match_turn_time_ms(500, 1_000.0, 1_100.0, 3), 33);
    assert_eq!(cpu_match_turn_time_ms(500, 1_000.0, 2_000.0, 0), 500);

    assert_eq!(mode_label_target(96, 3, 12), 8);
    assert_eq!(mode_label_target(96, 1, 12), 96);
    assert_eq!(
        cpu_training_position_target(96, 3, 2, 1, 2, 3, 2, 5, 4, 20, 12),
        8
    );
    assert_eq!(
        cpu_training_position_target(96, 1, 0, 0, 8, 0, 0, -2, 3, 9, 5),
        96
    );
    assert_eq!(cpu_training_budget_ms(120, 500, 12, 15_000), 15_000);
    assert_eq!(cpu_training_budget_ms(1, 10, 5, 0), 1_000);

    assert_eq!(
        cpu_training_position_search_config(5, 8_192),
        CpuTrainingPositionSearchConfig {
            depth: 2,
            nodes: 512,
        }
    );
    assert_eq!(
        cpu_training_position_search_config(1, 256),
        CpuTrainingPositionSearchConfig {
            depth: 1,
            nodes: 256,
        }
    );
    assert_eq!(
        cpu_training_position_search_config(0, -1),
        CpuTrainingPositionSearchConfig { depth: 1, nodes: 1 }
    );

    assert_eq!(MAX_CPU_TRAINING_CANDIDATES, 256);
    assert_eq!(MAX_CPU_TRAINING_ELITES, 4);
    assert_eq!(cpu_training_candidate_count(0), 1);
    assert_eq!(cpu_training_candidate_count(8), 8);
    assert_eq!(cpu_training_candidate_count(300), 256);
    assert_eq!(cpu_training_finalist_target(20, 1, 4, 12), 4);
    assert_eq!(cpu_training_finalist_target(20, 8, 4, 12), 8);
    assert_eq!(cpu_training_finalist_target(20, 1, 12, 0), 12);
    assert_eq!(cpu_training_finalist_target(3, 8, 12, 20), 3);
    assert_eq!(cpu_training_finalist_target(0, 8, 12, 20), 0);
    assert_eq!(cpu_training_elite_count(0), 1);
    assert_eq!(cpu_training_elite_count(3), 3);
    assert_eq!(cpu_training_elite_count(99), 4);
}

#[test]
fn cpu_screening_config_matches_browser_training_policy() {
    assert_eq!(
        cpu_screening_training_config(5, 4, 8_192, 16_384, 10_000),
        chronofish_engine::cpu::search::CpuScreeningTrainingConfig {
            cpu_depth: 2,
            depth: 2,
            cpu_nodes: 2_048,
            nodes: 4_096,
            cpu_training_time_ms: 2_500,
        }
    );
    assert_eq!(
        cpu_screening_training_config(1, 2, 1, 3, 1),
        chronofish_engine::cpu::search::CpuScreeningTrainingConfig {
            cpu_depth: 1,
            depth: 2,
            cpu_nodes: 1,
            nodes: 1,
            cpu_training_time_ms: 1,
        }
    );
    assert_eq!(
        cpu_screening_training_config(0, -5, 0, -1, 0),
        chronofish_engine::cpu::search::CpuScreeningTrainingConfig {
            cpu_depth: 1,
            depth: 1,
            cpu_nodes: 1,
            nodes: 1,
            cpu_training_time_ms: 1,
        }
    );
}

#[test]
fn cpu_move_agreement_matches_browser_training_policy() {
    let moves = vec![
        CpuTrainingMove {
            from_timeline_id: 1,
            from_time: 2,
            from_x: 3,
            from_y: 4,
            to_timeline_id: 1,
            to_time: 3,
            to_x: 4,
            to_y: 5,
        },
        CpuTrainingMove {
            from_timeline_id: -1,
            from_time: 8,
            from_x: 0,
            from_y: 7,
            to_timeline_id: 2,
            to_time: 9,
            to_x: 6,
            to_y: 1,
        },
    ];
    let reversed = vec![moves[1], moves[0]];

    assert_eq!(
        bot_training_moves_key(&moves),
        "1,2,3,4,1,3,4,5|-1,8,0,7,2,9,6,1"
    );
    assert_eq!(move_agreement_bonus(&moves, &moves), 25);
    assert_eq!(move_agreement_bonus(&moves, &reversed), 0);
    assert_eq!(move_agreement_bonus(&[], &[]), 0);
}

#[test]
fn cpu_reference_score_deltas_match_browser_training_policy() {
    let candidate_moves = vec![CpuTrainingMove {
        from_timeline_id: 1,
        from_time: 2,
        from_x: 3,
        from_y: 4,
        to_timeline_id: 1,
        to_time: 3,
        to_x: 4,
        to_y: 5,
    }];
    let other_moves = vec![CpuTrainingMove {
        from_timeline_id: 1,
        from_time: 2,
        from_x: 3,
        from_y: 4,
        to_timeline_id: 1,
        to_time: 3,
        to_x: 4,
        to_y: 6,
    }];

    assert_eq!(
        cpu_reference_score_delta(120, 100, &candidate_moves, &candidate_moves, 25),
        CpuReferenceScoreDelta {
            score: 45,
            near_draw: true,
        }
    );
    assert_eq!(
        cpu_reference_score_delta(190, 100, &candidate_moves, &other_moves, 25),
        CpuReferenceScoreDelta {
            score: 90,
            near_draw: false,
        }
    );
    assert_eq!(
        cpu_reference_score_delta(-20, 0, &[], &candidate_moves, -5),
        CpuReferenceScoreDelta {
            score: -20,
            near_draw: false,
        }
    );

    assert_eq!(cpu_reference_candidate_average(120, 3, 2, 0.8), 40.0);
    assert_eq!(cpu_reference_candidate_average(120, 3, 3, 0.8), 20.0);
    assert_eq!(cpu_reference_candidate_average(120, 0, 0, 0.8), 120.0);
}

#[test]
fn cpu_paired_match_scores_match_browser_training_policy() {
    assert_eq!(CPU_TRAINING_WIN_SCORE, 100_000);
    assert_eq!(cpu_training_no_move_score(true), -100_000);
    assert_eq!(cpu_training_no_move_score(false), 100_000);

    assert_eq!(cpu_training_winner_score(Some("white"), "white"), 100_000);
    assert_eq!(cpu_training_winner_score(Some("black"), "white"), -100_000);
    assert_eq!(cpu_training_winner_score(None, "white"), 0);

    assert_eq!(cpu_training_adjudication_score("white", "white", 125), 125);
    assert_eq!(cpu_training_adjudication_score("black", "white", 125), -125);
}

fn parameters(values: &[(&str, i32)]) -> CpuParameters {
    values
        .iter()
        .map(|(key, value)| ((*key).to_string(), *value))
        .collect()
}
