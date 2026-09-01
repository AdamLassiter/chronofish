use chronofish_engine::cpu::search::{
    bot_training_moves_key,
    breed_cpu_population,
    cpu_apply_turn_json,
    cpu_candidate_scoring_plan,
    cpu_candidate_scoring_should_continue,
    cpu_candidate_worker_count,
    cpu_fitness_entry_for_candidate,
    cpu_label_worker_count,
    cpu_match_remaining_searches,
    cpu_match_should_continue,
    cpu_match_turn_time_ms,
    cpu_paired_match_average_score,
    cpu_paired_match_candidate_colors,
    cpu_paired_match_deadline_ms,
    cpu_paired_match_total_matches,
    cpu_parameters_key,
    cpu_reference_candidate_average,
    cpu_reference_collection_should_continue,
    cpu_reference_comparison_count,
    cpu_reference_score_delta,
    cpu_reference_score_delta_from_result_json,
    cpu_reference_score_delta_json,
    cpu_reference_score_from_result_json,
    cpu_reference_should_continue,
    cpu_reference_worker_count,
    cpu_screening_game_count,
    cpu_screening_training_config,
    cpu_search_label_weight,
    cpu_training_adjudication_score,
    cpu_training_adjudication_score_from_result_json,
    cpu_training_budget_ms,
    cpu_training_candidate_count,
    cpu_training_candidate_improved,
    cpu_training_candidate_turn,
    cpu_training_elite_count,
    cpu_training_elites,
    cpu_training_finalist_candidates,
    cpu_training_finalist_target,
    cpu_training_generation_outcome,
    cpu_training_next_stagnation,
    cpu_training_no_move_score,
    cpu_training_position_search_config,
    cpu_training_position_target,
    cpu_training_position_worker_count,
    cpu_training_should_continue,
    cpu_training_winner_score,
    cpu_worker_search_config_json,
    cpu_worker_search_result_json,
    crossover_cpu_parameters,
    mode_label_target,
    move_agreement_bonus,
    mutate_cpu_parameters,
    rank_cpu_scored_candidates,
    search,
    unique_cpu_parameters,
    CpuFitnessEntry,
    CpuParameters,
    CpuReferenceScoreDelta,
    CpuScoredCandidate,
    CpuSearchRequest,
    CpuSearchStrategy,
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
        search_strategy: CpuSearchStrategy::Beam,
        ..CpuSearchRequest::default()
    })
    .expect("run native CPU search");

    assert_eq!(response.cpu_search, "heuristic");
    let value: serde_json::Value =
        serde_json::from_str(&response.result_json).expect("CPU search JSON should parse");
    assert_eq!(value["status"], "beam");
    assert!(value["moves"].as_array().is_some());
    assert!(value["principalVariation"].as_array().is_some());
    assert_eq!(value["depth"], 1);
}

#[test]
fn native_cpu_search_can_use_alpha_beta_when_requested() {
    let response = search(CpuSearchRequest {
        depth: 2,
        min_depth: Some(2),
        nodes: 20_000,
        time_ms: 1_000,
        search_strategy: CpuSearchStrategy::AlphaBeta,
        ..CpuSearchRequest::default()
    })
    .expect("run alpha-beta CPU search");

    let value: serde_json::Value =
        serde_json::from_str(&response.result_json).expect("CPU search JSON should parse");
    assert_eq!(value["status"], "ok");
    assert_eq!(value["depth"], 2);
    assert!(value["principalVariation"]
        .as_array()
        .is_some_and(|pv| pv.len() >= 2));
}

#[test]
fn native_cpu_beam_search_reports_its_actual_one_ply_depth() {
    let response = search(CpuSearchRequest {
        depth: 8,
        min_depth: Some(6),
        nodes: 200_000,
        time_ms: 1_000,
        search_strategy: CpuSearchStrategy::Beam,
        ..CpuSearchRequest::default()
    })
    .expect("run beam CPU search");

    let value: serde_json::Value =
        serde_json::from_str(&response.result_json).expect("CPU search JSON should parse");
    assert_eq!(value["status"], "beam");
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
fn cpu_worker_search_config_matches_browser_worker_contract() {
    let config = cpu_worker_search_config_json(
        r#"{
            "depth": 4.8,
            "minDepth": 9,
            "nodes": 127.9,
            "timeMs": 250.9
        }"#,
    )
    .expect("CPU worker search config");
    let value: serde_json::Value =
        serde_json::from_str(&config).expect("CPU worker search config JSON");
    assert_eq!(value["depth"], 4);
    assert_eq!(value["minDepth"], 4);
    assert_eq!(value["nodes"], 127);
    assert_eq!(value["timeMs"], 250);
    assert_eq!(value["searchStrategy"], "alpha-beta");

    let defaults = cpu_worker_search_config_json("{}").expect("default CPU worker search config");
    let value: serde_json::Value =
        serde_json::from_str(&defaults).expect("default CPU worker search config JSON");
    assert_eq!(value["depth"], 1);
    assert!(value.get("minDepth").is_none());
    assert_eq!(value["nodes"], 64);
    assert_eq!(value["timeMs"], 10_000);
    assert_eq!(value["searchStrategy"], "alpha-beta");

    let bounded = cpu_worker_search_config_json(
        r#"{
            "depth": 0,
            "minDepth": -5,
            "nodes": null,
            "timeMs": 0
        }"#,
    )
    .expect("bounded CPU worker search config");
    let value: serde_json::Value =
        serde_json::from_str(&bounded).expect("bounded CPU worker search config JSON");
    assert_eq!(value["depth"], 1);
    assert_eq!(value["minDepth"], 1);
    assert_eq!(value["nodes"], 64);
    assert_eq!(value["timeMs"], 1);
    assert_eq!(value["searchStrategy"], "alpha-beta");

    let alpha_beta = cpu_worker_search_config_json(r#"{"searchStrategy":"alpha-beta"}"#)
        .expect("alpha-beta CPU worker search config");
    let value: serde_json::Value =
        serde_json::from_str(&alpha_beta).expect("alpha-beta CPU worker search config JSON");
    assert_eq!(value["searchStrategy"], "alpha-beta");
}

#[test]
fn cpu_worker_search_result_matches_browser_worker_contract() {
    let result = cpu_worker_search_result_json(
        r#"{"status":"ok","moves":[{"from":{"timelineId":0,"time":0,"x":0,"y":1},"to":{"timelineId":0,"time":1,"x":0,"y":2}}],"score":12}"#,
    )
    .expect("CPU worker search result");
    let value: serde_json::Value =
        serde_json::from_str(&result).expect("CPU worker search result JSON");
    assert_eq!(value["cpuSearch"], "heuristic");
    assert_eq!(value["principalVariation"][0], value["moves"]);

    let empty = cpu_worker_search_result_json(r#"{"status":"ok","moves":[]}"#)
        .expect("empty CPU worker search result");
    let empty_value: serde_json::Value =
        serde_json::from_str(&empty).expect("empty CPU worker search result JSON");
    assert_eq!(
        empty_value["principalVariation"].as_array().map(Vec::len),
        Some(0)
    );

    let existing = cpu_worker_search_result_json(
        r#"{"status":"ok","moves":[],"principalVariation":[[{"from":{"timelineId":1,"time":0,"x":1,"y":1},"to":{"timelineId":1,"time":1,"x":1,"y":2}}]]}"#,
    )
    .expect("existing CPU worker search result");
    let existing_value: serde_json::Value =
        serde_json::from_str(&existing).expect("existing CPU worker search result JSON");
    assert_eq!(
        existing_value["principalVariation"]
            .as_array()
            .map(Vec::len),
        Some(1)
    );
}

#[test]
fn cpu_apply_turn_matches_browser_worker_contract() {
    let response = cpu_apply_turn_json(
        r#"{
            "game": { "turn": "white", "timelines": [] },
            "moves": []
        }"#,
    )
    .expect("CPU apply turn");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("CPU apply turn response JSON");
    assert_eq!(value["status"]["complete"], false);
    assert_eq!(value["status"]["terminal"], false);
    assert_eq!(value["status"]["nextTurn"], "white");
    assert_eq!(value["game"]["turn"], "white");
    assert!(value["game"]["timelines"].as_array().is_some());

    let omitted_moves = cpu_apply_turn_json(
        r#"{
            "game": { "turn": "white", "timelines": [] }
        }"#,
    )
    .expect("CPU apply turn without moves");
    let omitted_value: serde_json::Value =
        serde_json::from_str(&omitted_moves).expect("CPU apply turn response JSON");
    assert_eq!(omitted_value["status"]["complete"], false);
    assert_eq!(omitted_value["game"]["turn"], "white");
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
    assert_eq!(cpu_match_remaining_searches(12, 0), 13);
    assert_eq!(cpu_match_remaining_searches(12, 11), 2);
    assert_eq!(cpu_match_remaining_searches(12, 12), 1);
    assert_eq!(cpu_match_remaining_searches(12, 20), 1);
    assert!(cpu_match_should_continue(1_000.0, 1_001.0));
    assert!(!cpu_match_should_continue(1_001.0, 1_001.0));
    assert_eq!(
        cpu_paired_match_deadline_ms(1_000.0, 2_000.0, 4, 0),
        1_250.0
    );
    assert_eq!(
        cpu_paired_match_deadline_ms(1_000.0, 2_000.0, 4, 3),
        2_000.0
    );
    assert_eq!(cpu_paired_match_deadline_ms(1_000.0, 900.0, 4, 0), 900.0);
    assert_eq!(cpu_paired_match_total_matches(0), 0);
    assert_eq!(cpu_paired_match_total_matches(3), 6);
    assert_eq!(cpu_paired_match_total_matches(usize::MAX), usize::MAX);
    assert_eq!(
        cpu_paired_match_candidate_colors("white").expect("white color pair"),
        vec!["white", "black"]
    );
    assert_eq!(
        cpu_paired_match_candidate_colors("black").expect("black color pair"),
        vec!["black", "white"]
    );
    assert!(cpu_paired_match_candidate_colors("blue").is_err());
    assert_eq!(cpu_paired_match_average_score(45.0, 3), 15.0);
    assert!(cpu_paired_match_average_score(45.0, 0).is_nan());
    assert!(cpu_training_candidate_turn("white", "white"));
    assert!(!cpu_training_candidate_turn("black", "white"));

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
    assert_eq!(cpu_screening_game_count(0, 2), 0);
    assert_eq!(cpu_screening_game_count(12, 0), 1);
    assert_eq!(cpu_screening_game_count(12, 2), 2);
    assert_eq!(cpu_screening_game_count(2, 12), 2);
    assert_eq!(cpu_training_finalist_target(20, 1, 4, 12), 4);
    assert_eq!(cpu_training_finalist_target(20, 8, 4, 12), 8);
    assert_eq!(cpu_training_finalist_target(20, 1, 12, 0), 12);
    assert_eq!(cpu_training_finalist_target(3, 8, 12, 20), 3);
    assert_eq!(cpu_training_finalist_target(0, 8, 12, 20), 0);
    assert_eq!(cpu_training_elite_count(0), 1);
    assert_eq!(cpu_training_elite_count(3), 3);
    assert_eq!(cpu_training_elite_count(99), 4);
    assert!(cpu_training_candidate_improved(12.0, 10.0, 11.0));
    assert!(!cpu_training_candidate_improved(10.0, 10.0, 9.0));
    assert!(!cpu_training_candidate_improved(12.0, 10.0, 12.0));
    assert!(!cpu_training_candidate_improved(f64::NAN, 10.0, 0.0));
    assert_eq!(cpu_training_next_stagnation(12, true), 0);
    assert_eq!(cpu_training_next_stagnation(12, false), 13);
    assert_eq!(cpu_training_next_stagnation(usize::MAX, false), usize::MAX);
    assert!(cpu_training_should_continue(1_000.0, 2_000.0, 3, 8));
    assert!(!cpu_training_should_continue(2_000.0, 2_000.0, 3, 8));
    assert!(!cpu_training_should_continue(1_000.0, 2_000.0, 8, 8));
    assert!(cpu_candidate_scoring_should_continue(
        1_000.0, 2_000.0, 3, 8
    ));
    assert!(!cpu_candidate_scoring_should_continue(
        2_000.0, 2_000.0, 3, 8
    ));
    assert!(!cpu_candidate_scoring_should_continue(
        1_000.0, 2_000.0, 8, 8
    ));
    assert!(cpu_reference_collection_should_continue(
        1_000.0, 2_000.0, 3, 8
    ));
    assert!(!cpu_reference_collection_should_continue(
        2_000.0, 2_000.0, 3, 8
    ));
    assert!(!cpu_reference_collection_should_continue(
        1_000.0, 2_000.0, 8, 8
    ));
    assert_eq!(cpu_reference_comparison_count(12, 0), 12);
    assert_eq!(cpu_reference_comparison_count(12, 5), 5);
    assert_eq!(cpu_reference_comparison_count(3, 5), 3);
    assert!(cpu_reference_should_continue(1_000.0, 2_000.0, 3, 8));
    assert!(!cpu_reference_should_continue(2_000.0, 2_000.0, 3, 8));
    assert!(!cpu_reference_should_continue(1_000.0, 2_000.0, 8, 8));

    let baseline = parameters(&[("mobility", 1), ("tempo", 2)]);
    let low = parameters(&[("mobility", 2), ("tempo", 3)]);
    let high = parameters(&[("mobility", 3), ("tempo", 4)]);
    let tied = parameters(&[("mobility", 4), ("tempo", 5)]);
    let ranked = rank_cpu_scored_candidates(vec![
        CpuScoredCandidate {
            parameters: low.clone(),
            score: 3.0,
        },
        CpuScoredCandidate {
            parameters: high.clone(),
            score: 12.0,
        },
        CpuScoredCandidate {
            parameters: tied.clone(),
            score: 12.0,
        },
    ]);
    assert_eq!(ranked[0].parameters, high);
    assert_eq!(ranked[1].parameters, tied);
    assert_eq!(ranked[2].parameters, low);
    assert_eq!(
        cpu_training_elites(
            &[
                CpuScoredCandidate {
                    parameters: baseline.clone(),
                    score: 99.0,
                },
                CpuScoredCandidate {
                    parameters: low.clone(),
                    score: 3.0,
                },
                CpuScoredCandidate {
                    parameters: high.clone(),
                    score: 12.0,
                },
            ],
            &baseline,
            1,
        ),
        vec![high.clone()]
    );
    assert_eq!(
        cpu_training_finalist_candidates(
            &baseline,
            &[
                CpuScoredCandidate {
                    parameters: low.clone(),
                    score: 3.0,
                },
                CpuScoredCandidate {
                    parameters: baseline.clone(),
                    score: 99.0,
                },
                CpuScoredCandidate {
                    parameters: high.clone(),
                    score: 12.0,
                },
            ],
            2,
        ),
        vec![baseline.clone(), high.clone()]
    );
    let outcome = cpu_training_generation_outcome(
        &baseline,
        &[
            CpuScoredCandidate {
                parameters: baseline.clone(),
                score: 8.0,
            },
            CpuScoredCandidate {
                parameters: high.clone(),
                score: 12.0,
            },
        ],
        f64::NEG_INFINITY,
        11.0,
    );
    assert_eq!(outcome.baseline_score, 8.0);
    assert_eq!(
        outcome.winner.as_ref().map(|entry| &entry.parameters),
        Some(&high)
    );
    assert!(outcome.improved);
    let stale_baseline = cpu_training_generation_outcome(&baseline, &[], 7.0, 0.0);
    assert_eq!(stale_baseline.baseline_score, 7.0);
    assert!(stale_baseline.winner.is_none());
    assert!(!stale_baseline.improved);
    let scoring_plan = cpu_candidate_scoring_plan(
        &[low.clone(), high.clone(), low.clone(), tied.clone()],
        &[
            CpuFitnessEntry {
                key: cpu_parameters_key(&high),
                score: 42.0,
            },
            CpuFitnessEntry {
                key: cpu_parameters_key(&baseline),
                score: 99.0,
            },
        ],
    );
    assert_eq!(
        scoring_plan.unique_candidates,
        vec![low.clone(), high.clone(), tied.clone()]
    );
    assert_eq!(
        scoring_plan.cached_scores,
        vec![CpuScoredCandidate {
            parameters: high.clone(),
            score: 42.0,
        }]
    );
    assert_eq!(scoring_plan.uncached_candidates, vec![low, tied]);
    assert_eq!(scoring_plan.cache_hits, 1);
    assert_eq!(
        cpu_fitness_entry_for_candidate(&high, 13.5),
        CpuFitnessEntry {
            key: cpu_parameters_key(&high),
            score: 13.5,
        }
    );
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
fn cpu_reference_score_delta_json_accepts_browser_move_contract() {
    let response = cpu_reference_score_delta_json(
        r#"{
            "candidateScore": 120,
            "referenceScore": 100,
            "candidateMoves": [
                {
                    "from": { "timelineId": 1, "time": 2, "x": 3, "y": 4 },
                    "to": { "timelineId": 1, "time": 3, "x": 4, "y": 5 }
                }
            ],
            "referenceMoves": [
                {
                    "fromTimelineId": 1,
                    "fromTime": 2,
                    "fromX": 3,
                    "fromY": 4,
                    "toTimelineId": 1,
                    "toTime": 3,
                    "toX": 4,
                    "toY": 5
                }
            ],
            "drawWindow": 25
        }"#,
    )
    .expect("CPU reference score delta JSON");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("CPU reference score delta response JSON");
    assert_eq!(value["score"], 45);
    assert_eq!(value["nearDraw"], true);
}

#[test]
fn cpu_reference_score_helpers_accept_worker_search_results() {
    let response = cpu_reference_score_from_result_json(
        r#"{
            "result": {
                "score": 120,
                "moves": [{
                    "from": { "timelineId": 0, "time": 0, "x": 1, "y": 1 },
                    "to": { "timelineId": 0, "time": 1, "x": 1, "y": 2 }
                }]
            }
        }"#,
    )
    .expect("CPU reference score from result");
    let value: serde_json::Value =
        serde_json::from_str(&response).expect("CPU reference score JSON");
    assert_eq!(value["score"], 120);
    assert_eq!(value["moves"].as_array().map(Vec::len), Some(1));

    let fallback = cpu_reference_score_from_result_json(r#"{"result":null}"#)
        .expect("fallback CPU reference score");
    let fallback_value: serde_json::Value =
        serde_json::from_str(&fallback).expect("fallback CPU reference score JSON");
    assert_eq!(fallback_value["score"], 0);
    assert!(fallback_value.get("moves").is_none());

    let delta = cpu_reference_score_delta_from_result_json(
        r#"{
            "candidateResult": {
                "score": 120,
                "moves": [{
                    "from": { "timelineId": 0, "time": 0, "x": 1, "y": 1 },
                    "to": { "timelineId": 0, "time": 1, "x": 1, "y": 2 }
                }]
            },
            "referenceScore": 100,
            "referenceMoves": [{
                "from": { "timelineId": 0, "time": 0, "x": 1, "y": 1 },
                "to": { "timelineId": 0, "time": 1, "x": 1, "y": 2 }
            }],
            "drawWindow": 25
        }"#,
    )
    .expect("CPU reference score delta from result");
    let delta_value: serde_json::Value =
        serde_json::from_str(&delta).expect("CPU reference score delta JSON");
    assert_eq!(delta_value["score"], 45);
    assert_eq!(delta_value["nearDraw"], true);
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
    assert_eq!(
        cpu_training_adjudication_score_from_result_json(
            r#"{ "currentTurn": "black", "candidateColor": "white", "result": { "score": 125 } }"#
        )
        .expect("adjudication score from result"),
        -125
    );
    assert_eq!(
        cpu_training_adjudication_score_from_result_json(
            r#"{ "currentTurn": "white", "candidateColor": "white", "result": null }"#
        )
        .expect("neutral adjudication score from missing result"),
        0
    );
}

fn parameters(values: &[(&str, i32)]) -> CpuParameters {
    values
        .iter()
        .map(|(key, value)| ((*key).to_string(), *value))
        .collect()
}
