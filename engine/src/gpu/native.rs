use std::{collections::BTreeMap, sync::mpsc};

use burn::{backend::Wgpu, tensor::Tensor};
use wgpu::CompilationMessageType;

use super::{GpuKernel, WgslShader};

#[derive(Clone, Debug, PartialEq)]
pub struct NativeGpuBackendInfo {
    pub backend: &'static str,
    pub operation: &'static str,
    pub result: Vec<f32>,
}

impl std::fmt::Display for NativeGpuBackendInfo {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "native_gpu backend={} operation={} result={:?}",
            self.backend, self.operation, self.result
        )
    }
}

pub fn backend_info() -> Result<NativeGpuBackendInfo, String> {
    type B = Wgpu;
    let device = Default::default();
    let left = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
    let right = Tensor::<B, 1>::from_floats([0.5, 1.5, 2.5, 3.5], &device);
    let result = (left + right)
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| format!("native GPU tensor readback failed: {error:?}"))?;
    Ok(NativeGpuBackendInfo {
        backend: "burn-wgpu",
        operation: "tensor_add",
        result,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeShaderCompileReport {
    pub backend: &'static str,
    pub shaders: usize,
    pub kernels: usize,
}

impl std::fmt::Display for NativeShaderCompileReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "native_gpu_shader_compile backend={} shaders={} kernels={}",
            self.backend, self.shaders, self.kernels
        )
    }
}

pub fn compile_engine_shaders() -> Result<NativeShaderCompileReport, String> {
    futures_lite::future::block_on(async {
        let (device, _queue) = native_device().await?;

        let mut shader_count = 0;
        for shader in engine_shaders() {
            shader_count += 1;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader.name),
                source: wgpu::ShaderSource::Wgsl(shader.source.into()),
            });
            let compilation_info = module.get_compilation_info().await;
            let errors = compilation_info
                .messages
                .iter()
                .filter(|message| message.message_type == CompilationMessageType::Error)
                .map(|message| format_compile_message(shader.name, message))
                .collect::<Vec<_>>();
            if !errors.is_empty() {
                return Err(errors.join("; "));
            }
        }

        Ok(NativeShaderCompileReport {
            backend: "wgpu",
            shaders: shader_count,
            kernels: engine_kernels().count(),
        })
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeKernelCompileReport {
    pub backend: &'static str,
    pub shaders: usize,
    pub kernels: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeTurnStatusDispatchReport {
    pub backend: &'static str,
    pub result: [i32; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeSearchDispatchReport {
    pub backend: &'static str,
    pub frontier_root_summary: Vec<i32>,
    pub frontier_round: NativeFrontierRoundReport,
    pub turn_status: [i32; 4],
    pub frontier_reduced_scores: Vec<i32>,
    pub frontier_selected_indices: Vec<i32>,
    pub frontier_materialized_summary: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFrontierRoundReport {
    pub candidate_count: i32,
    pub selected_count: i32,
    pub root_color: i32,
    pub selected_indices: Vec<i32>,
    pub selected_moves: Vec<NativeFrontierMove>,
    pub first_candidate: Vec<i32>,
    pub summaries: Vec<i32>,
    pub state_stride: usize,
    pub next_states: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFrontierStateHeader {
    pub root: i32,
    pub score: i32,
    pub depth: i32,
    pub turn: i32,
    pub board_count: i32,
    pub plan_length: i32,
    pub complete: i32,
    pub terminal: i32,
    pub present: i32,
    pub pending: i32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFrontierMove {
    pub score: i32,
    pub root: i32,
    pub depth: i32,
    pub move_record: [i32; 8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeFrontierSearchReport {
    pub rounds: Vec<NativeFrontierRoundReport>,
}

struct NativeFrontierRoundInput {
    state_words: Vec<i32>,
    state_count: usize,
    max_boards: usize,
    state_stride: usize,
    root_color: i32,
    target_depth: i32,
    cycle_index: u32,
    frontier_width: usize,
    candidate_capacity: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeProjectFeaturesDispatchReport {
    pub backend: &'static str,
    pub result: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeProjectFeaturesBatchRequest {
    pub projection_size: usize,
    pub seed: u32,
    pub output_offset: usize,
    pub features: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NativeValuePredictionRequest {
    pub model: crate::gpu::training::CompactValueModel,
    pub features: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub struct NativeValueHeadTrainingRequest {
    pub model: crate::gpu::training::CompactValueModel,
    pub samples: Vec<crate::gpu::training::TrainingSample>,
    pub projected_features: Vec<f32>,
    pub config: crate::gpu::training::ValueHeadTrainingConfig,
    pub train_hidden_layers: bool,
}

#[derive(Clone, Debug)]
pub struct NativePolicyHeadTrainingRequest {
    pub model: crate::gpu::training::CompactValueModel,
    pub samples: Vec<crate::gpu::training::TrainingSample>,
    pub projected_features: Vec<f32>,
    pub config: crate::gpu::training::ValueHeadTrainingConfig,
}

impl std::fmt::Display for NativeTurnStatusDispatchReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "native_gpu_dispatch kernel=turn_status backend={} result={:?}",
            self.backend, self.result
        )
    }
}

impl std::fmt::Display for NativeSearchDispatchReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "native_gpu_dispatch backend={} frontier_root_summary={:?} frontier_round={{candidate_count:{} selected_count:{} selected_indices:{:?} selected_moves:{:?} materialized_states:{} state_headers:{:?}}} turn_status={:?} frontier_reduced_scores={:?} frontier_selected_indices={:?} frontier_materialized_summary={:?}",
            self.backend,
            self.frontier_root_summary,
            self.frontier_round.candidate_count,
            self.frontier_round.selected_count,
            self.frontier_round.selected_indices,
            self.frontier_round.selected_moves,
            self.frontier_round.materialized_state_count(),
            self.frontier_round.state_headers(),
            self.turn_status,
            self.frontier_reduced_scores,
            self.frontier_selected_indices,
            self.frontier_materialized_summary
        )
    }
}

impl NativeFrontierRoundReport {
    pub fn materialized_state_count(&self) -> usize {
        if self.state_stride == 0 {
            return 0;
        }
        self.next_states.len() / self.state_stride
    }

    pub fn best_selected_move(&self) -> Option<&NativeFrontierMove> {
        self.selected_moves
            .iter()
            .max_by_key(|candidate| candidate.score)
    }

    pub fn planned_root_moves(&self) -> Vec<NativeFrontierMove> {
        if self.state_stride == 0 {
            return Vec::new();
        }
        self.next_states
            .chunks_exact(self.state_stride)
            .filter_map(|state| {
                let plan_length = state[crate::gpu::search::FRONTIER_HEADER_PLAN_LENGTH];
                if plan_length <= 0 {
                    return None;
                }
                let move_start = crate::gpu::search::FRONTIER_PLAN_OFFSET;
                let move_end = move_start + crate::gpu::search::FRONTIER_MOVE_STRIDE;
                let move_record = state.get(move_start..move_end)?;
                Some(NativeFrontierMove {
                    score: state[crate::gpu::search::FRONTIER_HEADER_SCORE],
                    root: state[crate::gpu::search::FRONTIER_HEADER_ROOT],
                    depth: state[crate::gpu::search::FRONTIER_HEADER_DEPTH],
                    move_record: move_record.try_into().ok()?,
                })
            })
            .collect()
    }

    pub fn best_planned_root_move(&self) -> Option<NativeFrontierMove> {
        self.planned_root_moves()
            .into_iter()
            .max_by_key(|candidate| candidate.score)
    }

    pub fn state_headers(&self) -> Vec<NativeFrontierStateHeader> {
        if self.state_stride == 0 {
            return Vec::new();
        }
        self.next_states
            .chunks_exact(self.state_stride)
            .map(|state| NativeFrontierStateHeader {
                root: state[crate::gpu::search::FRONTIER_HEADER_ROOT],
                score: state[crate::gpu::search::FRONTIER_HEADER_SCORE],
                depth: state[crate::gpu::search::FRONTIER_HEADER_DEPTH],
                turn: state[crate::gpu::search::FRONTIER_HEADER_TURN],
                board_count: state[crate::gpu::search::FRONTIER_HEADER_BOARD_COUNT],
                plan_length: state[crate::gpu::search::FRONTIER_HEADER_PLAN_LENGTH],
                complete: state[crate::gpu::search::FRONTIER_HEADER_COMPLETE],
                terminal: state[crate::gpu::search::FRONTIER_HEADER_TERMINAL],
                present: state[crate::gpu::search::FRONTIER_HEADER_PRESENT_TIME],
                pending: state[crate::gpu::search::FRONTIER_HEADER_PENDING_BOARDS],
            })
            .collect()
    }
}

impl NativeFrontierSearchReport {
    pub fn minimax_root_count(&self) -> usize {
        self.backed_up_root_moves().len()
    }

    pub fn best_minimax_root_move(&self) -> Option<NativeFrontierMove> {
        let backed = self.backed_up_root_moves();
        if backed.is_empty() {
            return self
                .rounds
                .last()
                .and_then(NativeFrontierRoundReport::best_planned_root_move);
        }
        backed.into_iter().max_by_key(|candidate| candidate.score)
    }

    fn backed_up_root_moves(&self) -> Vec<NativeFrontierMove> {
        let Some(last) = self.rounds.last() else {
            return Vec::new();
        };
        if self.rounds.len() <= 1 {
            return last.planned_root_moves();
        }

        let mut worst_reply_by_root = BTreeMap::<i32, NativeFrontierMove>::new();
        for candidate in last.planned_root_moves() {
            worst_reply_by_root
                .entry(candidate.root)
                .and_modify(|current| {
                    if candidate.score < current.score {
                        *current = candidate.clone();
                    }
                })
                .or_insert(candidate);
        }
        worst_reply_by_root.into_values().collect()
    }
}

impl std::fmt::Display for NativeProjectFeaturesDispatchReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "native_gpu_dispatch kernel=project_features backend={} result={:?}",
            self.backend, self.result
        )
    }
}

pub fn dispatch_turn_status_smoke() -> Result<NativeTurnStatusDispatchReport, String> {
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("turn_status.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::TURN_STATUS_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("turn_status"),
            layout: None,
            module: &module,
            entry_point: Some("turn_status"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Records are timeline_id, owner, time, side_to_move. This fixture has
        // one active white board at present time 4 and one later black board.
        let board_records = i32_bytes(&[0, 0, 4, 1, 1, 0, 5, -1]);
        let params = uniform_bytes(&[2, 1, 0, 0]);
        let boards = storage_buffer(
            &device,
            &queue,
            "turn_status_boards",
            &board_records,
            wgpu::BufferUsages::STORAGE,
        );
        let result = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("turn_status_result"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = storage_buffer(
            &device,
            &queue,
            "turn_status_params",
            &params,
            wgpu::BufferUsages::UNIFORM,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("turn_status_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: boards.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("turn_status_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("turn_status_dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);

        let bytes = read_buffer(&device, &queue, &result, 16).await?;
        let result = bytes_to_i32_array_4(&bytes)?;
        let expected = [1, 1, 4, 1];
        if result != expected {
            return Err(format!(
                "turn_status smoke mismatch expected={expected:?} actual={result:?}"
            ));
        }
        Ok(NativeTurnStatusDispatchReport {
            backend: "wgpu",
            result,
        })
    })
}

pub fn dispatch_search_smoke() -> Result<NativeSearchDispatchReport, String> {
    let frontier_root_summary = encode_default_frontier_root_summary()?;
    let frontier_round = run_frontier_round(&crate::Game::new())?;
    let turn_status = dispatch_turn_status_smoke()?.result;
    let frontier_reduced_scores = dispatch_frontier_reduce_smoke()?;
    let frontier_selected_indices = dispatch_frontier_select_smoke()?;
    let frontier_materialized_summary = dispatch_frontier_materialize_smoke()?;
    Ok(NativeSearchDispatchReport {
        backend: "wgpu",
        frontier_root_summary,
        frontier_round,
        turn_status,
        frontier_reduced_scores,
        frontier_selected_indices,
        frontier_materialized_summary,
    })
}

fn encode_default_frontier_root_summary() -> Result<Vec<i32>, String> {
    let root = crate::gpu::search::encode_frontier_root(&crate::Game::new(), 1)?;
    let board = crate::gpu::search::FRONTIER_BOARD_OFFSET;
    Ok(vec![
        root.words[crate::gpu::search::FRONTIER_HEADER_PARENT],
        root.words[crate::gpu::search::FRONTIER_HEADER_ROOT],
        root.words[crate::gpu::search::FRONTIER_HEADER_TURN],
        root.words[crate::gpu::search::FRONTIER_HEADER_BOARD_COUNT],
        root.words[crate::gpu::search::FRONTIER_HEADER_PRESENT_TIME],
        root.words[crate::gpu::search::FRONTIER_HEADER_PENDING_BOARDS],
        root.hash_low,
        root.hash_high,
        root.words[board + crate::gpu::search::FRONTIER_BOARD_TIMELINE_ID],
        root.words[board + crate::gpu::search::FRONTIER_BOARD_TIME],
        root.words[board + crate::gpu::search::FRONTIER_BOARD_PENDING],
    ])
}

pub(crate) fn run_frontier_round(game: &crate::Game) -> Result<NativeFrontierRoundReport, String> {
    let max_boards = game
        .timelines
        .iter()
        .map(|timeline| timeline.boards.len())
        .sum::<usize>()
        .saturating_add(1)
        .max(2);
    let root = crate::gpu::search::encode_frontier_root(game, max_boards)?;
    let state_stride = crate::gpu::search::frontier_state_stride(max_boards);
    run_frontier_round_input(NativeFrontierRoundInput {
        root_color: root.words[crate::gpu::search::FRONTIER_HEADER_TURN],
        state_words: root.words,
        state_count: 1,
        max_boards,
        state_stride,
        target_depth: 1,
        cycle_index: 0,
        frontier_width: 4,
        candidate_capacity: 64,
    })
}

pub(crate) fn run_frontier_search(
    game: &crate::Game,
    depth: i32,
) -> Result<NativeFrontierSearchReport, String> {
    let first = run_frontier_round(game)?;
    if depth <= 1 {
        return Ok(NativeFrontierSearchReport {
            rounds: vec![first],
        });
    }
    let state_count = first.materialized_state_count();
    if state_count == 0 {
        return Ok(NativeFrontierSearchReport {
            rounds: vec![first],
        });
    }
    let max_boards = max_boards_from_state_stride(first.state_stride)?;
    let second = match run_frontier_round_input(NativeFrontierRoundInput {
        state_words: first.next_states.clone(),
        state_count,
        max_boards,
        state_stride: first.state_stride,
        root_color: first.root_color,
        target_depth: 2,
        cycle_index: 1,
        frontier_width: 4,
        candidate_capacity: 512,
    }) {
        Ok(report) => report,
        Err(message) if message.contains("produced no candidates") => {
            return Ok(NativeFrontierSearchReport {
                rounds: vec![first],
            });
        }
        Err(message) => return Err(message),
    };
    Ok(NativeFrontierSearchReport {
        rounds: vec![first, second],
    })
}

fn max_boards_from_state_stride(state_stride: usize) -> Result<usize, String> {
    if state_stride < crate::gpu::search::FRONTIER_BOARD_OFFSET {
        return Err(format!("frontier state stride {state_stride} is too small"));
    }
    let board_words = state_stride - crate::gpu::search::FRONTIER_BOARD_OFFSET;
    if board_words % crate::gpu::search::FRONTIER_BOARD_STRIDE != 0 {
        return Err(format!(
            "frontier state stride {state_stride} is not aligned to board stride"
        ));
    }
    Ok(board_words / crate::gpu::search::FRONTIER_BOARD_STRIDE)
}

fn run_frontier_round_input(
    input: NativeFrontierRoundInput,
) -> Result<NativeFrontierRoundReport, String> {
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        let expand_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier_expand.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::FRONTIER_EXPAND_SHADER.into()),
        });
        let select_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier_select.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::FRONTIER_SELECT_SHADER.into()),
        });
        let state_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier_state.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::FRONTIER_STATE_SHADER.into()),
        });
        let expand = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_expand_root"),
            layout: None,
            module: &expand_module,
            entry_point: Some("expand_frontier"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("EXPAND_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let hash = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_hash"),
            layout: None,
            module: &select_module,
            entry_point: Some("hash_candidates"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SELECT_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let order = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_order"),
            layout: None,
            module: &select_module,
            entry_point: Some("bucket_order"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SELECT_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let sort = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_sort"),
            layout: None,
            module: &select_module,
            entry_point: Some("bitonic_sort"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SELECT_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let unique = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_unique"),
            layout: None,
            module: &select_module,
            entry_point: Some("mark_unique"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SELECT_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let quota = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_parent_quota"),
            layout: None,
            module: &select_module,
            entry_point: Some("mark_parent_quota"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SELECT_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let compact = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_compact"),
            layout: None,
            module: &select_module,
            entry_point: Some("compact_selected"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SELECT_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });
        let fill = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_select"),
            layout: None,
            module: &select_module,
            entry_point: Some("fill_selection_underflow"),
            compilation_options: Default::default(),
            cache: None,
        });
        let materialize = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_round_materialize"),
            layout: None,
            module: &state_module,
            entry_point: Some("materialize_selected"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("MATERIALIZE_WORKGROUP_SIZE", 64.0)],
                ..Default::default()
            },
            cache: None,
        });

        let max_boards = input.max_boards;
        let state_stride = input.state_stride;
        let candidate_capacity = input.candidate_capacity;
        let frontier_width = input.frontier_width;
        let selection_capacity = 64usize;
        let candidate_stride = crate::gpu::search::FRONTIER_CANDIDATE_STRIDE;
        let delta_stride = crate::gpu::search::FRONTIER_DELTA_STRIDE;
        let states = storage_buffer(
            &device,
            &queue,
            "frontier_expand_root_states",
            &i32_bytes(&input.state_words),
            wgpu::BufferUsages::STORAGE,
        );
        let candidates = storage_buffer(
            &device,
            &queue,
            "frontier_expand_root_candidates",
            &i32_bytes(&vec![0; candidate_capacity * candidate_stride]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let deltas = storage_buffer(
            &device,
            &queue,
            "frontier_expand_root_deltas",
            &i32_bytes(&vec![0; candidate_capacity * delta_stride]),
            wgpu::BufferUsages::STORAGE,
        );
        let order_buffer = storage_buffer(
            &device,
            &queue,
            "frontier_round_order",
            &i32_bytes(&vec![-1; selection_capacity]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let selected = storage_buffer(
            &device,
            &queue,
            "frontier_round_selected",
            &i32_bytes(&vec![-1; frontier_width]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let eligibility = storage_buffer(
            &device,
            &queue,
            "frontier_round_eligibility",
            &u32_bytes(&vec![0; selection_capacity]),
            wgpu::BufferUsages::STORAGE,
        );
        let next_states = storage_buffer(
            &device,
            &queue,
            "frontier_round_next_states",
            &i32_bytes(&vec![0; frontier_width * state_stride]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let summaries = storage_buffer(
            &device,
            &queue,
            "frontier_round_summaries",
            &i32_bytes(&vec![
                0;
                frontier_width
                    * crate::gpu::search::FRONTIER_SUMMARY_STRIDE
            ]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let counters = storage_buffer(
            &device,
            &queue,
            "frontier_round_counters",
            &uniform_bytes(&[0, 0, 0, 0, 0, 0]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let expand_params = storage_buffer(
            &device,
            &queue,
            "frontier_expand_root_params",
            &uniform_bytes(&[
                input.state_count as u32,
                max_boards as u32,
                state_stride as u32,
                crate::gpu::search::FRONTIER_BOARD_OFFSET as u32,
                candidate_capacity as u32,
                candidate_stride as u32,
                delta_stride as u32,
                input.root_color as u32,
                input.target_depth as u32,
                input.cycle_index,
                0,
                (input.state_count * max_boards * 64) as u32,
                0,
                0,
                0,
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        let select_params = storage_buffer(
            &device,
            &queue,
            "frontier_round_select_params",
            &uniform_bytes(&[
                candidate_capacity as u32,
                frontier_width as u32,
                8,
                selection_capacity as u32,
                state_stride as u32,
                delta_stride as u32,
                input.cycle_index,
                100,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        let sort_stage = storage_buffer(
            &device,
            &queue,
            "frontier_round_sort_stage",
            &uniform_bytes(&[0, 0, 0, 0]),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let materialize_params = storage_buffer(
            &device,
            &queue,
            "frontier_round_materialize_params",
            &uniform_bytes(&[
                frontier_width as u32,
                max_boards as u32,
                state_stride as u32,
                crate::gpu::search::FRONTIER_BOARD_OFFSET as u32,
                crate::gpu::search::FRONTIER_PLAN_OFFSET as u32,
                delta_stride as u32,
                candidate_stride as u32,
                crate::gpu::search::FRONTIER_MAX_PLAN_MOVES as u32,
                crate::gpu::search::FRONTIER_HEADER_STRIDE as u32,
                0,
                0,
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        let expand_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_expand_root_bind_group"),
            layout: &expand.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: expand_params.as_entire_binding(),
                },
            ],
        });
        let hash_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_hash_bind_group"),
            layout: &hash.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
            ],
        });
        let order_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_order_bind_group"),
            layout: &order.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
            ],
        });
        let sort_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_sort_bind_group"),
            layout: &sort.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: sort_stage.as_entire_binding(),
                },
            ],
        });
        let unique_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_unique_bind_group"),
            layout: &unique.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eligibility.as_entire_binding(),
                },
            ],
        });
        let quota_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_quota_bind_group"),
            layout: &quota.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eligibility.as_entire_binding(),
                },
            ],
        });
        let compact_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_compact_bind_group"),
            layout: &compact.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eligibility.as_entire_binding(),
                },
            ],
        });
        let fill_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_fill_bind_group"),
            layout: &fill.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: select_params.as_entire_binding(),
                },
            ],
        });
        let materialize_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_round_materialize_bind_group"),
            layout: &materialize.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: selected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: next_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: summaries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: materialize_params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frontier_round_expand_select_materialize"),
        });
        let expand_workgroups = (input.state_count * max_boards * 64).div_ceil(64).max(1);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontier_round_expand_and_select"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&expand);
            pass.set_bind_group(0, &expand_group, &[]);
            pass.dispatch_workgroups(expand_workgroups as u32, 1, 1);
            pass.set_pipeline(&hash);
            pass.set_bind_group(0, &hash_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&order);
            pass.set_bind_group(0, &order_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);

        let mut k = 2u32;
        while k <= selection_capacity as u32 {
            let mut j = k / 2;
            while j > 0 {
                queue.write_buffer(&sort_stage, 0, &uniform_bytes(&[k, j, 0, 0]));
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frontier_round_sort"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("frontier_round_sort"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&sort);
                    pass.set_bind_group(0, &sort_group, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
                queue.submit([encoder.finish()]);
                j /= 2;
            }
            k *= 2;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frontier_round_finalize"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontier_round_finalize"),
                timestamp_writes: None,
            });
            for (pipeline, group) in [
                (&unique, &unique_group),
                (&quota, &quota_group),
                (&compact, &compact_group),
                (&fill, &fill_group),
                (&materialize, &materialize_group),
            ] {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        queue.submit([encoder.finish()]);

        let counters = bytes_to_i32_vec(&read_buffer(&device, &queue, &counters, 24).await?)?;
        if counters.first().copied().unwrap_or(0) <= 0 {
            return Err(format!(
                "frontier_round root produced no candidates counters={counters:?}"
            ));
        }
        if counters.get(1).copied().unwrap_or(0) <= 0 {
            return Err(format!(
                "frontier_round root selected no candidates counters={counters:?}"
            ));
        }
        let candidate_words = bytes_to_i32_vec(
            &read_buffer(
                &device,
                &queue,
                &candidates,
                (candidate_capacity * candidate_stride * std::mem::size_of::<i32>())
                    as wgpu::BufferAddress,
            )
            .await?,
        )?;
        let first_candidate = candidate_words
            .get(0..candidate_stride)
            .ok_or_else(|| "frontier_round candidate readback was empty".to_string())?
            .to_vec();
        let selected_indices = bytes_to_i32_vec(
            &read_buffer(
                &device,
                &queue,
                &selected,
                (frontier_width * std::mem::size_of::<i32>()) as wgpu::BufferAddress,
            )
            .await?,
        )?;
        let selected_moves = selected_indices
            .iter()
            .copied()
            .filter(|index| *index >= 0 && *index < counters[0])
            .filter_map(|index| {
                let base = index as usize * candidate_stride;
                let move_start = base + 8;
                let move_end = move_start + 8;
                let move_record = candidate_words.get(move_start..move_end)?;
                Some(NativeFrontierMove {
                    score: candidate_words[base + 2],
                    root: candidate_words[base + 1],
                    depth: candidate_words[base + 4],
                    move_record: move_record.try_into().ok()?,
                })
            })
            .collect::<Vec<_>>();
        if selected_moves.is_empty() {
            return Err(format!(
                "frontier_round selected no readable moves selected={selected_indices:?} counters={counters:?}"
            ));
        }
        let summaries = bytes_to_i32_vec(
            &read_buffer(
                &device,
                &queue,
                &summaries,
                (frontier_width
                    * crate::gpu::search::FRONTIER_SUMMARY_STRIDE
                    * std::mem::size_of::<i32>()) as wgpu::BufferAddress,
            )
            .await?,
        )?;
        let next_states = bytes_to_i32_vec(
            &read_buffer(
                &device,
                &queue,
                &next_states,
                (frontier_width * state_stride * std::mem::size_of::<i32>()) as wgpu::BufferAddress,
            )
            .await?,
        )?;

        Ok(NativeFrontierRoundReport {
            candidate_count: counters[0],
            selected_count: counters[1],
            root_color: input.root_color,
            selected_indices,
            selected_moves,
            first_candidate,
            summaries,
            state_stride,
            next_states,
        })
    })
}

fn dispatch_frontier_select_smoke() -> Result<Vec<i32>, String> {
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier_select.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::FRONTIER_SELECT_SHADER.into()),
        });
        let hash = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_hash"),
            layout: None,
            module: &module,
            entry_point: Some("hash_candidates"),
            compilation_options: Default::default(),
            cache: None,
        });
        let order = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_order"),
            layout: None,
            module: &module,
            entry_point: Some("bucket_order"),
            compilation_options: Default::default(),
            cache: None,
        });
        let sort = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_sort"),
            layout: None,
            module: &module,
            entry_point: Some("bitonic_sort"),
            compilation_options: Default::default(),
            cache: None,
        });
        let unique = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_unique"),
            layout: None,
            module: &module,
            entry_point: Some("mark_unique"),
            compilation_options: Default::default(),
            cache: None,
        });
        let quota = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_parent_quota"),
            layout: None,
            module: &module,
            entry_point: Some("mark_parent_quota"),
            compilation_options: Default::default(),
            cache: None,
        });
        let compact = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_compact"),
            layout: None,
            module: &module,
            entry_point: Some("compact_selected"),
            compilation_options: Default::default(),
            cache: None,
        });
        let fill = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_select"),
            layout: None,
            module: &module,
            entry_point: Some("fill_selection_underflow"),
            compilation_options: Default::default(),
            cache: None,
        });

        let candidate_count = 4usize;
        let candidate_stride = 24usize;
        let state_stride = 32usize;
        let delta_stride = 156usize;
        let max_scan = 4usize;
        let selected_limit = 2usize;
        let mut candidates = vec![0i32; candidate_count * candidate_stride];
        for (index, score) in [10, 40, 30, 20].iter().copied().enumerate() {
            let base = index * candidate_stride;
            candidates[base] = 0;
            candidates[base + 2] = score;
            candidates[base + 4] = 0;
            candidates[base + 8] = (index as i32) + 1;
            candidates[base + 16] = 0;
            candidates[base + 19] = 0;
            candidates[base + 21] = 0;
            candidates[base + 22] = 0;
            candidates[base + 23] = 6;
        }
        let mut parent_states = vec![0i32; state_stride];
        parent_states[3] = 0;
        parent_states[9] = 0x1234;
        parent_states[10] = 0x5678;
        let deltas = vec![0i32; candidate_count * delta_stride];
        let order_values = vec![-1i32; max_scan];
        let selected_values = vec![-1i32; selected_limit];
        let counters_values = vec![candidate_count as u32, 0, 0, 0, 0, 0];
        let eligibility_values = vec![0u32; max_scan];
        let params = uniform_bytes(&[
            candidate_count as u32,
            selected_limit as u32,
            selected_limit as u32,
            max_scan as u32,
            state_stride as u32,
            delta_stride as u32,
            0,
            100,
        ]);
        let inert_stage = uniform_bytes(&[0, 0, 0, 0]);

        let candidates = storage_buffer(
            &device,
            &queue,
            "frontier_select_candidates",
            &i32_bytes(&candidates),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let parent_states = storage_buffer(
            &device,
            &queue,
            "frontier_select_parent_states",
            &i32_bytes(&parent_states),
            wgpu::BufferUsages::STORAGE,
        );
        let deltas = storage_buffer(
            &device,
            &queue,
            "frontier_select_deltas",
            &i32_bytes(&deltas),
            wgpu::BufferUsages::STORAGE,
        );
        let order_buffer = storage_buffer(
            &device,
            &queue,
            "frontier_select_order",
            &i32_bytes(&order_values),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let selected = storage_buffer(
            &device,
            &queue,
            "frontier_select_selected",
            &i32_bytes(&selected_values),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let counters = storage_buffer(
            &device,
            &queue,
            "frontier_select_counters",
            &u32_bytes(&counters_values),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params = storage_buffer(
            &device,
            &queue,
            "frontier_select_params",
            &params,
            wgpu::BufferUsages::UNIFORM,
        );
        let inert_stage = storage_buffer(
            &device,
            &queue,
            "frontier_select_stage",
            &inert_stage,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let eligibility = storage_buffer(
            &device,
            &queue,
            "frontier_select_eligibility",
            &u32_bytes(&eligibility_values),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let hash_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_hash_bind_group"),
            layout: &hash.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parent_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let order_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_order_bind_group"),
            layout: &order.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parent_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let sort_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_sort_bind_group"),
            layout: &sort.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parent_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: inert_stage.as_entire_binding(),
                },
            ],
        });
        let unique_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_unique_bind_group"),
            layout: &unique.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eligibility.as_entire_binding(),
                },
            ],
        });
        let quota_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_quota_bind_group"),
            layout: &quota.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eligibility.as_entire_binding(),
                },
            ],
        });
        let compact_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_compact_bind_group"),
            layout: &compact.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: order_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: eligibility.as_entire_binding(),
                },
            ],
        });
        let fill_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_fill_bind_group"),
            layout: &fill.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: selected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frontier_select_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontier_select_dispatch"),
                timestamp_writes: None,
            });
            for (pipeline, group) in [(&hash, &hash_group), (&order, &order_group)] {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        queue.submit([encoder.finish()]);
        for (k, j) in [(2u32, 1u32), (4, 2), (4, 1)] {
            let stage = uniform_bytes(&[k, j, 0, 0]);
            queue.write_buffer(&inert_stage, 0, &stage);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frontier_sort_dispatch"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("frontier_sort_dispatch"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&sort);
                pass.set_bind_group(0, &sort_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit([encoder.finish()]);
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frontier_select_finalize_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontier_select_finalize_dispatch"),
                timestamp_writes: None,
            });
            for (pipeline, group) in [
                (&unique, &unique_group),
                (&quota, &quota_group),
                (&compact, &compact_group),
                (&fill, &fill_group),
            ] {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        queue.submit([encoder.finish()]);
        let byte_len = selected_limit
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| "frontier select readback size overflowed".to_string())?;
        let bytes =
            read_buffer(&device, &queue, &selected, byte_len as wgpu::BufferAddress).await?;
        let selected = bytes_to_i32_vec(&bytes)?;
        let expected = vec![1, 2];
        if selected != expected {
            return Err(format!(
                "frontier_select smoke mismatch expected={expected:?} actual={selected:?}"
            ));
        }
        Ok(selected)
    })
}

fn dispatch_frontier_materialize_smoke() -> Result<Vec<i32>, String> {
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier_state.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::FRONTIER_STATE_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_materialize"),
            layout: None,
            module: &module,
            entry_point: Some("materialize_selected"),
            compilation_options: Default::default(),
            cache: None,
        });

        let selected_count = 1usize;
        let max_boards = 1usize;
        let candidate_stride = 24usize;
        let delta_stride = 156usize;
        let header_stride = 16usize;
        let plan_offset = 32usize;
        let board_offset = 48usize;
        let state_stride = board_offset + max_boards * 78;
        let summary_stride = 12usize;
        let mut parent_states = vec![0i32; state_stride];
        parent_states[0] = -1;
        parent_states[1] = 0;
        parent_states[2] = 5;
        parent_states[3] = 1;
        parent_states[4] = 0;
        parent_states[5] = 0;
        parent_states[6] = 0;
        parent_states[7] = 0;
        parent_states[8] = 0;
        parent_states[9] = 0x100;
        parent_states[10] = 0x200;
        parent_states[11] = 2;
        parent_states[12] = -2;
        parent_states[13] = 4;
        parent_states[14] = 1;
        parent_states[15] = 0;

        let mut candidates = vec![0i32; candidate_stride];
        candidates[0] = 0;
        candidates[1] = 3;
        candidates[2] = 50;
        candidates[3] = 123;
        candidates[5] = 1;
        candidates[6] = 0x333;
        candidates[7] = 0x444;
        candidates[16] = 0;
        candidates[19] = 1;
        candidates[20] = 77;
        candidates[21] = 7;
        candidates[22] = 2;
        let deltas = vec![0i32; delta_stride];
        let selected = vec![0i32];
        let next_states = vec![0i32; state_stride * selected_count];
        let summaries = vec![0i32; summary_stride * selected_count];
        let counters = vec![0u32, selected_count as u32, 0, 0, 0, 0];
        let params = uniform_bytes(&[
            selected_count as u32,
            max_boards as u32,
            state_stride as u32,
            board_offset as u32,
            plan_offset as u32,
            delta_stride as u32,
            candidate_stride as u32,
            2,
            header_stride as u32,
            0,
            0,
            0,
        ]);

        let parent_states = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_parent_states",
            &i32_bytes(&parent_states),
            wgpu::BufferUsages::STORAGE,
        );
        let candidates = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_candidates",
            &i32_bytes(&candidates),
            wgpu::BufferUsages::STORAGE,
        );
        let deltas = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_deltas",
            &i32_bytes(&deltas),
            wgpu::BufferUsages::STORAGE,
        );
        let selected = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_selected",
            &i32_bytes(&selected),
            wgpu::BufferUsages::STORAGE,
        );
        let next_states = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_next_states",
            &i32_bytes(&next_states),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let summaries = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_summaries",
            &i32_bytes(&summaries),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let counters = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_counters",
            &u32_bytes(&counters),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params = storage_buffer(
            &device,
            &queue,
            "frontier_materialize_params",
            &params,
            wgpu::BufferUsages::UNIFORM,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_materialize_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: parent_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: candidates.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: deltas.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: selected.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: next_states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: summaries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frontier_materialize_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontier_materialize_dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let summary_byte_len = summary_stride
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| "frontier materialize summary readback size overflowed".to_string())?;
        let summary_bytes = read_buffer(
            &device,
            &queue,
            &summaries,
            summary_byte_len as wgpu::BufferAddress,
        )
        .await?;
        let summary = bytes_to_i32_vec(&summary_bytes)?;
        let expected = vec![3, 43, 1, 0, 0, 0, 0, 0x333, 0x444, 4, 1, 0];
        if summary != expected {
            return Err(format!(
                "frontier_materialize summary mismatch expected={expected:?} actual={summary:?}"
            ));
        }
        let state_byte_len = state_stride
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| "frontier materialize state readback size overflowed".to_string())?;
        let state_bytes = read_buffer(
            &device,
            &queue,
            &next_states,
            state_byte_len as wgpu::BufferAddress,
        )
        .await?;
        let state = bytes_to_i32_vec(&state_bytes)?;
        if state[0] != 0 || state[1] != 3 || state[2] != 43 || state[15] != 123 {
            return Err(format!(
                "frontier_materialize state mismatch parent={} root={} score={} last_neural={}",
                state[0], state[1], state[2], state[15]
            ));
        }
        Ok(summary)
    })
}

fn dispatch_frontier_reduce_smoke() -> Result<Vec<i32>, String> {
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frontier_state.wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::gpu::search::FRONTIER_STATE_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("frontier_reduce"),
            layout: None,
            module: &module,
            entry_point: Some("minimax_reduce_stage"),
            compilation_options: Default::default(),
            cache: None,
        });

        let state_count = 3usize;
        let state_stride = 32usize;
        let ancestry_offset = 16usize;
        let summary_stride = 12usize;
        let mut states = vec![0i32; state_count * state_stride];
        for (index, score) in [30, 10, 20].iter().copied().enumerate() {
            let base = index * state_stride;
            states[base + 2] = score;
            states[base + 3] = 1;
            states[base + ancestry_offset] = 7;
            states[base + ancestry_offset + 1] = 1;
        }
        let summaries = vec![0i32; state_count * summary_stride];
        let params = uniform_bytes(&[
            state_count as u32,
            state_stride as u32,
            ancestry_offset as u32,
            1,
            1,
            0,
            0,
            0,
        ]);
        let states = storage_buffer(
            &device,
            &queue,
            "frontier_reduce_states",
            &i32_bytes(&states),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let summaries = storage_buffer(
            &device,
            &queue,
            "frontier_reduce_summaries",
            &i32_bytes(&summaries),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params = storage_buffer(
            &device,
            &queue,
            "frontier_reduce_params",
            &params,
            wgpu::BufferUsages::UNIFORM,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frontier_reduce_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: states.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: summaries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("frontier_reduce_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("frontier_reduce_dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let byte_len = state_count
            .checked_mul(summary_stride)
            .and_then(|value| value.checked_mul(std::mem::size_of::<i32>()))
            .ok_or_else(|| "frontier reduce summary readback size overflowed".to_string())?;
        let bytes =
            read_buffer(&device, &queue, &summaries, byte_len as wgpu::BufferAddress).await?;
        let summaries = bytes_to_i32_vec(&bytes)?;
        let reduced = (0..state_count)
            .map(|index| summaries[index * summary_stride + 1])
            .collect::<Vec<_>>();
        let expected = vec![10, 10, 10];
        if reduced != expected {
            return Err(format!(
                "frontier_reduce smoke mismatch expected={expected:?} actual={reduced:?}"
            ));
        }
        Ok(reduced)
    })
}

pub fn dispatch_project_features_smoke() -> Result<NativeProjectFeaturesDispatchReport, String> {
    let features = vec![vec![0.0, 2.0, 0.0, -1.0, 0.5, 0.0]];
    let projection_size = 8usize;
    let seed = 0x9e37_79b9;
    let expected = crate::gpu::training::project_features(&features[0], projection_size, seed);
    let result = project_features_batch(NativeProjectFeaturesBatchRequest {
        projection_size,
        seed,
        output_offset: 0,
        features,
    })?;
    for (index, (actual, expected)) in result.iter().zip(expected.iter()).enumerate() {
        if (actual - expected).abs() > 0.000_001 {
            return Err(format!(
                "project_features smoke mismatch index={index} expected={expected} actual={actual}"
            ));
        }
    }
    Ok(NativeProjectFeaturesDispatchReport {
        backend: "wgpu",
        result,
    })
}

pub fn project_features_batch(
    request: NativeProjectFeaturesBatchRequest,
) -> Result<Vec<f32>, String> {
    validate_project_features_request(&request)?;
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        project_features_batch_on_device(&device, &queue, &request).await
    })
}

pub fn predict_values(request: NativeValuePredictionRequest) -> Result<Vec<f32>, String> {
    validate_value_prediction_request(&request)?;
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        let sample_count = request.features.len();
        let projected = project_features_batch_on_device(
            &device,
            &queue,
            &NativeProjectFeaturesBatchRequest {
                projection_size: request.model.projection_size as usize,
                seed: request.model.projection_seed,
                output_offset: 0,
                features: request.features.clone(),
            },
        )
        .await?;
        let hidden_features = hidden_features_batch_on_device(
            &device,
            &queue,
            &request.model,
            sample_count,
            projected,
        )
        .await?;
        predict_values_on_device(
            &device,
            &queue,
            &request.model,
            sample_count,
            &hidden_features,
        )
        .await
    })
}

pub fn train_value_head(
    request: NativeValueHeadTrainingRequest,
) -> Result<
    (
        crate::gpu::training::CompactValueModel,
        crate::gpu::training::ValueHeadTrainingReport,
    ),
    String,
> {
    validate_value_head_training_request(&request)?;
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        train_value_head_on_device(&device, &queue, request).await
    })
}

pub fn train_policy_head(
    request: NativePolicyHeadTrainingRequest,
) -> Result<
    (
        crate::gpu::training::CompactValueModel,
        crate::gpu::training::PolicyHeadTrainingReport,
    ),
    String,
> {
    validate_policy_head_training_request(&request)?;
    futures_lite::future::block_on(async {
        let (device, queue) = native_device().await?;
        train_policy_head_on_device(&device, &queue, request).await
    })
}

impl std::fmt::Display for NativeKernelCompileReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "native_gpu_kernel_compile backend={} shaders={} kernels={}",
            self.backend, self.shaders, self.kernels
        )
    }
}

pub fn compile_engine_kernels() -> Result<NativeKernelCompileReport, String> {
    futures_lite::future::block_on(async {
        let (device, _queue) = native_device().await?;
        let mut modules = Vec::new();
        for shader in engine_shaders() {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(shader.name),
                source: wgpu::ShaderSource::Wgsl(shader.source.into()),
            });
            modules.push((shader.name, module));
        }

        let mut kernel_count = 0;
        for kernel in engine_kernels() {
            kernel_count += 1;
            let module = modules
                .iter()
                .find(|(name, _module)| *name == kernel.shader)
                .map(|(_name, module)| module)
                .ok_or_else(|| {
                    format!(
                        "kernel {} references missing shader {}",
                        kernel.label, kernel.shader
                    )
                })?;
            let constants = kernel
                .constants
                .iter()
                .map(|(name, value)| (*name, *value as f64))
                .collect::<Vec<_>>();
            let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(kernel.label),
                layout: None,
                module,
                entry_point: Some(kernel.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            });
            if let Some(error) = error_scope.pop().await {
                return Err(format!(
                    "kernel {} shader {} entry {}: {error}",
                    kernel.label, kernel.shader, kernel.entry_point
                ));
            }
        }

        Ok(NativeKernelCompileReport {
            backend: "wgpu",
            shaders: modules.len(),
            kernels: kernel_count,
        })
    })
}

async fn native_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|error| format!("native GPU adapter request failed: {error:?}"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .map_err(|error| format!("native GPU device request failed: {error:?}"))
}

async fn project_features_batch_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: &NativeProjectFeaturesBatchRequest,
) -> Result<Vec<f32>, String> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("project_features.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::PROJECT_FEATURES_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("project_features"),
        layout: None,
        module: &module,
        entry_point: Some("project_features"),
        compilation_options: Default::default(),
        cache: None,
    });

    let feature_rows = request
        .features
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let sample_count = request.features.len();
    let input_size = request.features.first().map_or(0, Vec::len);
    let packed = crate::gpu::training::pack_sparse_feature_rows(&feature_rows, input_size)?;
    let output_rows = request.output_offset + sample_count;
    let params = uniform_bytes(&[
        sample_count as u32,
        input_size as u32,
        request.projection_size as u32,
        request.seed,
        request.output_offset as u32,
        0,
        0,
        0,
    ]);
    let offsets = storage_buffer(
        device,
        queue,
        "project_features_offsets",
        &u32_bytes(&packed.offsets),
        wgpu::BufferUsages::STORAGE,
    );
    let indices = storage_buffer(
        device,
        queue,
        "project_features_indices",
        &u32_bytes(&packed.indices),
        wgpu::BufferUsages::STORAGE,
    );
    let values = storage_buffer(
        device,
        queue,
        "project_features_values",
        &f32_bytes(&packed.values),
        wgpu::BufferUsages::STORAGE,
    );
    let projected_byte_len = output_rows
        .checked_mul(request.projection_size)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "projected feature buffer size overflowed".to_string())?;
    let projected = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("project_features_projected"),
        size: projected_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = storage_buffer(
        device,
        queue,
        "project_features_params",
        &params,
        wgpu::BufferUsages::UNIFORM,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("project_features_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: offsets.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: indices.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: values.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: projected.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("project_features_dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("project_features_dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            (sample_count as u32).div_ceil(16),
            (request.projection_size as u32).div_ceil(16),
            1,
        );
    }
    queue.submit([encoder.finish()]);

    let bytes = read_buffer(
        device,
        queue,
        &projected,
        projected_byte_len as wgpu::BufferAddress,
    )
    .await?;
    let all_rows = bytes_to_f32_vec(&bytes)?;
    let start = request.output_offset * request.projection_size;
    let end = start + sample_count * request.projection_size;
    Ok(all_rows[start..end].to_vec())
}

async fn predict_values_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &crate::gpu::training::CompactValueModel,
    sample_count: usize,
    hidden_features: &[f32],
) -> Result<Vec<f32>, String> {
    let input_size = model
        .hidden_layers
        .last()
        .map(|value| *value as usize)
        .unwrap_or(model.projection_size as usize);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forward_output.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::FORWARD_OUTPUT_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("forward_output"),
        layout: None,
        module: &module,
        entry_point: Some("forward_output"),
        compilation_options: Default::default(),
        cache: None,
    });
    let inputs = storage_buffer(
        device,
        queue,
        "forward_output_inputs",
        &f32_bytes(hidden_features),
        wgpu::BufferUsages::STORAGE,
    );
    let weights = storage_buffer(
        device,
        queue,
        "forward_output_weights",
        &f32_bytes(&model.output_weights),
        wgpu::BufferUsages::STORAGE,
    );
    let prediction_byte_len = sample_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "prediction buffer size overflowed".to_string())?;
    let predictions = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("forward_output_predictions"),
        size: prediction_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = storage_buffer(
        device,
        queue,
        "forward_output_params",
        &uniform_bytes(&[sample_count as u32, input_size as u32, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("forward_output_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: predictions.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("forward_output_dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("forward_output_dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((sample_count as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let bytes = read_buffer(
        device,
        queue,
        &predictions,
        prediction_byte_len as wgpu::BufferAddress,
    )
    .await?;
    let mut values = bytes_to_f32_vec(&bytes)?;
    for value in &mut values {
        *value = crate::gpu::training::bounded_value(*value * model.scale + model.bias);
    }
    Ok(values)
}

async fn hidden_features_batch_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &crate::gpu::training::CompactValueModel,
    sample_count: usize,
    projected: Vec<f32>,
) -> Result<Vec<f32>, String> {
    let projection_size = model.projection_size as usize;
    if model.hidden_layers.is_empty() {
        return Ok(projected);
    }
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forward_layer.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::FORWARD_LAYER_SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("forward_layer"),
        layout: None,
        module: &module,
        entry_point: Some("forward_layer"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut input_size = projection_size;
    let mut input_values = projected;
    let mut weight_offset = 0usize;
    for (layer_index, &output_size) in model.hidden_layers.iter().enumerate() {
        let output_size = output_size as usize;
        let row_size = input_size + 1;
        let layer_weight_count = output_size
            .checked_mul(row_size)
            .ok_or_else(|| format!("hidden layer {layer_index} weight count overflowed"))?;
        let weights_end = weight_offset
            .checked_add(layer_weight_count)
            .ok_or_else(|| format!("hidden layer {layer_index} weight range overflowed"))?;
        let expected_input_value_count = sample_count
            .checked_mul(input_size)
            .ok_or_else(|| format!("hidden layer {layer_index} input value count overflowed"))?;
        if input_values.len() != expected_input_value_count {
            return Err(format!(
                "hidden layer {layer_index} received {} input values, expected {expected_input_value_count}",
                input_values.len()
            ));
        }
        let output_value_count = sample_count
            .checked_mul(output_size)
            .ok_or_else(|| format!("hidden layer {layer_index} output value count overflowed"))?;
        let output_byte_len = output_value_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("hidden layer {layer_index} output buffer size overflowed"))?;

        let inputs = storage_buffer(
            device,
            queue,
            "forward_layer_inputs",
            &f32_bytes(&input_values),
            wgpu::BufferUsages::STORAGE,
        );
        let weights = storage_buffer(
            device,
            queue,
            "forward_layer_weights",
            &f32_bytes(&model.hidden_weights[weight_offset..weights_end]),
            wgpu::BufferUsages::STORAGE,
        );
        let outputs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("forward_layer_outputs"),
            size: output_byte_len as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = storage_buffer(
            device,
            queue,
            "forward_layer_params",
            &uniform_bytes(&[
                sample_count as u32,
                input_size as u32,
                output_size as u32,
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("forward_layer_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: inputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: outputs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("forward_layer_dispatch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("forward_layer_dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (sample_count as u32).div_ceil(16),
                (output_size as u32).div_ceil(16),
                1,
            );
        }
        queue.submit([encoder.finish()]);
        let bytes = read_buffer(
            device,
            queue,
            &outputs,
            output_byte_len as wgpu::BufferAddress,
        )
        .await?;
        input_values = bytes_to_f32_vec(&bytes)?;
        if input_values.len() != output_value_count {
            return Err(format!(
                "hidden layer {layer_index} read back {} values, expected {output_value_count}",
                input_values.len()
            ));
        }
        weight_offset = weights_end;
        input_size = output_size;
    }
    Ok(input_values)
}

async fn train_value_head_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: NativeValueHeadTrainingRequest,
) -> Result<
    (
        crate::gpu::training::CompactValueModel,
        crate::gpu::training::ValueHeadTrainingReport,
    ),
    String,
> {
    let sample_count = request.samples.len();
    let mut trained = request.model.clone();
    if request.train_hidden_layers && !request.model.hidden_layers.is_empty() {
        return train_value_model_with_hidden_layers_on_device(device, queue, request).await;
    }
    let hidden_features = hidden_features_batch_on_device(
        device,
        queue,
        &request.model,
        sample_count,
        request.projected_features,
    )
    .await?;
    let input_size = trained
        .hidden_layers
        .last()
        .map(|value| *value as usize)
        .unwrap_or(trained.projection_size as usize);
    let initial_loss = value_head_loss_flat(
        &hidden_features,
        input_size,
        &request.samples,
        &trained.output_weights,
    );
    let mut best_loss = initial_loss;
    let mut best_output_weights = trained.output_weights.clone();

    let output_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forward_output.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::FORWARD_OUTPUT_SHADER.into()),
    });
    let output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("forward_output"),
        layout: None,
        module: &output_module,
        entry_point: Some("forward_output"),
        compilation_options: Default::default(),
        cache: None,
    });
    let delta_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("output_delta.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::OUTPUT_DELTA_SHADER.into()),
    });
    let delta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("output_delta"),
        layout: None,
        module: &delta_module,
        entry_point: Some("output_delta"),
        compilation_options: Default::default(),
        cache: None,
    });
    let apply_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("apply_output.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::APPLY_OUTPUT_SHADER.into()),
    });
    let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("apply_output"),
        layout: None,
        module: &apply_module,
        entry_point: Some("apply_output"),
        compilation_options: Default::default(),
        cache: None,
    });

    let features = storage_buffer(
        device,
        queue,
        "native_value_train_features",
        &f32_bytes(&hidden_features),
        wgpu::BufferUsages::STORAGE,
    );
    let labels = request
        .samples
        .iter()
        .map(|sample| crate::gpu::training::bounded_value(sample.label))
        .collect::<Vec<_>>();
    let label_weights = request
        .samples
        .iter()
        .map(|sample| sample.label_weight.max(0.0))
        .collect::<Vec<_>>();
    let total_weight = label_weights.iter().sum::<f32>();
    let sample_indices = (0..sample_count)
        .map(|index| {
            u32::try_from(index).map_err(|_| "sample index exceeds GPU parameter range".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let label_buffer = storage_buffer(
        device,
        queue,
        "native_value_train_labels",
        &f32_bytes(&labels),
        wgpu::BufferUsages::STORAGE,
    );
    let label_weight_buffer = storage_buffer(
        device,
        queue,
        "native_value_train_label_weights",
        &f32_bytes(&label_weights),
        wgpu::BufferUsages::STORAGE,
    );
    let index_buffer = storage_buffer(
        device,
        queue,
        "native_value_train_indices",
        &u32_bytes(&sample_indices),
        wgpu::BufferUsages::STORAGE,
    );
    let mut output_weights = trained.output_weights.clone();
    let weight_buffer = storage_buffer(
        device,
        queue,
        "native_value_train_output_weights",
        &f32_bytes(&output_weights),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let velocity = vec![0.0f32; output_weights.len()];
    let velocity_buffer = storage_buffer(
        device,
        queue,
        "native_value_train_output_velocity",
        &f32_bytes(&velocity),
        wgpu::BufferUsages::STORAGE,
    );
    let prediction_byte_len = sample_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "prediction buffer size overflowed".to_string())?;
    let prediction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_value_train_predictions"),
        size: prediction_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_value_train_deltas"),
        size: prediction_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let forward_params = storage_buffer(
        device,
        queue,
        "native_value_train_forward_params",
        &uniform_bytes(&[sample_count as u32, input_size as u32, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let delta_params = storage_buffer(
        device,
        queue,
        "native_value_train_delta_params",
        &output_delta_params_bytes(sample_count as u32, total_weight),
        wgpu::BufferUsages::UNIFORM,
    );
    let apply_params = storage_buffer(
        device,
        queue,
        "native_value_train_apply_params",
        &apply_output_params_bytes(
            sample_count as u32,
            input_size as u32,
            request.config.learning_rate,
            request.config.weight_decay,
            request.config.momentum,
        ),
        wgpu::BufferUsages::UNIFORM,
    );

    let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_value_train_forward_bind_group"),
        layout: &output_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: prediction_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: forward_params.as_entire_binding(),
            },
        ],
    });
    let delta_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_value_train_delta_bind_group"),
        layout: &delta_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: prediction_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: label_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: delta_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: label_weight_buffer.as_entire_binding(),
            },
        ],
    });
    let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_value_train_apply_bind_group"),
        layout: &apply_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: apply_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: velocity_buffer.as_entire_binding(),
            },
        ],
    });

    let weight_byte_len = output_weights
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "output weight buffer size overflowed".to_string())?;
    for _ in 0..request.config.epochs.max(1) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("native_value_train_epoch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("native_value_train_epoch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&output_pipeline);
            pass.set_bind_group(0, &output_bind_group, &[]);
            pass.dispatch_workgroups((sample_count as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&delta_pipeline);
            pass.set_bind_group(0, &delta_bind_group, &[]);
            pass.dispatch_workgroups((sample_count as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&apply_pipeline);
            pass.set_bind_group(0, &apply_bind_group, &[]);
            pass.dispatch_workgroups(((input_size + 1) as u32).div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let bytes = read_buffer(
            device,
            queue,
            &weight_buffer,
            weight_byte_len as wgpu::BufferAddress,
        )
        .await?;
        output_weights = bytes_to_f32_vec(&bytes)?;
        let loss = value_head_loss_flat(
            &hidden_features,
            input_size,
            &request.samples,
            &output_weights,
        );
        if loss.is_finite() && loss < best_loss {
            best_loss = loss;
            best_output_weights.clone_from(&output_weights);
        }
    }

    trained.output_weights = best_output_weights;
    Ok((
        trained,
        crate::gpu::training::ValueHeadTrainingReport {
            initial_loss,
            final_loss: best_loss,
            samples: sample_count,
            epochs: request.config.epochs.max(1),
        },
    ))
}

async fn train_value_model_with_hidden_layers_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: NativeValueHeadTrainingRequest,
) -> Result<
    (
        crate::gpu::training::CompactValueModel,
        crate::gpu::training::ValueHeadTrainingReport,
    ),
    String,
> {
    let sample_count = request.samples.len();
    let mut trained = request.model.clone();
    let projection_size = trained.projection_size as usize;
    let initial_loss = value_loss_from_projected(
        &request.projected_features,
        projection_size,
        &request.samples,
        &trained,
    );
    let mut best_loss = initial_loss;
    let mut best_hidden_weights = trained.hidden_weights.clone();
    let mut best_output_weights = trained.output_weights.clone();

    let forward_layer_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forward_layer.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::FORWARD_LAYER_SHADER.into()),
    });
    let forward_layer_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("forward_layer"),
        layout: None,
        module: &forward_layer_module,
        entry_point: Some("forward_layer"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("forward_output.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::FORWARD_OUTPUT_SHADER.into()),
    });
    let output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("forward_output"),
        layout: None,
        module: &output_module,
        entry_point: Some("forward_output"),
        compilation_options: Default::default(),
        cache: None,
    });
    let output_delta_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("output_delta.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::OUTPUT_DELTA_SHADER.into()),
    });
    let output_delta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("output_delta"),
        layout: None,
        module: &output_delta_module,
        entry_point: Some("output_delta"),
        compilation_options: Default::default(),
        cache: None,
    });
    let hidden3_delta_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hidden3_delta.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::HIDDEN3_DELTA_SHADER.into()),
    });
    let hidden3_delta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hidden3_delta"),
        layout: None,
        module: &hidden3_delta_module,
        entry_point: Some("hidden3_delta"),
        compilation_options: Default::default(),
        cache: None,
    });
    let hidden_delta_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hidden_delta.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::HIDDEN_DELTA_SHADER.into()),
    });
    let hidden_delta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hidden_delta"),
        layout: None,
        module: &hidden_delta_module,
        entry_point: Some("hidden_delta"),
        compilation_options: Default::default(),
        cache: None,
    });
    let apply_output_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("apply_output.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::APPLY_OUTPUT_SHADER.into()),
    });
    let apply_output_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("apply_output"),
        layout: None,
        module: &apply_output_module,
        entry_point: Some("apply_output"),
        compilation_options: Default::default(),
        cache: None,
    });
    let apply_layer_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("apply_layer.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::APPLY_LAYER_SHADER.into()),
    });
    let apply_layer_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("apply_layer"),
        layout: None,
        module: &apply_layer_module,
        entry_point: Some("apply_layer"),
        compilation_options: Default::default(),
        cache: None,
    });

    let projected_buffer = storage_buffer(
        device,
        queue,
        "native_hidden_train_projected_features",
        &f32_bytes(&request.projected_features),
        wgpu::BufferUsages::STORAGE,
    );
    let labels = request
        .samples
        .iter()
        .map(|sample| crate::gpu::training::bounded_value(sample.label))
        .collect::<Vec<_>>();
    let label_weights = request
        .samples
        .iter()
        .map(|sample| sample.label_weight.max(0.0))
        .collect::<Vec<_>>();
    let total_weight = label_weights.iter().sum::<f32>();
    let sample_indices = (0..sample_count)
        .map(|index| {
            u32::try_from(index).map_err(|_| "sample index exceeds GPU parameter range".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let label_buffer = storage_buffer(
        device,
        queue,
        "native_hidden_train_labels",
        &f32_bytes(&labels),
        wgpu::BufferUsages::STORAGE,
    );
    let label_weight_buffer = storage_buffer(
        device,
        queue,
        "native_hidden_train_label_weights",
        &f32_bytes(&label_weights),
        wgpu::BufferUsages::STORAGE,
    );
    let index_buffer = storage_buffer(
        device,
        queue,
        "native_hidden_train_indices",
        &u32_bytes(&sample_indices),
        wgpu::BufferUsages::STORAGE,
    );

    let mut layer_weight_slices = Vec::new();
    let mut offset = 0usize;
    let mut input_size = projection_size;
    for &layer_size in &trained.hidden_layers {
        let layer_size = layer_size as usize;
        let weight_count = layer_size
            .checked_mul(input_size + 1)
            .ok_or_else(|| "hidden layer weight count overflowed".to_string())?;
        let end = offset
            .checked_add(weight_count)
            .ok_or_else(|| "hidden layer weight range overflowed".to_string())?;
        layer_weight_slices.push((offset, end, input_size, layer_size));
        offset = end;
        input_size = layer_size;
    }

    let mut weight_buffers = Vec::with_capacity(layer_weight_slices.len());
    let mut velocity_buffers = Vec::with_capacity(layer_weight_slices.len());
    let mut activation_buffers = Vec::with_capacity(layer_weight_slices.len());
    let mut delta_buffers = Vec::with_capacity(layer_weight_slices.len());
    let mut forward_param_buffers = Vec::with_capacity(layer_weight_slices.len());
    let mut apply_param_buffers = Vec::with_capacity(layer_weight_slices.len());
    for (layer_index, (start, end, layer_input_size, layer_output_size)) in
        layer_weight_slices.iter().copied().enumerate()
    {
        let weights = storage_buffer(
            device,
            queue,
            "native_hidden_train_layer_weights",
            &f32_bytes(&trained.hidden_weights[start..end]),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let velocity = vec![0.0f32; end - start];
        let velocity = storage_buffer(
            device,
            queue,
            "native_hidden_train_layer_velocity",
            &f32_bytes(&velocity),
            wgpu::BufferUsages::STORAGE,
        );
        let value_count = sample_count
            .checked_mul(layer_output_size)
            .ok_or_else(|| format!("hidden layer {layer_index} activation count overflowed"))?;
        let byte_len = value_count
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| format!("hidden layer {layer_index} activation buffer overflowed"))?;
        let activations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native_hidden_train_activations"),
            size: byte_len as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let deltas = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("native_hidden_train_deltas"),
            size: byte_len as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let forward_params = storage_buffer(
            device,
            queue,
            "native_hidden_train_forward_layer_params",
            &uniform_bytes(&[
                sample_count as u32,
                layer_input_size as u32,
                layer_output_size as u32,
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        let apply_params = storage_buffer(
            device,
            queue,
            "native_hidden_train_apply_layer_params",
            &layer_params_bytes(
                sample_count as u32,
                layer_input_size as u32,
                layer_output_size as u32,
                request.config.learning_rate,
                request.config.weight_decay,
                request.config.momentum,
            ),
            wgpu::BufferUsages::UNIFORM,
        );
        weight_buffers.push(weights);
        velocity_buffers.push(velocity);
        activation_buffers.push(activations);
        delta_buffers.push(deltas);
        forward_param_buffers.push(forward_params);
        apply_param_buffers.push(apply_params);
    }

    let output_weights = storage_buffer(
        device,
        queue,
        "native_hidden_train_output_weights",
        &f32_bytes(&trained.output_weights),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let output_velocity = vec![0.0f32; trained.output_weights.len()];
    let output_velocity = storage_buffer(
        device,
        queue,
        "native_hidden_train_output_velocity",
        &f32_bytes(&output_velocity),
        wgpu::BufferUsages::STORAGE,
    );
    let prediction_byte_len = sample_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "prediction buffer size overflowed".to_string())?;
    let prediction_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_hidden_train_predictions"),
        size: prediction_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_hidden_train_output_deltas"),
        size: prediction_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let final_hidden_size = *trained
        .hidden_layers
        .last()
        .ok_or_else(|| "hidden layer training requires at least one hidden layer".to_string())?
        as usize;
    let output_forward_params = storage_buffer(
        device,
        queue,
        "native_hidden_train_output_forward_params",
        &uniform_bytes(&[sample_count as u32, final_hidden_size as u32, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let output_delta_params = storage_buffer(
        device,
        queue,
        "native_hidden_train_output_delta_params",
        &output_delta_params_bytes(sample_count as u32, total_weight),
        wgpu::BufferUsages::UNIFORM,
    );
    let output_apply_params = storage_buffer(
        device,
        queue,
        "native_hidden_train_output_apply_params",
        &apply_output_params_bytes(
            sample_count as u32,
            final_hidden_size as u32,
            request.config.learning_rate,
            request.config.weight_decay,
            request.config.momentum,
        ),
        wgpu::BufferUsages::UNIFORM,
    );

    let mut forward_layer_bind_groups = Vec::with_capacity(layer_weight_slices.len());
    let mut apply_layer_bind_groups = Vec::with_capacity(layer_weight_slices.len());
    for layer_index in 0..layer_weight_slices.len() {
        let input_buffer = if layer_index == 0 {
            &projected_buffer
        } else {
            &activation_buffers[layer_index - 1]
        };
        forward_layer_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native_hidden_train_forward_layer_bind_group"),
            layout: &forward_layer_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: activation_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: forward_param_buffers[layer_index].as_entire_binding(),
                },
            ],
        }));
        apply_layer_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native_hidden_train_apply_layer_bind_group"),
            layout: &apply_layer_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: delta_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: apply_param_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: velocity_buffers[layer_index].as_entire_binding(),
                },
            ],
        }));
    }
    let output_forward_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_hidden_train_output_forward_bind_group"),
        layout: &output_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: activation_buffers
                    .last()
                    .ok_or_else(|| "hidden layer activation buffer missing".to_string())?
                    .as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: prediction_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_forward_params.as_entire_binding(),
            },
        ],
    });
    let output_delta_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_hidden_train_output_delta_bind_group"),
        layout: &output_delta_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: prediction_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: label_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_delta_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: label_weight_buffer.as_entire_binding(),
            },
        ],
    });
    let last_hidden_index = activation_buffers.len() - 1;
    let last_hidden_delta_params = storage_buffer(
        device,
        queue,
        "native_hidden_train_last_hidden_delta_params",
        &uniform_bytes(&[sample_count as u32, final_hidden_size as u32, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let last_hidden_delta_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_hidden_train_last_hidden_delta_bind_group"),
        layout: &hidden3_delta_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: activation_buffers[last_hidden_index].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: delta_buffers[last_hidden_index].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: last_hidden_delta_params.as_entire_binding(),
            },
        ],
    });
    let mut hidden_delta_bind_groups = Vec::new();
    let mut hidden_delta_param_buffers = Vec::new();
    for layer_index in 0..last_hidden_index {
        let current_size = trained.hidden_layers[layer_index] as usize;
        let next_size = trained.hidden_layers[layer_index + 1] as usize;
        let params = storage_buffer(
            device,
            queue,
            "native_hidden_train_hidden_delta_params",
            &uniform_bytes(&[
                sample_count as u32,
                current_size as u32,
                next_size as u32,
                0,
            ]),
            wgpu::BufferUsages::UNIFORM,
        );
        hidden_delta_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("native_hidden_train_hidden_delta_bind_group"),
            layout: &hidden_delta_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: activation_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: delta_buffers[layer_index + 1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffers[layer_index + 1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: delta_buffers[layer_index].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        }));
        hidden_delta_param_buffers.push(params);
    }
    let output_apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_hidden_train_output_apply_bind_group"),
        layout: &apply_output_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: activation_buffers[last_hidden_index].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_weights.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_apply_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_velocity.as_entire_binding(),
            },
        ],
    });

    let output_weight_byte_len = trained
        .output_weights
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "output weight buffer size overflowed".to_string())?;
    for _ in 0..request.config.epochs.max(1) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("native_hidden_train_epoch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("native_hidden_train_epoch"),
                timestamp_writes: None,
            });
            for (layer_index, (_start, _end, _input_size, output_size)) in
                layer_weight_slices.iter().copied().enumerate()
            {
                pass.set_pipeline(&forward_layer_pipeline);
                pass.set_bind_group(0, &forward_layer_bind_groups[layer_index], &[]);
                pass.dispatch_workgroups(
                    (sample_count as u32).div_ceil(16),
                    (output_size as u32).div_ceil(16),
                    1,
                );
            }
            pass.set_pipeline(&output_pipeline);
            pass.set_bind_group(0, &output_forward_bind_group, &[]);
            pass.dispatch_workgroups((sample_count as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&output_delta_pipeline);
            pass.set_bind_group(0, &output_delta_bind_group, &[]);
            pass.dispatch_workgroups((sample_count as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&hidden3_delta_pipeline);
            pass.set_bind_group(0, &last_hidden_delta_bind_group, &[]);
            pass.dispatch_workgroups(
                (sample_count as u32).div_ceil(16),
                (final_hidden_size as u32).div_ceil(16),
                1,
            );
            for layer_index in (0..last_hidden_index).rev() {
                pass.set_pipeline(&hidden_delta_pipeline);
                pass.set_bind_group(0, &hidden_delta_bind_groups[layer_index], &[]);
                pass.dispatch_workgroups(
                    (sample_count as u32).div_ceil(16),
                    trained.hidden_layers[layer_index].div_ceil(16),
                    1,
                );
            }
            pass.set_pipeline(&apply_output_pipeline);
            pass.set_bind_group(0, &output_apply_bind_group, &[]);
            pass.dispatch_workgroups(((final_hidden_size + 1) as u32).div_ceil(64), 1, 1);
            for layer_index in (0..layer_weight_slices.len()).rev() {
                let (_start, _end, layer_input_size, layer_output_size) =
                    layer_weight_slices[layer_index];
                pass.set_pipeline(&apply_layer_pipeline);
                pass.set_bind_group(0, &apply_layer_bind_groups[layer_index], &[]);
                pass.dispatch_workgroups(
                    ((layer_input_size + 1) as u32).div_ceil(16),
                    (layer_output_size as u32).div_ceil(16),
                    1,
                );
            }
        }
        queue.submit([encoder.finish()]);

        let mut hidden_weights = Vec::with_capacity(trained.hidden_weights.len());
        for (layer_index, (start, end, _input_size, _output_size)) in
            layer_weight_slices.iter().copied().enumerate()
        {
            let byte_len = (end - start)
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    format!("hidden layer {layer_index} weight readback size overflowed")
                })?;
            let bytes = read_buffer(
                device,
                queue,
                &weight_buffers[layer_index],
                byte_len as wgpu::BufferAddress,
            )
            .await?;
            hidden_weights.extend(bytes_to_f32_vec(&bytes)?);
        }
        let bytes = read_buffer(
            device,
            queue,
            &output_weights,
            output_weight_byte_len as wgpu::BufferAddress,
        )
        .await?;
        let output_weights_now = bytes_to_f32_vec(&bytes)?;
        let mut candidate = trained.clone();
        candidate.hidden_weights = hidden_weights;
        candidate.output_weights = output_weights_now;
        let loss = value_loss_from_projected(
            &request.projected_features,
            projection_size,
            &request.samples,
            &candidate,
        );
        if loss.is_finite() && loss < best_loss {
            best_loss = loss;
            best_hidden_weights.clone_from(&candidate.hidden_weights);
            best_output_weights.clone_from(&candidate.output_weights);
        }
    }

    trained.hidden_weights = best_hidden_weights;
    trained.output_weights = best_output_weights;
    Ok((
        trained,
        crate::gpu::training::ValueHeadTrainingReport {
            initial_loss,
            final_loss: best_loss,
            samples: sample_count,
            epochs: request.config.epochs.max(1),
        },
    ))
}

async fn train_policy_head_on_device(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    request: NativePolicyHeadTrainingRequest,
) -> Result<
    (
        crate::gpu::training::CompactValueModel,
        crate::gpu::training::PolicyHeadTrainingReport,
    ),
    String,
> {
    let sample_count = request.samples.len();
    let mut trained = request.model.clone();
    let hidden_features = hidden_features_batch_on_device(
        device,
        queue,
        &request.model,
        sample_count,
        request.projected_features,
    )
    .await?;
    let input_size = trained
        .hidden_layers
        .last()
        .map(|value| *value as usize)
        .unwrap_or(trained.projection_size as usize);
    let expected_weights = crate::gpu::training::POLICY_BUCKETS as usize * (input_size + 1);
    let mut policy_weights = if trained.policy_weights.len() == expected_weights {
        trained.policy_weights.clone()
    } else {
        vec![0.0; expected_weights]
    };
    let policy_indices = request
        .samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| {
            has_policy_training_target(sample)
                .then_some(index)
                .filter(|_| sample.label_weight.max(0.0) > 0.0)
        })
        .collect::<Vec<_>>();
    if policy_indices.is_empty() {
        trained.policy_weights = policy_weights;
        return Ok((
            trained,
            crate::gpu::training::PolicyHeadTrainingReport {
                initial_loss: f32::NAN,
                final_loss: f32::NAN,
                samples: 0,
                steps: 0,
            },
        ));
    }
    let batch_count = policy_indices.len();
    let initial_loss = policy_head_loss_flat(
        &hidden_features,
        input_size,
        &request.samples,
        &policy_weights,
        &policy_indices,
    );
    let mut best_loss = initial_loss;
    let mut best_policy_weights = policy_weights.clone();

    let policy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("policy.wgsl"),
        source: wgpu::ShaderSource::Wgsl(crate::gpu::training::POLICY_SHADER.into()),
    });
    let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("policy_forward"),
        layout: None,
        module: &policy_module,
        entry_point: Some("forward_policy"),
        compilation_options: Default::default(),
        cache: None,
    });
    let delta_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("policy_delta"),
        layout: None,
        module: &policy_module,
        entry_point: Some("policy_delta"),
        compilation_options: Default::default(),
        cache: None,
    });
    let apply_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("policy_apply"),
        layout: None,
        module: &policy_module,
        entry_point: Some("apply_policy"),
        compilation_options: Default::default(),
        cache: None,
    });

    let features = storage_buffer(
        device,
        queue,
        "native_policy_train_features",
        &f32_bytes(&hidden_features),
        wgpu::BufferUsages::STORAGE,
    );
    let targets = request
        .samples
        .iter()
        .map(|sample| policy_target(Some(sample)) as u32)
        .collect::<Vec<_>>();
    let sample_weights = request
        .samples
        .iter()
        .map(|sample| sample.label_weight.max(0.0))
        .collect::<Vec<_>>();
    let total_weight = policy_indices
        .iter()
        .map(|&index| sample_weights[index])
        .sum::<f32>();
    let batch_indices = policy_indices
        .iter()
        .map(|&index| {
            u32::try_from(index)
                .map_err(|_| "policy sample index exceeds GPU parameter range".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let target_buffer = storage_buffer(
        device,
        queue,
        "native_policy_train_targets",
        &u32_bytes(&targets),
        wgpu::BufferUsages::STORAGE,
    );
    let sample_weight_buffer = storage_buffer(
        device,
        queue,
        "native_policy_train_sample_weights",
        &f32_bytes(&sample_weights),
        wgpu::BufferUsages::STORAGE,
    );
    let policy_weight_buffer = storage_buffer(
        device,
        queue,
        "native_policy_train_weights",
        &f32_bytes(&policy_weights),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let weight_velocity = vec![0.0f32; policy_weights.len()];
    let velocity_buffer = storage_buffer(
        device,
        queue,
        "native_policy_train_velocity",
        &f32_bytes(&weight_velocity),
        wgpu::BufferUsages::STORAGE,
    );
    let batch_index_buffer = storage_buffer(
        device,
        queue,
        "native_policy_train_batch_indices",
        &u32_bytes(&batch_indices),
        wgpu::BufferUsages::STORAGE,
    );
    let logits_value_count = batch_count
        .checked_mul(crate::gpu::training::POLICY_BUCKETS as usize)
        .ok_or_else(|| "policy logits value count overflowed".to_string())?;
    let logits_byte_len = logits_value_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "policy logits buffer size overflowed".to_string())?;
    let logits_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_policy_train_logits"),
        size: logits_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let delta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_policy_train_deltas"),
        size: logits_byte_len as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let params = storage_buffer(
        device,
        queue,
        "native_policy_train_params",
        &policy_params_bytes(
            batch_count as u32,
            input_size as u32,
            crate::gpu::training::POLICY_BUCKETS,
            total_weight,
            request.config.learning_rate,
            request.config.weight_decay,
            request.config.momentum,
        ),
        wgpu::BufferUsages::UNIFORM,
    );

    let forward_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_policy_forward_bind_group"),
        layout: &forward_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: policy_weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: logits_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: batch_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params.as_entire_binding(),
            },
        ],
    });
    let delta_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_policy_delta_bind_group"),
        layout: &delta_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: target_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: sample_weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: logits_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: batch_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params.as_entire_binding(),
            },
        ],
    });
    let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("native_policy_apply_bind_group"),
        layout: &apply_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: features.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: policy_weight_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: delta_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: batch_index_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: velocity_buffer.as_entire_binding(),
            },
        ],
    });

    let weight_byte_len = policy_weights
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "policy weight buffer size overflowed".to_string())?;
    let steps = policy_training_steps(request.config.epochs);
    for _ in 0..steps {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("native_policy_train_step"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("native_policy_train_step"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&forward_pipeline);
            pass.set_bind_group(0, &forward_bind_group, &[]);
            pass.dispatch_workgroups(
                (batch_count as u32).div_ceil(16),
                crate::gpu::training::POLICY_BUCKETS.div_ceil(16),
                1,
            );
            pass.set_pipeline(&delta_pipeline);
            pass.set_bind_group(0, &delta_bind_group, &[]);
            pass.dispatch_workgroups((batch_count as u32).div_ceil(64), 1, 1);
            pass.set_pipeline(&apply_pipeline);
            pass.set_bind_group(0, &apply_bind_group, &[]);
            pass.dispatch_workgroups(
                ((input_size + 1) as u32).div_ceil(16),
                crate::gpu::training::POLICY_BUCKETS.div_ceil(16),
                1,
            );
        }
        queue.submit([encoder.finish()]);
        let bytes = read_buffer(
            device,
            queue,
            &policy_weight_buffer,
            weight_byte_len as wgpu::BufferAddress,
        )
        .await?;
        policy_weights = bytes_to_f32_vec(&bytes)?;
        let loss = policy_head_loss_flat(
            &hidden_features,
            input_size,
            &request.samples,
            &policy_weights,
            &policy_indices,
        );
        if loss.is_finite() && loss + 1e-6 < best_loss {
            best_loss = loss;
            best_policy_weights.clone_from(&policy_weights);
        }
    }

    trained.policy_weights = best_policy_weights;
    Ok((
        trained,
        crate::gpu::training::PolicyHeadTrainingReport {
            initial_loss,
            final_loss: best_loss,
            samples: policy_indices.len(),
            steps,
        },
    ))
}

fn validate_project_features_request(
    request: &NativeProjectFeaturesBatchRequest,
) -> Result<(), String> {
    if request.features.is_empty() {
        return Err("at least one feature row is required".to_string());
    }
    if request.projection_size == 0 {
        return Err("projection size must be greater than zero".to_string());
    }
    if request.features.len() > u32::MAX as usize {
        return Err("sample count exceeds GPU parameter range".to_string());
    }
    if request.projection_size > u32::MAX as usize {
        return Err("projection size exceeds GPU parameter range".to_string());
    }
    if request.output_offset > u32::MAX as usize {
        return Err("output offset exceeds GPU parameter range".to_string());
    }
    let input_size = request.features[0].len();
    if input_size > u32::MAX as usize {
        return Err("input size exceeds GPU parameter range".to_string());
    }
    for (index, features) in request.features.iter().enumerate() {
        if features.len() != input_size {
            return Err(format!(
                "feature row {index} has length {}, expected {input_size}",
                features.len()
            ));
        }
    }
    Ok(())
}

fn validate_value_prediction_request(request: &NativeValuePredictionRequest) -> Result<(), String> {
    if request.features.is_empty() {
        return Err("at least one feature row is required".to_string());
    }
    if request.features.len() > u32::MAX as usize {
        return Err("sample count exceeds GPU parameter range".to_string());
    }
    if request.model.projection_size == 0 {
        return Err("projection size must be greater than zero".to_string());
    }
    if request.model.projection_size > u32::MAX {
        return Err("projection size exceeds GPU parameter range".to_string());
    }
    let input_size = request
        .features
        .first()
        .map(Vec::len)
        .ok_or_else(|| "at least one feature row is required".to_string())?;
    for (index, features) in request.features.iter().enumerate() {
        if features.len() != input_size {
            return Err(format!(
                "feature row {index} has length {}, expected {input_size}",
                features.len()
            ));
        }
    }
    let mut hidden_input_size = request.model.projection_size as usize;
    let mut hidden_weight_count = 0usize;
    for (layer_index, &hidden_output_size) in request.model.hidden_layers.iter().enumerate() {
        if hidden_output_size == 0 {
            return Err(format!("hidden layer {layer_index} has zero output size"));
        }
        let hidden_output_size = hidden_output_size as usize;
        if hidden_output_size > u32::MAX as usize {
            return Err(format!(
                "hidden layer {layer_index} output size exceeds GPU parameter range"
            ));
        }
        let row_size = hidden_input_size
            .checked_add(1)
            .ok_or_else(|| format!("hidden layer {layer_index} row size overflowed"))?;
        let layer_weight_count = hidden_output_size
            .checked_mul(row_size)
            .ok_or_else(|| format!("hidden layer {layer_index} weight count overflowed"))?;
        hidden_weight_count = hidden_weight_count
            .checked_add(layer_weight_count)
            .ok_or_else(|| {
                format!("hidden layer {layer_index} cumulative weight count overflowed")
            })?;
        hidden_input_size = hidden_output_size;
    }
    if request.model.hidden_weights.len() != hidden_weight_count {
        return Err(format!(
            "GPU value model hidden layers have {} weights but expected {}.",
            request.model.hidden_weights.len(),
            hidden_weight_count
        ));
    }
    let output_size = request
        .model
        .hidden_layers
        .last()
        .map(|value| *value as usize)
        .unwrap_or(request.model.projection_size as usize);
    if request.model.output_weights.len() != output_size + 1 {
        return Err(format!(
            "GPU value model output head has {} weights but expected {}.",
            request.model.output_weights.len(),
            output_size + 1
        ));
    }
    if request.model.output_activation != crate::gpu::training::OutputActivation::Tanh {
        return Err(
            "native GPU value prediction currently requires tanh output activation".to_string(),
        );
    }
    Ok(())
}

fn validate_value_head_training_request(
    request: &NativeValueHeadTrainingRequest,
) -> Result<(), String> {
    if request.samples.is_empty() {
        return Err("native GPU value-head training requires at least one sample.".to_string());
    }
    if request.samples.len() > u32::MAX as usize {
        return Err("sample count exceeds GPU parameter range".to_string());
    }
    if request.model.projection_size == 0 {
        return Err("projection size must be greater than zero".to_string());
    }
    if request.model.projection_size > u32::MAX {
        return Err("projection size exceeds GPU parameter range".to_string());
    }
    let projection_size = request.model.projection_size as usize;
    let expected_projected_values = request
        .samples
        .len()
        .checked_mul(projection_size)
        .ok_or_else(|| "projected feature count overflowed".to_string())?;
    if request.projected_features.len() != expected_projected_values {
        return Err(format!(
            "native GPU value-head training got {} projected values but expected {expected_projected_values}.",
            request.projected_features.len()
        ));
    }
    let mut hidden_input_size = projection_size;
    let mut hidden_weight_count = 0usize;
    for (layer_index, &hidden_output_size) in request.model.hidden_layers.iter().enumerate() {
        if hidden_output_size == 0 {
            return Err(format!("hidden layer {layer_index} has zero output size"));
        }
        let hidden_output_size = hidden_output_size as usize;
        if hidden_output_size > u32::MAX as usize {
            return Err(format!(
                "hidden layer {layer_index} output size exceeds GPU parameter range"
            ));
        }
        let row_size = hidden_input_size
            .checked_add(1)
            .ok_or_else(|| format!("hidden layer {layer_index} row size overflowed"))?;
        let layer_weight_count = hidden_output_size
            .checked_mul(row_size)
            .ok_or_else(|| format!("hidden layer {layer_index} weight count overflowed"))?;
        hidden_weight_count = hidden_weight_count
            .checked_add(layer_weight_count)
            .ok_or_else(|| {
                format!("hidden layer {layer_index} cumulative weight count overflowed")
            })?;
        hidden_input_size = hidden_output_size;
    }
    if request.model.hidden_weights.len() != hidden_weight_count {
        return Err(format!(
            "GPU value model hidden layers have {} weights but expected {}.",
            request.model.hidden_weights.len(),
            hidden_weight_count
        ));
    }
    if request.model.output_weights.len() != hidden_input_size + 1 {
        return Err(format!(
            "GPU value model output head has {} weights but expected {}.",
            request.model.output_weights.len(),
            hidden_input_size + 1
        ));
    }
    if request.model.output_activation != crate::gpu::training::OutputActivation::Tanh {
        return Err(
            "native GPU value-head training currently requires tanh output activation".to_string(),
        );
    }
    Ok(())
}

fn validate_policy_head_training_request(
    request: &NativePolicyHeadTrainingRequest,
) -> Result<(), String> {
    if request.samples.is_empty() {
        return Err("native GPU policy-head training requires at least one sample.".to_string());
    }
    if request.samples.len() > u32::MAX as usize {
        return Err("sample count exceeds GPU parameter range".to_string());
    }
    if request.model.projection_size == 0 {
        return Err("projection size must be greater than zero".to_string());
    }
    if request.model.projection_size > u32::MAX {
        return Err("projection size exceeds GPU parameter range".to_string());
    }
    let projection_size = request.model.projection_size as usize;
    let expected_projected_values = request
        .samples
        .len()
        .checked_mul(projection_size)
        .ok_or_else(|| "projected feature count overflowed".to_string())?;
    if request.projected_features.len() != expected_projected_values {
        return Err(format!(
            "native GPU policy-head training got {} projected values but expected {expected_projected_values}.",
            request.projected_features.len()
        ));
    }
    let mut hidden_input_size = projection_size;
    let mut hidden_weight_count = 0usize;
    for (layer_index, &hidden_output_size) in request.model.hidden_layers.iter().enumerate() {
        if hidden_output_size == 0 {
            return Err(format!("hidden layer {layer_index} has zero output size"));
        }
        let hidden_output_size = hidden_output_size as usize;
        if hidden_output_size > u32::MAX as usize {
            return Err(format!(
                "hidden layer {layer_index} output size exceeds GPU parameter range"
            ));
        }
        let row_size = hidden_input_size
            .checked_add(1)
            .ok_or_else(|| format!("hidden layer {layer_index} row size overflowed"))?;
        let layer_weight_count = hidden_output_size
            .checked_mul(row_size)
            .ok_or_else(|| format!("hidden layer {layer_index} weight count overflowed"))?;
        hidden_weight_count = hidden_weight_count
            .checked_add(layer_weight_count)
            .ok_or_else(|| {
                format!("hidden layer {layer_index} cumulative weight count overflowed")
            })?;
        hidden_input_size = hidden_output_size;
    }
    if request.model.hidden_weights.len() != hidden_weight_count {
        return Err(format!(
            "GPU value model hidden layers have {} weights but expected {}.",
            request.model.hidden_weights.len(),
            hidden_weight_count
        ));
    }
    let expected_policy_weights = (crate::gpu::training::POLICY_BUCKETS as usize)
        .checked_mul(hidden_input_size + 1)
        .ok_or_else(|| "policy weight count overflowed".to_string())?;
    if !request.model.policy_weights.is_empty()
        && request.model.policy_weights.len() != expected_policy_weights
    {
        return Err(format!(
            "GPU policy model has {} weights but expected {}.",
            request.model.policy_weights.len(),
            expected_policy_weights
        ));
    }
    Ok(())
}

fn storage_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &'static str,
    bytes: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as wgpu::BufferAddress,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

async fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    byte_len: wgpu::BufferAddress,
) -> Result<Vec<u8>, String> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("native_readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("native_readback"),
    });
    encoder.copy_buffer_to_buffer(source, 0, &readback, 0, byte_len);
    queue.submit([encoder.finish()]);

    let slice = readback.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|error| format!("native GPU poll failed: {error:?}"))?;
    receiver
        .recv()
        .map_err(|error| format!("native GPU readback callback failed: {error}"))?
        .map_err(|error| format!("native GPU readback map failed: {error:?}"))?;
    let bytes = {
        let mapped = slice.get_mapped_range();
        mapped.to_vec()
    };
    readback.unmap();
    Ok(bytes)
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn uniform_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn output_delta_params_bytes(sample_count: u32, total_weight: f32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend(sample_count.to_le_bytes());
    bytes.extend(total_weight.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes
}

fn apply_output_params_bytes(
    sample_count: u32,
    input_size: u32,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend(sample_count.to_le_bytes());
    bytes.extend(input_size.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(learning_rate.to_le_bytes());
    bytes.extend(weight_decay.to_le_bytes());
    bytes.extend(momentum.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes
}

fn layer_params_bytes(
    sample_count: u32,
    input_size: u32,
    output_size: u32,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend(sample_count.to_le_bytes());
    bytes.extend(input_size.to_le_bytes());
    bytes.extend(output_size.to_le_bytes());
    bytes.extend(learning_rate.to_le_bytes());
    bytes.extend(weight_decay.to_le_bytes());
    bytes.extend(momentum.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes
}

fn policy_params_bytes(
    batch_count: u32,
    input_size: u32,
    bucket_count: u32,
    total_weight: f32,
    learning_rate: f32,
    weight_decay: f32,
    momentum: f32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(32);
    bytes.extend(batch_count.to_le_bytes());
    bytes.extend(input_size.to_le_bytes());
    bytes.extend(bucket_count.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(total_weight.max(0.0).to_le_bytes());
    bytes.extend(learning_rate.to_le_bytes());
    bytes.extend(weight_decay.to_le_bytes());
    bytes.extend(momentum.to_le_bytes());
    bytes
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn value_head_loss_flat(
    features: &[f32],
    input_size: usize,
    samples: &[crate::gpu::training::TrainingSample],
    weights: &[f32],
) -> f32 {
    let mut total = 0.0;
    let mut total_weight = 0.0;
    let bias_index = weights.len().saturating_sub(1);
    for (sample_index, sample) in samples.iter().enumerate() {
        let start = sample_index * input_size;
        let end = start + input_size;
        let Some(feature) = features.get(start..end) else {
            continue;
        };
        let mut logit = weights.get(bias_index).copied().unwrap_or(0.0);
        for (input, value) in feature.iter().enumerate() {
            logit += value * weights.get(input).copied().unwrap_or(0.0);
        }
        let prediction = logit.tanh();
        let weight = sample.label_weight.max(0.0);
        let error = prediction - crate::gpu::training::bounded_value(sample.label);
        total += weight * error * error;
        total_weight += weight;
    }
    if total_weight > 0.0 {
        total / total_weight
    } else {
        0.0
    }
}

fn value_loss_from_projected(
    projected: &[f32],
    projection_size: usize,
    samples: &[crate::gpu::training::TrainingSample],
    model: &crate::gpu::training::CompactValueModel,
) -> f32 {
    let features = projected
        .chunks(projection_size)
        .map(|row| crate::gpu::training::hidden_features_from_projected(row.to_vec(), model))
        .collect::<Vec<_>>();
    value_head_loss_flat(
        &features.iter().flatten().copied().collect::<Vec<_>>(),
        model
            .hidden_layers
            .last()
            .map(|value| *value as usize)
            .unwrap_or(projection_size),
        samples,
        &model.output_weights,
    )
}

fn policy_head_loss_flat(
    features: &[f32],
    input_size: usize,
    samples: &[crate::gpu::training::TrainingSample],
    weights: &[f32],
    indices: &[usize],
) -> f32 {
    let row_size = input_size + 1;
    let mut total = 0.0;
    let mut total_weight = 0.0;
    for &sample_index in indices {
        let start = sample_index * input_size;
        let end = start + input_size;
        let Some(feature) = features.get(start..end) else {
            continue;
        };
        let target = policy_target(samples.get(sample_index));
        let sample_weight = samples
            .get(sample_index)
            .map(|sample| sample.label_weight.max(0.0))
            .unwrap_or(0.0);
        if sample_weight <= 0.0 {
            continue;
        }
        let mut logits = vec![0.0; crate::gpu::training::POLICY_BUCKETS as usize];
        for (bucket, logit) in logits.iter_mut().enumerate() {
            let row = bucket * row_size;
            *logit = weights.get(row + input_size).copied().unwrap_or(0.0);
            for (input, value) in feature.iter().enumerate().take(input_size) {
                *logit += value * weights.get(row + input).copied().unwrap_or(0.0);
            }
        }
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denominator = logits
            .iter()
            .map(|logit| (*logit - max_logit).exp())
            .sum::<f32>()
            .max(1e-12);
        total += sample_weight * (denominator.ln() - (logits[target] - max_logit));
        total_weight += sample_weight;
    }
    if total_weight > 0.0 {
        total / total_weight
    } else {
        0.0
    }
}

fn policy_target(sample: Option<&crate::gpu::training::TrainingSample>) -> usize {
    sample
        .and_then(|sample| sample.policy)
        .unwrap_or(0)
        .min(crate::gpu::training::POLICY_BUCKETS - 1) as usize
}

fn has_policy_training_target(sample: &crate::gpu::training::TrainingSample) -> bool {
    sample.label_kind.as_deref() != Some("distilled") && sample.policy.is_some()
}

fn policy_training_steps(value_epochs: usize) -> usize {
    (value_epochs.saturating_add(63) / 64).clamp(16, 256)
}

fn bytes_to_i32_array_4(bytes: &[u8]) -> Result<[i32; 4], String> {
    if bytes.len() < 16 {
        return Err(format!(
            "native GPU readback returned {} bytes, expected at least 16",
            bytes.len()
        ));
    }
    let mut result = [0; 4];
    for (index, value) in result.iter_mut().enumerate() {
        let offset = index * 4;
        *value = i32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "native GPU readback chunk had invalid width".to_string())?,
        );
    }
    Ok(result)
}

fn bytes_to_i32_vec(bytes: &[u8]) -> Result<Vec<i32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "native GPU i32 readback returned {} bytes, expected a multiple of 4",
            bytes.len()
        ));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            Ok(i32::from_le_bytes(chunk.try_into().map_err(|_| {
                "native GPU i32 readback chunk had invalid width".to_string()
            })?))
        })
        .collect()
}

fn bytes_to_f32_vec(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if bytes.len() % 4 != 0 {
        return Err(format!(
            "native GPU f32 readback returned {} bytes, expected a multiple of 4",
            bytes.len()
        ));
    }
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            Ok(f32::from_le_bytes(chunk.try_into().map_err(|_| {
                "native GPU f32 readback chunk had invalid width".to_string()
            })?))
        })
        .collect()
}

fn engine_shaders() -> impl Iterator<Item = &'static WgslShader> {
    crate::gpu::search::SHADERS
        .iter()
        .chain(crate::gpu::training::SHADERS.iter())
}

fn engine_kernels() -> impl Iterator<Item = &'static GpuKernel> {
    crate::gpu::search::KERNELS
        .iter()
        .chain(crate::gpu::training::KERNELS.iter())
}

fn format_compile_message(shader_name: &str, message: &wgpu::CompilationMessage) -> String {
    let location = message
        .location
        .as_ref()
        .map(|location| format!(":{}:{}", location.line_number, location.line_position))
        .unwrap_or_default();
    format!("shader {shader_name}{location}: {}", message.message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_project_features_batch_matches_cpu_projection() {
        let features = vec![
            vec![0.0, 2.0, 0.0, -1.0, 0.5, 0.0],
            vec![1.0, 0.0, -0.25, 0.0, 0.0, 3.0],
        ];
        let projection_size = 8;
        let seed = 0x9e37_79b9;
        let projected = project_features_batch(NativeProjectFeaturesBatchRequest {
            projection_size,
            seed,
            output_offset: 1,
            features: features.clone(),
        })
        .expect("project features on native GPU");
        let expected = features
            .iter()
            .flat_map(|row| crate::gpu::training::project_features(row, projection_size, seed))
            .collect::<Vec<_>>();

        assert_eq!(projected.len(), expected.len());
        for (actual, expected) in projected.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.000_001);
        }
    }

    #[test]
    fn native_value_prediction_matches_cpu_model() {
        let model = crate::gpu::training::CompactValueModel {
            version: 4,
            projection_size: 4,
            projection_seed: 123,
            hidden_layers: vec![],
            hidden_weights: vec![],
            output_weights: vec![0.25, -0.5, 0.75, 0.1, -0.2],
            policy_logits: vec![],
            policy_weights: vec![],
            auxiliary_value_weights: vec![],
            scale: 1.5,
            bias: 0.1,
            output_activation: crate::gpu::training::OutputActivation::Tanh,
        };
        let features = vec![
            vec![0.0, 2.0, 0.0, -1.0, 0.5, 0.0],
            vec![1.0, 0.0, -0.25, 0.0, 0.0, 3.0],
        ];
        let predictions = predict_values(NativeValuePredictionRequest {
            model: model.clone(),
            features: features.clone(),
        })
        .expect("predict values on native GPU");
        let expected = model.predict_values(features.iter().map(Vec::as_slice));

        assert_eq!(predictions.len(), expected.len());
        for (actual, expected) in predictions.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.000_01);
        }
    }

    #[test]
    fn native_value_prediction_with_hidden_layer_matches_cpu_model() {
        let model = crate::gpu::training::CompactValueModel {
            version: 4,
            projection_size: 4,
            projection_seed: 456,
            hidden_layers: vec![3],
            hidden_weights: vec![
                0.2, -0.1, 0.05, 0.3, 0.01, -0.4, 0.25, 0.1, -0.2, 0.05, 0.15, 0.05, -0.05, 0.2,
                -0.03,
            ],
            output_weights: vec![0.3, -0.2, 0.4, 0.05],
            policy_logits: vec![],
            policy_weights: vec![],
            auxiliary_value_weights: vec![],
            scale: 1.2,
            bias: -0.05,
            output_activation: crate::gpu::training::OutputActivation::Tanh,
        };
        let features = vec![
            vec![0.0, 2.0, 0.0, -1.0, 0.5, 0.0],
            vec![1.0, 0.0, -0.25, 0.0, 0.0, 3.0],
        ];
        let predictions = predict_values(NativeValuePredictionRequest {
            model: model.clone(),
            features: features.clone(),
        })
        .expect("predict hidden-layer values on native GPU");
        let expected = model.predict_values(features.iter().map(Vec::as_slice));

        assert_eq!(predictions.len(), expected.len());
        for (actual, expected) in predictions.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 0.000_01);
        }
    }

    #[test]
    fn native_value_head_training_updates_output_weights() {
        let model = crate::gpu::training::CompactValueModel {
            version: 4,
            projection_size: 2,
            projection_seed: 789,
            hidden_layers: vec![],
            hidden_weights: vec![],
            output_weights: vec![0.0, 0.0, 0.0],
            policy_logits: vec![],
            policy_weights: vec![],
            auxiliary_value_weights: vec![],
            scale: 1.0,
            bias: 0.0,
            output_activation: crate::gpu::training::OutputActivation::Tanh,
        };
        let samples = vec![
            crate::gpu::training::TrainingSample {
                side_to_move: None,
                board_count: None,
                position_key: None,
                features: vec![],
                label: 0.8,
                label_kind: None,
                label_weight: 1.0,
                base_label_weight: None,
                label_mass: None,
                observation_count: None,
                policy: None,
                pseudo: None,
            },
            crate::gpu::training::TrainingSample {
                side_to_move: None,
                board_count: None,
                position_key: None,
                features: vec![],
                label: -0.8,
                label_kind: None,
                label_weight: 1.0,
                base_label_weight: None,
                label_mass: None,
                observation_count: None,
                policy: None,
                pseudo: None,
            },
        ];
        let (trained, report) = train_value_head(NativeValueHeadTrainingRequest {
            model,
            samples,
            projected_features: vec![1.0, 0.0, -1.0, 0.0],
            config: crate::gpu::training::ValueHeadTrainingConfig {
                learning_rate: 0.1,
                epochs: 32,
                weight_decay: 0.0,
                momentum: 0.0,
            },
            train_hidden_layers: false,
        })
        .expect("train value head on native GPU");

        assert_eq!(report.samples, 2);
        assert_eq!(report.epochs, 32);
        assert!(report.final_loss < report.initial_loss);
        assert!(trained.output_weights[0] > 0.0);
    }

    #[test]
    fn native_value_training_updates_hidden_layer_weights() {
        let initial_hidden_weights = vec![0.1, 0.0, 0.0, 0.0, 0.1, 0.0];
        let model = crate::gpu::training::CompactValueModel {
            version: 4,
            projection_size: 2,
            projection_seed: 789,
            hidden_layers: vec![2],
            hidden_weights: initial_hidden_weights.clone(),
            output_weights: vec![0.1, -0.1, 0.0],
            policy_logits: vec![],
            policy_weights: vec![],
            auxiliary_value_weights: vec![],
            scale: 1.0,
            bias: 0.0,
            output_activation: crate::gpu::training::OutputActivation::Tanh,
        };
        let samples = vec![
            crate::gpu::training::TrainingSample {
                side_to_move: None,
                board_count: None,
                position_key: None,
                features: vec![],
                label: 0.8,
                label_kind: None,
                label_weight: 1.0,
                base_label_weight: None,
                label_mass: None,
                observation_count: None,
                policy: None,
                pseudo: None,
            },
            crate::gpu::training::TrainingSample {
                side_to_move: None,
                board_count: None,
                position_key: None,
                features: vec![],
                label: -0.8,
                label_kind: None,
                label_weight: 1.0,
                base_label_weight: None,
                label_mass: None,
                observation_count: None,
                policy: None,
                pseudo: None,
            },
        ];
        let (trained, report) = train_value_head(NativeValueHeadTrainingRequest {
            model,
            samples,
            projected_features: vec![1.0, 0.0, 0.0, 1.0],
            config: crate::gpu::training::ValueHeadTrainingConfig {
                learning_rate: 0.1,
                epochs: 16,
                weight_decay: 0.0,
                momentum: 0.0,
            },
            train_hidden_layers: true,
        })
        .expect("train hidden layer on native GPU");

        assert_eq!(report.samples, 2);
        assert_eq!(report.epochs, 16);
        assert!(report.final_loss < report.initial_loss);
        assert_ne!(trained.hidden_weights, initial_hidden_weights);
    }

    #[test]
    fn native_policy_head_training_updates_policy_weights() {
        let model = crate::gpu::training::CompactValueModel {
            version: 4,
            projection_size: 2,
            projection_seed: 789,
            hidden_layers: vec![],
            hidden_weights: vec![],
            output_weights: vec![0.0, 0.0, 0.0],
            policy_logits: vec![],
            policy_weights: vec![],
            auxiliary_value_weights: vec![],
            scale: 1.0,
            bias: 0.0,
            output_activation: crate::gpu::training::OutputActivation::Tanh,
        };
        let samples = vec![
            crate::gpu::training::TrainingSample {
                side_to_move: None,
                board_count: None,
                position_key: None,
                features: vec![],
                label: 0.0,
                label_kind: None,
                label_weight: 1.0,
                base_label_weight: None,
                label_mass: None,
                observation_count: None,
                policy: Some(3),
                pseudo: None,
            },
            crate::gpu::training::TrainingSample {
                side_to_move: None,
                board_count: None,
                position_key: None,
                features: vec![],
                label: 0.0,
                label_kind: None,
                label_weight: 1.0,
                base_label_weight: None,
                label_mass: None,
                observation_count: None,
                policy: Some(7),
                pseudo: None,
            },
        ];
        let (trained, report) = train_policy_head(NativePolicyHeadTrainingRequest {
            model,
            samples,
            projected_features: vec![1.0, 0.0, -1.0, 0.0],
            config: crate::gpu::training::ValueHeadTrainingConfig {
                learning_rate: 0.1,
                epochs: 64,
                weight_decay: 0.0,
                momentum: 0.0,
            },
        })
        .expect("train policy head on native GPU");

        assert_eq!(report.samples, 2);
        assert_eq!(report.steps, 16);
        assert!(report.final_loss < report.initial_loss);
        assert_eq!(
            trained.policy_weights.len(),
            crate::gpu::training::POLICY_BUCKETS as usize * 3
        );
        assert!(trained.policy_weights.iter().any(|weight| *weight != 0.0));
    }
}
