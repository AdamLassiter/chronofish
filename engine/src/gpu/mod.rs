pub mod search;
pub mod training;

#[cfg(not(target_arch = "wasm32"))]
pub mod cli;

#[cfg(all(not(target_arch = "wasm32"), feature = "neural-wgpu"))]
pub mod native;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WgslShader {
    pub name: &'static str,
    pub source: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuKernelSet {
    CpuSearch,
    GpuSearch,
    GpuTraining,
}

impl GpuKernelSet {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuSearch => "cpu-search",
            Self::GpuSearch => "gpu-search",
            Self::GpuTraining => "gpu-training",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuKernel {
    pub set: GpuKernelSet,
    pub label: &'static str,
    pub shader: &'static str,
    pub entry_point: &'static str,
    pub constants: &'static [(&'static str, u32)],
}
