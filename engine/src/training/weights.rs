use super::*;
use crate::cpu::EvalWeights;

impl EvalWeights {
    pub(crate) fn to_json(self) -> String {
        serde_json::to_string(&self).expect("EvalWeights should serialize")
    }

    pub(crate) fn from_json(value: &str) -> Result<Self, String> {
        serde_json::from_str(value).map_err(|error| error.to_string())
    }
}

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.state
    }

    pub(crate) fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper.max(1)
    }

    pub(crate) fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}
