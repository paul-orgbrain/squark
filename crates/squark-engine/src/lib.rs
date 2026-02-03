#[derive(Debug, Clone, Copy)]
pub struct EngineParams {
    pub frequency_hz: f32,
    pub amplitude: f32,
}

impl Default for EngineParams {
    fn default() -> Self {
        Self {
            frequency_hz: 440.0,
            amplitude: 0.15,
        }
    }
}

/// A tiny realtime-safe synthesis core.
///
/// Design constraints:
/// - No allocations.
/// - No locks.
/// - Deterministic per-sample work.
#[derive(Debug, Clone, Copy)]
pub struct Engine {
    sample_rate_hz: f32,
    phase_radians: f32,
    params: EngineParams,
}

impl Engine {
    pub fn new(sample_rate_hz: f32, params: EngineParams) -> Self {
        let sample_rate_hz = sample_rate_hz.max(1.0);
        Self {
            sample_rate_hz,
            phase_radians: 0.0,
            params,
        }
    }

    pub fn sample_rate_hz(&self) -> f32 {
        self.sample_rate_hz
    }

    pub fn params(&self) -> EngineParams {
        self.params
    }

    pub fn set_params(&mut self, params: EngineParams) {
        self.params = params;
    }

    /// Generate the next mono sample in the range [-1, 1].
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let phase_increment = (self.params.frequency_hz * core::f32::consts::TAU) / self.sample_rate_hz;
        self.phase_radians += phase_increment;

        if self.phase_radians >= core::f32::consts::TAU {
            self.phase_radians -= core::f32::consts::TAU;
        }

        (self.phase_radians.sin() * self.params.amplitude).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_reasonable_output() {
        let mut engine = Engine::new(48_000.0, EngineParams::default());

        for _ in 0..10_000 {
            let s = engine.next_sample();
            assert!(s.is_finite());
            assert!(s >= -1.0 && s <= 1.0);
        }
    }
}
