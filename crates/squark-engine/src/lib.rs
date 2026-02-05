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

const MAX_MODES: usize = 10;
const POSITION_COUPLING_CLAMP: f32 = 1.0;
const VELOCITY_COUPLING_CLAMP: f32 = 1.0;
const OUTPUT_GAIN_RANGE: (f32, f32) = (0.05, 4.0);
pub const DEFAULT_INHARMONICITY: f32 = 1.0;
pub const DEFAULT_POSITION_COUPLING: f32 = 0.08;
pub const DEFAULT_VELOCITY_COUPLING: f32 = 0.02;
pub const DEFAULT_OUTPUT_GAIN: f32 = 1.0;

/// Realtime-safe mass-spring-damper engine integrated with semi-implicit Euler.
///
/// Design constraints:
/// - No allocations or locks.
/// - Deterministic per-sample work (one state update per `next_sample`).
#[derive(Debug)]
pub struct Engine {
    sample_rate_hz: f32,
    inv_sample_rate: f32,
    params: EngineParams,
    gate: bool,
    env: f32,
    attack_coeff: f32,
    release_coeff: f32,
    positions: [f32; MAX_MODES],
    velocities: [f32; MAX_MODES],
    mass: [f32; MAX_MODES],
    stiffness: [f32; MAX_MODES],
    mode_frequency: [f32; MAX_MODES],
    damping_ratio: f32,
    damping: [f32; MAX_MODES],
    mix_weights: [f32; MAX_MODES],
    coupling_position: [[f32; MAX_MODES]; MAX_MODES],
    coupling_velocity: [[f32; MAX_MODES]; MAX_MODES],
    excitation_gain: f32,
    output_gain: f32,
    inharmonicity: f32,
    position_coupling_base: f32,
    velocity_coupling_base: f32,
}

impl Engine {
    pub fn new(sample_rate_hz: f32, params: EngineParams) -> Self {
        let sample_rate_hz = sample_rate_hz.max(1.0);
        let attack_coeff = coeff_from_ms(sample_rate_hz, 2.0);
        let release_coeff = coeff_from_ms(sample_rate_hz, 40.0);
        let mut engine = Self {
            sample_rate_hz,
            inv_sample_rate: 1.0 / sample_rate_hz,
            params,
            gate: false,
            env: 0.0,
            attack_coeff,
            release_coeff,
            positions: [0.0; MAX_MODES],
            velocities: [0.0; MAX_MODES],
            mass: [1.0; MAX_MODES],
            stiffness: [0.0; MAX_MODES],
            mode_frequency: [0.0; MAX_MODES],
            damping_ratio: 0.02,
            damping: [0.0; MAX_MODES],
            mix_weights: [0.0; MAX_MODES],
            coupling_position: [[0.0; MAX_MODES]; MAX_MODES],
            coupling_velocity: [[0.0; MAX_MODES]; MAX_MODES],
            excitation_gain: 6.0,
            output_gain: DEFAULT_OUTPUT_GAIN,
            inharmonicity: DEFAULT_INHARMONICITY,
            position_coupling_base: DEFAULT_POSITION_COUPLING,
            velocity_coupling_base: DEFAULT_VELOCITY_COUPLING,
        };
        engine.rebuild_modes();
        engine
    }

    pub fn sample_rate_hz(&self) -> f32 {
        self.sample_rate_hz
    }

    pub fn params(&self) -> EngineParams {
        self.params
    }

    #[inline]
    pub fn set_frequency_hz(&mut self, frequency_hz: f32) {
        let frequency_hz = frequency_hz.max(1.0);
        self.params.frequency_hz = frequency_hz;
        self.rebuild_modes();
    }

    #[inline]
    pub fn set_amplitude(&mut self, amplitude: f32) {
        self.params.amplitude = amplitude;
    }

    #[inline]
    pub fn set_gate(&mut self, gate: bool) {
        self.gate = gate;
    }

    #[inline]
    pub fn set_attack_ms(&mut self, attack_ms: f32) {
        self.attack_coeff = coeff_from_ms(self.sample_rate_hz, attack_ms);
    }

    #[inline]
    pub fn set_release_ms(&mut self, release_ms: f32) {
        self.release_coeff = coeff_from_ms(self.sample_rate_hz, release_ms);
    }

    #[inline]
    pub fn set_damping_ratio(&mut self, damping_ratio: f32) {
        let clamped = damping_ratio.clamp(0.000_1, 10.0);
        self.damping_ratio = clamped;
        self.update_damping();
    }

    #[inline]
    pub fn set_inharmonicity(&mut self, inharmonicity: f32) {
        let clamped = inharmonicity.max(0.1);
        if (clamped - self.inharmonicity).abs() < 0.0001 {
            return;
        }
        self.inharmonicity = clamped;
        self.rebuild_modes();
    }

    #[inline]
    pub fn set_position_coupling_base(&mut self, value: f32) {
        let clamped = value.clamp(-POSITION_COUPLING_CLAMP, POSITION_COUPLING_CLAMP);
        if (clamped - self.position_coupling_base).abs() < 0.0001 {
            return;
        }
        self.position_coupling_base = clamped;
        self.rebuild_coupling();
    }

    #[inline]
    pub fn set_velocity_coupling_base(&mut self, value: f32) {
        let clamped = value.clamp(-VELOCITY_COUPLING_CLAMP, VELOCITY_COUPLING_CLAMP);
        if (clamped - self.velocity_coupling_base).abs() < 0.0001 {
            return;
        }
        self.velocity_coupling_base = clamped;
        self.rebuild_coupling();
    }

    #[inline]
    pub fn set_output_gain(&mut self, value: f32) {
        let (min, max) = OUTPUT_GAIN_RANGE;
        let clamped = value.clamp(min, max);
        self.output_gain = clamped;
    }

    #[inline]
    pub fn trigger(&mut self, amplitude: f32) {
        let energy = amplitude.max(0.0).min(1.0) * self.excitation_gain;
        for mode in 0..MAX_MODES {
            self.positions[mode] = 0.0;
            let harmonic = (mode + 1) as f32;
            let scale = 1.0 / harmonic;
            self.velocities[mode] = energy * scale;
        }
    }

    /// Semi-implicit Euler integration for a mass-spring-damper driven by `amplitude`.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let target = if self.gate {
            self.params.amplitude
        } else {
            0.0
        };
        let coeff = if self.gate {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.env += (target - self.env) * coeff;

        let prev_positions = self.positions;
        let prev_velocities = self.velocities;

        for mode in 0..MAX_MODES {
            let mut coupling_force = 0.0;
            for other in 0..MAX_MODES {
                if mode == other {
                    continue;
                }
                let pos_diff = prev_positions[other] - prev_positions[mode];
                let vel_diff = prev_velocities[other] - prev_velocities[mode];
                coupling_force += self.coupling_position[mode][other] * pos_diff;
                coupling_force += self.coupling_velocity[mode][other] * vel_diff;
            }

            let drive = self.env * self.stiffness[mode];
            let restoring = self.stiffness[mode] * prev_positions[mode];
            let damping_force = self.damping[mode] * prev_velocities[mode];
            let acceleration =
                (drive - restoring - damping_force + coupling_force) / self.mass[mode];

            let new_velocity = prev_velocities[mode] + acceleration * self.inv_sample_rate;
            self.velocities[mode] = new_velocity;
        }

        for mode in 0..MAX_MODES {
            self.positions[mode] += self.velocities[mode] * self.inv_sample_rate;
            self.positions[mode] = self.positions[mode].clamp(-2.0, 2.0);
        }

        let mut sample = 0.0;
        for mode in 0..MAX_MODES {
            sample += self.positions[mode] * self.mix_weights[mode];
        }

        (sample * self.output_gain).clamp(-1.0, 1.0)
    }

    fn rebuild_modes(&mut self) {
        let max_freq = self.max_mode_frequency();
        let mut weights = [0.0; MAX_MODES];
        for mode in 0..MAX_MODES {
            let harmonic = (mode + 1) as f32;
            let stretched = harmonic.powf(self.inharmonicity);
            let target = (self.params.frequency_hz * stretched).min(max_freq);
            self.mode_frequency[mode] = target;
            self.stiffness[mode] = stiffness_for_frequency(target, self.mass[mode]);
            weights[mode] = 1.0 / harmonic;
        }
        self.update_damping();
        self.normalize_mix_weights(weights);
        self.rebuild_coupling();
    }

    fn max_mode_frequency(&self) -> f32 {
        (0.45 * self.sample_rate_hz).max(2000.0)
    }

    fn update_damping(&mut self) {
        for mode in 0..MAX_MODES {
            self.damping[mode] = damping_for_frequency(
                self.mode_frequency[mode],
                self.mass[mode],
                self.damping_ratio,
            );
        }
    }

    fn normalize_mix_weights(&mut self, candidates: [f32; MAX_MODES]) {
        let sum: f32 = candidates.iter().copied().sum::<f32>().max(0.000_1);
        for mode in 0..MAX_MODES {
            self.mix_weights[mode] = candidates[mode] / sum;
        }
    }

    fn rebuild_coupling(&mut self) {
        for i in 0..MAX_MODES {
            for j in 0..MAX_MODES {
                if i == j {
                    self.coupling_position[i][j] = 0.0;
                    self.coupling_velocity[i][j] = 0.0;
                    continue;
                }
                let distance = i.abs_diff(j) as f32;
                let falloff = 1.0 / (1.0 + distance);
                let stiffness_scale = 0.5 * (self.stiffness[i] + self.stiffness[j]);
                let damping_scale = 0.5 * (self.damping[i] + self.damping[j]);
                self.coupling_position[i][j] =
                    self.position_coupling_base * stiffness_scale * falloff;
                self.coupling_velocity[i][j] =
                    self.velocity_coupling_base * damping_scale * falloff;
            }
        }
    }
}

#[inline]
fn coeff_from_ms(sample_rate_hz: f32, time_ms: f32) -> f32 {
    let time_s = (time_ms / 1000.0).max(0.000_001);
    1.0 - (-1.0 / (time_s * sample_rate_hz)).exp()
}

#[inline]
fn stiffness_for_frequency(frequency_hz: f32, mass: f32) -> f32 {
    let omega = core::f32::consts::TAU * frequency_hz;
    mass * omega * omega
}

#[inline]
fn damping_for_frequency(frequency_hz: f32, mass: f32, damping_ratio: f32) -> f32 {
    let omega = core::f32::consts::TAU * frequency_hz.max(1.0);
    2.0 * damping_ratio * mass * omega
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_spring_produces_finite_output() {
        let mut engine = Engine::new(48_000.0, EngineParams::default());
        engine.set_gate(true);
        engine.trigger(0.2);

        for _ in 0..20_000 {
            let s = engine.next_sample();
            assert!(s.is_finite());
            assert!(s >= -1.0 && s <= 1.0);
        }
    }
}
