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
const EXCITATION_GAIN_RANGE: (f32, f32) = (0.1, 20.0);
const OUTPUT_GAIN_RANGE: (f32, f32) = (0.05, 4.0);
const BREATH_PRESSURE_RANGE: (f32, f32) = (0.0, 4.0);
const BREATH_FEEDBACK_RANGE: (f32, f32) = (0.0, 1.5);
const BREATH_AUTO_LEVEL_RANGE: (f32, f32) = (0.0, 20.0);
const BREATH_NOISE_CUTOFF_HZ_RANGE: (f32, f32) = (50.0, 8000.0);
const BREATH_EMBOUCHURE_DELAY_MS_RANGE: (f32, f32) = (0.0, 10.0);
const BREATH_EMBOUCHURE_Q_RANGE: (f32, f32) = (0.3, 12.0);
const BREATH_EMBOUCHURE_TUNE_RATIO_RANGE: (f32, f32) = (0.85, 1.15);
const MAX_EMBOUCHURE_DELAY_MS: f32 = 12.0;
pub const DEFAULT_INHARMONICITY: f32 = 1.0;
pub const DEFAULT_POSITION_COUPLING: f32 = 0.0;
pub const DEFAULT_VELOCITY_COUPLING: f32 = 0.0;
pub const DEFAULT_EXCITATION_GAIN: f32 = 6.0;
pub const DEFAULT_OUTPUT_GAIN: f32 = 1.0;
pub const DEFAULT_BREATH_PRESSURE: f32 = 1.0;
pub const DEFAULT_BREATH_FEEDBACK: f32 = 0.8;
pub const DEFAULT_BREATH_AUTO_LEVEL: f32 = 6.0;
pub const DEFAULT_BREATH_NOISE_CUTOFF_HZ: f32 = 400.0;
pub const DEFAULT_BREATH_EMBOUCHURE_DELAY_MS: f32 = 1.2;
pub const DEFAULT_BREATH_EMBOUCHURE_Q: f32 = 2.0;
pub const DEFAULT_BREATH_EMBOUCHURE_TUNE_RATIO: f32 = 1.0;

#[derive(Debug, Clone, Copy)]
struct SvfBandpass {
    ic1eq: f32,
    ic2eq: f32,
    g: f32,
    k: f32,
}

impl SvfBandpass {
    fn new(sample_rate_hz: f32, center_hz: f32, q: f32) -> Self {
        let mut f = Self {
            ic1eq: 0.0,
            ic2eq: 0.0,
            g: 0.0,
            k: 1.0,
        };
        f.set_params(sample_rate_hz, center_hz, q);
        f
    }

    fn reset(&mut self) {
        self.ic1eq = 0.0;
        self.ic2eq = 0.0;
    }

    fn set_params(&mut self, sample_rate_hz: f32, center_hz: f32, q: f32) {
        let sr = sample_rate_hz.max(1.0);
        let nyquist = 0.49 * sr;
        let fc = center_hz.clamp(1.0, nyquist);
        let q = q.max(0.01);

        // g = tan(pi * fc / fs)
        self.g = (core::f32::consts::PI * fc / sr).tan();
        self.k = 1.0 / q;
    }

    #[inline]
    fn process(&mut self, v0: f32) -> f32 {
        // Andrew Simper-style SVF bandpass output.
        let g = self.g;
        let k = self.k;
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;
        let v3 = v0 - self.ic2eq;
        let v1 = a1 * self.ic1eq + a2 * v3;
        let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;
        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;
        v1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExciterMode {
    Strike,
    Breath,
}

impl ExciterMode {
    pub fn as_u8(self) -> u8 {
        match self {
            ExciterMode::Strike => 0,
            ExciterMode::Breath => 1,
        }
    }

    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => ExciterMode::Breath,
            _ => ExciterMode::Strike,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StrikeExciter {
    attack_samples: u32,
    release_samples: u32,
    attack_pos: u32,
    release_pos: u32,
    pulse_amp: f32,
}

impl StrikeExciter {
    fn new(sample_rate_hz: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            attack_samples: samples_from_ms(sample_rate_hz, attack_ms),
            release_samples: samples_from_ms(sample_rate_hz, release_ms),
            attack_pos: 0,
            release_pos: 0,
            pulse_amp: 0.0,
        }
    }

    fn set_attack_ms(&mut self, sample_rate_hz: f32, attack_ms: f32) {
        self.attack_samples = samples_from_ms(sample_rate_hz, attack_ms);
    }

    fn set_release_ms(&mut self, sample_rate_hz: f32, release_ms: f32) {
        self.release_samples = samples_from_ms(sample_rate_hz, release_ms);
    }

    fn trigger(&mut self, pulse_amp: f32) {
        self.pulse_amp = pulse_amp;
        self.attack_pos = 0;
        self.release_pos = 0;
    }

    #[inline]
    fn sample(&mut self) -> f32 {
        if self.pulse_amp == 0.0 {
            return 0.0;
        }

        if self.attack_pos < self.attack_samples {
            self.attack_pos = self.attack_pos.saturating_add(1);
            let t = self.attack_pos as f32 / self.attack_samples as f32;
            return self.pulse_amp * t;
        }

        if self.release_pos < self.release_samples {
            self.release_pos = self.release_pos.saturating_add(1);
            let remaining = (self.release_samples - self.release_pos) as f32;
            let t = remaining / self.release_samples as f32;
            return self.pulse_amp * t;
        }

        self.pulse_amp = 0.0;
        0.0
    }
}

#[derive(Debug, Clone, Copy)]
struct BreathExciter {
    attack_samples: u32,
    release_samples: u32,
    env: f32,
    noise_state: u32,
    noise_lp: f32,
}

impl BreathExciter {
    fn new(sample_rate_hz: f32, attack_ms: f32, release_ms: f32) -> Self {
        Self {
            attack_samples: samples_from_ms(sample_rate_hz, attack_ms),
            release_samples: samples_from_ms(sample_rate_hz, release_ms),
            env: 0.0,
            noise_state: 0x1234_5678,
            noise_lp: 0.0,
        }
    }

    fn set_attack_ms(&mut self, sample_rate_hz: f32, attack_ms: f32) {
        self.attack_samples = samples_from_ms(sample_rate_hz, attack_ms);
    }

    fn set_release_ms(&mut self, sample_rate_hz: f32, release_ms: f32) {
        self.release_samples = samples_from_ms(sample_rate_hz, release_ms);
    }

    fn reset(&mut self) {
        self.env = 0.0;
        self.noise_lp = 0.0;
    }

    #[inline]
    fn sample(
        &mut self,
        gate: bool,
        pressure: f32,
        feedback_signal: f32,
        output_level: f32,
        feedback_amount: f32,
        auto_level_strength: f32,
        noise_alpha: f32,
    ) -> f32 {
        // Pressure envelope (linear ramps in samples for simplicity).
        let target = if gate { 1.0 } else { 0.0 };
        let step = if gate {
            1.0 / self.attack_samples.max(1) as f32
        } else {
            1.0 / self.release_samples.max(1) as f32
        };
        self.env += (target - self.env) * step;

        // Simple xorshift noise in [-1, 1].
        let mut x = self.noise_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.noise_state = x;
        let n = (x as f32 / u32::MAX as f32) * 2.0 - 1.0;

        // Mild low-pass to remove extreme HF harshness (alpha derived from cutoff Hz).
        self.noise_lp += (n - self.noise_lp) * noise_alpha;

        // Nonlinear feedback + auto-leveling so more pressure doesn't just mean "louder".
        // This is not a full physical model (jet/reed), but gives a stable, controllable drive.
        let u = self.noise_lp + feedback_amount * feedback_signal;
        let nonlin = (pressure * u).tanh();
        let anti_gain = 1.0 / (1.0 + auto_level_strength * output_level);

        self.env * nonlin * anti_gain
    }
}

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
    exciter_mode: ExciterMode,
    strike_exciter: StrikeExciter,
    breath_exciter: BreathExciter,
    last_output: f32,
    output_level: f32,
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
    breath_pressure: f32,
    breath_feedback: f32,
    breath_auto_level: f32,
    breath_noise_cutoff_hz: f32,
    breath_noise_alpha: f32,
    breath_embouchure_delay_ms: f32,
    breath_embouchure_delay_samples: usize,
    breath_embouchure_q: f32,
    breath_embouchure_tune_ratio: f32,
    breath_feedback_filter: SvfBandpass,
    breath_feedback_delay: Vec<f32>,
    breath_feedback_write_idx: usize,
    inharmonicity: f32,
    position_coupling_base: f32,
    velocity_coupling_base: f32,
}

impl Engine {
    pub fn new(sample_rate_hz: f32, params: EngineParams) -> Self {
        let sample_rate_hz = sample_rate_hz.max(1.0);
        let strike_exciter = StrikeExciter::new(sample_rate_hz, 2.0, 40.0);
        let breath_exciter = BreathExciter::new(sample_rate_hz, 20.0, 80.0);
        let delay_len = samples_from_ms(sample_rate_hz, MAX_EMBOUCHURE_DELAY_MS) as usize + 2;
        let feedback_filter = SvfBandpass::new(
            sample_rate_hz,
            params.frequency_hz.max(1.0),
            DEFAULT_BREATH_EMBOUCHURE_Q,
        );
        let mut engine = Self {
            sample_rate_hz,
            inv_sample_rate: 1.0 / sample_rate_hz,
            params,
            gate: false,
            exciter_mode: ExciterMode::Strike,
            strike_exciter,
            breath_exciter,
            last_output: 0.0,
            output_level: 0.0,
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
            excitation_gain: DEFAULT_EXCITATION_GAIN,
            output_gain: DEFAULT_OUTPUT_GAIN,
            breath_pressure: DEFAULT_BREATH_PRESSURE,
            breath_feedback: DEFAULT_BREATH_FEEDBACK,
            breath_auto_level: DEFAULT_BREATH_AUTO_LEVEL,
            breath_noise_cutoff_hz: DEFAULT_BREATH_NOISE_CUTOFF_HZ,
            breath_noise_alpha: 0.0,
            breath_embouchure_delay_ms: DEFAULT_BREATH_EMBOUCHURE_DELAY_MS,
            breath_embouchure_delay_samples: 0,
            breath_embouchure_q: DEFAULT_BREATH_EMBOUCHURE_Q,
            breath_embouchure_tune_ratio: DEFAULT_BREATH_EMBOUCHURE_TUNE_RATIO,
            breath_feedback_filter: feedback_filter,
            breath_feedback_delay: vec![0.0; delay_len],
            breath_feedback_write_idx: 0,
            inharmonicity: DEFAULT_INHARMONICITY,
            position_coupling_base: DEFAULT_POSITION_COUPLING,
            velocity_coupling_base: DEFAULT_VELOCITY_COUPLING,
        };
        engine.set_breath_noise_cutoff_hz(DEFAULT_BREATH_NOISE_CUTOFF_HZ);
        engine.set_breath_embouchure_delay_ms(DEFAULT_BREATH_EMBOUCHURE_DELAY_MS);
        engine.set_breath_embouchure_q(DEFAULT_BREATH_EMBOUCHURE_Q);
        engine.set_breath_embouchure_tune_ratio(DEFAULT_BREATH_EMBOUCHURE_TUNE_RATIO);
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
        self.update_breath_embouchure_filter();
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
    pub fn set_exciter_mode(&mut self, mode: ExciterMode) {
        if mode == self.exciter_mode {
            return;
        }
        self.exciter_mode = mode;
        self.last_output = 0.0;
        self.output_level = 0.0;
        self.breath_exciter.reset();
        self.breath_feedback_filter.reset();
        self.breath_feedback_delay.fill(0.0);
        self.breath_feedback_write_idx = 0;
    }

    #[inline]
    pub fn set_attack_ms(&mut self, attack_ms: f32) {
        let attack_ms = attack_ms.max(0.000_1);
        self.strike_exciter
            .set_attack_ms(self.sample_rate_hz, attack_ms);
        self.breath_exciter
            .set_attack_ms(self.sample_rate_hz, attack_ms);
    }

    #[inline]
    pub fn set_release_ms(&mut self, release_ms: f32) {
        let release_ms = release_ms.max(0.000_1);
        self.strike_exciter
            .set_release_ms(self.sample_rate_hz, release_ms);
        self.breath_exciter
            .set_release_ms(self.sample_rate_hz, release_ms);
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
    pub fn set_excitation_gain(&mut self, value: f32) {
        let (min, max) = EXCITATION_GAIN_RANGE;
        self.excitation_gain = value.clamp(min, max);
    }

    #[inline]
    pub fn set_output_gain(&mut self, value: f32) {
        let (min, max) = OUTPUT_GAIN_RANGE;
        let clamped = value.clamp(min, max);
        self.output_gain = clamped;
    }

    #[inline]
    pub fn set_breath_pressure(&mut self, value: f32) {
        let (min, max) = BREATH_PRESSURE_RANGE;
        self.breath_pressure = value.clamp(min, max);
    }

    #[inline]
    pub fn set_breath_feedback(&mut self, value: f32) {
        let (min, max) = BREATH_FEEDBACK_RANGE;
        self.breath_feedback = value.clamp(min, max);
    }

    #[inline]
    pub fn set_breath_auto_level(&mut self, value: f32) {
        let (min, max) = BREATH_AUTO_LEVEL_RANGE;
        self.breath_auto_level = value.clamp(min, max);
    }

    #[inline]
    pub fn set_breath_noise_cutoff_hz(&mut self, value: f32) {
        let (min, max) = BREATH_NOISE_CUTOFF_HZ_RANGE;
        let clamped = value.clamp(min, max);
        let max_for_sr = (0.45 * self.sample_rate_hz).max(min);
        let cutoff = clamped.min(max_for_sr);
        self.breath_noise_cutoff_hz = cutoff;

        // One-pole low-pass alpha for cutoff Hz: alpha = 1 - exp(-2π fc / fs)
        let exp_term = (-core::f32::consts::TAU * cutoff / self.sample_rate_hz).exp();
        self.breath_noise_alpha = (1.0 - exp_term).clamp(0.0, 1.0);
    }

    #[inline]
    pub fn set_breath_embouchure_delay_ms(&mut self, value: f32) {
        let (min, max) = BREATH_EMBOUCHURE_DELAY_MS_RANGE;
        let clamped = value.clamp(min, max);
        self.breath_embouchure_delay_ms = clamped;
        self.breath_embouchure_delay_samples =
            samples_from_ms(self.sample_rate_hz, clamped).saturating_sub(1) as usize;
    }

    #[inline]
    pub fn set_breath_embouchure_q(&mut self, value: f32) {
        let (min, max) = BREATH_EMBOUCHURE_Q_RANGE;
        self.breath_embouchure_q = value.clamp(min, max);
        self.update_breath_embouchure_filter();
    }

    #[inline]
    pub fn set_breath_embouchure_tune_ratio(&mut self, value: f32) {
        let (min, max) = BREATH_EMBOUCHURE_TUNE_RATIO_RANGE;
        self.breath_embouchure_tune_ratio = value.clamp(min, max);
        self.update_breath_embouchure_filter();
    }

    fn update_breath_embouchure_filter(&mut self) {
        let center_hz = (self.params.frequency_hz * self.breath_embouchure_tune_ratio).max(1.0);
        self.breath_feedback_filter
            .set_params(self.sample_rate_hz, center_hz, self.breath_embouchure_q);
    }

    #[inline]
    pub fn trigger(&mut self, amplitude: f32) {
        match self.exciter_mode {
            ExciterMode::Strike => {
                for mode in 0..MAX_MODES {
                    self.positions[mode] = 0.0;
                    self.velocities[mode] = 0.0;
                }

                let pulse_amp = amplitude.max(0.0).min(1.0) * self.excitation_gain;
                self.strike_exciter.trigger(pulse_amp);
            }
            ExciterMode::Breath => {
                // For continuous excitation, avoid hard resets.
                self.last_output = 0.0;
                self.output_level = 0.0;
            }
        }
    }

    /// Semi-implicit Euler integration for a mass-spring-damper driven by `amplitude`.
    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let pressure =
            self.params.amplitude.max(0.0).min(1.0) * self.excitation_gain * self.breath_pressure;

        let delayed = if self.breath_embouchure_delay_samples == 0 {
            self.last_output
        } else {
            let len = self.breath_feedback_delay.len().max(1);
            let d = self.breath_embouchure_delay_samples.min(len - 1);
            let read_idx = (self.breath_feedback_write_idx + len - d) % len;
            self.breath_feedback_delay[read_idx]
        };
        let feedback_signal = self.breath_feedback_filter.process(delayed);

        let exciter_level = match self.exciter_mode {
            ExciterMode::Strike => self.strike_exciter.sample(),
            ExciterMode::Breath => self.breath_exciter.sample(
                self.gate,
                pressure,
                feedback_signal,
                self.output_level,
                self.breath_feedback,
                self.breath_auto_level,
                self.breath_noise_alpha,
            ),
        };

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

            let drive = exciter_level * self.stiffness[mode];
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

        if !self.breath_feedback_delay.is_empty() {
            self.breath_feedback_delay[self.breath_feedback_write_idx] = sample;
            self.breath_feedback_write_idx += 1;
            if self.breath_feedback_write_idx >= self.breath_feedback_delay.len() {
                self.breath_feedback_write_idx = 0;
            }
        }

        // Update level estimate before applying output gain so "anti loudness" isn't affected by it.
        let abs = sample.abs();
        self.output_level += (abs - self.output_level) * 0.01;
        self.last_output = sample;

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
fn samples_from_ms(sample_rate_hz: f32, time_ms: f32) -> u32 {
    let time_s = (time_ms / 1000.0).max(0.000_001);
    let samples = (time_s * sample_rate_hz).round();
    samples.max(1.0) as u32
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
