mod settings;
mod ui;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI8, AtomicU8, AtomicU32, Ordering};
use std::{env, sync::Arc};

use anyhow::Context as _;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use midir::{Ignore, MidiInput};
use parking_lot::Mutex;

use squark_engine::{
    DEFAULT_INHARMONICITY as ENGINE_DEFAULT_INHARMONICITY,
    DEFAULT_OUTPUT_GAIN as ENGINE_DEFAULT_OUTPUT_GAIN,
    DEFAULT_POSITION_COUPLING as ENGINE_DEFAULT_POSITION_COUPLING,
    DEFAULT_VELOCITY_COUPLING as ENGINE_DEFAULT_VELOCITY_COUPLING, Engine, EngineParams,
};

use settings::SettingsStore;

pub(crate) const DEFAULT_ATTACK_MS: f32 = 2.0;
pub(crate) const DEFAULT_RELEASE_MS: f32 = 400.0;
pub(crate) const DEFAULT_DAMPING_RATIO: f32 = 0.02;
pub(crate) const DEFAULT_INHARMONICITY: f32 = ENGINE_DEFAULT_INHARMONICITY;
pub(crate) const DEFAULT_POSITION_COUPLING: f32 = ENGINE_DEFAULT_POSITION_COUPLING;
pub(crate) const DEFAULT_VELOCITY_COUPLING: f32 = ENGINE_DEFAULT_VELOCITY_COUPLING;
pub(crate) const DEFAULT_OUTPUT_GAIN: f32 = ENGINE_DEFAULT_OUTPUT_GAIN;
const ATTACK_RANGE_MS: (f32, f32) = (0.5, 200.0);
const RELEASE_RANGE_MS: (f32, f32) = (20.0, 2000.0);
const DAMPING_RATIO_RANGE: (f32, f32) = (0.001, 0.2);
const INHARMONICITY_RANGE: (f32, f32) = (0.8, 1.4);
const POSITION_COUPLING_RANGE: (f32, f32) = (-1.0, 1.0);
const VELOCITY_COUPLING_RANGE: (f32, f32) = (-1.0, 1.0);
const OUTPUT_GAIN_RANGE: (f32, f32) = (0.05, 4.0);
const PEAK_HALF_LIFE_S: f32 = 0.5;
const PEAK_DECAY_TAU: f32 = PEAK_HALF_LIFE_S / core::f32::consts::LN_2;

#[derive(Debug)]
struct ControlBlock {
    gate: AtomicBool,
    frequency_bits: AtomicU32,
    amplitude_bits: AtomicU32,
    active_note: AtomicU8,
    trigger_counter: AtomicU32,
    attack_ms_bits: AtomicU32,
    release_ms_bits: AtomicU32,
    damping_ratio_bits: AtomicU32,
    peak_bits: AtomicU32,
    inharmonicity_bits: AtomicU32,
    position_coupling_bits: AtomicU32,
    velocity_coupling_bits: AtomicU32,
    output_gain_bits: AtomicU32,
}

impl ControlBlock {
    fn new(initial: EngineParams) -> Self {
        Self {
            gate: AtomicBool::new(false),
            frequency_bits: AtomicU32::new(initial.frequency_hz.to_bits()),
            amplitude_bits: AtomicU32::new(initial.amplitude.to_bits()),
            active_note: AtomicU8::new(0),
            trigger_counter: AtomicU32::new(0),
            attack_ms_bits: AtomicU32::new(DEFAULT_ATTACK_MS.to_bits()),
            release_ms_bits: AtomicU32::new(DEFAULT_RELEASE_MS.to_bits()),
            damping_ratio_bits: AtomicU32::new(DEFAULT_DAMPING_RATIO.to_bits()),
            peak_bits: AtomicU32::new(0.0f32.to_bits()),
            inharmonicity_bits: AtomicU32::new(DEFAULT_INHARMONICITY.to_bits()),
            position_coupling_bits: AtomicU32::new(DEFAULT_POSITION_COUPLING.to_bits()),
            velocity_coupling_bits: AtomicU32::new(DEFAULT_VELOCITY_COUPLING.to_bits()),
            output_gain_bits: AtomicU32::new(DEFAULT_OUTPUT_GAIN.to_bits()),
        }
    }

    #[inline]
    fn snapshot(&self) -> ControlSnapshot {
        ControlSnapshot {
            gate: self.gate.load(Ordering::Relaxed),
            frequency_hz: f32::from_bits(self.frequency_bits.load(Ordering::Relaxed)),
            amplitude: f32::from_bits(self.amplitude_bits.load(Ordering::Relaxed)),
            trigger_counter: self.trigger_counter.load(Ordering::Relaxed),
            attack_ms: f32::from_bits(self.attack_ms_bits.load(Ordering::Relaxed)),
            release_ms: f32::from_bits(self.release_ms_bits.load(Ordering::Relaxed)),
            damping_ratio: f32::from_bits(self.damping_ratio_bits.load(Ordering::Relaxed)),
            inharmonicity: f32::from_bits(self.inharmonicity_bits.load(Ordering::Relaxed)),
            position_coupling: f32::from_bits(self.position_coupling_bits.load(Ordering::Relaxed)),
            velocity_coupling: f32::from_bits(self.velocity_coupling_bits.load(Ordering::Relaxed)),
            output_gain: f32::from_bits(self.output_gain_bits.load(Ordering::Relaxed)),
        }
    }

    #[inline]
    fn trigger_counter(&self) -> u32 {
        self.trigger_counter.load(Ordering::Relaxed)
    }

    #[inline]
    fn attack_ms(&self) -> f32 {
        f32::from_bits(self.attack_ms_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_attack_ms(&self, attack_ms: f32) {
        self.attack_ms_bits
            .store(attack_ms.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn release_ms(&self) -> f32 {
        f32::from_bits(self.release_ms_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_release_ms(&self, release_ms: f32) {
        self.release_ms_bits
            .store(release_ms.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn damping_ratio(&self) -> f32 {
        f32::from_bits(self.damping_ratio_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_damping_ratio(&self, ratio: f32) {
        self.damping_ratio_bits
            .store(ratio.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn inharmonicity(&self) -> f32 {
        f32::from_bits(self.inharmonicity_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_inharmonicity(&self, value: f32) {
        self.inharmonicity_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn position_coupling(&self) -> f32 {
        f32::from_bits(self.position_coupling_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_position_coupling(&self, value: f32) {
        self.position_coupling_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn velocity_coupling(&self) -> f32 {
        f32::from_bits(self.velocity_coupling_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_velocity_coupling(&self, value: f32) {
        self.velocity_coupling_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn output_gain(&self) -> f32 {
        f32::from_bits(self.output_gain_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn set_output_gain(&self, value: f32) {
        self.output_gain_bits
            .store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn peak_level(&self) -> f32 {
        f32::from_bits(self.peak_bits.load(Ordering::Relaxed))
    }

    #[inline]
    fn update_peak(&self, block_peak: f32, block_duration_s: f32) {
        if !block_peak.is_finite() {
            return;
        }
        let decay = if block_duration_s <= 0.0 {
            0.0
        } else {
            (-block_duration_s / PEAK_DECAY_TAU).exp()
        };
        let current = self.peak_level();
        let decayed = current * decay;
        let next = block_peak.max(decayed);
        self.peak_bits
            .store(next.clamp(0.0, 1.5).to_bits(), Ordering::Relaxed);
    }

    #[inline]
    fn parameter_value(&self, id: ParameterId) -> f32 {
        match id {
            ParameterId::AttackMs => self.attack_ms(),
            ParameterId::ReleaseMs => self.release_ms(),
            ParameterId::DampingRatio => self.damping_ratio(),
            ParameterId::Inharmonicity => self.inharmonicity(),
            ParameterId::PositionCoupling => self.position_coupling(),
            ParameterId::VelocityCoupling => self.velocity_coupling(),
            ParameterId::OutputGain => self.output_gain(),
        }
    }

    #[inline]
    fn set_parameter_value(&self, id: ParameterId, value: f32) {
        match id {
            ParameterId::AttackMs => self.set_attack_ms(value),
            ParameterId::ReleaseMs => self.set_release_ms(value),
            ParameterId::DampingRatio => self.set_damping_ratio(value),
            ParameterId::Inharmonicity => self.set_inharmonicity(value),
            ParameterId::PositionCoupling => self.set_position_coupling(value),
            ParameterId::VelocityCoupling => self.set_velocity_coupling(value),
            ParameterId::OutputGain => self.set_output_gain(value),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ControlSnapshot {
    gate: bool,
    frequency_hz: f32,
    amplitude: f32,
    trigger_counter: u32,
    attack_ms: f32,
    release_ms: f32,
    damping_ratio: f32,
    inharmonicity: f32,
    position_coupling: f32,
    velocity_coupling: f32,
    output_gain: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterId {
    AttackMs,
    ReleaseMs,
    DampingRatio,
    Inharmonicity,
    PositionCoupling,
    VelocityCoupling,
    OutputGain,
}

const PARAMETER_ORDER: [ParameterId; 7] = [
    ParameterId::AttackMs,
    ParameterId::ReleaseMs,
    ParameterId::DampingRatio,
    ParameterId::Inharmonicity,
    ParameterId::PositionCoupling,
    ParameterId::VelocityCoupling,
    ParameterId::OutputGain,
];

impl ParameterId {
    fn as_i8(self) -> i8 {
        match self {
            ParameterId::AttackMs => 0,
            ParameterId::ReleaseMs => 1,
            ParameterId::DampingRatio => 2,
            ParameterId::Inharmonicity => 3,
            ParameterId::PositionCoupling => 4,
            ParameterId::VelocityCoupling => 5,
            ParameterId::OutputGain => 6,
        }
    }

    fn from_i8(value: i8) -> Option<Self> {
        match value {
            0 => Some(ParameterId::AttackMs),
            1 => Some(ParameterId::ReleaseMs),
            2 => Some(ParameterId::DampingRatio),
            3 => Some(ParameterId::Inharmonicity),
            4 => Some(ParameterId::PositionCoupling),
            5 => Some(ParameterId::VelocityCoupling),
            6 => Some(ParameterId::OutputGain),
            _ => None,
        }
    }

    fn all() -> &'static [ParameterId] {
        &PARAMETER_ORDER
    }

    fn label(self) -> &'static str {
        match self {
            ParameterId::AttackMs => "Attack (ms)",
            ParameterId::ReleaseMs => "Release (ms)",
            ParameterId::DampingRatio => "Damping ratio",
            ParameterId::Inharmonicity => "Inharmonicity",
            ParameterId::PositionCoupling => "Position coupling",
            ParameterId::VelocityCoupling => "Velocity coupling",
            ParameterId::OutputGain => "Output gain",
        }
    }

    fn range(self) -> (f32, f32) {
        match self {
            ParameterId::AttackMs => ATTACK_RANGE_MS,
            ParameterId::ReleaseMs => RELEASE_RANGE_MS,
            ParameterId::DampingRatio => DAMPING_RATIO_RANGE,
            ParameterId::Inharmonicity => INHARMONICITY_RANGE,
            ParameterId::PositionCoupling => POSITION_COUPLING_RANGE,
            ParameterId::VelocityCoupling => VELOCITY_COUPLING_RANGE,
            ParameterId::OutputGain => OUTPUT_GAIN_RANGE,
        }
    }

    fn clamp(self, value: f32) -> f32 {
        let (min, max) = self.range();
        value.clamp(min, max)
    }

    fn slider_range(self) -> std::ops::RangeInclusive<f32> {
        let (min, max) = self.range();
        min..=max
    }

    fn logarithmic(self) -> bool {
        matches!(
            self,
            ParameterId::AttackMs
                | ParameterId::ReleaseMs
                | ParameterId::DampingRatio
                | ParameterId::OutputGain
        )
    }

    fn format_value(self, value: f32) -> String {
        match self {
            ParameterId::AttackMs | ParameterId::ReleaseMs => format!("{value:.1} ms"),
            ParameterId::DampingRatio => format!("{value:.3}"),
            ParameterId::Inharmonicity => format!("{value:.3}×"),
            ParameterId::PositionCoupling | ParameterId::VelocityCoupling => {
                format!("{value:.3}")
            }
            ParameterId::OutputGain => format!("{value:.2}×"),
        }
    }

    fn value_from_cc(self, cc_value: u8) -> f32 {
        let norm = (cc_value as f32 / 127.0).clamp(0.0, 1.0);
        let (min, max) = self.range();
        min + (max - min) * norm
    }

    fn storage_key(self) -> &'static str {
        match self {
            ParameterId::AttackMs => "attack_ms",
            ParameterId::ReleaseMs => "release_ms",
            ParameterId::DampingRatio => "damping_ratio",
            ParameterId::Inharmonicity => "inharmonicity",
            ParameterId::PositionCoupling => "position_coupling",
            ParameterId::VelocityCoupling => "velocity_coupling",
            ParameterId::OutputGain => "output_gain",
        }
    }

    fn from_storage_key(key: &str) -> Option<Self> {
        match key {
            "attack_ms" => Some(ParameterId::AttackMs),
            "release_ms" => Some(ParameterId::ReleaseMs),
            "damping_ratio" => Some(ParameterId::DampingRatio),
            "inharmonicity" => Some(ParameterId::Inharmonicity),
            "position_coupling" => Some(ParameterId::PositionCoupling),
            "velocity_coupling" => Some(ParameterId::VelocityCoupling),
            "output_gain" => Some(ParameterId::OutputGain),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct MidiLearnState {
    pending: AtomicI8,
    assignments: Mutex<HashMap<ParameterId, u8>>,
}

impl MidiLearnState {
    fn new() -> Self {
        Self {
            pending: AtomicI8::new(-1),
            assignments: Mutex::new(HashMap::new()),
        }
    }

    fn load_assignments(&self, pairs: &[(ParameterId, u8)]) {
        let mut assignments = self.assignments.lock();
        assignments.clear();
        for (param, cc) in pairs {
            assignments.insert(*param, *cc);
        }
    }

    fn request(&self, param: ParameterId) {
        self.pending.store(param.as_i8(), Ordering::Relaxed);
    }

    fn pending(&self) -> Option<ParameterId> {
        ParameterId::from_i8(self.pending.load(Ordering::Relaxed))
    }

    fn take_pending(&self) -> Option<ParameterId> {
        ParameterId::from_i8(self.pending.swap(-1, Ordering::Relaxed))
    }

    fn set_mapping(&self, param: ParameterId, cc: u8) {
        let mut assignments = self.assignments.lock();
        assignments.insert(param, cc);
    }

    fn mapping_for(&self, param: ParameterId) -> Option<u8> {
        self.assignments.lock().get(&param).copied()
    }

    fn clear_mapping(&self, param: ParameterId) -> bool {
        let mut assignments = self.assignments.lock();
        assignments.remove(&param).is_some()
    }

    fn param_for_cc(&self, cc: u8) -> Option<ParameterId> {
        self.assignments
            .lock()
            .iter()
            .find_map(|(&param, &mapped_cc)| if mapped_cc == cc { Some(param) } else { None })
    }
}

struct MidiContext {
    control: Arc<ControlBlock>,
    midi_learn: Arc<MidiLearnState>,
    settings: Arc<SettingsStore>,
}

impl MidiContext {
    fn new(
        control: Arc<ControlBlock>,
        midi_learn: Arc<MidiLearnState>,
        settings: Arc<SettingsStore>,
    ) -> Self {
        Self {
            control,
            midi_learn,
            settings,
        }
    }
}

#[derive(Debug, Default)]
struct CliOptions {
    list_midi: bool,
    midi_port: Option<usize>,
    sample_rate_hz: Option<u32>,
    buffer_frames: Option<u32>,
}

fn parse_cli() -> anyhow::Result<CliOptions> {
    let mut opts = CliOptions::default();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--list-midi" => opts.list_midi = true,
            "--midi-port" => {
                let value = args
                    .next()
                    .context("--midi-port requires an index value (e.g. --midi-port 1)")?;
                opts.midi_port = Some(parse_port_index(&value)?);
            }
            "--sample-rate" => {
                let value = args
                    .next()
                    .context("--sample-rate requires a value in Hz (e.g. --sample-rate 48000)")?;
                opts.sample_rate_hz = Some(parse_positive_u32("sample rate", &value)?);
            }
            "--buffer-frames" => {
                let value = args
                    .next()
                    .context("--buffer-frames requires a frame count (e.g. --buffer-frames 128)")?;
                opts.buffer_frames = Some(parse_positive_u32("buffer frames", &value)?);
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ if arg.starts_with("--midi-port=") => {
                let value = &arg["--midi-port=".len()..];
                opts.midi_port = Some(parse_port_index(value)?);
            }
            _ if arg.starts_with("--sample-rate=") => {
                let value = &arg["--sample-rate=".len()..];
                opts.sample_rate_hz = Some(parse_positive_u32("sample rate", value)?);
            }
            _ if arg.starts_with("--buffer-frames=") => {
                let value = &arg["--buffer-frames=".len()..];
                opts.buffer_frames = Some(parse_positive_u32("buffer frames", value)?);
            }
            other => {
                anyhow::bail!("Unknown argument: {other}. Run with --help for usage.");
            }
        }
    }

    Ok(opts)
}

fn parse_port_index(value: &str) -> anyhow::Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("Invalid MIDI port index: {value}"))
}

fn parse_positive_u32(label: &str, value: &str) -> anyhow::Result<u32> {
    let parsed = value
        .parse::<u32>()
        .with_context(|| format!("Invalid {label} value: {value}"))?;
    anyhow::ensure!(parsed > 0, "{label} must be > 0");
    Ok(parsed)
}

fn print_usage() {
    println!("squark-app usage:");
    println!("  cargo run -p squark-app --release -- [OPTIONS]\n");
    println!("Options:");
    println!("  --list-midi           List available MIDI inputs and exit");
    println!("  --midi-port <index>   Force a specific MIDI input (0-based)");
    println!("  --sample-rate <Hz>    Request a specific sample rate (default 48000)");
    println!("  --buffer-frames <N>   Request buffer size in frames (default 128)");
    println!("  -h, --help            Show this help text");
}

fn main() -> anyhow::Result<()> {
    let cli = parse_cli()?;

    if cli.list_midi {
        list_midi_ports()?;
        return Ok(());
    }

    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("No default output audio device available")?;

    let requested_sample_rate = cli.sample_rate_hz.unwrap_or(48_000);
    let supported_config = choose_supported_config(&device, requested_sample_rate)
        .or_else(|_| device.default_output_config())
        .context("Failed to query output config")?;
    let sample_format = supported_config.sample_format();

    let mut config: cpal::StreamConfig = supported_config.into();
    let channels = config.channels as usize;

    // For sub-10ms targets, you'll typically want fixed buffers like 64/128.
    // Not all backends/devices will accept this; we fall back if stream creation fails.
    if let Some(buffer_frames) = cli.buffer_frames {
        config.buffer_size = cpal::BufferSize::Fixed(buffer_frames);
    } else {
        config.buffer_size = cpal::BufferSize::Fixed(128);
    }

    println!(
        "Output device: {}",
        device.name().unwrap_or_else(|_| "<unknown>".to_string())
    );
    println!(
        "Config request: {} Hz, {} channels, {:?}, buffer {:?}",
        config.sample_rate.0, config.channels, sample_format, config.buffer_size
    );

    let settings = Arc::new(SettingsStore::load()?);
    let snapshot = settings.snapshot();

    let params = EngineParams::default();
    let control = Arc::new(ControlBlock::new(params));
    control.set_attack_ms(snapshot.attack_ms);
    control.set_release_ms(snapshot.release_ms);
    control.set_damping_ratio(snapshot.damping_ratio);
    control.set_inharmonicity(snapshot.inharmonicity);
    control.set_position_coupling(snapshot.position_coupling);
    control.set_velocity_coupling(snapshot.velocity_coupling);
    control.set_output_gain(snapshot.output_gain);

    let midi_learn = Arc::new(MidiLearnState::new());
    midi_learn.load_assignments(&snapshot.midi_assignments);

    // Start MIDI input on a non-audio thread. The callback only touches atomics.
    // Keep the connection in scope so it stays active.
    let _midi_conn = match start_midi(
        control.clone(),
        midi_learn.clone(),
        settings.clone(),
        cli.midi_port,
    ) {
        Ok(conn) => conn,
        Err(err) => {
            eprintln!("MIDI disabled: {err}");
            None
        }
    };

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            build_stream::<f32>(&device, &config, channels, params, control.clone())?
        }
        cpal::SampleFormat::I16 => {
            build_stream::<i16>(&device, &config, channels, params, control.clone())?
        }
        cpal::SampleFormat::U16 => {
            build_stream::<u16>(&device, &config, channels, params, control.clone())?
        }
        other => anyhow::bail!("Unsupported sample format: {other:?}"),
    };

    stream.play().context("Failed to start audio stream")?;
    println!("Running. Close the UI window or hit Ctrl+C to quit.");
    println!("Tip: pass --list-midi / --midi-port / --buffer-frames / --sample-rate as needed.");

    ui::launch(control.clone(), midi_learn.clone(), settings.clone())?;

    Ok(())
}

fn list_midi_ports() -> anyhow::Result<()> {
    let mut midi_in = MidiInput::new("squark")?;
    midi_in.ignore(Ignore::None);

    let in_ports = midi_in.ports();
    if in_ports.is_empty() {
        println!("No MIDI inputs found.");
        return Ok(());
    }

    println!("MIDI inputs:");
    for (idx, port) in in_ports.iter().enumerate() {
        let name = midi_in
            .port_name(port)
            .unwrap_or_else(|_| "<unknown>".to_string());
        println!("  [{idx}] {name}");
    }

    Ok(())
}

fn start_midi(
    control: Arc<ControlBlock>,
    midi_learn: Arc<MidiLearnState>,
    settings: Arc<SettingsStore>,
    midi_port_override: Option<usize>,
) -> anyhow::Result<Option<midir::MidiInputConnection<MidiContext>>> {
    let mut midi_in = MidiInput::new("squark")?;
    midi_in.ignore(Ignore::None);

    let in_ports = midi_in.ports();
    if in_ports.is_empty() {
        println!("No MIDI input ports found (audio will still run).");
        println!("(Plug in a controller, or enable an IAC bus on macOS.)");
        return Ok(None);
    }

    if midi_port_override.is_none() {
        println!("MIDI inputs:");
        for (idx, port) in in_ports.iter().enumerate() {
            let name = midi_in
                .port_name(port)
                .unwrap_or_else(|_| "<unknown>".to_string());
            println!("  [{idx}] {name}");
        }
    }

    let selected_idx = midi_port_override
        .or_else(|| {
            env::var("MIDI_PORT")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or(0)
        .min(in_ports.len() - 1);
    let selected_port = &in_ports[selected_idx];
    let selected_name = midi_in
        .port_name(selected_port)
        .unwrap_or_else(|_| "<unknown>".to_string());

    println!("Connecting MIDI input [{selected_idx}] {selected_name}");
    let ctx = MidiContext::new(control, midi_learn, settings);
    Ok(Some(midi_in.connect(
        selected_port,
        "squark-midi-in",
        midi_callback,
        ctx,
    )?))
}

fn midi_callback(_timestamp_ms: u64, message: &[u8], ctx: &mut MidiContext) {
    // Basic channel voice messages only.
    if message.is_empty() {
        return;
    }

    let status = message[0] & 0xF0;

    // Note On: 0x90, Note Off: 0x80
    match status {
        0x90 => {
            if message.len() < 3 {
                return;
            }
            let note = message[1];
            let vel = message[2];
            if vel == 0 {
                note_off(&ctx.control, note);
            } else {
                note_on(&ctx.control, note, vel);
            }
        }
        0x80 => {
            if message.len() < 3 {
                return;
            }
            let note = message[1];
            note_off(&ctx.control, note);
        }
        0xB0 => {
            if message.len() < 3 {
                return;
            }
            let controller = message[1];
            let value = message[2];
            handle_cc(ctx, controller, value);
        }
        _ => {}
    }
}

fn handle_cc(ctx: &MidiContext, controller: u8, value: u8) {
    if let Some(param) = ctx.midi_learn.take_pending() {
        ctx.midi_learn.set_mapping(param, controller);
        println!("Mapped {} to CC {controller}", param.label());
        if let Err(err) = ctx.settings.update_mapping(param, controller) {
            eprintln!("Warning: Failed to persist mapping: {err}");
        }
    }

    if let Some(param) = ctx.midi_learn.param_for_cc(controller) {
        let mapped_value = param.value_from_cc(value);
        ctx.control.set_parameter_value(param, mapped_value);
        if let Err(err) = ctx.settings.update_parameter(param, mapped_value) {
            eprintln!("Warning: Failed to persist parameter update: {err}");
        }
    }
}

#[inline]
fn note_on(control: &Arc<ControlBlock>, note: u8, velocity: u8) {
    let freq = midi_note_to_hz(note);
    let amp = (velocity as f32 / 127.0) * 0.25;

    control.active_note.store(note, Ordering::Relaxed);
    control
        .frequency_bits
        .store(freq.to_bits(), Ordering::Relaxed);
    control
        .amplitude_bits
        .store(amp.to_bits(), Ordering::Relaxed);
    control.gate.store(true, Ordering::Relaxed);
    control.trigger_counter.fetch_add(1, Ordering::Relaxed);
}

#[inline]
fn note_off(control: &Arc<ControlBlock>, note: u8) {
    // Simple monophonic behavior: only release if this is the active note.
    let active = control.active_note.load(Ordering::Relaxed);
    if active == note {
        control.gate.store(false, Ordering::Relaxed);
    }
}

#[inline]
fn midi_note_to_hz(note: u8) -> f32 {
    // A4 = MIDI 69 = 440Hz
    let semitones = note as f32 - 69.0;
    440.0 * (2.0_f32).powf(semitones / 12.0)
}

fn choose_supported_config(
    device: &cpal::Device,
    sample_rate_hz: u32,
) -> anyhow::Result<cpal::SupportedStreamConfig> {
    let mut best: Option<(i32, cpal::SupportedStreamConfig)> = None;

    for range in device.supported_output_configs()? {
        if range.min_sample_rate().0 > sample_rate_hz || range.max_sample_rate().0 < sample_rate_hz
        {
            continue;
        }

        let cfg = range.with_sample_rate(cpal::SampleRate(sample_rate_hz));
        let score = config_score(&cfg);

        match &best {
            Some((best_score, _)) if *best_score >= score => {}
            _ => best = Some((score, cfg)),
        }
    }

    best.map(|(_, cfg)| cfg)
        .context("No supported output config matches requested sample rate")
}

fn config_score(cfg: &cpal::SupportedStreamConfig) -> i32 {
    let channels_score = match cfg.channels() {
        2 => 1000,
        1 => 500,
        _ => 0,
    };

    let format_score = match cfg.sample_format() {
        cpal::SampleFormat::F32 => 30,
        cpal::SampleFormat::I16 => 20,
        cpal::SampleFormat::U16 => 10,
        _ => 0,
    };

    channels_score + format_score
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    params: EngineParams,
    control: Arc<ControlBlock>,
) -> anyhow::Result<cpal::Stream>
where
    T: FromF32 + cpal::SizedSample + Send + 'static,
{
    let err_fn = |err| eprintln!("stream error: {err}");

    let requested_buffer = config.buffer_size.clone();

    let try_build = |cfg: &cpal::StreamConfig| {
        let mut engine = Engine::new(cfg.sample_rate.0 as f32, params);
        let control = control.clone();
        let mut trigger_seen = control.trigger_counter();
        device.build_output_stream(
            cfg,
            move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                // Realtime-safe: just atomic loads + pure math.
                let snapshot = control.snapshot();
                engine.set_gate(snapshot.gate);
                engine.set_frequency_hz(snapshot.frequency_hz);
                engine.set_amplitude(snapshot.amplitude);
                engine.set_attack_ms(snapshot.attack_ms);
                engine.set_release_ms(snapshot.release_ms);
                engine.set_damping_ratio(snapshot.damping_ratio);
                engine.set_inharmonicity(snapshot.inharmonicity);
                engine.set_position_coupling_base(snapshot.position_coupling);
                engine.set_velocity_coupling_base(snapshot.velocity_coupling);
                engine.set_output_gain(snapshot.output_gain);
                if snapshot.trigger_counter != trigger_seen {
                    engine.trigger(snapshot.amplitude);
                    trigger_seen = snapshot.trigger_counter;
                }
                let block_peak = write_interleaved::<T>(&mut engine, data, channels);
                let frames = if channels == 0 {
                    0
                } else {
                    data.len() / channels
                };
                if frames > 0 {
                    let block_duration = frames as f32 / engine.sample_rate_hz().max(1.0);
                    control.update_peak(block_peak, block_duration);
                }
            },
            err_fn,
            None,
        )
    };

    // First try with requested buffer size.
    if let Ok(stream) = try_build(config) {
        log_buffer_acceptance("requested", &config.buffer_size);
        return Ok(stream);
    }

    // Fall back to whatever the backend wants.
    let mut fallback = config.clone();
    fallback.buffer_size = cpal::BufferSize::Default;
    eprintln!(
        "Requested buffer {:?} not supported by device; falling back to default",
        requested_buffer
    );
    let stream = try_build(&fallback).context("Failed to build audio stream")?;
    log_buffer_acceptance("fallback", &fallback.buffer_size);
    Ok(stream)
}

fn log_buffer_acceptance(label: &str, buffer_size: &cpal::BufferSize) {
    match buffer_size {
        cpal::BufferSize::Fixed(n) => println!("Audio buffer ({label}): {n} frames"),
        cpal::BufferSize::Default => println!("Audio buffer ({label}): device default"),
    }
}

#[inline]
fn write_interleaved<T: FromF32>(engine: &mut Engine, out: &mut [T], channels: usize) -> f32 {
    if channels == 0 {
        return 0.0;
    }

    let mut block_peak = 0.0f32;
    let frames = out.len() / channels;
    for frame_idx in 0..frames {
        let s = engine.next_sample();
        block_peak = block_peak.max(s.abs());
        for ch in 0..channels {
            let idx = frame_idx * channels + ch;
            out[idx] = FromF32::from_f32(s);
        }
    }

    block_peak
}

trait FromF32: Copy {
    fn from_f32(s: f32) -> Self;
}

impl FromF32 for f32 {
    #[inline]
    fn from_f32(s: f32) -> Self {
        s.clamp(-1.0, 1.0)
    }
}

impl FromF32 for i16 {
    #[inline]
    fn from_f32(s: f32) -> Self {
        (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
    }
}

impl FromF32 for u16 {
    #[inline]
    fn from_f32(s: f32) -> Self {
        let s = s.clamp(-1.0, 1.0);
        (((s * 0.5) + 0.5) * u16::MAX as f32) as u16
    }
}
