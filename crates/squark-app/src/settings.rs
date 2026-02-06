use std::collections::HashMap;
use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

use anyhow::{Context, Result};
use directories::ProjectDirs;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::ParameterId;

const SETTINGS_FILE_NAME: &str = "settings.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredSettings {
    #[serde(default = "default_exciter_mode")]
    exciter_mode: u8,
    attack_ms: f32,
    release_ms: f32,
    damping_ratio: f32,
    #[serde(default = "default_inharmonicity")]
    inharmonicity: f32,
    #[serde(default = "default_position_coupling")]
    position_coupling: f32,
    #[serde(default = "default_velocity_coupling")]
    velocity_coupling: f32,
    #[serde(default = "default_excitation_gain")]
    excitation_gain: f32,
    #[serde(default = "default_output_gain")]
    output_gain: f32,
    #[serde(default = "default_breath_pressure")]
    breath_pressure: f32,
    #[serde(default = "default_breath_noise_cutoff_hz")]
    breath_noise_cutoff_hz: f32,
    #[serde(default = "default_breath_feedback")]
    breath_feedback: f32,
    #[serde(default = "default_breath_auto_level")]
    breath_auto_level: f32,
    midi_cc: HashMap<String, u8>,
}

impl Default for StoredSettings {
    fn default() -> Self {
        Self {
            exciter_mode: default_exciter_mode(),
            attack_ms: crate::DEFAULT_ATTACK_MS,
            release_ms: crate::DEFAULT_RELEASE_MS,
            damping_ratio: crate::DEFAULT_DAMPING_RATIO,
            inharmonicity: crate::DEFAULT_INHARMONICITY,
            position_coupling: crate::DEFAULT_POSITION_COUPLING,
            velocity_coupling: crate::DEFAULT_VELOCITY_COUPLING,
            excitation_gain: crate::DEFAULT_EXCITATION_GAIN,
            output_gain: crate::DEFAULT_OUTPUT_GAIN,
            breath_pressure: crate::DEFAULT_BREATH_PRESSURE,
            breath_noise_cutoff_hz: crate::DEFAULT_BREATH_NOISE_CUTOFF_HZ,
            breath_feedback: crate::DEFAULT_BREATH_FEEDBACK,
            breath_auto_level: crate::DEFAULT_BREATH_AUTO_LEVEL,
            midi_cc: HashMap::new(),
        }
    }
}

pub(crate) struct SettingsSnapshot {
    pub exciter_mode: u8,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub damping_ratio: f32,
    pub inharmonicity: f32,
    pub position_coupling: f32,
    pub velocity_coupling: f32,
    pub excitation_gain: f32,
    pub output_gain: f32,
    pub breath_pressure: f32,
    pub breath_noise_cutoff_hz: f32,
    pub breath_feedback: f32,
    pub breath_auto_level: f32,
    pub midi_assignments: Vec<(ParameterId, u8)>,
}

impl From<StoredSettings> for SettingsSnapshot {
    fn from(value: StoredSettings) -> Self {
        let mut midi_assignments = Vec::new();
        for (key, cc) in value.midi_cc.iter() {
            if let Some(param) = ParameterId::from_storage_key(key) {
                midi_assignments.push((param, *cc));
            }
        }

        Self {
            exciter_mode: value.exciter_mode.min(1),
            attack_ms: ParameterId::AttackMs.clamp(value.attack_ms),
            release_ms: ParameterId::ReleaseMs.clamp(value.release_ms),
            damping_ratio: ParameterId::DampingRatio.clamp(value.damping_ratio),
            inharmonicity: ParameterId::Inharmonicity.clamp(value.inharmonicity),
            position_coupling: ParameterId::PositionCoupling.clamp(value.position_coupling),
            velocity_coupling: ParameterId::VelocityCoupling.clamp(value.velocity_coupling),
            excitation_gain: ParameterId::ExcitationGain.clamp(value.excitation_gain),
            output_gain: ParameterId::OutputGain.clamp(value.output_gain),
            breath_pressure: ParameterId::BreathPressure.clamp(value.breath_pressure),
            breath_noise_cutoff_hz: ParameterId::BreathNoiseCutoffHz
                .clamp(value.breath_noise_cutoff_hz),
            breath_feedback: ParameterId::BreathFeedback.clamp(value.breath_feedback),
            breath_auto_level: ParameterId::BreathAutoLevel.clamp(value.breath_auto_level),
            midi_assignments,
        }
    }
}

pub(crate) struct SettingsStore {
    path: PathBuf,
    state: Mutex<StoredSettings>,
}

impl SettingsStore {
    pub(crate) fn load() -> Result<Self> {
        let path = resolve_settings_path()?;
        let stored = read_settings(&path).unwrap_or_default();
        Ok(Self {
            path,
            state: Mutex::new(stored),
        })
    }

    pub(crate) fn snapshot(&self) -> SettingsSnapshot {
        let data = self.state.lock().clone();
        SettingsSnapshot::from(data)
    }

    pub(crate) fn update_parameter(&self, param: ParameterId, value: f32) -> Result<()> {
        let clamped = param.clamp(value);
        {
            let mut data = self.state.lock();
            match param {
                ParameterId::AttackMs => data.attack_ms = clamped,
                ParameterId::ReleaseMs => data.release_ms = clamped,
                ParameterId::DampingRatio => data.damping_ratio = clamped,
                ParameterId::Inharmonicity => data.inharmonicity = clamped,
                ParameterId::PositionCoupling => data.position_coupling = clamped,
                ParameterId::VelocityCoupling => data.velocity_coupling = clamped,
                ParameterId::ExcitationGain => data.excitation_gain = clamped,
                ParameterId::OutputGain => data.output_gain = clamped,
                ParameterId::BreathPressure => data.breath_pressure = clamped,
                ParameterId::BreathNoiseCutoffHz => data.breath_noise_cutoff_hz = clamped,
                ParameterId::BreathFeedback => data.breath_feedback = clamped,
                ParameterId::BreathAutoLevel => data.breath_auto_level = clamped,
            }
        }
        self.persist_current_state()
    }

    pub(crate) fn update_exciter_mode(&self, mode: u8) -> Result<()> {
        {
            let mut data = self.state.lock();
            data.exciter_mode = mode.min(1);
        }
        self.persist_current_state()
    }

    pub(crate) fn update_mapping(&self, param: ParameterId, cc: u8) -> Result<()> {
        {
            let mut data = self.state.lock();
            data.midi_cc.insert(param.storage_key().to_string(), cc);
        }
        self.persist_current_state()
    }

    pub(crate) fn remove_mapping(&self, param: ParameterId) -> Result<()> {
        {
            let mut data = self.state.lock();
            data.midi_cc.remove(param.storage_key());
        }
        self.persist_current_state()
    }

    pub(crate) fn reset_parameters(&self) -> Result<()> {
        {
            let mut data = self.state.lock();
            data.exciter_mode = default_exciter_mode();
            data.attack_ms = crate::DEFAULT_ATTACK_MS;
            data.release_ms = crate::DEFAULT_RELEASE_MS;
            data.damping_ratio = crate::DEFAULT_DAMPING_RATIO;
            data.inharmonicity = crate::DEFAULT_INHARMONICITY;
            data.position_coupling = crate::DEFAULT_POSITION_COUPLING;
            data.velocity_coupling = crate::DEFAULT_VELOCITY_COUPLING;
            data.excitation_gain = crate::DEFAULT_EXCITATION_GAIN;
            data.output_gain = crate::DEFAULT_OUTPUT_GAIN;
            data.breath_pressure = crate::DEFAULT_BREATH_PRESSURE;
            data.breath_noise_cutoff_hz = crate::DEFAULT_BREATH_NOISE_CUTOFF_HZ;
            data.breath_feedback = crate::DEFAULT_BREATH_FEEDBACK;
            data.breath_auto_level = crate::DEFAULT_BREATH_AUTO_LEVEL;
        }
        self.persist_current_state()
    }

    fn persist_current_state(&self) -> Result<()> {
        let data = self.state.lock().clone();
        let serialized = serde_json::to_vec_pretty(&data)?;
        let tmp_path = self.path.with_extension("tmp");
        fs::write(&tmp_path, &serialized).with_context(|| {
            format!(
                "Failed to write temporary settings file: {}",
                tmp_path.display()
            )
        })?;
        fs::rename(&tmp_path, &self.path).with_context(|| {
            format!("Failed to replace settings file at {}", self.path.display())
        })?;
        Ok(())
    }
}

fn read_settings(path: &PathBuf) -> Option<StoredSettings> {
    match fs::read(path) {
        Ok(bytes) => match serde_json::from_slice::<StoredSettings>(&bytes) {
            Ok(parsed) => Some(parsed),
            Err(err) => {
                eprintln!(
                    "Warning: Failed to parse settings at {}: {err}",
                    path.display()
                );
                None
            }
        },
        Err(err) if err.kind() == io::ErrorKind::NotFound => None,
        Err(err) => {
            eprintln!(
                "Warning: Failed to read settings at {}: {err}",
                path.display()
            );
            None
        }
    }
}

fn resolve_settings_path() -> Result<PathBuf> {
    if let Some(proj_dirs) = ProjectDirs::from("com", "OrgBrain", "Squark") {
        let dir = proj_dirs.config_dir().to_path_buf();
        fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create config directory {}", dir.display()))?;
        return Ok(dir.join(SETTINGS_FILE_NAME));
    }

    let fallback = env::current_dir()
        .context("Failed to resolve current directory for fallback settings path")?
        .join(".squark");
    fs::create_dir_all(&fallback).with_context(|| {
        format!(
            "Failed to create fallback settings directory {}",
            fallback.display()
        )
    })?;
    Ok(fallback.join(SETTINGS_FILE_NAME))
}

fn default_inharmonicity() -> f32 {
    crate::DEFAULT_INHARMONICITY
}

fn default_position_coupling() -> f32 {
    crate::DEFAULT_POSITION_COUPLING
}

fn default_velocity_coupling() -> f32 {
    crate::DEFAULT_VELOCITY_COUPLING
}

fn default_excitation_gain() -> f32 {
    crate::DEFAULT_EXCITATION_GAIN
}

fn default_output_gain() -> f32 {
    crate::DEFAULT_OUTPUT_GAIN
}

fn default_breath_pressure() -> f32 {
    crate::DEFAULT_BREATH_PRESSURE
}

fn default_breath_noise_cutoff_hz() -> f32 {
    crate::DEFAULT_BREATH_NOISE_CUTOFF_HZ
}

fn default_breath_feedback() -> f32 {
    crate::DEFAULT_BREATH_FEEDBACK
}

fn default_breath_auto_level() -> f32 {
    crate::DEFAULT_BREATH_AUTO_LEVEL
}

fn default_exciter_mode() -> u8 {
    0
}
