use std::time::Duration;

use anyhow::Context as _;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use squark_engine::{Engine, EngineParams};

fn main() -> anyhow::Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .context("No default output audio device available")?;

    let supported_config = choose_supported_config(&device, 48_000)
        .or_else(|_| device.default_output_config())
        .context("Failed to query output config")?;
    let sample_format = supported_config.sample_format();

    let mut config: cpal::StreamConfig = supported_config.into();
    let channels = config.channels as usize;

    // For sub-10ms targets, you'll typically want fixed buffers like 64/128.
    // Not all backends/devices will accept this; we fall back if stream creation fails.
    config.buffer_size = cpal::BufferSize::Fixed(128);

    println!("Output device: {}", device.name().unwrap_or_else(|_| "<unknown>".to_string()));
    println!(
        "Config: {} Hz, {} channels, {:?}",
        config.sample_rate.0, config.channels, sample_format
    );

    let params = EngineParams::default();

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, channels, params)?,
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, channels, params)?,
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, channels, params)?,
        other => anyhow::bail!("Unsupported sample format: {other:?}"),
    };

    stream.play().context("Failed to start audio stream")?;
    println!("Running. Press Ctrl+C to stop.");

    loop {
        std::thread::sleep(Duration::from_secs(1));
    }
}

fn choose_supported_config(device: &cpal::Device, sample_rate_hz: u32) -> anyhow::Result<cpal::SupportedStreamConfig> {
    let mut best: Option<(i32, cpal::SupportedStreamConfig)> = None;

    for range in device.supported_output_configs()? {
        if range.min_sample_rate().0 > sample_rate_hz || range.max_sample_rate().0 < sample_rate_hz {
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
) -> anyhow::Result<cpal::Stream>
where
    T: FromF32 + cpal::SizedSample + Send + 'static,
{
    let err_fn = |err| eprintln!("stream error: {err}");

    let try_build = |cfg: &cpal::StreamConfig| {
        let mut engine = Engine::new(cfg.sample_rate.0 as f32, params);
        device.build_output_stream(
            cfg,
            move |data: &mut [T], _info: &cpal::OutputCallbackInfo| {
                write_interleaved::<T>(&mut engine, data, channels);
            },
            err_fn,
            None,
        )
    };

    // First try with requested buffer size.
    if let Ok(stream) = try_build(config) {
        return Ok(stream);
    }

    // Fall back to whatever the backend wants.
    let mut fallback = config.clone();
    fallback.buffer_size = cpal::BufferSize::Default;
    try_build(&fallback).context("Failed to build audio stream")
}

#[inline]
fn write_interleaved<T: FromF32>(engine: &mut Engine, out: &mut [T], channels: usize) {
    if channels == 0 {
        return;
    }

    let frames = out.len() / channels;
    for frame_idx in 0..frames {
        let s = engine.next_sample();
        for ch in 0..channels {
            let idx = frame_idx * channels + ch;
            out[idx] = FromF32::from_f32(s);
        }
    }
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
