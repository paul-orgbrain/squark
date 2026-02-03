# squark

A low-level, realtime sound synthesis engine.

This repo is structured so you can develop on macOS (CoreAudio) and move to a 64-bit Raspberry Pi (Linux/aarch64) with minimal changes:

- Engine core: `crates/squark-engine` (no allocations/locks in the audio callback path)
- Audio host app: `crates/squark-app` (CPAL output stream)

## Realtime notes (sub-10ms)

To hit sub-10ms end-to-end latency, you typically need:

- 48kHz sample rate
- Small buffers (e.g. 128 frames ≈ 2.7ms; 64 frames ≈ 1.3ms)
- Avoid allocations/locks/syscalls in the audio callback
- On Linux: realtime scheduling + a low-latency kernel and/or `PREEMPT_RT` helps a lot

`squark-app` requests a 128-frame buffer where the backend supports it, and falls back gracefully if not.

## Prereqs (macOS)

Rust toolchain (recommended via rustup):

1. Install rustup:

   `brew install rustup`

2. Install the stable toolchain:

   `rustup default stable`

3. Ensure Cargo is on your PATH in new terminals:

   `echo 'source "$HOME/.cargo/env"' >> ~/.zshrc`

## Run on macOS

From the repo root:

`source "$HOME/.cargo/env" && cargo run -p squark-app --release`

You should hear a quiet 440Hz sine.

## Build + run on Raspberry Pi (64-bit)

The simplest workflow is to build on the Pi itself (it avoids cross-compiling headaches with ALSA/JACK system deps).

On the Pi (aarch64 OS recommended):

1. Install build deps:

   `sudo apt update && sudo apt install -y build-essential pkg-config libasound2-dev`

2. Install Rust via rustup (recommended):

   `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`

3. Build + run:

   `source "$HOME/.cargo/env" && cargo build -p squark-app --release && ./target/release/squark-app`

### Low-latency tips (Pi/Linux)

- Set CPU governor:
  - `sudo apt install -y cpufrequtils`
  - `sudo cpufreq-set -g performance`
- If you use PipeWire, consider running via JACK compatibility or install JACK directly.
- For best results, use a low-latency kernel and enable realtime scheduling (rtkit).

## Optional: cross-compile + deploy from macOS

Cross-compiling Linux audio targets can require a sysroot with ALSA headers/libs. If you want the workflow anyway:

- Build for `aarch64-unknown-linux-gnu` (you’ll need a working cross toolchain or Docker-based tooling)
- Deploy with:
  - `PI_HOST=pi.local PI_USER=pi ./scripts/pi_deploy.sh`
  - `PI_HOST=pi.local PI_USER=pi ./scripts/pi_run.sh`
