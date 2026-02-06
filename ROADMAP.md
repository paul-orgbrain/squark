# Roadmap

## Goal

Broaden Squark beyond the current coupled modal resonator by introducing richer exciters and eventually additional resonator types (e.g., waveguides). Each step should remain realtime-safe and composable so we can mix and match building blocks to form new instruments.

## Immediate Focus: Exciter Abstraction

1. Extract the current envelope/drive logic into an `Exciter` module that outputs a per-sample drive signal consumed by the resonator. This keeps the modal engine agnostic to how energy is injected.
2. Implement at least two exciter modes:
   - **Strike**: today's fast attack/decay envelope (baseline behavior).
   - **Breath/Bow**: slower attack, noise-rich signal with controllable pressure and jitter to emulate continuous energy input.
3. Route engine drive through the exciter output instead of directly using the amplitude envelope. Ensure no allocations and keep parameter updates lock-free.

## UI & MIDI Plumbing

1. Add an exciter selector plus mode-specific controls (e.g., pressure, noise color, bow stiffness) to the egui panel.
2. Extend `ParameterId`, persistence, and MIDI Learn so each exciter control can be stored/mapped just like the existing parameters.
3. Provide sensible defaults and ranges to keep the exciters stable when automated.

## Waveguide Prototype

1. With the exciter interface in place, build a simple waveguide resonator (e.g., Karplus-Strong or a bidirectional delay with loss filters) that also consumes the exciter output.
2. Allow selecting between Modal and Waveguide resonators (or blending them) to explore hybrid instruments.
3. Share common parameter plumbing so future resonators can plug in easily.

## Longer-Term Directions

- Combine multiple exciters/resonators in a small graph to mimic complex instruments (e.g., breath exciter driving both a waveguide body and the existing modal cluster).
- Investigate physical-inspired friction models (stick-slip) for bowing and pressure-dependent turbulence for breath.
- Explore exporting the core DSP to other platforms (e.g., wasm) once the architecture is modular enough.

## Incremental Delivery Plan

1. Implement exciter abstraction + breath/bow prototype.
2. Wire UI/MIDI/persistence for new exciter parameters.
3. Prototype a waveguide resonator fed by the shared exciter.
4. Add routing/blending controls between resonators.
5. Iterate on advanced exciters (bow friction curves, breath turbulence) and multi-resonator instrument presets.
