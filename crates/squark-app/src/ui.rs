use std::{sync::Arc, time::Duration};

use anyhow::anyhow;
use eframe::{self, egui};

use crate::{ControlBlock, MidiLearnState, ParameterId, settings::SettingsStore};

pub fn launch(
    control: Arc<ControlBlock>,
    midi_learn: Arc<MidiLearnState>,
    settings: Arc<SettingsStore>,
) -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([560.0, 520.0])
            .with_min_inner_size([420.0, 360.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Squark controls",
        native_options,
        Box::new(move |_cc| {
            Box::new(SquarkApp::new(
                control.clone(),
                midi_learn.clone(),
                settings.clone(),
            ))
        }),
    )
    .map_err(|err| anyhow!("eframe error: {err}"))?;

    Ok(())
}

struct SquarkApp {
    control: Arc<ControlBlock>,
    midi_learn: Arc<MidiLearnState>,
    settings: Arc<SettingsStore>,
}

impl SquarkApp {
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

    fn draw_parameter(&self, ui: &mut egui::Ui, param: ParameterId) {
        let mut value = self.control.parameter_value(param);
        let mut slider = egui::Slider::new(&mut value, param.slider_range()).text(param.label());
        if param.logarithmic() {
            slider = slider.logarithmic(true);
        }
        if ui.add(slider).changed() {
            let clamped = param.clamp(value);
            self.control.set_parameter_value(param, clamped);
            if let Err(err) = self.settings.update_parameter(param, clamped) {
                eprintln!("Warning: Failed to persist parameter update: {err}");
            }
        }

        ui.horizontal(|row| {
            row.label(param.format_value(value));
            let pending = self.midi_learn.pending();
            let listening = pending == Some(param);
            let other_pending = pending.is_some() && !listening;
            let button_label = if listening {
                "Listening…"
            } else {
                "MIDI Learn"
            };
            if row
                .add_enabled(!other_pending, egui::Button::new(button_label))
                .clicked()
            {
                if listening {
                    let _ = self.midi_learn.take_pending();
                } else {
                    let removed = self.midi_learn.clear_mapping(param);
                    if removed {
                        if let Err(err) = self.settings.remove_mapping(param) {
                            eprintln!(
                                "Warning: Failed to persist MIDI unlearn for {}: {err}",
                                param.label()
                            );
                        }
                    }
                    self.midi_learn.request(param);
                }
            }

            match self.midi_learn.mapping_for(param) {
                Some(cc) => row.label(format!("CC {cc}")),
                None => row.label("Unassigned"),
            };
        });
    }

    fn draw_peak_meter(&self, ui: &mut egui::Ui) {
        let peak = self.control.peak_level().clamp(0.0, 1.0);
        let db = if peak <= 0.000_01 {
            "-inf".to_string()
        } else {
            format!("{:.1} dBFS", 20.0 * peak.log10())
        };
        ui.label("Output peak");
        ui.add(
            egui::ProgressBar::new(peak)
                .desired_width(ui.available_width())
                .fill(if peak < 0.9 {
                    egui::Color32::from_rgb(80, 200, 120)
                } else {
                    egui::Color32::from_rgb(220, 90, 90)
                })
                .text(db),
        );
    }
}

impl eframe::App for SquarkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(16));
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Squark mass-spring");
            ui.label("Shape envelopes and resonator couplings or map them to MIDI CCs.");
            if ui.button("Reset parameters").clicked() {
                self.control.reset_parameters();
                if let Err(err) = self.settings.reset_parameters() {
                    eprintln!("Warning: Failed to persist parameter reset: {err}");
                }
            }
            ui.separator();

            for &param in ParameterId::all() {
                self.draw_parameter(ui, param);
                ui.add_space(8.0);
            }

            ui.separator();
            self.draw_peak_meter(ui);

            if let Some(param) = self.midi_learn.pending() {
                ui.colored_label(
                    egui::Color32::LIGHT_GREEN,
                    format!("Listening for CC assignment for {}", param.label()),
                );
            }
        });
    }
}
