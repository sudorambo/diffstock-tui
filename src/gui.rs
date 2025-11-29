use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use crate::app::{App, AppState};
use chrono::TimeZone;

pub struct GuiApp {
    app: App,
}

impl GuiApp {
    pub fn new(app: App) -> Self {
        Self { app }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.app.tick();

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.app.state {
                AppState::Input => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(50.0);
                        ui.heading("DiffStock");
                        ui.add_space(20.0);
                        ui.label("Enter stock symbol:");
                        
                        let response = ui.text_edit_singleline(&mut self.app.input);
                        if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                             self.app.trigger_fetch();
                        }
                        
                        ui.add_space(10.0);
                        if ui.button("Predict").clicked() {
                            self.app.trigger_fetch();
                        }
                        
                        if let Some(err) = &self.app.error_msg {
                            ui.add_space(10.0);
                            ui.colored_label(egui::Color32::RED, err);
                        }
                    });
                }
                AppState::Loading => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(100.0);
                        ui.spinner();
                        ui.label("Fetching data...");
                    });
                    ctx.request_repaint();
                }
                AppState::Forecasting => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(100.0);
                        ui.label("Running Inference...");
                        ui.add(egui::ProgressBar::new(self.app.progress as f32));
                    });
                    ctx.request_repaint();
                }
                AppState::Dashboard => {
                    ui.horizontal(|ui| {
                        if ui.button("Back").clicked() {
                            self.app.state = AppState::Input;
                            self.app.input.clear();
                            self.app.stock_data = None;
                            self.app.forecast = None;
                        }
                        if let Some(data) = &self.app.stock_data {
                            ui.label(format!("Symbol: {}", data.symbol));
                        }
                    });
                    
                    let plot = Plot::new("stock_plot")
                        .legend(egui_plot::Legend::default())
                        .x_axis_formatter(|x, _range| {
                            let dt = chrono::Utc.timestamp_opt(x.value as i64, 0).unwrap();
                            dt.format("%Y-%m-%d").to_string()
                        })
                        .view_aspect(2.0);
                        
                    plot.show(ui, |plot_ui| {
                        if let Some(data) = &self.app.stock_data {
                            let points: PlotPoints = data.history.iter()
                                .map(|c| [c.date.timestamp() as f64, c.close])
                                .collect();
                            plot_ui.line(Line::new(points).name("History"));
                        }
                        
                        if let Some(forecast) = &self.app.forecast {
                            // Assuming forecast times are also timestamps
                            let p50: PlotPoints = forecast.p50.iter().map(|&(x, y)| [x, y]).collect();
                            let p10: PlotPoints = forecast.p10.iter().map(|&(x, y)| [x, y]).collect();
                            let p90: PlotPoints = forecast.p90.iter().map(|&(x, y)| [x, y]).collect();

                            plot_ui.line(Line::new(p50).name("Median").color(egui::Color32::GREEN));
                            plot_ui.line(Line::new(p10).name("P10").style(egui_plot::LineStyle::Dashed { length: 10.0 }).color(egui::Color32::LIGHT_GREEN));
                            plot_ui.line(Line::new(p90).name("P90").style(egui_plot::LineStyle::Dashed { length: 10.0 }).color(egui::Color32::LIGHT_GREEN));
                        }
                    });
                }
            }
        });
    }
}
