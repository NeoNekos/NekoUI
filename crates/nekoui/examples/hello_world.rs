use nekoui::{
    Application, Color, Direction, EdgeInsets, Length, ParentElement, WindowOptions, div, text,
};

fn main() -> Result<(), nekoui::Error> {
    env_logger::init();
    Application::new().run(|app| {
        app.open_window(
            WindowOptions::new().title("NekoUI").size(960, 640),
            |_window, _app| {
                div()
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .padding(EdgeInsets::all(24.0))
                    .direction(Direction::Column)
                    .gap(16.0)
                    .background(Color::rgb(0xF5F3EF))
                    .child(
                        div()
                            .padding(EdgeInsets::all(20.0))
                            .background(Color::rgb(0x1F2937))
                            .child(
                                text("Hello world!")
                                    .font_size(32.0)
                                    .color(Color::rgb(0xFFFFFF)),
                            ),
                    )
                    .child(
                        div()
                            .padding(EdgeInsets {
                                top: 0.0,
                                right: 0.0,
                                bottom: 0.0,
                                left: 20.0,
                            })
                            .background(Color::rgb(0xFFFFFF))
                            .child(
                                text("这是一个简单示例喵 🍥")
                                    .font_size(24.0)
                                    .line_height(28.0)
                                    .color(Color::rgb(0x111827)),
                            ),
                    )
            },
        )?;
        Ok(())
    })
}
