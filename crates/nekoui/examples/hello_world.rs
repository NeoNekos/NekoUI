use nekoui::prelude::*;
use tracing_subscriber::EnvFilter;

#[derive(Debug)]
struct HelloWorld;

impl Render for HelloWorld {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .p(px(12.0))
            .w(fill())
            .h(fill())
            .bg(Color::rgb(0xFF, 0xFF, 0xFF))
            .child(
                div()
                    .w(px(300.))
                    .h(px(50.))
                    .m(px(20.))
                    .radius(px(8.))
                    .bg(Color::rgb(0x00, 0xFA, 0xFF)),
            )
            // .child(text("Hello NekoUI 🍥🔴🔵🟢😅❇️ Test words").font_size(px(24.0)))
            .child(text("The longest word 你好世界这段是中文，こんにちはこの段落は日本語です in any of the major \
            English language dictionaries is pneumonoultramicroscopicsilicovolcanoconiosis, a word that \
            refers to a lung disease contracted from the inhalation of very fine silica particles, 🍥🔴🔵🟢😅❇️ \
            a url https://github.com/Yamrc/NekoUI/blob/master/crates/nekoui/examples/hello_world.rs#L25, \
            specifically from a volcano; medically, it is the same as silicosis.").font_size(px(24.0)))
    }
}

fn run_example() {
    Application::new()
        .run(|cx| {
            cx.windows()
                .open(WindowOptions::new().title("Hello NekoUI"), |_| HelloWorld)?;
            Ok(())
        })
        .unwrap();
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .with_thread_names(true)
        .init();

    run_example();
}
