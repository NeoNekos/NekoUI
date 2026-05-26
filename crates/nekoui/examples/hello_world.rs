use nekoui::prelude::*;

#[derive(Debug)]
struct HelloWorld;

impl Render for HelloWorld {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .p(px(12.0))
            .w(fill())
            .bg(Color::rgb(0xF8, 0xFA, 0xFC))
            .child(text("Hello NekoUI").font_size(px(18.0)))
    }
}

fn main() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new().title("Hello NekoUI"), |_| HelloWorld)?;
            Ok(())
        })
        .unwrap();

    println!("{:#?}", run.performance_report());
}
