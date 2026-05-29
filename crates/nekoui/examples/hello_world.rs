use nekoui::prelude::*;

// Foundation-only example: this opens a native OS window and runs until the
// window is closed. Windows has a private D3D11 backend that draws supported
// sRGB solid rectangles plus v0 static monochrome glyph masks.

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
                    .bg(Color::rgb(0x00, 0x1F, 0xFF)),
            )
            .child(text("Hello NekoUI").font_size(px(18.0)))
    }
}

fn main() -> NekoResult<()> {
    Application::new().run(|cx| {
        cx.windows()
            .open(WindowOptions::new().title("Hello NekoUI"), |_| HelloWorld)?;
        Ok(())
    })
}
