use nekoui::prelude::*;
use tracing_subscriber::EnvFilter;

const BLACK: Color = Color::rgb(0x00, 0x00, 0x00);
const WHITE: Color = Color::rgb(0xFF, 0xFF, 0xFF);
const BLUE_WASH: Color = Color::rgb(0xF0, 0xF7, 0xFF);
const GREEN_WASH: Color = Color::rgb(0xEA, 0xFA, 0xEE);
const BLUE_MARK: Color = Color::rgb(0x99, 0xC7, 0xFF);
const GREEN_MARK: Color = Color::rgb(0x8A, 0xD6, 0x96);

const BORDER_1: Length = Length::Px(1.0);
const SPACE_2: Length = Length::Px(8.0);
const SPACE_3: Length = Length::Px(12.0);
const SPACE_4: Length = Length::Px(16.0);
const TITLE_SIZE: Length = Length::Px(18.0);
const BODY_SIZE: Length = Length::Px(14.0);
const LABEL_SIZE: Length = Length::Px(16.0);
const VERTICAL_VIEWPORT_WIDTH: Length = Length::Px(460.0);
const VERTICAL_VIEWPORT_HEIGHT: Length = Length::Px(420.0);
const VERTICAL_CONTENT_HEIGHT: Length = Length::Px(1500.0);
const HORIZONTAL_VIEWPORT_WIDTH: Length = Length::Px(360.0);
const HORIZONTAL_VIEWPORT_HEIGHT: Length = Length::Px(180.0);
const HORIZONTAL_CONTENT_WIDTH: Length = Length::Px(1600.0);
const HORIZONTAL_CONTENT_HEIGHT: Length = Length::Px(150.0);

#[derive(Debug)]
struct NestedScroll;

impl Render for NestedScroll {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .w(fill())
            .h(fill())
            .p(SPACE_4)
            .bg(WHITE)
            .child(
                text("Nested scroll overflow demo")
                    .font_size(TITLE_SIZE)
                    .text_color(BLACK),
            )
            .child(
                text("Wheel-scroll clipped overflow demo; no custom scrollbar styling or momentum demo.")
                    .mt(SPACE_2)
                    .mb(SPACE_3)
                    .font_size(BODY_SIZE)
                    .text_color(BLACK)
                    .line_clamp(2),
            )
            .child(vertical_scroll_shell())
    }
}

fn vertical_scroll_shell() -> impl IntoElement {
    div()
        .w(VERTICAL_VIEWPORT_WIDTH)
        .bg(BLACK)
        .p(BORDER_1)
        .child(
            div()
                .w(fill())
                .h(VERTICAL_VIEWPORT_HEIGHT)
                .overflow(Overflow::Scroll)
                .bg(WHITE)
                .child(vertical_scroll_content()),
        )
}

fn vertical_scroll_content() -> impl IntoElement {
    div()
        .w(fill())
        .h(VERTICAL_CONTENT_HEIGHT)
        .p(SPACE_4)
        .bg(BLUE_WASH)
        .child(
            div()
                .w(fill())
                .h(BORDER_1)
                .mb(SPACE_3)
                .bg(BLUE_MARK),
        )
        .child(
            text("Scroll Vertical")
                .font_size(LABEL_SIZE)
                .text_color(BLACK),
        )
        .child(
            text("The blue panel is taller than its viewport, so wheel input moves the retained scroll offset without relayout.")
                .mt(SPACE_2)
                .mb(SPACE_4)
                .font_size(BODY_SIZE)
                .text_color(BLACK)
                .line_clamp(2),
        )
        .child(horizontal_scroll_shell())
        .child(
            text("More vertical content continues below the nested horizontal viewport.")
                .mt(SPACE_4)
                .font_size(BODY_SIZE)
                .text_color(BLACK),
        )
}

fn horizontal_scroll_shell() -> impl IntoElement {
    div()
        .w(HORIZONTAL_VIEWPORT_WIDTH)
        .bg(BLACK)
        .p(BORDER_1)
        .child(
            div()
                .w(fill())
                .h(HORIZONTAL_VIEWPORT_HEIGHT)
                .overflow(Overflow::Scroll)
                .bg(WHITE)
                .child(
                    div()
                        .w(HORIZONTAL_CONTENT_WIDTH)
                        .h(HORIZONTAL_CONTENT_HEIGHT)
                        .p(SPACE_4)
                        .bg(GREEN_WASH)
                        .child(div().w(fill()).h(BORDER_1).mb(SPACE_3).bg(GREEN_MARK))
                        .child(
                            text("Scroll Horizontal")
                                .font_size(LABEL_SIZE)
                                .text_color(BLACK),
                        )
                        .child(
                            text("The green child is much wider than the nested viewport.")
                                .mt(SPACE_2)
                                .font_size(BODY_SIZE)
                                .text_color(BLACK),
                        ),
                ),
        )
}

fn run_example() {
    Application::new()
        .run(|cx| {
            cx.windows()
                .open(WindowOptions::new().title("NekoUI Nested Scroll"), |_| {
                    NestedScroll
                })?;
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
