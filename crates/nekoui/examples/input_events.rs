use nekoui::prelude::*;
use tracing_subscriber::EnvFilter;

const RECENT_LIMIT: usize = 8;

const BLACK: Color = Color::rgb(0x00, 0x00, 0x00);
const WHITE: Color = Color::rgb(0xFF, 0xFF, 0xFF);
const PANEL_GREY: Color = Color::rgb(0xAA, 0xAA, 0xAA);
const SURFACE_GREY: Color = Color::rgb(0xEE, 0xEE, 0xEE);
const YELLOW: Color = Color::rgb(0xFF, 0xEA, 0x00);

const SPACE_1: Length = Length::Px(4.0);
const SPACE_2: Length = Length::Px(8.0);
const SPACE_3: Length = Length::Px(12.0);

const BORDER_1: Length = Length::Px(1.0);
const TOP_BAR_HEIGHT: Length = Length::Px(36.0);
const RESET_WIDTH: Length = Length::Px(72.0);
const RESET_HEIGHT: Length = Length::Px(28.0);
const FIELD_HEIGHT: Length = Length::Px(76.0);
const TITLE_SIZE: Length = Length::Px(16.0);
const FIELD_TITLE_SIZE: Length = Length::Px(18.0);
const BODY_SIZE: Length = Length::Px(14.0);
const ROW_SIZE: Length = Length::Px(13.0);

#[derive(Debug, Default)]
struct InputEventsState {
    key_downs: usize,
    key_ups: usize,
    recent: Vec<String>,
}

impl InputEventsState {
    fn record_key_down(&mut self, event: impl Into<String>) {
        self.key_downs += 1;
        self.push_recent(event);
    }

    fn record_key_up(&mut self, event: impl Into<String>) {
        self.key_ups += 1;
        self.push_recent(event);
    }

    fn reset(&mut self) {
        self.key_downs = 0;
        self.key_ups = 0;
        self.recent.clear();
    }

    fn push_recent(&mut self, event: impl Into<String>) {
        self.recent.push(event.into());
        if self.recent.len() > RECENT_LIMIT {
            self.recent.remove(0);
        }
    }
}

#[derive(Debug)]
struct InputEventsExample {
    state: Entity<InputEventsState>,
}

impl Render for InputEventsExample {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let (key_downs, key_ups, recent) = self
            .state
            .read(cx, |state| {
                (state.key_downs, state.key_ups, state.recent.clone())
            })
            .unwrap();

        div()
            .key("input-events-root")
            .w(fill())
            .h(fill())
            .bg(PANEL_GREY)
            .child(top_bar(self.state.clone(), key_downs, key_ups))
            .child(div().key("top-divider").h(BORDER_1).w(fill()).bg(BLACK))
            .child(input_field(self.state.clone()))
            .child(recent_heading(recent.len()))
            .children(recent.into_iter().rev().enumerate().map(recent_row))
    }
}

fn top_bar(state: Entity<InputEventsState>, key_downs: usize, key_ups: usize) -> impl IntoElement {
    div()
        .key("top-bar")
        .display(Display::Flex)
        .w(fill())
        .h(TOP_BAR_HEIGHT)
        .px(SPACE_2)
        .py(SPACE_1)
        .gap(SPACE_2)
        .bg(WHITE)
        .child(
            text(format!(
                "Keyboard events     downs: {key_downs}     ups: {key_ups}"
            ))
            .key("title")
            .w(fill())
            .font_size(TITLE_SIZE)
            .text_color(BLACK)
            .line_clamp(1),
        )
        .child(
            div()
                .key("reset-button")
                .w(RESET_WIDTH)
                .h(RESET_HEIGHT)
                .px(SPACE_2)
                .py(SPACE_1)
                .bg(YELLOW)
                .focusable(true)
                .on_click_with(move |_event, cx| {
                    state.update(cx, |state, cx| {
                        state.reset();
                        cx.notify();
                        Ok(())
                    })?;
                    Ok(())
                })
                .child(text("Reset").font_size(TITLE_SIZE).text_color(BLACK)),
        )
}

fn input_field(state: Entity<InputEventsState>) -> impl IntoElement {
    div()
        .key("field-shell")
        .m(SPACE_3)
        .bg(BLACK)
        .p(BORDER_1)
        .child(
            div()
                .key("field")
                .w(fill())
                .h(FIELD_HEIGHT)
                .p(SPACE_2)
                .bg(WHITE)
                .focusable(true)
                .on_key_down_with({
                    let state = state.clone();
                    move |event, cx| {
                        let formatted = format_key_event(event);
                        state.update(cx, |state, cx| {
                            state.record_key_down(formatted);
                            cx.notify();
                            Ok(())
                        })?;
                        Ok(())
                    }
                })
                .on_key_up_with(move |event, cx| {
                    let formatted = format_key_event(event);
                    state.update(cx, |state, cx| {
                        state.record_key_up(formatted);
                        cx.notify();
                        Ok(())
                    })?;
                    Ok(())
                })
                .child(
                    text("Focusable key event logger")
                        .key("field-title")
                        .font_size(FIELD_TITLE_SIZE)
                        .text_color(BLACK),
                )
                .child(
                    text("Click this white field to focus it. It logs key down/up facts only; it is not a text editor.")
                        .key("field-copy")
                        .mt(SPACE_1)
                        .font_size(BODY_SIZE)
                        .text_color(BLACK)
                        .line_clamp(2),
                ),
        )
}

fn recent_heading(count: usize) -> impl IntoElement {
    div()
        .key("recent-heading")
        .mx(SPACE_3)
        .mb(SPACE_1)
        .bg(SURFACE_GREY)
        .p(SPACE_2)
        .child(
            text(format!("Recent keystroke rows ({count}/{RECENT_LIMIT})"))
                .font_size(BODY_SIZE)
                .text_color(BLACK),
        )
}

fn recent_row((index, event): (usize, String)) -> impl IntoElement {
    div()
        .key(format!("recent-row-{index}"))
        .mx(SPACE_3)
        .mb(SPACE_1)
        .bg(BLACK)
        .p(BORDER_1)
        .child(
            div()
                .key(format!("recent-row-inner-{index}"))
                .w(fill())
                .bg(WHITE)
                .p(SPACE_2)
                .child(
                    text(format!("{:02}  {event}", index + 1))
                        .font_size(ROW_SIZE)
                        .text_color(BLACK)
                        .line_clamp(1),
                ),
        )
}

fn format_key_event(event: &KeyEvent) -> String {
    format!(
        "kind={:?} logical_kind={} logical={} physical={} modifiers={} repeat={} synthetic={}",
        event.kind(),
        event.logical_key().kind_name(),
        event.logical_key().name(),
        event.physical_key().name(),
        event.modifiers().bits(),
        event.repeat(),
        event.synthetic()
    )
}

fn run_example() {
    Application::new()
        .run(|cx| {
            let state = cx.new_entity(|_| InputEventsState::default());
            cx.windows()
                .open(WindowOptions::new().title("NekoUI Input Events"), |_| {
                    InputEventsExample { state }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_down_and_up_counts_are_recorded() {
        let mut state = InputEventsState::default();

        state.record_key_down("kind=Down logical=Enter");
        state.record_key_up("kind=Up logical=Enter");

        assert_eq!(state.key_downs, 1);
        assert_eq!(state.key_ups, 1);
        assert_eq!(state.recent.len(), 2);
        assert!(state.recent[0].contains("kind=Down"));
        assert!(state.recent[1].contains("kind=Up"));
    }

    #[test]
    fn recent_events_are_capped_to_latest_rows() {
        let mut state = InputEventsState::default();

        for index in 0..(RECENT_LIMIT + 3) {
            state.push_recent(format!("event-{index}"));
        }

        assert_eq!(state.recent.len(), RECENT_LIMIT);
        assert_eq!(state.recent.first().map(String::as_str), Some("event-3"));
        assert_eq!(state.recent.last().map(String::as_str), Some("event-10"));
    }

    #[test]
    fn reset_clears_counts_and_recent_rows() {
        let mut state = InputEventsState::default();

        state.record_key_down("kind=Down logical=Space");
        state.record_key_up("kind=Up logical=Space");
        state.reset();

        assert_eq!(state.key_downs, 0);
        assert_eq!(state.key_ups, 0);
        assert!(state.recent.is_empty());
    }
}
