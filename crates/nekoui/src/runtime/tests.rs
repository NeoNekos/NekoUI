use std::cell::Cell;
use std::rc::Rc;

use crate::app::{Application, Context, Entity, Render};
use crate::diagnostic::{DirtyLane, DirtyLanes};
use crate::element::{Element, ElementKind, IntoElement, div, input, text};
use crate::error::{ErrorKind, NekoError};
use crate::interaction::{
    ImeInput, ImePreeditInput, Key, KeyEvent, KeyInput, Modifiers, PhysicalKey, PointerInput,
    ScrollDelta, ScrollPhase, TextInput, TextRange, WheelInput, WindowFocusInput,
};
use crate::layout::LayoutSize;
use crate::layout::text_viewport_placement;
use crate::layout::{LayoutPoint, LayoutRect};
use crate::platform::{ImePlatformRequest, PlatformFact};
use crate::runtime::Runtime;
use crate::runtime::command::RuntimeCommand;
use crate::style::{Display, Overflow, StyleExt, px};
use crate::window::WindowOptions;

#[derive(Debug)]
struct TestRoot {
    root: Element,
}

impl TestRoot {
    fn new(root: impl IntoElement) -> Self {
        Self {
            root: root.into_element(),
        }
    }
}

impl Render for TestRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.root.clone()
    }
}

#[derive(Debug)]
struct CountingRoot {
    renders: Rc<Cell<usize>>,
}

impl Render for CountingRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().key("root").child(text("stable").key("label"))
    }
}

#[derive(Debug)]
struct PointerLabelRoot {
    label: Entity<String>,
}

impl Render for PointerLabelRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let current = self.label.read(cx, Clone::clone).unwrap();
        div()
            .key("root")
            .child(text(current).key("target").on_click_with({
                let label = self.label.clone();
                move |_event, cx| {
                    label.update(cx, |label, cx| {
                        *label = "after".to_owned();
                        cx.notify();
                        Ok(())
                    })?;
                    Ok(())
                }
            }))
    }
}

#[derive(Debug)]
struct DeferredPointerNotifyRoot {
    label: Entity<String>,
    renders: Rc<Cell<usize>>,
    renders_seen_in_handler: Rc<Cell<usize>>,
}

impl Render for DeferredPointerNotifyRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let current = self.label.read(cx, Clone::clone).unwrap();
        div()
            .key("root")
            .child(text(current).key("target").on_click_with({
                let label = self.label.clone();
                let renders = self.renders.clone();
                let renders_seen_in_handler = self.renders_seen_in_handler.clone();
                move |_event, cx| {
                    label.update(cx, |label, cx| {
                        *label = "after".to_owned();
                        cx.notify();
                        Ok(())
                    })?;
                    renders_seen_in_handler.set(renders.get());
                    Ok(())
                }
            }))
    }
}

#[derive(Debug)]
struct HidingPointerRoot {
    visible: Entity<bool>,
    clicks: Rc<Cell<usize>>,
}

impl Render for HidingPointerRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let visible = self.visible.read(cx, |visible| *visible).unwrap();
        let target = text("target")
            .key("target")
            .h(px(20.0))
            .on_pointer_down_with({
                let visible = self.visible.clone();
                move |_event, cx| {
                    visible.update(cx, |visible, cx| {
                        *visible = false;
                        cx.notify();
                        Ok(())
                    })?;
                    Ok(())
                }
            })
            .on_click({
                let clicks = self.clicks.clone();
                move |_| clicks.set(clicks.get() + 1)
            });

        if visible {
            div().key("root").child(target)
        } else {
            div().key("root").child(target.display(Display::None))
        }
    }
}

#[derive(Debug)]
struct FailingPointerDownRoot {
    clicks: Rc<Cell<usize>>,
}

impl Render for FailingPointerDownRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div().key("root").child(
            text("target")
                .key("target")
                .h(px(20.0))
                .focusable(true)
                .on_pointer_down_with(|_event, _cx| {
                    Err(NekoError::diagnostic("pointer down failed"))
                })
                .on_click({
                    let clicks = self.clicks.clone();
                    move |_| clicks.set(clicks.get() + 1)
                }),
        )
    }
}

#[derive(Debug)]
struct FailingPointerUpRoot {
    clicks: Rc<Cell<usize>>,
    fail_once: Rc<Cell<bool>>,
}

impl Render for FailingPointerUpRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div().key("root").child(
            text("target")
                .key("target")
                .h(px(20.0))
                .on_pointer_up_with({
                    let fail_once = self.fail_once.clone();
                    move |_event, _cx| {
                        if fail_once.replace(false) {
                            Err(NekoError::diagnostic("pointer up failed"))
                        } else {
                            Ok(())
                        }
                    }
                })
                .on_click({
                    let clicks = self.clicks.clone();
                    move |_| clicks.set(clicks.get() + 1)
                }),
        )
    }
}

#[derive(Debug)]
struct KeyboardFocusRoot {
    key_downs: Rc<Cell<usize>>,
    key_ups: Rc<Cell<usize>>,
}

impl Render for KeyboardFocusRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .key("root")
            .child(
                text("focusable")
                    .key("focusable")
                    .h(px(20.0))
                    .focusable(true)
                    .on_key_down({
                        let key_downs = self.key_downs.clone();
                        move |event: &KeyEvent| {
                            assert_eq!(event.kind(), crate::interaction::KeyInputKind::Down);
                            assert_eq!(event.logical_key(), &Key::named("Enter"));
                            key_downs.set(key_downs.get() + 1);
                        }
                    })
                    .on_key_up({
                        let key_ups = self.key_ups.clone();
                        move |event: &KeyEvent| {
                            assert_eq!(event.kind(), crate::interaction::KeyInputKind::Up);
                            assert_eq!(event.logical_key(), &Key::named("Enter"));
                            key_ups.set(key_ups.get() + 1);
                        }
                    }),
            )
            .child(
                text("other")
                    .key("other")
                    .h(px(20.0))
                    .focusable(true)
                    .on_key_down({
                        let key_downs = self.key_downs.clone();
                        move |_| key_downs.set(key_downs.get() + 100)
                    }),
            )
    }
}

#[derive(Debug)]
struct NonFocusableKeyboardRoot {
    key_downs: Rc<Cell<usize>>,
}

impl Render for NonFocusableKeyboardRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .key("root")
            .child(text("target").key("target").h(px(20.0)).on_key_down({
                let key_downs = self.key_downs.clone();
                move |_| key_downs.set(key_downs.get() + 1)
            }))
    }
}

#[derive(Debug)]
struct HidingFocusableRoot {
    visible: Entity<bool>,
    key_downs: Rc<Cell<usize>>,
}

impl Render for HidingFocusableRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        let visible = self.visible.read(cx, |visible| *visible).unwrap();
        let target = text("target")
            .key("target")
            .h(px(20.0))
            .focusable(true)
            .on_key_down({
                let key_downs = self.key_downs.clone();
                move |_| key_downs.set(key_downs.get() + 1)
            });

        if visible {
            div().key("root").child(target)
        } else {
            div().key("root").child(target.display(Display::None))
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FocusMutationMode {
    Visible,
    Replaced,
    Destroyed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TextInputMutationMode {
    Visible,
    DisplayNone,
}

#[derive(Debug)]
struct MutatingFocusableRoot {
    mode: Entity<FocusMutationMode>,
    key_downs: Rc<Cell<usize>>,
}

impl Render for MutatingFocusableRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        match self.mode.read(cx, |mode| *mode).unwrap() {
            FocusMutationMode::Visible => div().key("root").child(
                text("target")
                    .key("target")
                    .h(px(20.0))
                    .focusable(true)
                    .on_key_down({
                        let key_downs = self.key_downs.clone();
                        move |_| key_downs.set(key_downs.get() + 1)
                    }),
            ),
            FocusMutationMode::Replaced => div().key("root").child(
                div()
                    .key("target")
                    .h(px(20.0))
                    .focusable(true)
                    .on_key_down({
                        let key_downs = self.key_downs.clone();
                        move |_| key_downs.set(key_downs.get() + 100)
                    }),
            ),
            FocusMutationMode::Destroyed => div().key("root"),
        }
    }
}

#[derive(Debug)]
struct MutatingInputRoot {
    mode: Rc<Cell<TextInputMutationMode>>,
}

impl Render for MutatingInputRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        let field = input("hi").key("field").h(px(20.0));
        match self.mode.get() {
            TextInputMutationMode::Visible => div().key("root").child(field),
            TextInputMutationMode::DisplayNone => {
                div().key("root").child(field.display(Display::None))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScrollMutationMode {
    Visible,
    DisplayNone,
    Replaced,
    Destroyed,
}

#[derive(Debug)]
struct MutatingScrollRoot {
    mode: Entity<ScrollMutationMode>,
}

impl Render for MutatingScrollRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        match self.mode.read(cx, |mode| *mode).unwrap() {
            ScrollMutationMode::Visible => div().key("root").child(
                div()
                    .key("scroll")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(text("content").key("content").h(px(220.0))),
            ),
            ScrollMutationMode::DisplayNone => div().key("root").child(
                div()
                    .key("scroll")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .display(Display::None)
                    .child(text("content").key("content").h(px(220.0))),
            ),
            ScrollMutationMode::Replaced => div()
                .key("root")
                .child(text("replacement").key("scroll").w(px(100.0)).h(px(100.0))),
            ScrollMutationMode::Destroyed => div().key("root"),
        }
    }
}

#[test]
fn runtime_drains_commands_fifo_with_monotonic_sequence() {
    let mut runtime = Runtime::new();
    let first = runtime.enqueue(RuntimeCommand::Notify);
    let second = runtime.enqueue(RuntimeCommand::Notify);

    let processed = runtime.drain_all().unwrap();

    assert_eq!(processed, [first, second]);
    assert!(first.raw() < second.raw());
}

#[test]
fn notify_marks_dirty_and_coalesces_redraws() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div()))
        .unwrap();

    runtime.request_notify();
    runtime.request_notify();
    runtime.drain_all().unwrap();

    let scheduler_state = runtime
        .state()
        .scheduler()
        .window_state(window.id())
        .unwrap();
    assert!(
        scheduler_state
            .dirty_lanes()
            .contains(DirtyLane::Build.flag())
    );
    assert!(
        !scheduler_state
            .dirty_lanes()
            .contains(DirtyLane::Paint.flag())
    );
    assert!(scheduler_state.pending_redraw());
    assert_eq!(runtime.state().retained_dirty().len(), 3);
    assert_eq!(runtime.performance_report().coalesced_redraws, 1);
}

#[test]
fn multiple_notify_commands_coalesce_to_one_root_render() {
    let mut runtime = Runtime::new();
    let renders = Rc::new(Cell::new(0));
    runtime
        .open_window(WindowOptions::new(), {
            let renders = renders.clone();
            move |_| CountingRoot { renders }
        })
        .unwrap();

    runtime.request_notify();
    runtime.request_notify();
    runtime.request_notify();
    runtime.drain_all().unwrap();

    assert_eq!(renders.get(), 2);
    assert_eq!(runtime.performance_report().notify_requests, 3);
}

#[test]
fn runtime_entity_keys_do_not_reuse_released_slots() {
    let mut runtime = Runtime::new();
    let first = runtime.reserve_entity_key();
    runtime.insert_reserved_entity(first, 1_i32);
    let second = runtime.reserve_entity_key();

    assert_ne!(first.raw_id(), second.raw_id());
}

#[test]
fn deferred_text_measurement_publishes_diagnostics_without_layout_snapshot() {
    let mut runtime = Runtime::new();
    runtime
        .state_mut()
        .font_manager_mut()
        .clear_fallback_for_test();

    let error = runtime
        .open_window(
            WindowOptions::new().logical_size(LayoutSize::new(120.0, 120.0)),
            |_| TestRoot::new(div().child(text("deferred text").key("label"))),
        )
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Diagnostic);
    let stats = runtime.state().last_layout_pass();
    assert!(stats.text_measure.query_count > 0);
    assert_eq!(
        stats.text_measure.deferred_count,
        stats.text_measure.query_count
    );
    assert_eq!(
        stats.layout_deferred_count,
        stats.text_measure.deferred_count
    );
    assert_eq!(
        stats.layout_blocked_on_text_count,
        stats.text_measure.deferred_count
    );
    assert_eq!(
        runtime.diagnostics().snapshot().counter("layout.defer"),
        stats.layout_deferred_count
    );
    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("layout.blocked_on_text"),
        stats.layout_blocked_on_text_count
    );
    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("text.measure.deferred"),
        stats.text_measure.deferred_count
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("text.measure.duration_micros")
            > 0
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "text.measure.query"
                    && record
                        .fields
                        .get("result")
                        .is_some_and(|value| value == "deferred")
                    && record.fields.contains_key("node_id")
                    && !record.fields.contains_key("text")
            })
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "layout.blocked_on_text"
                    && record
                        .fields
                        .get("result")
                        .is_some_and(|value| value == "deferred")
                    && record.fields.contains_key("reason")
            })
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "layout.defer"
                    && record.fields.contains_key("reason")
                    && record.fields.contains_key("blocked_on_text_count")
            })
    );
}

#[test]
fn pointer_input_commands_are_discrete_fifo_facts() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    let down = runtime.enqueue(RuntimeCommand::PointerInput {
        handle: window.into(),
        input: crate::interaction::PointerInput::down(LayoutPoint::new(1.0, 1.0)),
    });
    let up = runtime.enqueue(RuntimeCommand::PointerInput {
        handle: window.into(),
        input: crate::interaction::PointerInput::up(LayoutPoint::new(1.0, 1.0)),
    });
    let processed = runtime.drain_all().unwrap();

    assert_eq!(processed, [down, up]);
    assert!(down.raw() < up.raw());
    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("input.pointer_fact"),
        2
    );
}

#[test]
fn synthetic_pointer_click_updates_entity_and_rerenders() {
    let run = Application::new()
        .run_test(|cx| {
            let label = cx.new_entity(|_| "before".to_owned());
            let window = cx.windows().open(WindowOptions::new(), {
                let label = label.clone();
                move |_| PointerLabelRoot { label }
            })?;

            cx.windows()
                .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
            cx.windows()
                .pointer_up(window, LayoutPoint::new(1.0, 1.0))?;

            assert_eq!(label.read(cx, Clone::clone)?, "after");
            assert_eq!(
                cx.windows()
                    .retained_snapshot(window)?
                    .find_by_key("target")
                    .unwrap()
                    .text(),
                Some("after")
            );
            Ok(())
        })
        .unwrap();

    assert_eq!(run.diagnostics().counter("input.click_derived"), 1);
    assert!(run.diagnostics().counter("input.dispatch") >= 1);
}

#[test]
fn pointer_handler_notify_rerenders_after_handler_returns() {
    let renders = Rc::new(Cell::new(0));
    let renders_seen_in_handler = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let renders = renders.clone();
            let renders_seen_in_handler = renders_seen_in_handler.clone();
            move |cx| {
                let label = cx.new_entity(|_| "before".to_owned());
                let window = cx.windows().open(WindowOptions::new(), {
                    let renders = renders.clone();
                    let renders_seen_in_handler = renders_seen_in_handler.clone();
                    move |_| DeferredPointerNotifyRoot {
                        label,
                        renders,
                        renders_seen_in_handler,
                    }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows()
                    .pointer_up(window, LayoutPoint::new(1.0, 1.0))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(renders_seen_in_handler.get(), 1);
    assert_eq!(renders.get(), 2);
    assert_eq!(
        run.retained_snapshot(run.windows()[0].handle())
            .unwrap()
            .find_by_key("target")
            .unwrap()
            .text(),
        Some("after")
    );
}

#[test]
fn stale_pressed_target_cleanup_before_up_prevents_click() {
    let clicks = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let clicks = clicks.clone();
            move |cx| {
                let visible = cx.new_entity(|_| true);
                let window = cx.windows().open(WindowOptions::new(), {
                    let clicks = clicks.clone();
                    move |_| HidingPointerRoot { visible, clicks }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                assert!(
                    !cx.windows()
                        .retained_snapshot(window)?
                        .find_by_key("target")
                        .unwrap()
                        .participation()
                        .hit_test()
                );
                cx.windows()
                    .pointer_up(window, LayoutPoint::new(1.0, 1.0))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(clicks.get(), 0);
    assert_eq!(run.diagnostics().counter("input.stale_target"), 1);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.stale_target"
            && record
                .fields
                .get("state_kind")
                .is_some_and(|value| value == "pressed_cleanup")
    }));
}

#[test]
fn pointer_down_up_on_different_targets_does_not_click() {
    let clicks = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let clicks = clicks.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let clicks = clicks.clone();
                    move |_| {
                        TestRoot::new(
                            div()
                                .key("root")
                                .child(text("first").key("first").h(px(20.0)).on_click({
                                    let clicks = clicks.clone();
                                    move |_| clicks.set(clicks.get() + 1)
                                }))
                                .child(text("second").key("second").h(px(20.0))),
                        )
                    }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows()
                    .pointer_up(window, LayoutPoint::new(1.0, 25.0))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(clicks.get(), 0);
    assert_eq!(run.diagnostics().counter("input.click_derived"), 0);
}

#[test]
fn invalid_pointer_coordinates_return_invalid_input_without_hit_or_miss() {
    for position in [
        LayoutPoint::new(f32::NAN, 1.0),
        LayoutPoint::new(f32::INFINITY, 1.0),
        LayoutPoint::new(1.0, f32::NEG_INFINITY),
    ] {
        let error = Application::new()
            .run_test(|cx| {
                let window = cx
                    .windows()
                    .open(WindowOptions::new(), |_| TestRoot::new(div().key("root")))?;
                cx.windows()
                    .pointer_input(window, PointerInput::move_to(position))?;
                Ok(())
            })
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }

    let run = Application::new()
        .run_test(|cx| {
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| TestRoot::new(div().key("root")))?;
            let error = cx
                .windows()
                .pointer_input(
                    window,
                    PointerInput::move_to(LayoutPoint::new(f32::NAN, 1.0)),
                )
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            Ok(())
        })
        .unwrap();

    assert_eq!(run.diagnostics().counter("input.hit"), 0);
    assert_eq!(run.diagnostics().counter("input.miss"), 0);
    assert_eq!(run.diagnostics().counter("input.pointer_fact"), 0);
    assert!(run.diagnostics().records().iter().all(|record| {
        record.operation != "input.dispatch"
            || record
                .fields
                .get("event_kind")
                .is_none_or(|value| value != "hit_test")
    }));
}

#[test]
fn failing_pointer_down_does_not_leave_pressed_state_or_derive_click() {
    let clicks = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let clicks = clicks.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let clicks = clicks.clone();
                    move |_| FailingPointerDownRoot { clicks }
                })?;

                let error = cx
                    .windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))
                    .unwrap_err();
                assert_eq!(error.kind(), ErrorKind::Diagnostic);

                cx.windows()
                    .pointer_up(window, LayoutPoint::new(1.0, 1.0))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(clicks.get(), 0);
    assert_eq!(run.diagnostics().counter("input.click_derived"), 0);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "pointer_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "handler_error")
            && record
                .fields
                .get("error_kind")
                .is_some_and(|value| value == "Diagnostic")
    }));
}

#[test]
fn failing_pointer_move_rolls_back_hover_position_before_wheel() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("root")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(
                        text("content")
                            .key("content")
                            .h(px(250.0))
                            .on_pointer_move_with(|_event, _cx| {
                                Err(NekoError::diagnostic("pointer move failed"))
                            }),
                    ),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let target_node = retained.find_by_key("root").unwrap();
    let target =
        crate::interaction::InteractionTarget::new(target_node.id(), target_node.generation());

    let error = runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Diagnostic);
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 80.0), ScrollPhase::Moved),
        })
        .unwrap();

    assert_eq!(
        runtime.scroll_offset(window, target).unwrap(),
        LayoutPoint::ZERO
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "scroll.intent"
                    && record
                        .fields
                        .get("result")
                        .is_some_and(|value| value == "miss")
                    && record
                        .fields
                        .get("reason")
                        .is_some_and(|value| value == "no_hover_position")
            })
    );
}

#[test]
fn failing_pointer_up_clears_pressed_state_and_does_not_click_on_retry() {
    let clicks = Rc::new(Cell::new(0));
    let fail_once = Rc::new(Cell::new(true));
    let run = Application::new()
        .run_test({
            let clicks = clicks.clone();
            let fail_once = fail_once.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let clicks = clicks.clone();
                    let fail_once = fail_once.clone();
                    move |_| FailingPointerUpRoot { clicks, fail_once }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                let error = cx
                    .windows()
                    .pointer_up(window, LayoutPoint::new(1.0, 1.0))
                    .unwrap_err();
                assert_eq!(error.kind(), ErrorKind::Diagnostic);
                cx.windows()
                    .pointer_up(window, LayoutPoint::new(1.0, 1.0))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(clicks.get(), 0);
    assert_eq!(run.diagnostics().counter("input.click_derived"), 0);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "pointer_up")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "handler_error")
    }));
}

#[test]
fn pointer_down_on_explicit_focusable_target_sets_keyboard_focus_after_handler() {
    let key_downs = Rc::new(Cell::new(0));
    let key_ups = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            let key_ups = key_ups.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let key_downs = key_downs.clone();
                    let key_ups = key_ups.clone();
                    move |_| KeyboardFocusRoot { key_downs, key_ups }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                cx.windows()
                    .key_input(window, KeyInput::up(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 1);
    assert_eq!(key_ups.get(), 1);
    assert_eq!(run.diagnostics().counter("focus.transition"), 1);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "focus.transition"
            && record
                .fields
                .get("domain")
                .is_some_and(|value| value == "keyboard")
            && record
                .fields
                .get("reason")
                .is_some_and(|value| value == "pointer_down_default")
    }));
}

#[test]
fn pointer_down_on_non_focusable_target_does_not_focus_or_dispatch_keys() {
    let key_downs = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let key_downs = key_downs.clone();
                    move |_| NonFocusableKeyboardRoot { key_downs }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 0);
    assert_eq!(run.diagnostics().counter("focus.transition"), 0);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "key_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "no_focus")
    }));
}

#[test]
fn key_down_up_dispatch_to_current_keyboard_focus_only() {
    let key_downs = Rc::new(Cell::new(0));
    let key_ups = Rc::new(Cell::new(0));
    Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            let key_ups = key_ups.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let key_downs = key_downs.clone();
                    let key_ups = key_ups.clone();
                    move |_| KeyboardFocusRoot { key_downs, key_ups }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                cx.windows()
                    .key_input(window, KeyInput::up(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 1);
    assert_eq!(key_ups.get(), 1);
}

#[test]
fn key_fact_without_focus_is_processed_but_not_dispatched_to_handler() {
    let key_downs = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let key_downs = key_downs.clone();
                    move |_| NonFocusableKeyboardRoot { key_downs }
                })?;

                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 0);
    assert_eq!(run.diagnostics().counter("input.key_fact"), 1);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "key_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "no_focus")
    }));
}

#[test]
fn synthetic_keys_are_ignored_even_when_target_is_focused() {
    let key_downs = Rc::new(Cell::new(0));
    let key_ups = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            let key_ups = key_ups.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let key_downs = key_downs.clone();
                    let key_ups = key_ups.clone();
                    move |_| KeyboardFocusRoot { key_downs, key_ups }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows().key_input(
                    window,
                    KeyInput::down(Key::character("a")).with_synthetic(true),
                )?;
                cx.windows().key_input(
                    window,
                    KeyInput::up(Key::character("a")).with_synthetic(true),
                )?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 0);
    assert_eq!(key_ups.get(), 0);
    assert_eq!(run.diagnostics().counter("input.key_fact"), 2);
    let synthetic_ignored_count = run
        .diagnostics()
        .records()
        .iter()
        .filter(|record| {
            record.operation == "input.dispatch"
                && record
                    .fields
                    .get("result")
                    .is_some_and(|value| value == "synthetic_ignored")
                && record
                    .fields
                    .get("synthetic")
                    .is_some_and(|value| value == "true")
        })
        .count();
    assert_eq!(synthetic_ignored_count, 2);
}

#[test]
fn failing_pointer_down_prevents_keyboard_focus_default_and_rolls_back_pressed() {
    let clicks = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let clicks = clicks.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let clicks = clicks.clone();
                    move |_| FailingPointerDownRoot { clicks }
                })?;

                let error = cx
                    .windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))
                    .unwrap_err();
                assert_eq!(error.kind(), ErrorKind::Diagnostic);
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(run.diagnostics().counter("focus.transition"), 0);
    assert_eq!(run.diagnostics().counter("input.click_derived"), 0);
}

#[test]
fn focused_target_display_none_clears_keyboard_focus_and_records_stale() {
    let key_downs = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            move |cx| {
                let visible = cx.new_entity(|_| true);
                let window = cx.windows().open(WindowOptions::new(), {
                    let visible = visible.clone();
                    let key_downs = key_downs.clone();
                    move |_| HidingFocusableRoot { visible, key_downs }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                visible.update(cx, |visible, cx| {
                    *visible = false;
                    cx.notify();
                    Ok(())
                })?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 0);
    assert_eq!(run.diagnostics().counter("input.stale_target"), 2);
    assert_eq!(run.diagnostics().counter("runtime.stale_drop"), 2);
    assert_eq!(run.diagnostics().counter("focus.transition"), 2);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.stale_target"
            && record
                .fields
                .get("state_kind")
                .is_some_and(|value| value == "keyboard_focus_cleanup")
            && record.fields.contains_key("actual_generation")
    }));
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "key_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "no_focus")
    }));
}

#[test]
fn focused_target_replaced_clears_keyboard_focus_and_records_stale() {
    let key_downs = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            move |cx| {
                let mode = cx.new_entity(|_| FocusMutationMode::Visible);
                let window = cx.windows().open(WindowOptions::new(), {
                    let mode = mode.clone();
                    let key_downs = key_downs.clone();
                    move |_| MutatingFocusableRoot { mode, key_downs }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                mode.update(cx, |mode, cx| {
                    *mode = FocusMutationMode::Replaced;
                    cx.notify();
                    Ok(())
                })?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 0);
    assert_eq!(run.diagnostics().counter("input.stale_target"), 2);
    assert_eq!(run.diagnostics().counter("runtime.stale_drop"), 2);
    assert_eq!(run.diagnostics().counter("focus.transition"), 2);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.stale_target"
            && record
                .fields
                .get("state_kind")
                .is_some_and(|value| value == "keyboard_focus_cleanup")
            && record
                .fields
                .get("actual_generation")
                .is_some_and(|value| value == "none")
    }));
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "focus.transition"
            && record
                .fields
                .get("domain")
                .is_some_and(|value| value == "keyboard")
            && record
                .fields
                .get("reason")
                .is_some_and(|value| value == "stale_cleanup")
    }));
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "key_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "no_focus")
    }));
}

#[test]
fn focused_target_destroyed_clears_keyboard_focus_and_records_stale() {
    let key_downs = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            move |cx| {
                let mode = cx.new_entity(|_| FocusMutationMode::Visible);
                let window = cx.windows().open(WindowOptions::new(), {
                    let mode = mode.clone();
                    let key_downs = key_downs.clone();
                    move |_| MutatingFocusableRoot { mode, key_downs }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                mode.update(cx, |mode, cx| {
                    *mode = FocusMutationMode::Destroyed;
                    cx.notify();
                    Ok(())
                })?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 0);
    assert_eq!(run.diagnostics().counter("input.stale_target"), 2);
    assert_eq!(run.diagnostics().counter("runtime.stale_drop"), 2);
    assert_eq!(run.diagnostics().counter("focus.transition"), 2);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.stale_target"
            && record
                .fields
                .get("state_kind")
                .is_some_and(|value| value == "keyboard_focus_cleanup")
            && record
                .fields
                .get("actual_generation")
                .is_some_and(|value| value == "none")
    }));
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "focus.transition"
            && record
                .fields
                .get("domain")
                .is_some_and(|value| value == "keyboard")
            && record
                .fields
                .get("reason")
                .is_some_and(|value| value == "stale_cleanup")
    }));
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "key_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "no_focus")
    }));
}

#[test]
fn window_focus_false_does_not_clear_logical_keyboard_focus() {
    let key_downs = Rc::new(Cell::new(0));
    let key_ups = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let key_downs = key_downs.clone();
            let key_ups = key_ups.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let key_downs = key_downs.clone();
                    let key_ups = key_ups.clone();
                    move |_| KeyboardFocusRoot { key_downs, key_ups }
                })?;

                cx.windows()
                    .pointer_down(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows()
                    .window_focus_changed(window, WindowFocusInput::new(false))?;
                cx.windows()
                    .key_input(window, KeyInput::down(Key::named("Enter")))?;
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(key_downs.get(), 1);
    assert_eq!(run.diagnostics().counter("window.focus_fact"), 1);
    assert!(run.diagnostics().records().iter().any(|record| {
        record.operation == "window.focus_fact"
            && record
                .fields
                .get("focused")
                .is_some_and(|value| value == "false")
    }));
}

#[test]
fn platform_facts_are_fifo_runtime_ingress() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    let close_requested =
        runtime.enqueue(RuntimeCommand::PlatformFact(PlatformFact::CloseRequested {
            handle: window.into(),
        }));
    let resize = runtime.enqueue(RuntimeCommand::PlatformFact(
        PlatformFact::LogicalSizeChanged {
            handle: window.into(),
            logical_size: LayoutSize::new(320.0, 240.0),
        },
    ));
    let redraw = runtime.enqueue(RuntimeCommand::PlatformFact(
        PlatformFact::RedrawRequested {
            handle: window.into(),
        },
    ));

    let processed = runtime.drain_all().unwrap();

    assert_eq!(processed.len(), 4);
    assert_eq!(processed[0], close_requested);
    assert_eq!(processed[1], resize);
    assert_eq!(processed[2], redraw);
    assert!(processed[3].raw() > redraw.raw());
    let snapshot = runtime.diagnostics().snapshot();
    assert_eq!(snapshot.counter("platform.fact_queued"), 4);
    assert_eq!(snapshot.counter("platform.fact_processed"), 4);
}

#[test]
fn platform_close_requested_defaults_to_accepted_close() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::CloseRequested {
            handle: window.into(),
        })
        .unwrap();

    assert_eq!(
        runtime
            .state()
            .window(window.into())
            .unwrap()
            .lifecycle()
            .name(),
        "closing"
    );
    assert_eq!(runtime.take_platform_close_requests(), vec![window.any()]);
}

#[test]
fn close_confirmed_and_destroyed_are_separate_from_close_request() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::CloseRequested {
            handle: window.into(),
        })
        .unwrap();
    runtime.validate_window(window).unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::CloseConfirmed {
            handle: window.into(),
        })
        .unwrap();
    runtime.validate_window(window).unwrap();
    assert_eq!(
        runtime
            .state()
            .window(window.into())
            .unwrap()
            .lifecycle()
            .name(),
        "closing"
    );

    runtime
        .ingest_platform_fact(PlatformFact::Destroyed {
            handle: window.into(),
        })
        .unwrap();

    assert_eq!(
        runtime.validate_window(window).unwrap_err().kind(),
        ErrorKind::Stale
    );
}

#[test]
fn restore_does_not_reopen_closing_window_renderability() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::CloseRequested {
            handle: window.into(),
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::Restored {
            handle: window.into(),
        })
        .unwrap();

    let record = runtime.state().window(window.into()).unwrap();
    assert_eq!(record.lifecycle().name(), "closing");
    assert_eq!(record.renderability().name(), "closing");
}

#[test]
fn platform_window_created_marks_native_created() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::WindowCreated {
            handle: window.into(),
        })
        .unwrap();

    let record = runtime.state().window(window.into()).unwrap();
    assert!(record.native_created());
    assert_eq!(record.lifecycle().name(), "created_hidden");
    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("platform.window_created"),
        1
    );
}

#[test]
fn physical_zero_size_enters_not_renderable_without_redraw_spin() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::PhysicalSizeChanged {
            handle: window.into(),
            physical_size: crate::platform::PhysicalSize::new(0, 480),
        })
        .unwrap();

    let scheduler_state = runtime
        .state()
        .scheduler()
        .window_state(window.id())
        .unwrap();
    assert_eq!(scheduler_state.renderability().name(), "zero_size");
    assert!(!scheduler_state.pending_redraw());
    assert!(scheduler_state.suppressed_redraw_count() >= 1);
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("runtime.not_renderable")
            >= 1
    );
}

#[test]
fn restore_after_zero_size_requests_one_platform_redraw() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::PhysicalSizeChanged {
            handle: window.into(),
            physical_size: crate::platform::PhysicalSize::new(0, 480),
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::PhysicalSizeChanged {
            handle: window.into(),
            physical_size: crate::platform::PhysicalSize::new(640, 480),
        })
        .unwrap();

    let redraws = runtime.take_platform_redraw_requests();
    assert_eq!(redraws, vec![window.any()]);
    assert_eq!(
        runtime
            .state()
            .window(window.into())
            .unwrap()
            .renderability()
            .name(),
        "renderable"
    );
}

#[test]
fn platform_destroyed_makes_handle_stale() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::Destroyed {
            handle: window.into(),
        })
        .unwrap();

    assert_eq!(
        runtime.validate_window(window).unwrap_err().kind(),
        ErrorKind::Stale
    );
}

#[test]
fn platform_resize_and_scale_update_viewport_and_dirty_lanes() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::LogicalSizeChanged {
            handle: window.into(),
            logical_size: LayoutSize::new(320.0, 240.0),
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::ScaleFactorChanged {
            handle: window.into(),
            scale_factor: 2.0,
        })
        .unwrap();

    let record = runtime.state().window(window.into()).unwrap();
    assert_eq!(
        record.viewport().logical_size(),
        LayoutSize::new(320.0, 240.0)
    );
    assert_eq!(record.viewport().scale_factor(), 2.0);
    let lanes = runtime.state().reported_dirty_lanes(window.id());
    assert!(lanes.contains(DirtyLane::Layout.flag()));
    assert!(lanes.contains(DirtyLane::Surface.flag()));
    assert!(lanes.contains(DirtyLane::Paint.flag()));
}

#[test]
fn redraw_requested_consumes_pending_redraw() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::LogicalSizeChanged {
            handle: window.into(),
            logical_size: LayoutSize::new(320.0, 240.0),
        })
        .unwrap();
    assert!(
        runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .pending_redraw()
    );

    runtime
        .ingest_platform_fact(PlatformFact::RedrawRequested {
            handle: window.into(),
        })
        .unwrap();

    assert!(
        !runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .pending_redraw()
    );
}

#[test]
fn platform_resize_defers_frame_work_until_redraw_requested() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();
    let initial_report = runtime.performance_report();

    runtime
        .ingest_platform_fact(PlatformFact::LogicalSizeChanged {
            handle: window.into(),
            logical_size: LayoutSize::new(320.0, 240.0),
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::LogicalSizeChanged {
            handle: window.into(),
            logical_size: LayoutSize::new(640.0, 480.0),
        })
        .unwrap();

    let after_resize = runtime.performance_report();
    assert_eq!(
        after_resize.layout.pass_count,
        initial_report.layout.pass_count
    );
    assert_eq!(
        after_resize.scene.compile_count,
        initial_report.scene.compile_count
    );
    assert_eq!(
        after_resize.render.frame_graph_count,
        initial_report.render.frame_graph_count
    );
    assert_eq!(
        runtime
            .state()
            .window(window.into())
            .unwrap()
            .viewport()
            .logical_size(),
        LayoutSize::new(640.0, 480.0)
    );

    runtime
        .ingest_platform_fact(PlatformFact::RedrawRequested {
            handle: window.into(),
        })
        .unwrap();

    let after_redraw = runtime.performance_report();
    assert_eq!(
        after_redraw.layout.pass_count,
        initial_report.layout.pass_count + 1
    );
    assert_eq!(
        after_redraw.scene.compile_count,
        initial_report.scene.compile_count + 1
    );
    assert_eq!(
        after_redraw.render.frame_graph_count,
        initial_report.render.frame_graph_count + 1
    );
    assert_eq!(
        runtime
            .layout_snapshot(window)
            .unwrap()
            .viewport()
            .logical_size(),
        LayoutSize::new(640.0, 480.0)
    );
}

#[test]
fn platform_scale_defers_frame_work_until_redraw_requested() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();
    let initial_report = runtime.performance_report();

    runtime
        .ingest_platform_fact(PlatformFact::ScaleFactorChanged {
            handle: window.into(),
            scale_factor: 1.5,
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::ScaleFactorChanged {
            handle: window.into(),
            scale_factor: 2.0,
        })
        .unwrap();

    let after_scale = runtime.performance_report();
    assert_eq!(
        after_scale.layout.pass_count,
        initial_report.layout.pass_count
    );
    assert_eq!(
        after_scale.scene.compile_count,
        initial_report.scene.compile_count
    );
    assert_eq!(
        after_scale.render.frame_graph_count,
        initial_report.render.frame_graph_count
    );
    assert_eq!(
        runtime
            .state()
            .window(window.into())
            .unwrap()
            .viewport()
            .scale_factor(),
        2.0
    );

    runtime
        .ingest_platform_fact(PlatformFact::RedrawRequested {
            handle: window.into(),
        })
        .unwrap();

    let after_redraw = runtime.performance_report();
    assert_eq!(
        after_redraw.layout.pass_count,
        initial_report.layout.pass_count + 1
    );
    assert_eq!(
        after_redraw.scene.compile_count,
        initial_report.scene.compile_count + 1
    );
    assert_eq!(
        after_redraw.render.frame_graph_count,
        initial_report.render.frame_graph_count + 1
    );
    assert_eq!(
        runtime
            .layout_snapshot(window)
            .unwrap()
            .viewport()
            .scale_factor(),
        2.0
    );
}

#[test]
fn platform_pointer_fact_uses_runtime_input_path() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::PointerInput {
            handle: window.into(),
            input: crate::interaction::PointerInput::move_to(LayoutPoint::new(1.0, 1.0)),
        })
        .unwrap();

    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("input.pointer_fact"),
        1
    );
}

#[test]
fn key_down_up_platform_facts_are_discrete_fifo_facts() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    let down = runtime.enqueue(RuntimeCommand::PlatformFact(PlatformFact::KeyInput {
        handle: window.into(),
        input: KeyInput::down(Key::named("Enter"))
            .with_physical_key(PhysicalKey::code("Enter"))
            .with_modifiers(Modifiers::new(true, false, false, false)),
    }));
    let up = runtime.enqueue(RuntimeCommand::PlatformFact(PlatformFact::KeyInput {
        handle: window.into(),
        input: KeyInput::up(Key::named("Enter")).with_physical_key(PhysicalKey::code("Enter")),
    }));

    let processed = runtime.drain_all().unwrap();

    assert_eq!(processed, [down, up]);
    assert!(down.raw() < up.raw());
    let snapshot = runtime.diagnostics().snapshot();
    assert_eq!(snapshot.counter("platform.fact_queued"), 2);
    assert_eq!(snapshot.counter("platform.fact_processed"), 2);
    assert_eq!(snapshot.counter("input.key_fact"), 2);
    let key_records: Vec<_> = snapshot
        .records()
        .iter()
        .filter(|record| record.operation == "input.key_fact")
        .collect();
    assert_eq!(key_records.len(), 2);
    assert!(
        key_records[0]
            .fields
            .get("kind")
            .is_some_and(|value| value == "down")
    );
    assert!(
        key_records[1]
            .fields
            .get("kind")
            .is_some_and(|value| value == "up")
    );
}

#[test]
fn keydown_is_not_text_input_and_synthetic_key_does_not_invoke_behavior() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| CountingRoot {
            renders: Rc::new(Cell::new(0)),
        })
        .unwrap();

    let before = runtime.performance_report();
    runtime
        .ingest_platform_fact(PlatformFact::KeyInput {
            handle: window.into(),
            input: KeyInput::down(Key::character("a")).with_synthetic(true),
        })
        .unwrap();
    let after = runtime.performance_report();

    assert_eq!(after.notify_requests, before.notify_requests);
    assert_eq!(after.retained.diff_count, before.retained.diff_count);
    assert_eq!(after.text.measure_count, before.text.measure_count);
    let snapshot = runtime.diagnostics().snapshot();
    assert_eq!(snapshot.counter("input.key_fact"), 1);
    assert_eq!(snapshot.counter("input.dispatch"), 1);
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "input.key_fact"
            && record
                .fields
                .get("logical_kind")
                .is_some_and(|value| value == "character")
            && record
                .fields
                .get("logical_key")
                .is_some_and(|value| value == "a")
            && record
                .fields
                .get("synthetic")
                .is_some_and(|value| value == "true")
    }));
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "input.dispatch"
            && record
                .fields
                .get("event_kind")
                .is_some_and(|value| value == "key_down")
            && record
                .fields
                .get("result")
                .is_some_and(|value| value == "synthetic_ignored")
    }));
}

#[test]
fn text_input_platform_fact_inserts_into_focused_input_without_logical_key_text() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::KeyInput {
            handle: window.into(),
            input: KeyInput::down(Key::character("x")),
        })
        .unwrap();
    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some(String::new()),
    );

    runtime
        .ingest_platform_fact(PlatformFact::TextInput {
            handle: window.into(),
            input: TextInput::commit("é"),
        })
        .unwrap();

    let field = runtime
        .retained_snapshot(window)
        .unwrap()
        .find_by_key("field")
        .unwrap()
        .clone();
    assert_eq!(field.display_text(), Some("é".to_owned()));
    assert_eq!(
        field.text_block().unwrap().selection(),
        TextRange::collapsed("é".len())
    );
    let diagnostics = runtime.diagnostics().snapshot();
    assert_eq!(diagnostics.counter("text.input_fact"), 1);
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "text.input_fact"
            && record
                .fields
                .get("kind")
                .is_some_and(|value| value == "text_input")
    }));
}

#[test]
fn backspace_deletes_previous_utf8_char_in_focused_input() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("aé").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::KeyInput {
            handle: window.into(),
            input: KeyInput::down(Key::named("Backspace")),
        })
        .unwrap();

    let field = runtime
        .retained_snapshot(window)
        .unwrap()
        .find_by_key("field")
        .unwrap()
        .clone();
    assert_eq!(field.display_text(), Some("a".to_owned()));
    assert_eq!(
        field.text_block().unwrap().selection(),
        TextRange::collapsed(1)
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "text.edit"
                    && record
                        .fields
                        .get("kind")
                        .is_some_and(|value| value == "delete_backward")
                    && record
                        .fields
                        .get("result")
                        .is_some_and(|value| value == "mutated")
            })
    );
}

#[test]
fn ime_preedit_and_commit_update_text_input_and_candidate_rect_from_text_layout() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("hi").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    let requests = runtime.ime_requests(window).unwrap();
    assert!(
        requests
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: true }))
    );
    assert!(
        requests
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Purpose { .. }))
    );
    let cursor_area = requests
        .iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(*rect),
            _ => None,
        })
        .expect("text input focus should publish an IME cursor area");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let text_layout = field_layout.text_layout().unwrap();
    let expected_cursor_area =
        text_viewport_placement(ElementKind::Input, field_layout.content_rect(), text_layout)
            .visible_caret_rect();
    assert_eq!(cursor_area, expected_cursor_area);
    assert_eq!(cursor_area.width(), 1.0);
    assert!(cursor_area.x() > field_layout.content_rect().x());

    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Preedit(ImePreeditInput::new(
                "文",
                Some(TextRange::collapsed("文".len())),
            )),
        })
        .unwrap();
    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi文".to_owned()),
    );

    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Commit(TextInput::commit("!")),
        })
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi!".to_owned()),
    );
    let semantic = runtime.semantic_snapshot(window).unwrap();
    let field = semantic.find_by_key("field").unwrap();
    assert_eq!(field.value(), Some("hi!"));
    assert!(field.state().editable());
    assert!(field.state().focused());
    assert_eq!(field.state().selection(), Some(TextRange::collapsed(3)));
    assert_eq!(field.state().composition(), None);
    assert_eq!(field.state().composition_cursor(), None);
    let diagnostics = runtime.diagnostics().snapshot();
    assert_eq!(diagnostics.counter("ime.preedit"), 1);
    assert_eq!(diagnostics.counter("ime.commit"), 1);
    assert_eq!(diagnostics.counter("text.input_fact"), 2);
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "text.input_fact"
            && record
                .fields
                .get("kind")
                .is_some_and(|value| value == "preedit")
            && record
                .fields
                .get("text_len")
                .is_some_and(|value| value == &"文".len().to_string())
    }));
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "text.input_fact"
            && record
                .fields
                .get("kind")
                .is_some_and(|value| value == "ime_commit")
            && record
                .fields
                .get("text_len")
                .is_some_and(|value| value == "1")
    }));
}

#[test]
fn ime_cursor_area_refreshes_after_text_mutation_redraw_not_before() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("hi").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    let initial_cursor_area = runtime
        .take_platform_ime_requests(window.into())
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("text input focus should publish initial cursor area");

    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Preedit(ImePreeditInput::new(
                "文",
                Some(TextRange::collapsed("文".len())),
            )),
        })
        .unwrap();
    assert!(
        runtime
            .take_platform_ime_requests(window.into())
            .unwrap()
            .into_iter()
            .all(|request| !matches!(request, ImePlatformRequest::CursorArea { .. }))
    );

    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    let refreshed_cursor_area = runtime
        .take_platform_ime_requests(window.into())
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("new layout publication should refresh cursor area");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let expected = text_viewport_placement(
        ElementKind::Input,
        field_layout.content_rect(),
        field_layout.text_layout().unwrap(),
    )
    .visible_caret_rect();

    assert_ne!(refreshed_cursor_area, initial_cursor_area);
    assert_eq!(refreshed_cursor_area, expected);
}

#[test]
fn ime_candidate_rect_for_long_input_uses_visible_caret() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    input("AAAA AAAA AAAA AAAA")
                        .key("field")
                        .w(px(36.0))
                        .font_size(px(12.0)),
                ),
            )
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    let cursor_area = runtime
        .ime_requests(window)
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("text input focus should publish a cursor area");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let content = field_layout.content_rect();
    let text_layout = field_layout.text_layout().unwrap();
    let placement = text_viewport_placement(ElementKind::Input, content, text_layout);

    assert!(placement.input_inline_scroll() > 0.0);
    assert_eq!(cursor_area, placement.visible_caret_rect());
    assert!(cursor_area.x() >= content.x());
    assert!(cursor_area.x() + cursor_area.width() <= content.x() + content.width());
}

#[test]
fn ime_candidate_rect_for_tiny_input_is_clamped_to_viewport() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    input("AAAA AAAA AAAA")
                        .key("field")
                        .w(px(0.5))
                        .font_size(px(12.0)),
                ),
            )
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(0.25, 1.0)))
        .unwrap();

    let cursor_area = runtime
        .ime_requests(window)
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("text input focus should publish a cursor area");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let content = field_layout.content_rect();
    let text_layout = field_layout.text_layout().unwrap();
    let placement = text_viewport_placement(ElementKind::Input, content, text_layout);

    assert_eq!(cursor_area, placement.visible_caret_rect());
    assert!(cursor_area.x() >= content.x());
    assert!(cursor_area.x() + cursor_area.width() <= content.x() + content.width());
    assert_eq!(cursor_area.width(), content.width());
}

#[test]
fn ime_candidate_rect_for_emoji_input_uses_effective_line_box() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    input("Hello 😀")
                        .key("field")
                        .h(px(120.0))
                        .font_size(px(48.0)),
                ),
            )
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    let cursor_area = runtime
        .ime_requests(window)
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("text input focus should publish a cursor area");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let text_layout = field_layout.text_layout().unwrap();
    let placement =
        text_viewport_placement(ElementKind::Input, field_layout.content_rect(), text_layout);
    let line = text_layout.lines()[0];

    assert_eq!(text_layout.metrics().line_count, 1);
    assert!(line.height() >= line.required_height());
    assert_eq!(cursor_area, placement.visible_caret_rect());
    assert_eq!(cursor_area.height(), line.height());
}

#[test]
fn ime_disabled_after_commit_keeps_text_input_focus_and_does_not_disable_platform_ime() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("hi").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    let _ = runtime.take_platform_ime_requests(window.into()).unwrap();
    let field = runtime
        .retained_snapshot(window)
        .unwrap()
        .find_by_key("field")
        .unwrap()
        .clone();
    let target = crate::interaction::InteractionTarget::new(field.id(), field.generation());
    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Preedit(ImePreeditInput::new("ni", None)),
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Commit(TextInput::commit("你")),
        })
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Disabled,
        })
        .unwrap();

    assert_eq!(
        runtime
            .state()
            .interaction(window.id())
            .unwrap()
            .text_input_focus(),
        Some(target),
    );
    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi你".to_owned()),
    );
    let ime_requests = runtime.take_platform_ime_requests(window.into()).unwrap();
    assert!(
        ime_requests
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: true }))
    );
    assert!(
        !ime_requests
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: false }))
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "ime.transition"
                    && record
                        .fields
                        .get("transition")
                        .is_some_and(|value| value == "disabled")
                    && record
                        .fields
                        .get("target_id")
                        .is_some_and(|value| value != "none")
            })
    );
}

#[test]
fn ime_disabled_after_active_preedit_refreshes_cursor_area_after_redraw_not_before() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("hi").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    let _ = runtime.take_platform_ime_requests(window.into()).unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Preedit(ImePreeditInput::new(
                "文",
                Some(TextRange::collapsed("文".len())),
            )),
        })
        .unwrap();
    let _ = runtime.take_platform_ime_requests(window.into()).unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Disabled,
        })
        .unwrap();
    let immediate_requests = runtime.take_platform_ime_requests(window.into()).unwrap();
    assert!(
        immediate_requests
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: true }))
    );
    assert!(
        immediate_requests
            .iter()
            .all(|request| !matches!(request, ImePlatformRequest::CursorArea { .. }))
    );

    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    let refreshed_cursor_area = runtime
        .take_platform_ime_requests(window.into())
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("new layout publication should refresh cursor area after disabled preedit clear");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let expected = text_viewport_placement(
        ElementKind::Input,
        field_layout.content_rect(),
        field_layout.text_layout().unwrap(),
    )
    .visible_caret_rect();

    assert_eq!(refreshed_cursor_area, expected);
}

#[test]
fn ime_candidate_rect_tracks_scroll_offset_and_shaped_caret_geometry() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    div()
                        .key("scroll")
                        .w(px(120.0))
                        .h(px(40.0))
                        .overflow(Overflow::Scroll)
                        .child(div().h(px(80.0)).child(input("hello").key("field"))),
                ),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let scroll_node = retained.find_by_key("scroll").unwrap();
    let scroll_target =
        crate::interaction::InteractionTarget::new(scroll_node.id(), scroll_node.generation());
    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 5.0), ScrollPhase::Moved),
        })
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    assert_eq!(
        runtime.scroll_offset(window, scroll_target).unwrap().y(),
        5.0
    );
    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    let cursor_area = runtime
        .ime_requests(window)
        .unwrap()
        .into_iter()
        .find_map(|request| match request {
            ImePlatformRequest::CursorArea { rect } => Some(rect),
            _ => None,
        })
        .expect("text input focus should publish a cursor area");
    let layout = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout.find_by_key("field").unwrap();
    let text_layout = field_layout.text_layout().unwrap();
    let expected =
        text_viewport_placement(ElementKind::Input, field_layout.content_rect(), text_layout)
            .visible_caret_rect()
            .translate(0.0, -5.0);

    assert_eq!(cursor_area, expected);
}

#[test]
fn stale_text_input_target_clears_focus_composition_and_disables_ime() {
    let mut runtime = Runtime::new();
    let mode = Rc::new(Cell::new(TextInputMutationMode::Visible));
    let window = runtime
        .open_window(WindowOptions::new(), {
            let mode = mode.clone();
            move |_| MutatingInputRoot { mode }
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    let _ = runtime.take_platform_ime_requests(window.into()).unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Preedit(ImePreeditInput::new("文", None)),
        })
        .unwrap();
    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi文".to_owned()),
    );

    mode.set(TextInputMutationMode::DisplayNone);
    runtime.request_notify();
    runtime.drain_all().unwrap();

    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi".to_owned()),
    );
    assert_eq!(
        runtime
            .state()
            .interaction(window.id())
            .unwrap()
            .text_input_focus(),
        None,
    );
    assert!(
        runtime
            .take_platform_ime_requests(window.into())
            .unwrap()
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: false }))
    );
    let diagnostics = runtime.diagnostics().snapshot();
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "input.stale_target"
            && record
                .fields
                .get("state_kind")
                .is_some_and(|value| value == "text_input_focus_cleanup")
    }));
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "focus.transition"
            && record
                .fields
                .get("domain")
                .is_some_and(|value| value == "text_input")
            && record
                .fields
                .get("reason")
                .is_some_and(|value| value == "stale_cleanup")
    }));
}

#[test]
fn window_blur_clears_text_input_focus_composition_and_disables_platform_ime() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(input("hi").key("field")))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    let _ = runtime.take_platform_ime_requests(window.into()).unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::ImeInput {
            handle: window.into(),
            input: ImeInput::Preedit(ImePreeditInput::new(
                "文",
                Some(TextRange::collapsed("文".len())),
            )),
        })
        .unwrap();
    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi文".to_owned()),
    );

    runtime
        .ingest_platform_fact(PlatformFact::WindowFocusChanged {
            handle: window.into(),
            input: WindowFocusInput::new(false),
        })
        .unwrap();

    assert_eq!(
        runtime
            .retained_snapshot(window)
            .unwrap()
            .find_by_key("field")
            .unwrap()
            .display_text(),
        Some("hi".to_owned()),
    );
    assert_eq!(
        runtime
            .state()
            .interaction(window.id())
            .unwrap()
            .text_input_focus(),
        None,
    );
    assert!(
        runtime
            .take_platform_ime_requests(window.into())
            .unwrap()
            .iter()
            .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: false }))
    );
    let diagnostics = runtime.diagnostics().snapshot();
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "focus.transition"
            && record
                .fields
                .get("domain")
                .is_some_and(|value| value == "text_input")
            && record
                .fields
                .get("reason")
                .is_some_and(|value| value == "window_unfocused")
    }));
}

#[test]
fn pointer_down_away_from_input_clears_text_focus_composition_and_disables_ime() {
    for (point, reason) in [
        (LayoutPoint::new(99.0, 99.0), "pointer_down_miss"),
        (LayoutPoint::new(1.0, 24.0), "pointer_down_non_focusable"),
    ] {
        let mut runtime = Runtime::new();
        let window = runtime
            .open_window(WindowOptions::new(), |_| {
                TestRoot::new(
                    div()
                        .key("root")
                        .w(px(40.0))
                        .h(px(40.0))
                        .child(input("hi").key("field").h(px(20.0)))
                        .child(div().key("surface").h(px(20.0))),
                )
            })
            .unwrap();

        runtime
            .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
            .unwrap();
        let _ = runtime.take_platform_ime_requests(window.into()).unwrap();
        runtime
            .ingest_platform_fact(PlatformFact::ImeInput {
                handle: window.into(),
                input: ImeInput::Preedit(ImePreeditInput::new("文", None)),
            })
            .unwrap();
        assert_eq!(
            runtime
                .retained_snapshot(window)
                .unwrap()
                .find_by_key("field")
                .unwrap()
                .display_text(),
            Some("hi文".to_owned()),
        );

        runtime
            .pointer_input(window, PointerInput::down(point))
            .unwrap();

        assert_eq!(
            runtime
                .retained_snapshot(window)
                .unwrap()
                .find_by_key("field")
                .unwrap()
                .display_text(),
            Some("hi".to_owned()),
        );
        assert_eq!(
            runtime
                .state()
                .interaction(window.id())
                .unwrap()
                .text_input_focus(),
            None,
        );
        assert!(
            runtime
                .take_platform_ime_requests(window.into())
                .unwrap()
                .iter()
                .any(|request| matches!(request, ImePlatformRequest::Allowed { allowed: false }))
        );
        let diagnostics = runtime.diagnostics().snapshot();
        assert!(diagnostics.records().iter().any(|record| {
            record.operation == "focus.transition"
                && record
                    .fields
                    .get("domain")
                    .is_some_and(|value| value == "text_input")
                && record
                    .fields
                    .get("reason")
                    .is_some_and(|value| value == reason)
        }));
    }
}

#[test]
fn ime_candidate_rect_ignores_display_none_retained_siblings_during_layout_lookup() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    div()
                        .key("scroll")
                        .w(px(120.0))
                        .h(px(40.0))
                        .overflow(Overflow::Scroll)
                        .child(
                            div()
                                .key("hidden")
                                .display(Display::None)
                                .child(text("hidden")),
                        )
                        .child(div().h(px(80.0)).child(input("hello").key("field"))),
                ),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    assert!(retained.find_by_key("hidden").is_some());
    let layout = runtime.layout_snapshot(window).unwrap();
    assert!(layout.find_by_key("hidden").is_none());

    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 5.0), ScrollPhase::Moved),
        })
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    assert!(runtime.ime_requests(window).unwrap().iter().any(|request| {
        matches!(request, ImePlatformRequest::CursorArea { rect } if rect.y() < layout.find_by_key("field").unwrap().content_rect().y())
    }));
}

#[test]
fn modifiers_changed_updates_runtime_input_state_and_diagnostics() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();
    let modifiers = Modifiers::new(true, true, false, false);

    runtime
        .ingest_platform_fact(PlatformFact::ModifiersChanged {
            handle: window.into(),
            modifiers,
        })
        .unwrap();

    assert_eq!(
        runtime
            .state()
            .interaction(window.id())
            .unwrap()
            .modifiers(),
        modifiers
    );
    let snapshot = runtime.diagnostics().snapshot();
    assert_eq!(snapshot.counter("input.modifiers_fact"), 1);
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "input.modifiers_fact"
            && record
                .fields
                .get("shift")
                .is_some_and(|value| value == "true")
            && record
                .fields
                .get("ctrl")
                .is_some_and(|value| value == "true")
            && record
                .fields
                .get("logo")
                .is_some_and(|value| value == "false")
            && record
                .fields
                .get("command")
                .is_some_and(|value| value == &cfg!(not(target_os = "macos")).to_string())
    }));
}

#[test]
fn wheel_line_pixel_units_and_phase_are_preserved() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("root")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(text("wide").key("child").w(px(100.0)).h(px(200.0))),
            )
        })
        .unwrap();
    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::lines(1.0, -2.0), ScrollPhase::Started),
        })
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(3.5, -4.5), ScrollPhase::Cancelled),
        })
        .unwrap();

    let snapshot = runtime.diagnostics().snapshot();
    assert_eq!(snapshot.counter("input.wheel_fact"), 2);
    let records: Vec<_> = snapshot
        .records()
        .iter()
        .filter(|record| record.operation == "input.wheel_fact")
        .collect();
    assert_eq!(records.len(), 2);
    assert!(
        records[0]
            .fields
            .get("delta_unit")
            .is_some_and(|value| value == "lines")
    );
    assert!(
        records[0]
            .fields
            .get("phase")
            .is_some_and(|value| value == "started")
    );
    assert!(
        records[1]
            .fields
            .get("delta_unit")
            .is_some_and(|value| value == "pixels")
    );
    assert!(
        records[1]
            .fields
            .get("phase")
            .is_some_and(|value| value == "cancelled")
    );
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "scroll.intent"
            && record
                .fields
                .get("delta_unit")
                .is_some_and(|value| value == "lines")
            && record
                .fields
                .get("phase")
                .is_some_and(|value| value == "started")
    }));
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "scroll.intent"
            && record
                .fields
                .get("delta_unit")
                .is_some_and(|value| value == "pixels")
            && record
                .fields
                .get("phase")
                .is_some_and(|value| value == "cancelled")
    }));
}

#[test]
fn wheel_over_scrollable_target_updates_offset_and_clamps() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("root")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(text("content").key("content").w(px(100.0)).h(px(250.0))),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let target_node = retained.find_by_key("root").unwrap();
    let target =
        crate::interaction::InteractionTarget::new(target_node.id(), target_node.generation());

    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 80.0), ScrollPhase::Moved),
        })
        .unwrap();
    assert_eq!(runtime.scroll_offset(window, target).unwrap().y(), 80.0);

    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 1000.0), ScrollPhase::Moved),
        })
        .unwrap();
    assert_eq!(runtime.scroll_offset(window, target).unwrap().y(), 150.0);
    let snapshot = runtime.diagnostics().snapshot();
    assert!(snapshot.counter("scroll.offset") >= 2);
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "scroll.clamp"
            && record
                .fields
                .get("clamped_y")
                .is_some_and(|value| value == "150")
    }));
}

#[test]
fn scroll_offset_change_does_not_advance_layout_generation() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("root")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(text("top").key("top").h(px(100.0)))
                    .child(text("content").key("content").h(px(120.0))),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let content_id = retained.find_by_key("content").unwrap().id();
    let layout_generation_before = runtime.layout_snapshot(window).unwrap().generation();
    let scene_generation_before = runtime.scene_snapshot(window).unwrap().generation();
    let report_before = runtime.performance_report();

    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .state_mut()
        .scheduler_mut()
        .take_dirty_lanes(window.id(), DirtyLanes::all());
    runtime.state_mut().clear_consumed_dirty_lanes(window.id());
    let lanes_before = runtime.state().reported_dirty_lanes(window.id());
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 40.0), ScrollPhase::Moved),
        })
        .unwrap();

    let layout_generation_after_wheel = runtime.layout_snapshot(window).unwrap().generation();
    let scene_after_wheel = runtime.scene_snapshot(window).unwrap();
    let report_after_wheel = runtime.performance_report();
    assert_eq!(layout_generation_after_wheel, layout_generation_before);
    assert_eq!(scene_after_wheel.generation(), scene_generation_before);
    assert_eq!(
        report_after_wheel.layout.pass_count,
        report_before.layout.pass_count
    );
    assert_eq!(
        report_after_wheel.scene.compile_count,
        report_before.scene.compile_count
    );
    assert_ne!(
        scene_after_wheel
            .hit_test()
            .hit_test(LayoutPoint::new(1.0, 1.0))
            .map(|entry| entry.node_id()),
        Some(content_id)
    );
    let lanes_after = runtime.state().reported_dirty_lanes(window.id());
    let added_lanes = lanes_after - lanes_before;
    assert!(!added_lanes.contains(DirtyLane::Layout.flag()));
    assert!(!added_lanes.contains(DirtyLane::Style.flag()));
    assert!(!added_lanes.contains(DirtyLane::Text.flag()));
    assert!(added_lanes.contains(DirtyLane::Paint.flag()));
    assert!(added_lanes.contains(DirtyLane::Semantics.flag()));
    assert!(
        runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .pending_redraw()
    );

    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    let scene_after_redraw = runtime.scene_snapshot(window).unwrap();
    let report_after_redraw = runtime.performance_report();
    assert_ne!(scene_after_redraw.generation(), scene_generation_before);
    assert!(report_after_redraw.scene.compile_count > report_after_wheel.scene.compile_count);
    assert_eq!(
        scene_after_redraw
            .hit_test()
            .hit_test(LayoutPoint::new(1.0, 80.0))
            .map(|entry| entry.node_id()),
        Some(content_id)
    );
}

#[test]
fn scroll_geometry_probe_reports_surface_geometry_after_scroll_without_layout() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("scroll")
                    .w(px(100.0))
                    .h(px(50.0))
                    .overflow(Overflow::Scroll)
                    .child(input("hello").key("field").h(px(20.0)))
                    .child(text("filler").key("filler").h(px(100.0))),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let scroll = retained.find_by_key("scroll").unwrap();
    let field = retained.find_by_key("field").unwrap();
    let scroll_target =
        crate::interaction::InteractionTarget::new(scroll.id(), scroll.generation());
    let field_target = crate::interaction::InteractionTarget::new(field.id(), field.generation());

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    let _ = runtime.take_platform_ime_requests(window.into()).unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .state_mut()
        .scheduler_mut()
        .take_dirty_lanes(window.id(), DirtyLanes::all());
    runtime.state_mut().clear_consumed_dirty_lanes(window.id());

    let layout_before = runtime.layout_snapshot(window).unwrap();
    let field_layout = layout_before.find_by_key("field").unwrap();
    let field_border_before = field_layout.border_rect();
    let expected_cursor_before = text_viewport_placement(
        ElementKind::Input,
        field_layout.content_rect(),
        field_layout.text_layout().unwrap(),
    )
    .visible_caret_rect();
    let layout_generation_before = layout_before.generation();
    let scene_generation_before = runtime.scene_snapshot(window).unwrap().generation();
    let report_before = runtime.performance_report();

    runtime
        .wheel_input(
            window,
            WheelInput::new(ScrollDelta::pixels(0.0, 10.0), ScrollPhase::Moved),
        )
        .unwrap();

    let after_wheel_probe = runtime
        .scroll_geometry_probe(
            window,
            scroll_target,
            field_target,
            LayoutPoint::new(1.0, 1.0),
        )
        .unwrap();
    let scroll_geometry = after_wheel_probe
        .scroll
        .expect("scroll target should report layout scroll geometry");
    let expected_cursor_after = expected_cursor_before.translate(0.0, -10.0);
    let lanes_after_wheel = runtime.state().reported_dirty_lanes(window.id());
    let report_after_wheel = runtime.performance_report();

    assert_eq!(
        runtime.layout_snapshot(window).unwrap().generation(),
        layout_generation_before
    );
    assert_eq!(
        runtime.scene_snapshot(window).unwrap().generation(),
        scene_generation_before
    );
    assert_eq!(
        report_after_wheel.layout.pass_count,
        report_before.layout.pass_count
    );
    assert_eq!(
        report_after_wheel.scene.compile_count,
        report_before.scene.compile_count
    );
    assert_eq!(after_wheel_probe.scroll_target, scroll_target);
    assert_eq!(after_wheel_probe.observed_target, field_target);
    assert_eq!(
        scroll_geometry.viewport,
        LayoutRect::new(0.0, 0.0, 100.0, 50.0)
    );
    assert_eq!(scroll_geometry.content_extent.height(), 120.0);
    assert_eq!(scroll_geometry.max_offset.y(), 70.0);
    assert_eq!(
        after_wheel_probe.current_offset,
        LayoutPoint::new(0.0, 10.0)
    );
    assert_eq!(
        after_wheel_probe.ime_caret_rect,
        Some(expected_cursor_after)
    );
    assert_eq!(
        after_wheel_probe.ime_candidate_rect,
        Some(expected_cursor_after)
    );
    assert!(lanes_after_wheel.contains(DirtyLane::Paint.flag()));
    assert!(lanes_after_wheel.contains(DirtyLane::Semantics.flag()));
    assert!(!lanes_after_wheel.contains(DirtyLane::Layout.flag()));

    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }

    let after_redraw_probe = runtime
        .scroll_geometry_probe(
            window,
            scroll_target,
            field_target,
            LayoutPoint::new(1.0, 1.0),
        )
        .unwrap();
    let report_after_redraw = runtime.performance_report();
    let diagnostics = runtime.diagnostics().snapshot();

    assert_eq!(
        runtime.layout_snapshot(window).unwrap().generation(),
        layout_generation_before
    );
    assert_eq!(
        report_after_redraw.layout.pass_count,
        report_before.layout.pass_count
    );
    assert!(report_after_redraw.scene.compile_count > report_after_wheel.scene.compile_count);
    assert_eq!(
        after_redraw_probe.current_offset,
        LayoutPoint::new(0.0, 10.0)
    );
    assert_eq!(after_redraw_probe.hit_target, Some(field_target));
    assert!(after_redraw_probe.hit_path.contains(&scroll_target));
    assert!(after_redraw_probe.hit_path.contains(&field_target));
    assert!(
        after_redraw_probe
            .paint_bounds
            .contains(&expected_cursor_after)
    );
    assert_eq!(
        after_redraw_probe.semantic_bounds,
        Some(field_border_before.translate(0.0, -10.0))
    );
    assert_eq!(
        after_redraw_probe.ime_candidate_rect,
        Some(expected_cursor_after)
    );
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "ime.cursor_area"
            && record
                .fields
                .get("target_id")
                .is_some_and(|value| value == &field.id().raw().to_string())
            && record
                .fields
                .get("rect_y")
                .is_some_and(|value| value == &expected_cursor_after.y().to_string())
    }));
    assert!(diagnostics.records().iter().any(|record| {
        record.operation == "scroll.offset"
            && record
                .fields
                .get("target_id")
                .is_some_and(|value| value == &scroll.id().raw().to_string())
            && record
                .fields
                .get("new_y")
                .is_some_and(|value| value == "10")
            && record
                .fields
                .get("max_y")
                .is_some_and(|value| value == "70")
    }));
}

#[test]
fn semantic_dirty_lane_consumes_on_existing_redraw_without_scene_recompile() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").child(text("label").key("label")))
        })
        .unwrap();
    let diagnostics_before = runtime.diagnostics().snapshot();
    let semantic_before = diagnostics_before.counter("semantics.build");
    let scene_before = diagnostics_before.counter("scene.compile");

    runtime.state_mut().clear_consumed_dirty_lanes(window.id());
    runtime
        .state_mut()
        .scheduler_mut()
        .mark_dirty(window.id(), DirtyLane::Semantics);
    runtime
        .state_mut()
        .scheduler_mut()
        .request_redraw(window.id());
    runtime
        .ingest_platform_fact(PlatformFact::RedrawRequested {
            handle: window.into(),
        })
        .unwrap();

    let diagnostics_after = runtime.diagnostics().snapshot();
    assert_eq!(
        diagnostics_after.counter("semantics.build"),
        semantic_before + 1
    );
    assert_eq!(diagnostics_after.counter("scene.compile"), scene_before);
    assert!(
        runtime
            .state()
            .reported_dirty_lanes(window.id())
            .contains(DirtyLane::Semantics.flag())
    );
    assert!(
        !runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .dirty_lanes()
            .contains(DirtyLane::Semantics.flag())
    );
}

#[test]
fn semantic_only_focus_change_publishes_snapshot_without_platform_redraw_or_scene_recompile() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div().key("root").child(
                    div()
                        .key("focusable")
                        .focusable(true)
                        .w(px(20.0))
                        .h(px(20.0)),
                ),
            )
        })
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    runtime.state_mut().clear_consumed_dirty_lanes(window.id());
    let report_before = runtime.performance_report();
    let semantic_builds_before = runtime.diagnostics().snapshot().counter("semantics.build");
    let focused_before = runtime
        .semantic_snapshot(window)
        .unwrap()
        .find_by_key("focusable")
        .unwrap()
        .state()
        .focused();
    assert!(!focused_before);

    runtime
        .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))
        .unwrap();

    let report_after = runtime.performance_report();
    let semantic_builds_after = runtime.diagnostics().snapshot().counter("semantics.build");
    let semantic_after = runtime.semantic_snapshot(window).unwrap();
    let focusable_after = semantic_after.find_by_key("focusable").unwrap();

    assert!(focusable_after.state().focused());
    assert_eq!(semantic_builds_after, semantic_builds_before + 1);
    assert_eq!(
        report_after.scene.compile_count,
        report_before.scene.compile_count
    );
    assert_eq!(
        report_after.render.frame_graph_count,
        report_before.render.frame_graph_count
    );
    assert!(runtime.take_platform_redraw_requests().is_empty());
    assert!(
        runtime
            .state()
            .reported_dirty_lanes(window.id())
            .contains(DirtyLane::Semantics.flag())
    );
    assert!(
        !runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .dirty_lanes()
            .contains(DirtyLane::Semantics.flag())
    );
    assert!(
        !runtime
            .state()
            .reported_dirty_lanes(window.id())
            .contains(DirtyLane::Paint.flag())
    );
}

#[test]
fn semantic_only_window_focus_change_publishes_snapshot_without_platform_redraw_or_scene_recompile()
{
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();
    for handle in runtime.take_platform_redraw_requests() {
        runtime
            .ingest_platform_fact(PlatformFact::RedrawRequested { handle })
            .unwrap();
    }
    runtime.state_mut().clear_consumed_dirty_lanes(window.id());
    let report_before = runtime.performance_report();
    let semantic_builds_before = runtime.diagnostics().snapshot().counter("semantics.build");
    let window_focused_before = runtime
        .semantic_snapshot(window)
        .unwrap()
        .root()
        .unwrap()
        .state()
        .window_focused();
    assert!(!window_focused_before);

    runtime
        .ingest_platform_fact(PlatformFact::WindowFocusChanged {
            handle: window.into(),
            input: WindowFocusInput::new(true),
        })
        .unwrap();

    let report_after = runtime.performance_report();
    let semantic_builds_after = runtime.diagnostics().snapshot().counter("semantics.build");
    let semantic_after = runtime.semantic_snapshot(window).unwrap();

    assert!(semantic_after.root().unwrap().state().window_focused());
    assert_eq!(semantic_builds_after, semantic_builds_before + 1);
    assert_eq!(
        report_after.scene.compile_count,
        report_before.scene.compile_count
    );
    assert_eq!(
        report_after.render.frame_graph_count,
        report_before.render.frame_graph_count
    );
    assert!(runtime.take_platform_redraw_requests().is_empty());
    assert!(
        runtime
            .state()
            .reported_dirty_lanes(window.id())
            .contains(DirtyLane::Semantics.flag())
    );
    assert!(
        !runtime
            .state()
            .scheduler()
            .window_state(window.id())
            .unwrap()
            .dirty_lanes()
            .contains(DirtyLane::Semantics.flag())
    );
    assert!(
        !runtime
            .state()
            .reported_dirty_lanes(window.id())
            .contains(DirtyLane::Paint.flag())
    );
}

#[test]
fn wheel_over_non_scrollable_target_records_no_scroll_target() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(div().key("root").w(px(100.0)).h(px(100.0)))
        })
        .unwrap();

    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 40.0), ScrollPhase::Moved),
        })
        .unwrap();

    assert_eq!(
        runtime
            .state()
            .interaction(window.id())
            .unwrap()
            .scroll_offsets_len(),
        0
    );
    assert!(
        runtime
            .diagnostics()
            .snapshot()
            .records()
            .iter()
            .any(|record| {
                record.operation == "scroll.intent"
                    && record
                        .fields
                        .get("result")
                        .is_some_and(|value| value == "no_scroll_target")
            })
    );
}

#[test]
fn nearest_scrollable_ancestor_is_chosen_from_hit_path() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| {
            TestRoot::new(
                div()
                    .key("outer")
                    .w(px(100.0))
                    .h(px(100.0))
                    .overflow(Overflow::Scroll)
                    .child(
                        div()
                            .key("inner")
                            .w(px(100.0))
                            .h(px(220.0))
                            .child(text("leaf").key("leaf").h(px(20.0))),
                    ),
            )
        })
        .unwrap();
    let retained = runtime.retained_snapshot(window).unwrap();
    let outer = retained.find_by_key("outer").unwrap();
    let inner = retained.find_by_key("inner").unwrap();
    let outer_target = crate::interaction::InteractionTarget::new(outer.id(), outer.generation());
    let inner_target = crate::interaction::InteractionTarget::new(inner.id(), inner.generation());

    runtime
        .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))
        .unwrap();
    runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(0.0, 40.0), ScrollPhase::Moved),
        })
        .unwrap();

    assert_eq!(
        runtime.scroll_offset(window, outer_target).unwrap().y(),
        40.0
    );
    assert_eq!(
        runtime.scroll_offset(window, inner_target).unwrap(),
        LayoutPoint::ZERO
    );
}

#[test]
fn display_none_replaced_and_destroyed_scroll_targets_clear_offsets() {
    for mode in [
        ScrollMutationMode::DisplayNone,
        ScrollMutationMode::Replaced,
        ScrollMutationMode::Destroyed,
    ] {
        let run = Application::new()
            .run_test(move |cx| {
                let mode_entity = cx.new_entity(|_| ScrollMutationMode::Visible);
                let window = cx.windows().open(WindowOptions::new(), {
                    let mode_entity = mode_entity.clone();
                    move |_| MutatingScrollRoot { mode: mode_entity }
                })?;
                cx.windows()
                    .pointer_move(window, LayoutPoint::new(1.0, 1.0))?;
                cx.windows().wheel_input(
                    window,
                    WheelInput::new(ScrollDelta::pixels(0.0, 40.0), ScrollPhase::Moved),
                )?;
                mode_entity.update(cx, |mode_entity, cx| {
                    *mode_entity = mode;
                    cx.notify();
                    Ok(())
                })?;
                Ok(())
            })
            .unwrap();

        let window = run.windows()[0].handle();
        let diagnostics = run.diagnostics();
        assert!(diagnostics.records().iter().any(|record| {
            record.operation == "input.stale_target"
                && record
                    .fields
                    .get("state_kind")
                    .is_some_and(|value| value == "scroll_offset_cleanup")
        }));
        assert_eq!(
            run.scene_snapshot(window)
                .unwrap()
                .hit_test()
                .entries()
                .iter()
                .filter(|entry| {
                    let retained = run.retained_snapshot(window).unwrap();
                    retained
                        .find_by_key("scroll")
                        .is_some_and(|node| node.id() == entry.node_id())
                })
                .count(),
            if mode == ScrollMutationMode::Replaced {
                1
            } else {
                0
            }
        );
    }
}

#[test]
fn invalid_wheel_delta_is_rejected_before_fact_processing() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    let error = runtime
        .ingest_platform_fact(PlatformFact::WheelInput {
            handle: window.into(),
            input: WheelInput::new(ScrollDelta::pixels(f32::NAN, 1.0), ScrollPhase::Moved),
        })
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(
        runtime.diagnostics().snapshot().counter("input.wheel_fact"),
        0
    );
}

#[test]
fn window_focus_fact_is_processed_distinctly_from_element_focus() {
    let mut runtime = Runtime::new();
    let window = runtime
        .open_window(WindowOptions::new(), |_| TestRoot::new(div().key("root")))
        .unwrap();

    runtime
        .ingest_platform_fact(PlatformFact::WindowFocusChanged {
            handle: window.into(),
            input: WindowFocusInput::new(true),
        })
        .unwrap();

    let interaction = runtime.state().interaction(window.id()).unwrap();
    assert!(interaction.window_focused());
    assert_eq!(interaction.hover(), None);
    assert_eq!(interaction.pressed(), None);
    assert_eq!(interaction.keyboard_focus(), None);
    let snapshot = runtime.diagnostics().snapshot();
    assert_eq!(snapshot.counter("window.focus_fact"), 1);
    assert_eq!(snapshot.counter("focus.transition"), 1);
    assert_eq!(snapshot.counter("input.dispatch"), 0);
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "window.focus_fact"
            && record
                .fields
                .get("focused")
                .is_some_and(|value| value == "true")
    }));
    assert!(snapshot.records().iter().any(|record| {
        record.operation == "focus.transition"
            && record
                .fields
                .get("domain")
                .is_some_and(|value| value == "window")
            && record
                .fields
                .get("from")
                .is_some_and(|value| value == "unfocused")
            && record
                .fields
                .get("to")
                .is_some_and(|value| value == "focused")
    }));
}

#[test]
fn public_input_types_do_not_leak_winit_types() {
    fn assert_type_name<T>() {
        assert!(!std::any::type_name::<T>().contains("winit"));
    }

    assert_type_name::<KeyInput>();
    assert_type_name::<Key>();
    assert_type_name::<PhysicalKey>();
    assert_type_name::<Modifiers>();
    assert_type_name::<WheelInput>();
    assert_type_name::<ScrollDelta>();
    assert_type_name::<ScrollPhase>();
    assert_type_name::<WindowFocusInput>();
}

#[test]
fn platform_wake_and_exit_are_runtime_ingress_facts() {
    let mut runtime = Runtime::new();

    let wake = runtime.enqueue(RuntimeCommand::PlatformFact(PlatformFact::Wake));
    let exit = runtime.enqueue(RuntimeCommand::PlatformFact(PlatformFact::Exit));
    let processed = runtime.drain_all().unwrap();

    assert_eq!(processed, [wake, exit]);
    assert_eq!(
        runtime
            .diagnostics()
            .snapshot()
            .counter("platform.fact_processed"),
        2
    );
}

#[test]
fn internal_run_test_harness_still_works() {
    let run = crate::app::Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new().title("Internal Harness"), |_| {
                    TestRoot::new(div().key("root"))
                })?;
            Ok(())
        })
        .unwrap();

    assert_eq!(run.windows().len(), 1);
    assert_eq!(run.windows()[0].title(), "Internal Harness");
}
