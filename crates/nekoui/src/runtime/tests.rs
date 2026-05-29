use std::cell::Cell;
use std::rc::Rc;

use crate::app::{Application, Context, Entity, Render};
use crate::diagnostic::DirtyLane;
use crate::element::{Element, IntoElement, div, text};
use crate::error::{ErrorKind, NekoError};
use crate::interaction::PointerInput;
use crate::layout::LayoutPoint;
use crate::layout::LayoutSize;
use crate::platform::PlatformFact;
use crate::runtime::Runtime;
use crate::runtime::command::RuntimeCommand;
use crate::style::{Display, StyleExt, px};
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
