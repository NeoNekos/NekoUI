use std::cell::Cell;
use std::rc::Rc;

use crate::app::{Context, Render};
use crate::diagnostic::DirtyLane;
use crate::element::{Element, IntoElement, div, text};
use crate::error::ErrorKind;
use crate::layout::LayoutPoint;
use crate::layout::LayoutSize;
use crate::runtime::Runtime;
use crate::runtime::command::RuntimeCommand;
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
