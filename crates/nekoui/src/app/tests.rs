use std::cell::RefCell;
use std::rc::Rc;

use super::{
    App, Application, Entity, EventEmitter, LastWindowBehavior, Render, TaskResult, UiExecutor,
};
use crate::element::SpecArena;
use crate::element::{IntoElement, ParentElement};
use crate::input::TextInputEvent;
use crate::platform::window::WindowInfoSeed;
use crate::window::{WindowId, WindowInfo, WindowOptions, WindowSize};

fn test_app() -> App {
    App::new(Vec::new())
}

fn test_window(logical: WindowSize, physical: WindowSize, scale_factor: f64) -> WindowInfo {
    WindowInfo::from_options(
        WindowId::new(),
        &WindowOptions::default(),
        WindowInfoSeed {
            content_size: logical,
            frame_size: Some(logical),
            physical_size: physical,
            scale_factor,
            position: None,
            current_display: None,
        },
    )
}

#[derive(Default)]
struct Counter {
    value: usize,
}

impl EventEmitter<usize> for Counter {}

#[derive(Default)]
struct RootView;

impl Render for RootView {
    fn render(
        &mut self,
        _window: &WindowInfo,
        _cx: &mut super::Context<'_, Self>,
    ) -> impl IntoElement {
        crate::text("root")
    }
}

#[test]
fn application_defaults_to_exit_when_last_window_closes() {
    let application = Application::new();
    assert_eq!(
        application.last_window_behavior,
        LastWindowBehavior::ExitEventLoop
    );
}

#[test]
fn open_window_queues_request() {
    let mut app = test_app();
    let handle = app
        .open_window(WindowOptions::new().title("Neko"), |_window, cx| {
            cx.new_view(|_| RootView)
        })
        .unwrap();

    let requests = app.drain_window_requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].id, handle.id());
}

#[test]
fn update_round_trips_entity_state() {
    let app = test_app();
    let entity = app.insert_entity(String::from("neko"));

    let updated = app
        .update(entity, |value, cx| {
            value.push_str(" ui");
            cx.notify();
            value.clone()
        })
        .unwrap();

    assert_eq!(updated, "neko ui");
}

#[test]
fn observe_marks_subscriber_view_dirty_after_source_notify() {
    let mut app = test_app();
    let source: Entity<Counter> = app.insert_entity(Counter::default());
    let observer = app.insert_entity(Counter::default());

    let _subscription = app
        .update(observer, |_state, cx| {
            cx.observe(&source, |observer_state, _, cx| {
                observer_state.value += 1;
                cx.notify();
            })
            .unwrap()
        })
        .unwrap();

    app.update(source, |state, cx| {
        state.value += 1;
        cx.notify();
    })
    .unwrap();

    let runtime = app.process_runtime().unwrap();
    assert!(runtime.dirty_views.is_empty());
    let observer_value = app.update(observer, |state, _| state.value).unwrap();
    assert_eq!(observer_value, 1);
}

#[test]
fn subscribe_and_emit_deliver_typed_events() {
    let mut app = test_app();
    let source: Entity<Counter> = app.insert_entity(Counter::default());
    let observer = app.insert_entity(Counter::default());

    let _subscription = app
        .update(observer, |_state, cx| {
            cx.subscribe(&source, |observer_state, _, event: &usize, cx| {
                observer_state.value += *event;
                cx.notify();
            })
            .unwrap()
        })
        .unwrap();

    app.update(source, |_state, cx| cx.emit(7usize).unwrap())
        .unwrap();

    let runtime = app.process_runtime().unwrap();
    assert!(runtime.dirty_views.is_empty());
    let observer_value = app.update(observer, |state, _| state.value).unwrap();
    assert_eq!(observer_value, 7);
}

#[test]
fn subscribe_text_input_delivers_commit_events() {
    #[derive(Default)]
    struct InputSource;
    impl EventEmitter<TextInputEvent> for InputSource {}

    #[derive(Default)]
    struct InputSink {
        committed: String,
    }

    let app = test_app();
    let source: Entity<InputSource> = app.insert_entity(InputSource);
    let sink = app.insert_entity(InputSink::default());

    let _subscription = app
        .update(sink, |_state, cx| {
            cx.subscribe_text_input(&source, |sink_state, _, event, _| {
                if let TextInputEvent::Commit { text, .. } = event {
                    sink_state.committed.push_str(text.as_ref());
                }
            })
            .unwrap()
        })
        .unwrap();

    app.dispatch_text_input_event(
        source.id(),
        &TextInputEvent::Commit {
            source: source.id(),
            target: crate::InputNodeId::new(),
            text: "ok".into(),
        },
    )
        .unwrap();

    let committed = app.update(sink, |state, _| state.committed.clone()).unwrap();
    assert_eq!(committed, "ok");
}

#[test]
fn subscribe_text_input_target_filters_node_id() {
    #[derive(Default)]
    struct InputSource;
    impl EventEmitter<TextInputEvent> for InputSource {}

    #[derive(Default)]
    struct InputSink {
        committed: String,
    }

    let app = test_app();
    let source: Entity<InputSource> = app.insert_entity(InputSource);
    let sink = app.insert_entity(InputSink::default());
    let target = crate::InputNodeId::new();
    let other = crate::InputNodeId::new();

    let _subscription = app
        .update(sink, |_state, cx| {
            cx.subscribe_text_input_target(&source, target, |sink_state, _, event, _| {
                if let TextInputEvent::Commit { text, .. } = event {
                    sink_state.committed.push_str(text.as_ref());
                }
            })
            .unwrap()
        })
        .unwrap();

    app.dispatch_text_input_event(
        source.id(),
        &TextInputEvent::Commit {
            source: source.id(),
            target: other,
            text: "no".into(),
        },
    )
    .unwrap();
    app.dispatch_text_input_event(
        source.id(),
        &TextInputEvent::Commit {
            source: source.id(),
            target,
            text: "ok".into(),
        },
    )
    .unwrap();

    let committed = app.update(sink, |state, _| state.committed.clone()).unwrap();
    assert_eq!(committed, "ok");
}

#[test]
fn view_can_subscribe_to_its_own_text_input() {
    #[derive(Default)]
    struct InputView {
        subscription: Option<super::Subscription>,
        committed: String,
    }

    impl Render for InputView {
        fn render(
            &mut self,
            _window: &WindowInfo,
            cx: &mut super::Context<'_, Self>,
        ) -> impl IntoElement {
            let view = cx.entity();
            if self.subscription.is_none() {
                self.subscription = cx
                    .subscribe_text_input(&view, |state, _, event, _| {
                        if let TextInputEvent::Commit { text, .. } = event {
                            state.committed.push_str(text.as_ref());
                        }
                    })
                    .ok();
            }
            crate::div()
                .child(crate::text("input").text_input(crate::TextInputPurpose::Normal))
        }
    }

    let app = test_app();
    let view = app.insert_view(InputView::default());
    let root = crate::div().child(view).into_any_element();
    let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
    let mut arena = SpecArena::new();
    let _ = app.build_root_spec(&window, &root, &mut arena).unwrap();

    app.dispatch_text_input_event(
        view.id(),
        &TextInputEvent::Commit {
            source: view.id(),
            target: crate::InputNodeId::new(),
            text: "ok".into(),
        },
    )
    .unwrap();

    let committed = app
        .update(view.entity(), |state, _| state.committed.clone())
        .unwrap();
    assert_eq!(committed, "ok");
}

#[test]
fn dropping_subscription_stops_event_delivery() {
    let mut app = test_app();
    let source: Entity<Counter> = app.insert_entity(Counter::default());
    let observer = app.insert_entity(Counter::default());

    let subscription = app
        .update(observer, |_state, cx| {
            cx.subscribe(&source, |observer_state, _, event: &usize, _| {
                observer_state.value += *event;
            })
            .unwrap()
        })
        .unwrap();
    drop(subscription);

    app.update(source, |_state, cx| cx.emit(3usize).unwrap())
        .unwrap();
    app.process_runtime().unwrap();

    let observer_value = app.update(observer, |state, _| state.value).unwrap();
    assert_eq!(observer_value, 0);
}

#[test]
fn ui_executor_runs_tasks_only_when_drained() {
    let executor = UiExecutor::new();
    let task = executor.spawn(async { 42usize });
    assert!(task.try_recv().is_none());
    executor.run_pending();
    assert!(matches!(task.recv(), TaskResult::Ready(42)));
}

#[test]
fn cached_resolution_rerenders_only_dirty_nested_view() {
    struct ChildView {
        renders: Rc<RefCell<usize>>,
    }

    impl Render for ChildView {
        fn render(
            &mut self,
            _window: &WindowInfo,
            _cx: &mut super::Context<'_, Self>,
        ) -> impl IntoElement {
            *self.renders.borrow_mut() += 1;
            crate::text("child")
        }
    }

    struct ParentView {
        renders: Rc<RefCell<usize>>,
        child: super::View<ChildView>,
    }

    impl Render for ParentView {
        fn render(
            &mut self,
            _window: &WindowInfo,
            _cx: &mut super::Context<'_, Self>,
        ) -> impl IntoElement {
            *self.renders.borrow_mut() += 1;
            crate::div().child(self.child)
        }
    }

    let app = test_app();
    let parent_renders = Rc::new(RefCell::new(0));
    let child_renders = Rc::new(RefCell::new(0));
    let child = app.insert_view(ChildView {
        renders: child_renders.clone(),
    });
    let parent = app.insert_view(ParentView {
        renders: parent_renders.clone(),
        child,
    });
    let root = crate::div().child(parent).into_any_element();
    let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
    let mut arena = SpecArena::new();
    let built = app.build_root_spec(&window, &root, &mut arena).unwrap();
    assert_eq!(*parent_renders.borrow(), 1);
    assert_eq!(*child_renders.borrow(), 1);
    assert!(built.referenced_views.contains(&parent.id()));
    assert!(built.referenced_views.contains(&child.id()));

    let rebuilt = app.build_root_spec(&window, &root, &mut arena).unwrap();
    assert_eq!(*parent_renders.borrow(), 2);
    assert_eq!(*child_renders.borrow(), 2);
    assert!(rebuilt.referenced_views.contains(&parent.id()));
    assert!(rebuilt.referenced_views.contains(&child.id()));
}

#[test]
fn spec_arena_is_reused_between_builds() {
    let app = test_app();
    let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
    let root = crate::div().child(crate::text("a")).into_any_element();
    let mut arena = SpecArena::new();

    let built = app.build_root_spec(&window, &root, &mut arena).unwrap();
    let first_len = arena.len();
    assert_eq!(first_len, 2);
    assert!(matches!(
        arena.node(built.root).kind,
        crate::element::SpecKind::Div
    ));

    let root = crate::div()
        .child(crate::text("a"))
        .child(crate::text("b"))
        .into_any_element();
    let rebuilt = app.build_root_spec(&window, &root, &mut arena).unwrap();
    let second_len = arena.len();

    assert_eq!(second_len, 3);
    assert!(matches!(
        arena.node(rebuilt.root).kind,
        crate::element::SpecKind::Div
    ));
}
