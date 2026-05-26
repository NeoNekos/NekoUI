use std::cell::Cell;
use std::path::Path;
use std::rc::Rc;

use nekoui::ErrorKind;
use nekoui::diagnostic::DirtyLane;
use nekoui::layout::LayoutSize;
use nekoui::prelude::*;
use nekoui::scene::{PaintFragmentKind, PaintScene, SceneGeneration};
use nekoui::style::OutputParticipation;

#[derive(Debug)]
struct TestView {
    root: Element,
}

impl TestView {
    fn new(root: impl IntoElement) -> Self {
        Self {
            root: root.into_element(),
        }
    }
}

impl Render for TestView {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.root.clone()
    }
}

#[derive(Debug)]
struct CounterRoot {
    label: String,
}

impl Render for CounterRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .key("root")
            .child(text(self.label.clone()).key("label"))
    }
}

#[derive(Debug)]
struct ValidatingRoot {
    valid: bool,
}

impl Render for ValidatingRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        let color = if self.valid {
            Color::rgb(0, 0, 0)
        } else {
            Color::oklch(f32::NAN, 0.1, 20.0)
        };
        div().key("root").bg(color)
    }
}

#[derive(Debug)]
struct NotifyOnlyRoot;

impl Render for NotifyOnlyRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div().key("root").child(text("stable").key("label"))
    }
}

#[derive(Debug)]
struct SharedLabelRoot {
    label: Entity<String>,
}

impl Render for SharedLabelRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
            .key("root")
            .child(text(self.label.read(cx, Clone::clone).unwrap()).key("label"))
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
struct EntityRoot {
    root: Entity<Element>,
}

impl Render for EntityRoot {
    fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
        self.root.read(cx, Clone::clone).unwrap()
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
fn public_api_reports_core_runtime_performance() {
    let run = Application::new()
        .run_test(|cx| {
            let window =
                cx.windows().open(
                    WindowOptions::new().title("Smoke"),
                    |_| TestView::new(div()),
                )?;
            cx.notify();
            cx.windows().close(window)?;
            let error = cx.windows().validate(window).unwrap_err();

            assert_eq!(error.kind(), ErrorKind::Stale);
            Ok(())
        })
        .unwrap();

    let report = run.performance_report();

    assert_eq!(report.windows_alive, 0);
    assert_eq!(report.stale_handle_errors, 1);
    assert!(report.command_ingress.commands_enqueued >= 2);
    assert!(report.command_ingress.commands_processed >= 2);
    assert!(report.dirty_lanes.iter().any(|window| {
        window.lanes.contains(DirtyLane::Build.flag())
            && window.lanes.contains(DirtyLane::Semantics.flag())
    }));
}

#[test]
fn public_prelude_exposes_entity_context_and_subscription() {
    struct Observer {
        count: i32,
        _subscription: Subscription,
    }

    let run = Application::new()
        .run_test(|cx| {
            let source: Entity<i32> = cx.new_entity(|_| 1);
            let weak: WeakEntity<i32> = source.downgrade();
            let observer = cx.new_entity(|entity_cx: &mut Context<'_, Observer>| {
                let subscription = entity_cx.observe(&source, |observer, source, cx| {
                    observer.count += source.read(cx, |value| *value).unwrap();
                });
                Observer {
                    count: 0,
                    _subscription: subscription,
                }
            });

            assert_eq!(weak.upgrade(cx)?.read(cx, |value| *value)?, 1);
            source.update(cx, |value, entity_cx| {
                *value += 1;
                entity_cx.notify();
                Ok(())
            })?;
            assert_eq!(observer.read(cx, |observer| observer.count)?, 2);
            Ok(())
        })
        .unwrap();

    assert_eq!(run.performance_report().notify_requests, 1);
    assert!(run.diagnostics().counter("api.entity.created") >= 2);
    assert!(run.diagnostics().counter("api.entity.notification_flushed") >= 1);
}

#[test]
fn public_element_and_style_api_builds_pure_declarations() {
    let root = div()
        .key("root")
        .padding(px(12.0))
        .p(px(8.0))
        .pl(px(4.0))
        .m(px(10.0))
        .ml(px(6.0))
        .gap(px(3.0))
        .width(fill())
        .w(fill())
        .display(Display::Flex)
        .background(Color::rgb(1, 2, 3))
        .bg(Color::rgb(4, 5, 6))
        .opacity(opacity(0.75))
        .child(
            text("Hello NekoUI")
                .key("label")
                .font_size(px(18.0))
                .text_color(Color::rgb(7, 8, 9))
                .line_clamp(2),
        )
        .into_element();

    assert_eq!(root.kind(), ElementKind::Div);
    assert_eq!(root.key().unwrap().as_str(), "root");
    assert_eq!(root.children()[0].kind(), ElementKind::Text);
    assert_eq!(root.children()[0].text(), Some("Hello NekoUI"));
    assert_eq!(root.style().layout().padding().left, Some(px(4.0)));
    assert_eq!(root.style().layout().padding().top, Some(px(8.0)));
    assert_eq!(root.style().layout().margin().left, Some(px(6.0)));
    assert_eq!(root.style().layout().margin().top, Some(px(10.0)));
    assert_eq!(root.style().layout().gap(), Some(px(3.0)));
    assert_eq!(root.style().layout().width(), Some(Dimension::Fill));
    assert_eq!(root.style().layout().display(), Some(Display::Flex));
    assert_eq!(
        root.style().visual().background(),
        Some(Color::rgb(4, 5, 6))
    );
    assert_eq!(root.style().visual().opacity(), Some(Opacity::new(0.75)));
    assert_eq!(root.children()[0].style().text().max_lines(), Some(2));
    assert_eq!(
        root.children()[0].style().text().text_overflow(),
        Some(TextOverflow::Ellipsis)
    );

    let participation = OutputParticipation::excluded();
    assert!(!participation.layout());
    assert!(!participation.paint());
    assert!(!participation.hit_test());
    assert!(!participation.semantics());
}

#[test]
fn run_test_exposes_read_only_retained_and_style_snapshots() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new().title("Root"), |_| {
                TestView::new(
                    div()
                        .key("root")
                        .p(px(12.0))
                        .bg(Color::rgb(240, 241, 242))
                        .child(
                            text("Hello")
                                .key("label")
                                .font_size(px(20.0))
                                .text_color(Color::rgb(10, 20, 30)),
                        ),
                )
            })?;
            Ok(())
        })
        .unwrap();
    let window = run.windows()[0].handle();
    let retained = run.retained_snapshot(window).unwrap();
    let style = run.style_snapshot(window).unwrap();
    let label = retained.find_by_key("label").unwrap();

    assert_eq!(retained.node_count(), 2);
    assert_eq!(label.text(), Some("Hello"));
    assert_eq!(label.resolved_style().text().font_size(), px(20.0));
    assert_eq!(style.node_count(), 2);
    assert!(retained.root().unwrap().participation().layout());
    assert!(style.root().unwrap().participation().semantics());
    assert_eq!(
        style.root().unwrap().children()[0]
            .resolved()
            .text()
            .text_color(),
        Color::rgb(10, 20, 30)
    );
    assert_eq!(run.performance_report().retained.node_count, 2);
    assert_eq!(run.performance_report().style.resolved_node_count, 2);
    assert!(
        run.performance_report()
            .phase_durations
            .contains_key("retained.diff")
    );
}

#[test]
fn run_test_exposes_read_only_layout_snapshot() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(
                WindowOptions::new()
                    .title("Layout")
                    .logical_size(LayoutSize::new(320.0, 240.0)),
                |_| {
                    TestView::new(
                        div()
                            .key("root")
                            .p(px(10.0))
                            .gap(px(5.0))
                            .w(fill())
                            .child(text("A").key("first").h(px(20.0)))
                            .child(text("B").key("second").h(px(30.0))),
                    )
                },
            )?;
            Ok(())
        })
        .unwrap();
    let window = run.windows()[0].handle();
    let layout = run.layout_snapshot(window).unwrap();
    let root = layout.root().unwrap();
    let first = layout.find_by_key("first").unwrap();
    let second = layout.find_by_key("second").unwrap();

    assert_eq!(layout.node_count(), 3);
    assert_eq!(layout.viewport().logical_size().width(), 320.0);
    assert_eq!(root.border_rect().width(), 320.0);
    assert_eq!(root.content_rect().x(), 10.0);
    assert_eq!(root.content_rect().width(), 300.0);
    assert_eq!(first.border_rect().y(), 10.0);
    assert_eq!(second.border_rect().y(), 35.0);
    assert_eq!(run.performance_report().layout.node_count, 3);
    assert!(run.performance_report().layout.pass_count >= 1);
    assert!(
        run.performance_report()
            .phase_durations
            .contains_key("layout.pass")
    );
}

#[test]
fn run_test_exposes_read_only_scene_snapshot_without_backend() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(
                WindowOptions::new()
                    .title("Scene")
                    .logical_size(LayoutSize::new(320.0, 240.0)),
                |_| {
                    TestView::new(
                        div()
                            .key("root")
                            .w(fill())
                            .bg(Color::rgb(240, 241, 242))
                            .child(text("Hello scene").key("label")),
                    )
                },
            )?;
            Ok(())
        })
        .unwrap();
    let window = run.windows()[0].handle();
    let scene: PaintScene = run.scene_snapshot(window).unwrap();
    let generation: SceneGeneration = scene.generation();
    let scene_report: nekoui::diagnostic::ScenePerformanceReport = run.performance_report().scene;

    assert_eq!(scene.hit_test().entries().len(), 2);
    assert!(scene.fragments().len() >= 2);
    assert!(generation.viewport_generation() >= 1);
    assert!(!generation.style_generation().facts().is_empty());
    assert!(!generation.text_generation().facts().is_empty());
    assert!(matches!(
        scene.fragments()[0].kind(),
        PaintFragmentKind::Rect { .. }
    ));
    assert!(run.performance_report().scene.compile_count >= 1);
    assert_eq!(scene_report.published_node_count, 2);
    assert_eq!(scene_report.node_count, 2);
    assert_eq!(scene_report.last_compile.node_count, 2);
    assert_eq!(
        scene_report.last_compile.fragment_count,
        scene.fragments().len()
    );
    assert!(run.diagnostics().counter("scene.compile") >= 1);
    assert!(
        run.diagnostics()
            .records()
            .iter()
            .any(|record| record.operation == "scene.compile"
                && record.fields.contains_key("fragment_count"))
    );
}

#[test]
fn root_view_initial_mount_publishes_all_core_snapshots() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(
                WindowOptions::new().logical_size(LayoutSize::new(320.0, 240.0)),
                |_| CounterRoot {
                    label: "first".to_owned(),
                },
            )?;
            Ok(())
        })
        .unwrap();
    let window = run.windows()[0].handle();

    assert_eq!(run.retained_snapshot(window).unwrap().node_count(), 2);
    assert_eq!(run.style_snapshot(window).unwrap().node_count(), 2);
    assert_eq!(run.layout_snapshot(window).unwrap().node_count(), 2);
    assert_eq!(run.scene_snapshot(window).unwrap().stats().node_count, 2);
}

#[test]
fn root_view_state_update_and_notify_rerenders_snapshots() {
    let run = Application::new()
        .run_test(|cx| {
            let label = cx.new_entity(|_| "before".to_owned());
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| SharedLabelRoot {
                    label: label.clone(),
                })?;

            assert_eq!(
                cx.windows()
                    .retained_snapshot(window)?
                    .find_by_key("label")
                    .unwrap()
                    .text(),
                Some("before")
            );
            label.update(cx, |label, cx| {
                *label = "after".to_owned();
                cx.notify();
                Ok(())
            })?;
            assert_eq!(
                cx.windows()
                    .retained_snapshot(window)?
                    .find_by_key("label")
                    .unwrap()
                    .text(),
                Some("after")
            );
            Ok(())
        })
        .unwrap();

    assert!(run.performance_report().retained.diff_count >= 2);
}

#[test]
fn root_view_entity_update_and_notify_rerenders_mounted_window() {
    let run = Application::new()
        .run_test(|cx| {
            let window = cx.windows().open(WindowOptions::new(), |_| CounterRoot {
                label: "before".to_owned(),
            })?;
            let root = cx.windows().root_view(window)?;

            root.update(cx, |root, cx| {
                root.label = "after".to_owned();
                cx.notify();
                Ok(())
            })?;

            assert_eq!(
                cx.windows()
                    .retained_snapshot(window)?
                    .find_by_key("label")
                    .unwrap()
                    .text(),
                Some("after")
            );
            Ok(())
        })
        .unwrap();

    assert!(run.performance_report().retained.diff_count >= 2);
}

#[test]
fn root_view_entity_is_restored_after_render_validation_failure() {
    Application::new()
        .run_test(|cx| {
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| ValidatingRoot { valid: true })?;
            let root = cx.windows().root_view(window)?;

            let error = root
                .update(cx, |root, cx| {
                    root.valid = false;
                    cx.notify();
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidInput);

            root.update(cx, |root, cx| {
                root.valid = true;
                cx.notify();
                Ok(())
            })?;
            assert_eq!(cx.windows().retained_snapshot(window)?.node_count(), 1);
            Ok(())
        })
        .unwrap();
}

#[test]
fn multiple_notify_calls_coalesce_root_rerender() {
    let renders = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let renders = renders.clone();
            move |cx| {
                cx.windows().open(WindowOptions::new(), {
                    let renders = renders.clone();
                    move |_| CountingRoot { renders }
                })?;
                cx.notify();
                cx.notify();
                Ok(())
            }
        })
        .unwrap();

    assert_eq!(renders.get(), 2);
    assert_eq!(run.performance_report().notify_requests, 2);
}

#[test]
fn notify_only_unchanged_render_does_not_create_downstream_work() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new(), |_| NotifyOnlyRoot)?;
            cx.notify();
            cx.notify();
            Ok(())
        })
        .unwrap();

    assert_eq!(run.diagnostics().counter("layout.pass"), 1);
    assert_eq!(run.diagnostics().counter("scene.compile"), 1);
    assert_eq!(run.performance_report().retained.last_diff.created, 0);
    assert_eq!(run.performance_report().notify_requests, 2);
}

#[test]
fn scene_snapshot_updates_generation_after_resize_and_root_notify() {
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| div().key("root").child(text("A")).into_element());
            let window = cx.windows().open(
                WindowOptions::new().logical_size(LayoutSize::new(200.0, 100.0)),
                |_| EntityRoot { root: root.clone() },
            )?;
            let first_scene = cx.windows().scene_snapshot(window)?;
            cx.windows().resize(window, LayoutSize::new(300.0, 100.0))?;
            let resized_scene = cx.windows().scene_snapshot(window)?;
            root.update(cx, |root, cx| {
                *root = div().key("root").child(text("B")).into_element();
                cx.notify();
                Ok(())
            })?;
            let updated_scene = cx.windows().scene_snapshot(window)?;

            assert!(
                resized_scene.generation().layout_generation()
                    > first_scene.generation().layout_generation()
            );
            assert_ne!(updated_scene.generation(), resized_scene.generation());
            Ok(())
        })
        .unwrap();

    assert!(run.performance_report().scene.compile_count >= 3);
}

#[test]
fn public_synthetic_pointer_click_updates_entity_and_rerenders() {
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
fn public_pointer_input_typed_api_smoke_preserves_discrete_fact_order() {
    let run = Application::new()
        .run_test(|cx| {
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| TestView::new(div().key("root")))?;
            cx.windows()
                .pointer_input(window, PointerInput::move_to(LayoutPoint::new(1.0, 1.0)))?;
            cx.windows()
                .pointer_input(window, PointerInput::down(LayoutPoint::new(1.0, 1.0)))?;
            cx.windows()
                .pointer_input(window, PointerInput::cancel(LayoutPoint::new(1.0, 1.0)))?;
            cx.windows()
                .pointer_input(window, PointerInput::up(LayoutPoint::new(1.0, 1.0)))?;
            Ok(())
        })
        .unwrap();
    let diagnostics = run.diagnostics();
    let kinds = diagnostics
        .records()
        .iter()
        .filter(|record| record.operation == "input.pointer_fact")
        .filter_map(|record| record.fields.get("kind"))
        .map(|kind| kind.to_string())
        .collect::<Vec<_>>();

    assert_eq!(kinds, ["move", "down", "cancel", "up"]);
    assert_eq!(run.diagnostics().counter("input.pointer_fact"), 4);
    assert_eq!(run.diagnostics().counter("input.click_derived"), 0);
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
fn stale_pressed_target_hidden_by_notify_is_dropped_before_up_click() {
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
fn public_pointer_down_up_different_targets_does_not_click() {
    let clicks = Rc::new(Cell::new(0));
    let run = Application::new()
        .run_test({
            let clicks = clicks.clone();
            move |cx| {
                let window = cx.windows().open(WindowOptions::new(), {
                    let clicks = clicks.clone();
                    move |_| {
                        TestView::new(
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
fn invalid_pointer_coordinates_return_typed_invalid_input_without_hit_or_miss() {
    for position in [
        LayoutPoint::new(f32::NAN, 1.0),
        LayoutPoint::new(f32::INFINITY, 1.0),
        LayoutPoint::new(1.0, f32::NEG_INFINITY),
    ] {
        let error = Application::new()
            .run_test(|cx| {
                let window = cx
                    .windows()
                    .open(WindowOptions::new(), |_| TestView::new(div().key("root")))?;
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
                .open(WindowOptions::new(), |_| TestView::new(div().key("root")))?;
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
fn invalid_opacity_values_return_typed_invalid_input() {
    for opacity_value in [f32::NAN, -0.1, 1.1] {
        let error = Application::new()
            .run_test(|cx| {
                cx.windows().open(WindowOptions::new(), |_| {
                    TestView::new(div().opacity(opacity(opacity_value)))
                })?;
                Ok(())
            })
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn oklch_color_declarations_are_preserved_without_srgb_conversion() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| {
                TestView::new(
                    div().key("root").bg(Color::oklch(0.72, 0.18, 145.0)).child(
                        text("OKLCH")
                            .key("label")
                            .text_color(Color::oklcha(0.64, 0.12, 35.0, 0.5)),
                    ),
                )
            })?;
            Ok(())
        })
        .unwrap();
    let style = run.style_snapshot(run.windows()[0].handle()).unwrap();
    let root = style.root().unwrap();
    let label = &root.children()[0];

    assert_eq!(
        root.resolved()
            .visual()
            .background()
            .unwrap()
            .oklch_channels(),
        Some((0.72, 0.18, 145.0, 1.0))
    );
    assert_eq!(
        root.resolved()
            .visual()
            .background()
            .unwrap()
            .srgb_channels(),
        None
    );
    assert_eq!(
        label.resolved().text().text_color().oklch_channels(),
        Some((0.64, 0.12, 35.0, 0.5))
    );
    assert_eq!(label.resolved().text().text_color().red(), None);
}

#[test]
fn invalid_oklch_values_return_typed_invalid_input() {
    for color in [
        Color::oklch(f32::NAN, 0.1, 20.0),
        Color::oklch(1.1, 0.1, 20.0),
        Color::oklch(0.5, -0.1, 20.0),
        Color::oklch(0.5, 0.1, f32::INFINITY),
        Color::oklcha(0.5, 0.1, 20.0, 1.1),
    ] {
        let error = Application::new()
            .run_test(|cx| {
                cx.windows()
                    .open(WindowOptions::new(), |_| TestView::new(div().bg(color)))?;
                Ok(())
            })
            .unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidInput);
    }
}

#[test]
fn transitional_root_setting_api_is_absent_from_active_files() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crate should live below the workspace root");
    let needles = [
        concat!("set", "_root"),
        concat!("Set", "Root"),
        concat!("process", "_set", "_root"),
        concat!("set", "_window", "_root"),
    ];

    for root in [
        manifest_dir.join("src"),
        manifest_dir.join("tests"),
        manifest_dir.join("examples"),
        repo_root.join("docs"),
    ] {
        assert_no_needles(&root, &needles);
    }
}

#[test]
fn layout_uses_border_box_padding_margin_and_gap_in_block_flow() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| {
                TestView::new(
                    div()
                        .key("root")
                        .w(px(120.0))
                        .p(px(10.0))
                        .gap(px(4.0))
                        .child(text("A").key("a").h(px(12.0)).m(px(2.0)))
                        .child(text("B").key("b").h(px(8.0))),
                )
            })?;
            Ok(())
        })
        .unwrap();
    let layout = run.layout_snapshot(run.windows()[0].handle()).unwrap();
    let root = layout.root().unwrap();
    let first = layout.find_by_key("a").unwrap();
    let second = layout.find_by_key("b").unwrap();

    assert_eq!(root.border_rect().width(), 120.0);
    assert_eq!(root.content_rect().width(), 100.0);
    assert_eq!(first.margin_rect().x(), 10.0);
    assert_eq!(first.border_rect().x(), 12.0);
    assert_eq!(second.border_rect().y(), 30.0);
}

#[test]
fn display_none_omits_layout_output_but_preserves_retained_identity() {
    let mut hidden_id = None;
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| {
                div()
                    .key("root")
                    .child(text("visible").key("visible"))
                    .child(text("hidden").key("hidden"))
                    .into_element()
            });
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| EntityRoot { root: root.clone() })?;
            hidden_id = Some(
                cx.windows()
                    .retained_snapshot(window)?
                    .find_by_key("hidden")
                    .unwrap()
                    .id(),
            );
            root.update(cx, |root, cx| {
                *root = div()
                    .key("root")
                    .child(text("visible").key("visible"))
                    .child(text("hidden").key("hidden").display(Display::None))
                    .into_element();
                cx.notify();
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
    let window = run.windows()[0].handle();
    let retained = run.retained_snapshot(window).unwrap();
    let layout = run.layout_snapshot(window).unwrap();

    assert_eq!(
        retained.find_by_key("hidden").unwrap().id(),
        hidden_id.unwrap()
    );
    assert_eq!(layout.generation().unwrap().raw(), 2);
    assert!(layout.find_by_key("visible").is_some());
    assert!(layout.find_by_key("hidden").is_none());
}

#[test]
fn layout_snapshot_rejects_closed_window_handle_as_stale() {
    let mut saved_window = None;
    let error = Application::new()
        .run_test(|cx| {
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| TestView::new(div().key("root")))?;
            saved_window = Some(window);
            cx.windows().close(window)?;
            Ok(())
        })
        .unwrap()
        .layout_snapshot(saved_window.unwrap())
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Stale);
}

#[test]
fn viewport_resize_recomputes_layout_with_latest_size() {
    let run = Application::new()
        .run_test(|cx| {
            let window = cx.windows().open(
                WindowOptions::new().logical_size(LayoutSize::new(200.0, 100.0)),
                |_| TestView::new(div().key("root").w(fill()).child(text("A"))),
            )?;
            cx.windows().resize(window, LayoutSize::new(360.0, 100.0))?;
            Ok(())
        })
        .unwrap();
    let layout = run.layout_snapshot(run.windows()[0].handle()).unwrap();

    assert_eq!(layout.viewport().logical_size().width(), 360.0);
    assert_eq!(layout.root().unwrap().border_rect().width(), 360.0);
    assert!(run.performance_report().layout.pass_count >= 2);
}

#[test]
fn invalid_viewport_and_layout_values_return_typed_invalid_input() {
    let open_error = Application::new()
        .run_test(|cx| {
            cx.windows().open(
                WindowOptions::new().logical_size(LayoutSize::new(f32::NAN, 100.0)),
                |_| TestView::new(div()),
            )?;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(open_error.kind(), ErrorKind::InvalidInput);

    let resize_error = Application::new()
        .run_test(|cx| {
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| TestView::new(div()))?;
            cx.windows().resize(window, LayoutSize::new(-1.0, 100.0))?;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(resize_error.kind(), ErrorKind::InvalidInput);

    let scale_error = Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new().scale_factor(0.0), |_| {
                    TestView::new(div())
                })?;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(scale_error.kind(), ErrorKind::InvalidInput);

    let length_error = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| {
                TestView::new(div().w(px(f32::INFINITY)))
            })?;
            Ok(())
        })
        .unwrap_err();
    assert_eq!(length_error.kind(), ErrorKind::InvalidInput);
}

#[test]
fn retained_diff_preserves_and_replaces_identity_by_key_kind_and_position() {
    let mut saved_window = None;
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| {
                div()
                    .key("root")
                    .child(text("A").key("stable"))
                    .child(text("B"))
                    .into_element()
            });
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| EntityRoot { root: root.clone() })?;
            saved_window = Some(window);
            let first_stable = cx.performance_report().retained.last_diff.created;
            assert_eq!(first_stable, 3);

            root.update(cx, |root, cx| {
                *root = div()
                    .key("root")
                    .child(text("A2").key("stable"))
                    .child(text("B2"))
                    .into_element();
                cx.notify();
                Ok(())
            })?;
            root.update(cx, |root, cx| {
                *root = div()
                    .key("root")
                    .child(div().key("stable"))
                    .child(text("B3"))
                    .into_element();
                cx.notify();
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
    let window = saved_window.unwrap();
    let snapshot = run.retained_snapshot(window).unwrap();

    assert_eq!(
        snapshot.find_by_key("stable").unwrap().kind(),
        ElementKind::Div
    );
    assert_eq!(
        run.performance_report().retained.last_diff.kind_mismatches,
        1
    );
    assert!(
        run.diagnostics()
            .records()
            .iter()
            .any(|record| record.operation == "retained.kind_mismatch")
    );
}

#[test]
fn duplicate_sibling_keys_diagnose_and_preserve_first_match() {
    let mut saved_window = None;
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| {
                div()
                    .key("root")
                    .child(text("A").key("dup"))
                    .child(text("B").key("other"))
                    .into_element()
            });
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| EntityRoot { root: root.clone() })?;
            saved_window = Some(window);
            root.update(cx, |root, cx| {
                *root = div()
                    .key("root")
                    .child(text("A2").key("dup"))
                    .child(text("B2").key("dup"))
                    .into_element();
                cx.notify();
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
    let snapshot = run.retained_snapshot(saved_window.unwrap()).unwrap();
    let root = snapshot.root().unwrap();

    assert_ne!(root.children()[0].id(), root.children()[1].id());
    assert_eq!(
        run.performance_report().retained.last_diff.duplicate_keys,
        1
    );
    assert!(
        run.diagnostics()
            .records()
            .iter()
            .any(|record| record.operation == "retained.duplicate_key")
    );
}

#[test]
fn style_dirty_classification_keeps_visual_layout_and_text_lanes_separate() {
    let mut saved_window = None;
    let run = Application::new()
        .run_test(|cx| {
            let root = cx.new_entity(|_| div().key("root").into_element());
            let window = cx
                .windows()
                .open(WindowOptions::new(), |_| EntityRoot { root: root.clone() })?;
            saved_window = Some(window);
            root.update(cx, |root, cx| {
                *root = div().key("root").bg(Color::rgb(1, 2, 3)).into_element();
                cx.notify();
                Ok(())
            })?;
            let visual_lanes = cx.performance_report().dirty_lanes[0].lanes;
            assert!(visual_lanes.contains(DirtyLane::Paint.flag()));
            assert!(visual_lanes.contains(DirtyLane::Style.flag()));

            root.update(cx, |root, cx| {
                *root = div().key("root").w(fill()).into_element();
                cx.notify();
                Ok(())
            })?;
            root.update(cx, |root, cx| {
                *root = div()
                    .key("root")
                    .child(text("Hello").key("label").font_size(px(22.0)))
                    .into_element();
                cx.notify();
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
    let lanes = run
        .performance_report()
        .dirty_lanes
        .iter()
        .find(|entry| entry.window == saved_window.unwrap().id())
        .unwrap()
        .lanes;

    assert!(lanes.contains(DirtyLane::Layout.flag()));
    assert!(lanes.contains(DirtyLane::Text.flag()));
}

#[test]
fn public_probe_snapshot_exposes_structured_counters() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new(), |_| TestView::new(div()))?;
            cx.notify();
            Ok(())
        })
        .unwrap();
    let probe = run.probe_snapshot();

    assert!(probe.diagnostics().counter("runtime.command_queued") >= 1);
    assert!(probe.performance().notify_requests >= 1);
}

#[test]
fn text_measurement_projects_diagnostics_and_performance() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(
                WindowOptions::new().logical_size(LayoutSize::new(120.0, 120.0)),
                |_| {
                    TestView::new(
                        div()
                            .key("root")
                            .w(fill())
                            .child(text("Hello text diagnostics").key("label")),
                    )
                },
            )?;
            Ok(())
        })
        .unwrap();

    let performance = run.performance_report();
    assert!(performance.layout.text_query_count > 0);
    assert!(performance.text.measure_count > 0);
    assert!(performance.text.total_duration.as_nanos() > 0);
    assert_eq!(
        performance.layout.text_cache_misses,
        performance.text.cache_misses
    );
    assert_eq!(performance.layout.deferred_count, 0);
    assert_eq!(performance.layout.blocked_on_text_count, 0);
    assert_eq!(performance.text.deferred_count, 0);
    assert_eq!(performance.text.failed_count, 0);
    assert!(run.diagnostics().counter("layout.measure_text") > 0);
    assert!(run.diagnostics().counter("text.measure") > 0);
    assert!(run.diagnostics().counter("text.measure.duration_micros") > 0);
    assert!(
        run.diagnostics()
            .records()
            .iter()
            .any(|record| record.operation == "text.measure"
                && record.fields.contains_key("duration_micros"))
    );
    assert!(
        run.diagnostics()
            .records()
            .iter()
            .any(|record| record.operation == "text.measure.query"
                && record.fields.contains_key("node_id")
                && record.fields.contains_key("result")
                && !record.fields.contains_key("text"))
    );
}

#[test]
fn text_layout_uses_grapheme_clusters_for_intrinsic_width() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| {
                TestView::new(
                    div()
                        .key("root")
                        .display(nekoui::style::Display::Flex)
                        .w(px(100.0))
                        .child(text("👨‍👩‍👧‍👦a").key("emoji").font_size(px(10.0))),
                )
            })?;
            Ok(())
        })
        .unwrap();
    let layout = run.layout_snapshot(run.windows()[0].handle()).unwrap();
    let emoji = layout.find_by_key("emoji").unwrap();

    assert_eq!(emoji.border_rect().width(), 10.0);
    assert!(emoji.border_rect().width() < 35.0);
}

#[test]
fn text_min_content_width_is_owned_by_text_measurement() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| {
                TestView::new(
                    div()
                        .key("root")
                        .display(nekoui::style::Display::Flex)
                        .w(px(100.0))
                        .child(text("ab").key("short").font_size(px(10.0)))
                        .child(text("abcd").key("long").font_size(px(10.0))),
                )
            })?;
            Ok(())
        })
        .unwrap();
    let layout = run.layout_snapshot(run.windows()[0].handle()).unwrap();
    let short = layout.find_by_key("short").unwrap();
    let long = layout.find_by_key("long").unwrap();

    assert_eq!(short.border_rect().width(), 10.0);
    assert_eq!(long.border_rect().width(), 20.0);
}

fn assert_no_needles(root: &Path, needles: &[&str]) {
    if !root.exists() {
        return;
    }

    let mut pending = vec![root.to_path_buf()];
    while let Some(path) = pending.pop() {
        let metadata = std::fs::metadata(&path).unwrap();
        if metadata.is_dir() {
            for entry in std::fs::read_dir(&path).unwrap() {
                let entry = entry.unwrap();
                let path = entry.path();
                if path
                    .components()
                    .any(|component| component.as_os_str() == "old_things")
                {
                    continue;
                }
                pending.push(path);
            }
            continue;
        }

        if !matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "md")
        ) {
            continue;
        }

        let contents = std::fs::read_to_string(&path).unwrap();
        for needle in needles {
            assert!(
                !contents.contains(needle),
                "found forbidden `{needle}` in {}",
                path.display()
            );
        }
    }
}
