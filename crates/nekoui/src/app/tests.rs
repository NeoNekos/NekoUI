use crate::app::{Application, Context, Entity, Render, Subscription, WeakEntity};
use crate::diagnostic::DirtyLane;
use crate::element::{IntoElement, div};
use crate::error::ErrorKind;
use crate::window::WindowOptions;

#[derive(Debug)]
struct EmptyRoot;

impl Render for EmptyRoot {
    fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
        div()
    }
}

#[test]
fn startup_transaction_can_open_zero_windows() {
    let run = Application::new().run_test(|_cx| Ok(())).unwrap();

    assert_eq!(run.windows().len(), 0);
    assert_eq!(run.performance_report().windows_alive, 0);
}

#[test]
fn startup_transaction_records_logical_window() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows()
                .open(WindowOptions::new().title("Hello NekoUI"), |_| EmptyRoot)?;
            Ok(())
        })
        .unwrap();

    assert_eq!(run.windows().len(), 1);
    assert_eq!(run.windows()[0].title(), "Hello NekoUI");
}

#[test]
fn closed_window_handle_returns_typed_stale() {
    let error = Application::new()
        .run_test(|cx| {
            let handle = cx.windows().open(WindowOptions::new(), |_| EmptyRoot)?;
            cx.windows().close(handle)?;
            cx.windows().validate(handle)
        })
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Stale);
}

#[test]
fn notify_reports_dirty_lanes_and_performance_counters() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| EmptyRoot)?;
            cx.notify();
            cx.notify();
            Ok(())
        })
        .unwrap();
    let report = run.performance_report();

    assert_eq!(report.notify_requests, 2);
    assert!(report.coalesced_redraws >= 1);
    assert_eq!(report.dirty_lanes.len(), 1);
    assert!(
        report.dirty_lanes[0]
            .lanes
            .contains(DirtyLane::Build.flag())
    );
    assert!(
        report.dirty_lanes[0]
            .lanes
            .contains(DirtyLane::Paint.flag())
    );
}

#[test]
fn diagnostics_snapshot_contains_structured_counters() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| EmptyRoot)?;
            cx.notify();
            Ok(())
        })
        .unwrap();
    let snapshot = run.diagnostics();

    assert!(snapshot.counter("runtime.command_queued") >= 1);
    assert!(snapshot.counter("runtime.command_processed") >= 1);
}

#[test]
fn entity_creation_read_and_update_use_transactions() {
    let run = Application::new()
        .run_test(|cx| {
            let counter = cx.new_entity(|_| 1_i32);

            assert_eq!(counter.read(cx, |value| *value)?, 1);
            counter.update(cx, |value, entity_cx| {
                *value += 1;
                entity_cx.notify();
                Ok(())
            })?;
            assert_eq!(counter.read(cx, |value| *value)?, 2);
            Ok(())
        })
        .unwrap();

    assert_eq!(run.performance_report().notify_requests, 1);
}

#[test]
fn entity_notify_uses_existing_dirty_integration() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new(), |_| EmptyRoot)?;
            let counter = cx.new_entity(|_| 1_i32);
            counter.update(cx, |_value, entity_cx| {
                entity_cx.notify();
                Ok(())
            })?;
            Ok(())
        })
        .unwrap();
    let report = run.performance_report();

    assert_eq!(report.notify_requests, 1);
    assert_eq!(report.dirty_lanes.len(), 1);
    assert!(
        report.dirty_lanes[0]
            .lanes
            .contains(DirtyLane::Build.flag())
    );
    assert!(
        report.dirty_lanes[0]
            .lanes
            .contains(DirtyLane::Paint.flag())
    );
}

#[test]
fn weak_entity_reports_stale_after_last_entity_drop() {
    let error = Application::new()
        .run_test(|cx| {
            let weak = {
                let counter = cx.new_entity(|_| 1_i32);
                counter.downgrade()
            };

            weak.update(cx, |value, _cx| {
                *value += 1;
                Ok(())
            })
        })
        .unwrap_err();

    assert_eq!(error.kind(), ErrorKind::Stale);
}

#[test]
fn observe_callbacks_flush_after_outer_transaction() {
    struct Observer {
        values: Vec<i32>,
        _subscription: Subscription,
    }

    let run = Application::new()
        .run_test(|cx| {
            let source = cx.new_entity(|_| 0_i32);
            let observer = cx.new_entity(|entity_cx: &mut Context<'_, Observer>| {
                let subscription = entity_cx.observe(&source, |observer, source, cx| {
                    let source_value = source.read(cx, |value| *value).unwrap();
                    assert_eq!(source_value, 1);
                    observer.values.push(1);
                });
                Observer {
                    values: Vec::new(),
                    _subscription: subscription,
                }
            });

            source.update(cx, |value, source_cx| {
                *value += 1;
                source_cx.notify();
                assert_eq!(
                    observer.read(source_cx, |observer| observer.values.len())?,
                    0
                );
                Ok(())
            })?;

            assert_eq!(
                observer.read(cx, |observer| observer.values.clone())?,
                vec![1]
            );
            Ok(())
        })
        .unwrap();

    assert_eq!(run.performance_report().notify_requests, 1);
}

#[test]
fn dropping_subscription_cancels_observer_callback() {
    Application::new()
        .run_test(|cx| {
            let source = cx.new_entity(|_| 0_i32);
            let observer = cx.new_entity(|entity_cx: &mut Context<'_, Vec<i32>>| {
                let subscription = entity_cx.observe(&source, |values, _source, _cx| {
                    values.push(1);
                });
                drop(subscription);
                Vec::new()
            });

            source.update(cx, |value, source_cx| {
                *value += 1;
                source_cx.notify();
                Ok(())
            })?;

            assert_eq!(observer.read(cx, |values| values.len())?, 0);
            let diagnostics = cx.diagnostics();
            assert_eq!(diagnostics.counter("api.subscription.cancelled"), 1);
            Ok(())
        })
        .unwrap();
}

#[test]
fn subscription_can_cancel_itself_during_callback() {
    struct SelfCancellingObserver {
        calls: usize,
        subscription: Option<Subscription>,
    }

    Application::new()
        .run_test(|cx| {
            let source = cx.new_entity(|_| 0_i32);
            let observer = cx.new_entity(|entity_cx: &mut Context<'_, SelfCancellingObserver>| {
                let subscription = entity_cx.observe(&source, |observer, _source, _cx| {
                    observer.calls += 1;
                    observer.subscription.take();
                });
                SelfCancellingObserver {
                    calls: 0,
                    subscription: Some(subscription),
                }
            });

            source.update(cx, |value, source_cx| {
                *value += 1;
                source_cx.notify();
                Ok(())
            })?;
            assert_eq!(observer.read(cx, |observer| observer.calls)?, 1);

            source.update(cx, |value, source_cx| {
                *value += 1;
                source_cx.notify();
                Ok(())
            })?;
            assert_eq!(observer.read(cx, |observer| observer.calls)?, 1);
            assert_eq!(cx.diagnostics().counter("api.subscription.cancelled"), 2);
            Ok(())
        })
        .unwrap();
}

#[test]
fn released_entity_slot_does_not_alias_new_entity() {
    Application::new()
        .run_test(|cx| {
            let stale = {
                let first = cx.new_entity(|_| 1_i32);
                first.downgrade()
            };
            let second = cx.new_entity(|_| 2_i32);

            assert_eq!(second.read(cx, |value| *value)?, 2);
            assert_eq!(stale.upgrade(cx).unwrap_err().kind(), ErrorKind::Stale);
            Ok(())
        })
        .unwrap();
}

#[test]
fn entity_context_subscription_are_available_from_prelude_surface() {
    fn assert_prelude_surface<T: 'static>(
        _entity: &Entity<T>,
        _weak: &WeakEntity<T>,
        _subscription: Option<Subscription>,
    ) {
    }

    Application::new()
        .run_test(|cx| {
            let entity = cx.new_entity(|_| 1_i32);
            let weak = entity.downgrade();
            assert_prelude_surface(&entity, &weak, None);
            Ok(())
        })
        .unwrap();
}

#[test]
fn entity_api_diagnostics_are_visible_by_dotted_names() {
    let run = Application::new()
        .run_test(|cx| {
            let source = cx.new_entity(|_| 0_i32);
            let observer = cx.new_entity(|entity_cx: &mut Context<'_, ObserverDiagnostics>| {
                let subscription = entity_cx.observe(&source, |observer, source, cx| {
                    observer.seen = source.read(cx, |value| *value).unwrap();
                });
                ObserverDiagnostics {
                    seen: 0,
                    _subscription: subscription,
                }
            });

            source.read(cx, |_| ())?;
            source.update(cx, |value, entity_cx| {
                *value = 7;
                entity_cx.notify();
                Ok(())
            })?;
            let weak = {
                let released = cx.new_entity(|_| 1_i32);
                released.downgrade()
            };
            assert_eq!(weak.upgrade(cx).unwrap_err().kind(), ErrorKind::Stale);
            assert_eq!(observer.read(cx, |observer| observer.seen)?, 7);
            Ok(())
        })
        .unwrap();
    let diagnostics = run.diagnostics();

    assert!(diagnostics.counter("api.entity.created") >= 3);
    assert!(diagnostics.counter("api.entity.read") >= 3);
    assert!(diagnostics.counter("api.entity.update") >= 1);
    assert!(diagnostics.counter("api.entity.notify") >= 1);
    assert!(diagnostics.counter("api.entity.notification_flushed") >= 1);
    assert!(diagnostics.counter("api.handle.stale") >= 1);
}

struct ObserverDiagnostics {
    seen: i32,
    _subscription: Subscription,
}

mod internal_harness_public_api_tests {
    use crate::app::{Application, Context, Render};
    use crate::diagnostic::DirtyLane;
    use crate::element::{IntoElement, div, text};
    use crate::error::ErrorKind;
    use crate::layout::LayoutSize;
    use crate::style::{Color, StyleExt, fill, px};
    use crate::window::WindowOptions;

    #[derive(Debug)]
    struct TestRoot {
        label: String,
    }

    impl Render for TestRoot {
        fn render(&mut self, _cx: &mut Context<'_, Self>) -> impl IntoElement {
            div()
                .key("root")
                .p(px(12.0))
                .w(fill())
                .bg(Color::rgb(240, 241, 242))
                .child(text(self.label.clone()).key("label").font_size(px(20.0)))
        }
    }

    #[test]
    fn internal_harness_exposes_read_only_core_snapshots() {
        let run = Application::new()
            .run_test(|cx| {
                cx.windows().open(
                    WindowOptions::new()
                        .title("Snapshots")
                        .logical_size(LayoutSize::new(320.0, 240.0)),
                    |_| TestRoot {
                        label: "Hello".to_owned(),
                    },
                )?;
                Ok(())
            })
            .unwrap();

        let window = run.windows()[0].handle();
        let retained = run.retained_snapshot(window).unwrap();
        let style = run.style_snapshot(window).unwrap();
        let layout = run.layout_snapshot(window).unwrap();
        let scene = run.scene_snapshot(window).unwrap();

        assert_eq!(retained.node_count(), 2);
        assert_eq!(retained.find_by_key("label").unwrap().text(), Some("Hello"));
        assert_eq!(style.node_count(), 2);
        assert_eq!(
            layout.viewport().logical_size(),
            LayoutSize::new(320.0, 240.0)
        );
        assert_eq!(scene.stats().node_count, 2);
        assert_eq!(run.probe_snapshot().performance().retained.node_count, 2);
        assert!(run.performance_report().retained.node_count >= 2);
        assert!(run.diagnostics().counter("scene.compile") >= 1);
    }

    #[test]
    fn internal_harness_reports_runtime_performance_and_stale_handles() {
        let run = Application::new()
            .run_test(|cx| {
                let window = cx
                    .windows()
                    .open(WindowOptions::new().title("Smoke"), |_| TestRoot {
                        label: "Smoke".to_owned(),
                    })?;
                cx.notify();
                cx.windows().close(window)?;
                assert_eq!(
                    cx.windows().validate(window).unwrap_err().kind(),
                    ErrorKind::Stale
                );
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
    fn internal_harness_rerenders_after_entity_update_and_notify() {
        let run = Application::new()
            .run_test(|cx| {
                let window = cx.windows().open(WindowOptions::new(), |_| TestRoot {
                    label: "before".to_owned(),
                })?;
                let root = cx.windows().root_view(window)?;
                root.update(cx, |root, cx| {
                    root.label = "after".to_owned();
                    cx.notify();
                    Ok(())
                })?;
                cx.windows().resize(window, LayoutSize::new(480.0, 320.0))?;
                Ok(())
            })
            .unwrap();

        let window = run.windows()[0].handle();
        assert_eq!(
            run.layout_snapshot(window)
                .unwrap()
                .viewport()
                .logical_size(),
            LayoutSize::new(480.0, 320.0)
        );
        assert!(run.performance_report().notify_requests >= 1);
        assert!(run.performance_report().layout.pass_count >= 2);
    }
}

mod internal_million_node_stress_tests {
    use crate::app::{Application, Context, Entity, Render};
    use crate::element::{Element, IntoElement, div, text};
    use crate::layout::LayoutSize;
    use crate::style::{StyleExt, fill, px};
    use crate::window::WindowOptions;

    #[derive(Clone, Copy, Debug)]
    enum StressShape {
        Simple,
        Complex,
    }

    #[derive(Debug)]
    struct StressRoot {
        root: Entity<Element>,
    }

    impl Render for StressRoot {
        fn render(&mut self, cx: &mut Context<'_, Self>) -> impl IntoElement {
            self.root.read(cx, Clone::clone).unwrap()
        }
    }

    fn retained_style_branch(
        depth: usize,
        width: usize,
        index: usize,
        style_version: u8,
    ) -> Element {
        if depth == 0 {
            return text(format!("leaf-{index}"))
                .key(format!("leaf-{index}"))
                .font_size(px(12.0 + f32::from(style_version)))
                .into_element();
        }

        let mut node = div()
            .key(format!("node-{depth}-{index}"))
            .p(px((depth % 5) as f32 + f32::from(style_version)))
            .bg(crate::style::Color::rgb(
                (depth % 255) as u8,
                (index % 255) as u8,
                64 + style_version,
            ));
        for child in 0..width {
            node = node.child(retained_style_branch(
                depth - 1,
                width,
                index * width + child,
                style_version,
            ));
        }
        node.into_element()
    }

    fn pruned_retained_branch(depth: usize, width: usize, index: usize) -> Element {
        if depth == 0 {
            return text(format!("leaf-{index}"))
                .key(format!("leaf-{index}"))
                .into_element();
        }

        let mut node = div()
            .key(format!("node-{depth}-{index}"))
            .p(px((depth % 5) as f32));
        for child in 0..width - 1 {
            node = node.child(pruned_retained_branch(
                depth - 1,
                width,
                index * width + child,
            ));
        }
        node.into_element()
    }

    fn stress_branch(depth: usize, width: usize, index: usize, shape: StressShape) -> Element {
        if depth == 0 {
            let leaf = text(format!("leaf-{index}")).key(format!("leaf-{index}"));
            return match shape {
                StressShape::Simple => leaf.into_element(),
                StressShape::Complex => {
                    leaf.font_size(px(12.0 + (index % 3) as f32)).into_element()
                }
            };
        }

        let mut node = div().key(format!("node-{depth}-{index}"));
        if matches!(shape, StressShape::Complex) {
            node = node
                .p(px((depth % 5) as f32))
                .gap(px((index % 3) as f32))
                .bg(crate::style::Color::rgb(
                    (depth % 255) as u8,
                    (index % 255) as u8,
                    96,
                ));
        }
        for child in 0..width {
            node = node.child(stress_branch(
                depth - 1,
                width,
                index * width + child,
                shape,
            ));
        }
        node.into_element()
    }

    fn text_heavy_branch(depth: usize, width: usize, index: usize, text_len: usize) -> Element {
        if depth == 0 {
            let payload = format!("leaf-{index}-{}", "text".repeat(text_len));
            return text(payload)
                .key(format!("text-leaf-{index}"))
                .font_size(px(12.0 + (index % 5) as f32))
                .into_element();
        }

        let mut node = div()
            .key(format!("text-node-{depth}-{index}"))
            .w(fill())
            .p(px((depth % 3) as f32));
        for child in 0..width {
            node = node.child(text_heavy_branch(
                depth - 1,
                width,
                index * width + child,
                text_len,
            ));
        }
        node.into_element()
    }

    fn run_layout_stress(depth: usize, width: usize, shape: StressShape, minimum_nodes: usize) {
        let run = Application::new()
            .run_test(|cx| {
                let root = cx.new_entity(|_| stress_branch(depth, width, 1, shape));
                cx.windows()
                    .open(WindowOptions::new(), |_| StressRoot { root: root.clone() })?;
                Ok(())
            })
            .unwrap();
        let report = run.performance_report();

        assert!(report.layout.node_count >= minimum_nodes);
        assert_eq!(report.layout.pass_count, 1);
    }

    fn run_scene_stress(depth: usize, width: usize, shape: StressShape, minimum_nodes: usize) {
        let run = Application::new()
            .run_test(|cx| {
                let root = cx.new_entity(|_| stress_branch(depth, width, 1, shape));
                cx.windows()
                    .open(WindowOptions::new(), |_| StressRoot { root: root.clone() })?;
                Ok(())
            })
            .unwrap();
        let report = run.performance_report();

        assert!(report.scene.node_count >= minimum_nodes);
        assert!(report.scene.compile_count >= 1);
        assert!(report.scene.last_compile.fragment_count > 0);
        assert_eq!(
            report.scene.fragment_count,
            report.scene.last_compile.fragment_count
        );
        assert!(report.scene.hit_test_entry_count >= minimum_nodes);
        assert!(report.phase_durations.contains_key("scene.compile"));
    }

    fn run_text_heavy_stress(depth: usize, width: usize, minimum_nodes: usize) {
        let run = Application::new()
            .run_test(|cx| {
                let root = cx.new_entity(|_| text_heavy_branch(depth, width, 1, 6));
                cx.windows().open(
                    WindowOptions::new().logical_size(LayoutSize::new(640.0, 480.0)),
                    |_| StressRoot { root: root.clone() },
                )?;
                Ok(())
            })
            .unwrap();
        let report = run.performance_report();

        assert!(report.layout.node_count >= minimum_nodes);
        assert!(report.scene.node_count >= minimum_nodes);
        assert!(report.scene.compile_count >= 1);
        assert!(report.scene.fragment_count > 0);
        assert!(report.scene.hit_test_entry_count >= minimum_nodes);
        assert!(report.phase_durations.contains_key("scene.compile"));
        assert!(report.scene.resource_demand_count > 0);
        assert!(report.layout.text_query_count > 0);
        assert!(report.text.measure_count > 0);
    }

    #[test]
    #[ignore = "manual internal million-node retained/style stress harness"]
    fn internal_million_node_retained_style_stress() {
        let width = 10;
        let depth = 6;
        let run = Application::new()
            .run_test(|cx| {
                let root = cx.new_entity(|_| retained_style_branch(depth, width, 1, 0));
                cx.windows()
                    .open(WindowOptions::new().title("Million Node Stress"), |_| {
                        StressRoot { root: root.clone() }
                    })?;
                let first = cx.performance_report().retained.last_diff.clone();
                assert!(first.created >= 1_000_000);
                assert_eq!(first.preserved, 0);
                assert_eq!(first.destroyed, 0);

                root.update(cx, |root, cx| {
                    *root = retained_style_branch(depth, width, 1, 1);
                    cx.notify();
                    Ok(())
                })?;
                let update = cx.performance_report().retained.last_diff.clone();
                assert!(update.preserved >= 1_000_000);
                assert_eq!(update.created, 0);
                assert_eq!(update.replaced, 0);
                assert_eq!(update.destroyed, 0);

                root.update(cx, |root, cx| {
                    *root = pruned_retained_branch(depth, width, 1);
                    cx.notify();
                    Ok(())
                })?;
                let delete = cx.performance_report().retained.last_diff.clone();
                assert!(delete.preserved >= 100_000);
                assert!(delete.destroyed > 0);
                assert_eq!(delete.created, 0);
                Ok(())
            })
            .unwrap();

        let report = run.performance_report();
        assert!(report.retained.last_diff.destroyed > 0);
        assert!(report.retained.node_count > 100_000);
        assert!(report.retained.node_count < 1_000_000);
    }

    #[test]
    #[ignore = "manual internal deterministic layout/scene stress harness"]
    fn internal_ten_thousand_node_layout_scene_stress() {
        let run = Application::new()
            .run_test(|cx| {
                let root = cx.new_entity(|_| stress_branch(4, 10, 1, StressShape::Complex));
                cx.windows()
                    .open(WindowOptions::new(), |_| StressRoot { root: root.clone() })?;
                Ok(())
            })
            .unwrap();
        let report = run.performance_report();

        assert!(report.layout.node_count >= 10_000);
        assert!(report.scene.node_count >= 10_000);
        assert!(report.scene.fragment_count > 0);
        assert!(report.phase_durations.contains_key("scene.compile"));
    }

    #[test]
    #[ignore = "manual internal 10k simple layout stress harness"]
    fn internal_ten_thousand_simple_layout_stress() {
        run_layout_stress(4, 10, StressShape::Simple, 10_000);
    }

    #[test]
    #[ignore = "manual internal 10k complex layout stress harness"]
    fn internal_ten_thousand_complex_layout_stress() {
        run_layout_stress(4, 10, StressShape::Complex, 10_000);
    }

    #[test]
    #[ignore = "manual internal 100k simple layout stress harness"]
    fn internal_hundred_thousand_simple_layout_stress() {
        run_layout_stress(5, 10, StressShape::Simple, 100_000);
    }

    #[test]
    #[ignore = "manual internal 100k complex layout stress harness"]
    fn internal_hundred_thousand_complex_layout_stress() {
        run_layout_stress(5, 10, StressShape::Complex, 100_000);
    }

    #[test]
    #[ignore = "manual internal 1m simple layout stress harness"]
    fn internal_million_simple_layout_stress() {
        run_layout_stress(6, 10, StressShape::Simple, 1_000_000);
    }

    #[test]
    #[ignore = "manual internal 1m complex layout stress harness"]
    fn internal_million_complex_layout_stress() {
        run_layout_stress(6, 10, StressShape::Complex, 1_000_000);
    }

    #[test]
    #[ignore = "manual internal 10k simple scene stress harness"]
    fn internal_ten_thousand_simple_scene_stress() {
        run_scene_stress(4, 10, StressShape::Simple, 10_000);
    }

    #[test]
    #[ignore = "manual internal 10k complex scene stress harness"]
    fn internal_ten_thousand_complex_scene_stress() {
        run_scene_stress(4, 10, StressShape::Complex, 10_000);
    }

    #[test]
    #[ignore = "manual internal 100k simple scene stress harness"]
    fn internal_hundred_thousand_simple_scene_stress() {
        run_scene_stress(5, 10, StressShape::Simple, 100_000);
    }

    #[test]
    #[ignore = "manual internal 100k complex scene stress harness"]
    fn internal_hundred_thousand_complex_scene_stress() {
        run_scene_stress(5, 10, StressShape::Complex, 100_000);
    }

    #[test]
    #[ignore = "manual internal 1m simple scene stress harness"]
    fn internal_million_simple_scene_stress() {
        run_scene_stress(6, 10, StressShape::Simple, 1_000_000);
    }

    #[test]
    #[ignore = "manual internal 1m complex scene stress harness"]
    fn internal_million_complex_scene_stress() {
        run_scene_stress(6, 10, StressShape::Complex, 1_000_000);
    }

    #[test]
    #[ignore = "manual internal 10k text-heavy layout/scene stress harness"]
    fn internal_ten_thousand_text_heavy_layout_stress() {
        run_text_heavy_stress(4, 10, 10_000);
    }

    #[test]
    #[ignore = "manual internal 100k text-heavy layout/scene stress harness"]
    fn internal_hundred_thousand_text_heavy_layout_stress() {
        run_text_heavy_stress(5, 10, 100_000);
    }

    #[test]
    #[ignore = "manual internal 1m text-heavy layout/scene stress harness"]
    fn internal_million_text_heavy_layout_stress() {
        run_text_heavy_stress(6, 10, 1_000_000);
    }
}
