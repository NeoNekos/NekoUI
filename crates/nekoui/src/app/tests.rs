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
