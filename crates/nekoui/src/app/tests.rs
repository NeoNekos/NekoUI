use crate::app::Application;
use crate::diagnostic::DirtyLane;
use crate::error::ErrorKind;
use crate::window::WindowOptions;

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
                .open(WindowOptions::new().title("Hello NekoUI"))?;
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
            let handle = cx.windows().open(WindowOptions::new())?;
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
            cx.windows().open(WindowOptions::new())?;
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
            cx.windows().open(WindowOptions::new())?;
            cx.notify();
            Ok(())
        })
        .unwrap();
    let snapshot = run.diagnostics();

    assert!(snapshot.counter("runtime.command_queued") >= 2);
    assert!(snapshot.counter("runtime.command_processed") >= 2);
}
