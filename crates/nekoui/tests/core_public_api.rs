use nekoui::ErrorKind;
use nekoui::diagnostic::DirtyLane;
use nekoui::prelude::*;

#[test]
fn public_api_reports_core_runtime_performance() {
    let run = Application::new()
        .run_test(|cx| {
            let window = cx.windows().open(WindowOptions::new().title("Smoke"))?;
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
    assert!(report.command_ingress.commands_enqueued >= 3);
    assert!(report.command_ingress.commands_processed >= 3);
    assert!(report.dirty_lanes.iter().any(|window| {
        window.lanes.contains(DirtyLane::Build.flag())
            && window.lanes.contains(DirtyLane::Semantics.flag())
    }));
}

#[test]
fn public_probe_snapshot_exposes_structured_counters() {
    let run = Application::new()
        .run_test(|cx| {
            cx.windows().open(WindowOptions::new())?;
            cx.notify();
            Ok(())
        })
        .unwrap();
    let probe = run.probe_snapshot();

    assert!(probe.diagnostics().counter("runtime.command_queued") >= 2);
    assert!(probe.performance().notify_requests >= 1);
}
