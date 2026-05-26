use crate::diagnostic::DirtyLane;
use crate::runtime::Runtime;
use crate::runtime::command::RuntimeCommand;
use crate::window::WindowOptions;

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
    let window = runtime.open_window(WindowOptions::new()).unwrap();

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
        scheduler_state
            .dirty_lanes()
            .contains(DirtyLane::Paint.flag())
    );
    assert!(scheduler_state.pending_redraw());
    assert_eq!(runtime.state().retained_dirty().len(), 3);
    assert_eq!(runtime.performance_report().coalesced_redraws, 2);
}
