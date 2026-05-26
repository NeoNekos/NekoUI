use crate::app::{AppContext, TestRun};
use crate::error::NekoResult;
use crate::runtime::Runtime;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Application;

impl Application {
    pub fn new() -> Self {
        Self
    }

    pub fn run_test(
        self,
        run: impl FnOnce(&mut AppContext<'_>) -> NekoResult<()>,
    ) -> NekoResult<TestRun> {
        let mut runtime = Runtime::new();
        {
            let mut cx = AppContext::new(&mut runtime);
            run(&mut cx)?;
        }
        runtime.drain_all()?;
        Ok(TestRun::new(runtime))
    }
}
