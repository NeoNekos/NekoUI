use crate::app::AppContext;
#[cfg(test)]
use crate::app::run_result::TestRun;
use crate::error::NekoResult;
use crate::platform::ApplicationPlatform;
#[cfg(test)]
use crate::runtime::Runtime;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Application;

impl Application {
    /// Creates a new application launch object.
    pub fn new() -> Self {
        Self
    }

    pub fn run(self, run: impl FnOnce(&mut AppContext<'_>) -> NekoResult<()>) -> NekoResult<()> {
        ApplicationPlatform::run(self, run)
    }

    #[cfg(test)]
    pub(crate) fn run_test(
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
