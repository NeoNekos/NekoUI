#![forbid(unsafe_code)]

pub mod app;
pub mod diagnostic;
pub mod error;
pub mod prelude;
pub mod retained;
mod runtime;
pub mod window;

pub use app::{AppContext, Application, TestRun};
pub use error::{ErrorKind, NekoError, NekoResult};
