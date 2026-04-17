use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Runtime(#[from] RuntimeError),
    #[error(transparent)]
    Platform(#[from] PlatformError),
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    #[error("entity {0} was not found")]
    EntityNotFound(u64),
    #[error("entity {0} has a different concrete type")]
    TypeMismatch(u64),
    #[error("queued event payload type did not match its subscription")]
    EventTypeMismatch,
}

#[derive(Debug, Error)]
#[error("{message}")]
pub struct PlatformError {
    message: String,
}

impl PlatformError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}
