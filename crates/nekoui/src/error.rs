use std::borrow::Cow;

use thiserror::Error;

pub type NekoResult<T> = Result<T, NekoError>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ErrorKind {
    Cancelled,
    Unavailable,
    Unsupported,
    InvalidInput,
    Stale,
    NotRenderable,
    ResourceFailure,
    BackendLost,
    Diagnostic,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("{kind:?}: {message}")]
pub struct NekoError {
    kind: ErrorKind,
    message: Cow<'static, str>,
}

impl NekoError {
    pub fn new(kind: ErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn cancelled(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Cancelled, message)
    }

    pub fn unavailable(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Unavailable, message)
    }

    pub fn unsupported(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Unsupported, message)
    }

    pub fn invalid_input(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::InvalidInput, message)
    }

    pub fn stale(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Stale, message)
    }

    pub fn not_renderable(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::NotRenderable, message)
    }

    pub fn resource_failure(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::ResourceFailure, message)
    }

    pub fn backend_lost(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::BackendLost, message)
    }

    pub fn diagnostic(message: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Diagnostic, message)
    }
}

#[cfg(test)]
mod tests {
    use super::{ErrorKind, NekoError, NekoResult};

    #[test]
    fn typed_error_keeps_category() {
        let error = NekoError::stale("window handle is closed");

        assert_eq!(error.kind(), ErrorKind::Stale);
        assert_eq!(error.message(), "window handle is closed");
    }

    #[test]
    fn result_alias_preserves_typed_error() {
        fn fail() -> NekoResult<()> {
            Err(NekoError::unavailable("current window"))
        }

        assert_eq!(fail().unwrap_err().kind(), ErrorKind::Unavailable);
    }
}
