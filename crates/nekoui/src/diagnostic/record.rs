use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::error::ErrorKind;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DiagnosticArea {
    App,
    Runtime,
    Window,
    Retained,
    Style,
    Layout,
    Scene,
    Semantics,
    Render,
    Gpu,
    Input,
    Text,
    Scheduler,
    Diagnostic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum DiagnosticSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Audit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticRecord {
    pub area: DiagnosticArea,
    pub severity: DiagnosticSeverity,
    pub category: ErrorKind,
    pub operation: &'static str,
    pub message: Cow<'static, str>,
    pub fields: BTreeMap<Cow<'static, str>, Cow<'static, str>>,
}

impl DiagnosticRecord {
    pub fn new(
        area: DiagnosticArea,
        severity: DiagnosticSeverity,
        category: ErrorKind,
        operation: &'static str,
        message: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            area,
            severity,
            category,
            operation,
            message: message.into(),
            fields: BTreeMap::new(),
        }
    }

    pub fn with_field(
        mut self,
        key: impl Into<Cow<'static, str>>,
        value: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }
}
