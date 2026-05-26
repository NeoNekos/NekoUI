use crate::retained::RetainedIdentity;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DirtyCause {
    WindowOpened,
    WindowClosed,
    AppNotified,
    RetainedChanged,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedDirty {
    pub identity: Option<RetainedIdentity>,
    pub cause: DirtyCause,
}
