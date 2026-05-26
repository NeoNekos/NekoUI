use crate::window::{AnyWindowHandle, WindowOptions};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CommandId(u64);

impl CommandId {
    pub(crate) fn new(raw: u64) -> Self {
        Self(raw)
    }

    #[cfg(test)]
    pub fn raw(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeCommand {
    Notify,
    Window(WindowCommand),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WindowCommand {
    Open {
        handle: AnyWindowHandle,
        options: WindowOptions,
    },
    RequestClose {
        handle: AnyWindowHandle,
    },
    Close {
        handle: AnyWindowHandle,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequencedCommand {
    id: CommandId,
    command: RuntimeCommand,
}

impl SequencedCommand {
    pub fn new(id: CommandId, command: RuntimeCommand) -> Self {
        Self { id, command }
    }

    pub fn id(&self) -> CommandId {
        self.id
    }

    pub fn into_inner(self) -> RuntimeCommand {
        self.command
    }
}
