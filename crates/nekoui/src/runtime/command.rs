use crate::interaction::PointerInput;
use crate::layout::LayoutSize;
use crate::platform::PlatformFact;
use crate::runtime::subscription_store::SubscriptionKey;
use crate::window::AnyWindowHandle;

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

#[derive(Clone, Debug, PartialEq)]
pub enum RuntimeCommand {
    Notify,
    NotifySubscription {
        subscription: SubscriptionKey,
    },
    PointerInput {
        handle: AnyWindowHandle,
        input: PointerInput,
    },
    PlatformFact(PlatformFact),
    Window(WindowCommand),
}

#[derive(Clone, Debug, PartialEq)]
pub enum WindowCommand {
    RequestClose {
        handle: AnyWindowHandle,
    },
    Close {
        handle: AnyWindowHandle,
    },
    Resize {
        handle: AnyWindowHandle,
        logical_size: LayoutSize,
    },
}

#[derive(Clone, Debug, PartialEq)]
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

    pub fn command(&self) -> &RuntimeCommand {
        &self.command
    }

    pub fn into_inner(self) -> RuntimeCommand {
        self.command
    }
}
