use crate::SharedString;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SemanticsRole {
    #[default]
    Generic,
    Text,
    Button,
    TextInput,
    Image,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SemanticsState {
    pub role: SemanticsRole,
    pub label: Option<SharedString>,
    pub value: Option<SharedString>,
    pub hidden: bool,
    pub disabled: bool,
}
