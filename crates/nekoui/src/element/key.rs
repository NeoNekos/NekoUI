use std::borrow::Cow;

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ElementKey(Cow<'static, str>);

impl ElementKey {
    pub fn new(value: impl Into<Cow<'static, str>>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&'static str> for ElementKey {
    fn from(value: &'static str) -> Self {
        Self::new(value)
    }
}

impl From<String> for ElementKey {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<Cow<'static, str>> for ElementKey {
    fn from(value: Cow<'static, str>) -> Self {
        Self::new(value)
    }
}
