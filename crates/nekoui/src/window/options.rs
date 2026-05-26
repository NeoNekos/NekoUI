use std::borrow::Cow;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WindowOptions {
    pub(crate) title: Cow<'static, str>,
}

impl Default for WindowOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowOptions {
    pub fn new() -> Self {
        Self {
            title: Cow::Borrowed("NekoUI"),
        }
    }

    pub fn title(mut self, title: impl Into<Cow<'static, str>>) -> Self {
        self.title = title.into();
        self
    }

    pub fn title_text(&self) -> &str {
        &self.title
    }
}
