use std::borrow::Cow;

use crate::layout::{LayoutSize, Viewport};

#[derive(Clone, Debug, PartialEq)]
pub struct WindowOptions {
    pub(crate) title: Cow<'static, str>,
    pub(crate) viewport: Viewport,
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
            viewport: Viewport::default(),
        }
    }

    pub fn title(mut self, title: impl Into<Cow<'static, str>>) -> Self {
        self.title = title.into();
        self
    }

    pub fn title_text(&self) -> &str {
        &self.title
    }

    pub fn viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = viewport;
        self
    }

    pub fn logical_size(mut self, size: LayoutSize) -> Self {
        self.viewport = self
            .viewport
            .next_generation(size, self.viewport.scale_factor());
        self
    }

    pub fn scale_factor(mut self, scale_factor: f32) -> Self {
        self.viewport = self
            .viewport
            .next_generation(self.viewport.logical_size(), scale_factor);
        self
    }

    pub fn viewport_value(&self) -> Viewport {
        self.viewport
    }
}
