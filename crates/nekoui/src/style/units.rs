use crate::error::{NekoError, NekoResult};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Length {
    Px(f32),
}

impl Length {
    pub const ZERO: Self = Self::Px(0.0);

    pub fn px(value: f32) -> Self {
        Self::Px(value)
    }

    pub fn as_px(self) -> f32 {
        match self {
            Self::Px(value) => value,
        }
    }

    pub(crate) fn validate_non_negative(self, label: &'static str) -> NekoResult<()> {
        let value = self.as_px();
        if value.is_finite() && value >= 0.0 {
            Ok(())
        } else {
            Err(NekoError::invalid_input(format!(
                "{label} must be finite and non-negative"
            )))
        }
    }
}

pub fn px(value: f32) -> Length {
    Length::px(value)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Dimension {
    #[default]
    Auto,
    Fill,
    Length(Length),
}

impl From<Length> for Dimension {
    fn from(value: Length) -> Self {
        Self::Length(value)
    }
}

impl Dimension {
    pub(crate) fn validate_for_layout(self, label: &'static str) -> NekoResult<()> {
        match self {
            Self::Auto | Self::Fill => Ok(()),
            Self::Length(length) => length.validate_non_negative(label),
        }
    }
}

pub fn auto() -> Dimension {
    Dimension::Auto
}

pub fn fill() -> Dimension {
    Dimension::Fill
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edges<T> {
    pub top: T,
    pub right: T,
    pub bottom: T,
    pub left: T,
}

impl<T: Copy> Edges<T> {
    pub fn all(value: T) -> Self {
        Self {
            top: value,
            right: value,
            bottom: value,
            left: value,
        }
    }
}

impl<T: Default + Copy> Default for Edges<T> {
    fn default() -> Self {
        Self::all(T::default())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CornerRadii<T> {
    pub top_left: T,
    pub top_right: T,
    pub bottom_right: T,
    pub bottom_left: T,
}

impl<T: Copy> CornerRadii<T> {
    pub fn all(value: T) -> Self {
        Self {
            top_left: value,
            top_right: value,
            bottom_right: value,
            bottom_left: value,
        }
    }
}

impl<T: Default + Copy> Default for CornerRadii<T> {
    fn default() -> Self {
        Self::all(T::default())
    }
}
