use crate::error::{NekoError, NekoResult};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    space: ColorSpace,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorSpace {
    Srgb {
        red: u8,
        green: u8,
        blue: u8,
        alpha: u8,
    },
    Oklch {
        lightness: f32,
        chroma: f32,
        hue: f32,
        alpha: f32,
    },
}

impl Color {
    pub const BLACK: Self = Self::rgba(0, 0, 0, 255);
    pub const TRANSPARENT: Self = Self::rgba(0, 0, 0, 0);

    pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self::rgba(red, green, blue, 255)
    }

    pub const fn rgba(red: u8, green: u8, blue: u8, alpha: u8) -> Self {
        Self {
            space: ColorSpace::Srgb {
                red,
                green,
                blue,
                alpha,
            },
        }
    }

    pub const fn oklch(lightness: f32, chroma: f32, hue: f32) -> Self {
        Self::oklcha(lightness, chroma, hue, 1.0)
    }

    pub const fn oklcha(lightness: f32, chroma: f32, hue: f32, alpha: f32) -> Self {
        Self {
            space: ColorSpace::Oklch {
                lightness,
                chroma,
                hue,
                alpha,
            },
        }
    }

    pub fn color_space(self) -> ColorSpace {
        self.space
    }

    pub fn srgb_channels(self) -> Option<(u8, u8, u8, u8)> {
        match self.space {
            ColorSpace::Srgb {
                red,
                green,
                blue,
                alpha,
            } => Some((red, green, blue, alpha)),
            ColorSpace::Oklch { .. } => None,
        }
    }

    pub fn oklch_channels(self) -> Option<(f32, f32, f32, f32)> {
        match self.space {
            ColorSpace::Srgb { .. } => None,
            ColorSpace::Oklch {
                lightness,
                chroma,
                hue,
                alpha,
            } => Some((lightness, chroma, hue, alpha)),
        }
    }

    pub(crate) fn validate_input(self) -> NekoResult<()> {
        match self.space {
            ColorSpace::Srgb { .. } => Ok(()),
            ColorSpace::Oklch {
                lightness,
                chroma,
                hue,
                alpha,
            } => {
                if !lightness.is_finite() || !(0.0..=1.0).contains(&lightness) {
                    return Err(NekoError::invalid_input(
                        "OKLCH lightness must be finite and in the 0.0..=1.0 range",
                    ));
                }
                if !chroma.is_finite() || chroma < 0.0 {
                    return Err(NekoError::invalid_input(
                        "OKLCH chroma must be finite and non-negative",
                    ));
                }
                if !hue.is_finite() {
                    return Err(NekoError::invalid_input("OKLCH hue must be finite"));
                }
                if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
                    return Err(NekoError::invalid_input(
                        "OKLCH alpha must be finite and in the 0.0..=1.0 range",
                    ));
                }
                Ok(())
            }
        }
    }

    pub fn red(self) -> Option<u8> {
        self.srgb_channels().map(|channels| channels.0)
    }

    pub fn green(self) -> Option<u8> {
        self.srgb_channels().map(|channels| channels.1)
    }

    pub fn blue(self) -> Option<u8> {
        self.srgb_channels().map(|channels| channels.2)
    }

    pub fn alpha(self) -> Option<u8> {
        self.srgb_channels().map(|channels| channels.3)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opacity(f32);

impl Opacity {
    pub const TRANSPARENT: Self = Self(0.0);
    pub const OPAQUE: Self = Self(1.0);

    pub const fn new(value: f32) -> Self {
        Self(value)
    }

    pub fn as_f32(self) -> f32 {
        self.0
    }

    pub(crate) fn validate_input(self) -> NekoResult<()> {
        if self.0.is_finite() && (0.0..=1.0).contains(&self.0) {
            Ok(())
        } else {
            Err(NekoError::invalid_input(
                "opacity must be finite and in the 0.0..=1.0 range",
            ))
        }
    }
}

impl Default for Opacity {
    fn default() -> Self {
        Self::OPAQUE
    }
}

impl From<f32> for Opacity {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

pub fn opacity(value: f32) -> Opacity {
    Opacity::new(value)
}
