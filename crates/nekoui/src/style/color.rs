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

    pub const fn from_rgb_hex(rgb: u32) -> Self {
        assert!(rgb <= 0x00ff_ffff, "rgb hex must be 0xRRGGBB");
        Self::rgba(
            ((rgb >> 16) & 0xff) as u8,
            ((rgb >> 8) & 0xff) as u8,
            (rgb & 0xff) as u8,
            0xff,
        )
    }

    pub const fn from_rgba_hex(rgba: u32) -> Self {
        Self::rgba(
            ((rgba >> 24) & 0xff) as u8,
            ((rgba >> 16) & 0xff) as u8,
            ((rgba >> 8) & 0xff) as u8,
            (rgba & 0xff) as u8,
        )
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

    #[cfg(any(test, target_os = "windows"))]
    pub(crate) fn to_current_backend_sdr_srgb_rgba(self) -> Option<[f32; 4]> {
        match self.space {
            ColorSpace::Srgb {
                red,
                green,
                blue,
                alpha,
            } => Some([
                red as f32 / 255.0,
                green as f32 / 255.0,
                blue as f32 / 255.0,
                alpha as f32 / 255.0,
            ]),
            ColorSpace::Oklch {
                lightness,
                chroma,
                hue,
                alpha,
            } => {
                self.validate_input().ok()?;
                let hue = (hue as f64).rem_euclid(360.0).to_radians();
                let chroma = chroma as f64;
                let oklab_a = chroma * hue.cos();
                let oklab_b = chroma * hue.sin();
                let lightness = lightness as f64;
                let l = lightness + 0.396_337_777_4 * oklab_a + 0.215_803_757_3 * oklab_b;
                let m = lightness - 0.105_561_345_8 * oklab_a - 0.063_854_172_8 * oklab_b;
                let s = lightness - 0.089_484_177_5 * oklab_a + 1.291_485_548_0 * oklab_b;
                let l = l * l * l;
                let m = m * m * m;
                let s = s * s * s;
                let red = 4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s;
                let green = -1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s;
                let blue = -0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s;
                Some([
                    linear_srgb_to_sdr_channel(red)?,
                    linear_srgb_to_sdr_channel(green)?,
                    linear_srgb_to_sdr_channel(blue)?,
                    alpha,
                ])
            }
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

#[cfg(any(test, target_os = "windows"))]
fn linear_srgb_to_sdr_channel(value: f64) -> Option<f32> {
    if !value.is_finite() {
        return None;
    }
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        12.92 * value
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    encoded.is_finite().then_some(encoded as f32)
}

pub const fn rgb(rgb: u32) -> Color {
    Color::from_rgb_hex(rgb)
}

pub const fn rgba(rgba: u32) -> Color {
    Color::from_rgba_hex(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ErrorKind;

    #[test]
    fn current_backend_sdr_srgb_preserves_direct_srgb() {
        assert_eq!(
            Color::rgba(12, 34, 56, 78).to_current_backend_sdr_srgb_rgba(),
            Some([12.0 / 255.0, 34.0 / 255.0, 56.0 / 255.0, 78.0 / 255.0,])
        );
    }

    #[test]
    fn current_backend_sdr_srgb_converts_oklch() {
        assert_rgba_close(
            Color::oklcha(0.5, 0.1, 120.0, 0.25)
                .to_current_backend_sdr_srgb_rgba()
                .unwrap(),
            [0.420_088_9, 0.328_834_68, 0.593_882_74, 0.25],
        );
    }

    #[test]
    fn current_backend_sdr_srgb_clamps_out_of_gamut_oklch() {
        assert_eq!(
            Color::oklch(0.5, 10.0, 0.0).to_current_backend_sdr_srgb_rgba(),
            Some([1.0, 0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn invalid_oklch_validation_stays_invalid_input() {
        for color in [
            Color::oklch(f32::NAN, 0.1, 0.0),
            Color::oklch(0.5, -0.1, 0.0),
            Color::oklch(0.5, 0.1, f32::INFINITY),
            Color::oklcha(0.5, 0.1, 0.0, f32::NAN),
        ] {
            assert_eq!(
                color.validate_input().unwrap_err().kind(),
                ErrorKind::InvalidInput
            );
            assert_eq!(color.to_current_backend_sdr_srgb_rgba(), None);
        }
    }

    #[test]
    fn rgb_hex_alias_uses_rrggbb_with_opaque_alpha() {
        assert_eq!(rgb(0xffffff).srgb_channels(), Some((255, 255, 255, 255)));
        assert_eq!(
            rgb(0x123456).srgb_channels(),
            Some((0x12, 0x34, 0x56, 0xff))
        );
    }

    #[test]
    fn rgba_hex_alias_uses_rrggbbaa() {
        assert_eq!(
            rgba(0x12345678).srgb_channels(),
            Some((0x12, 0x34, 0x56, 0x78))
        );
    }

    #[test]
    fn hex_constructors_work_in_const_contexts() {
        const FREE_RGB: Color = rgb(0x123456);
        const FREE_RGBA: Color = rgba(0x12345678);
        const ASSOCIATED_RGB: Color = Color::from_rgb_hex(0xffffff);
        const ASSOCIATED_RGBA: Color = Color::from_rgba_hex(0x12345678);

        assert_eq!(FREE_RGB.srgb_channels(), Some((0x12, 0x34, 0x56, 0xff)));
        assert_eq!(FREE_RGBA.srgb_channels(), Some((0x12, 0x34, 0x56, 0x78)));
        assert_eq!(ASSOCIATED_RGB.srgb_channels(), Some((255, 255, 255, 255)));
        assert_eq!(
            ASSOCIATED_RGBA.srgb_channels(),
            Some((0x12, 0x34, 0x56, 0x78))
        );
    }

    #[test]
    #[should_panic(expected = "rgb hex must be 0xRRGGBB")]
    fn rgb_hex_rejects_values_outside_rrggbb() {
        let _ = rgb(0x0100_0000);
    }

    fn assert_rgba_close(actual: [f32; 4], expected: [f32; 4]) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 0.000_001,
                "{actual} != {expected}"
            );
        }
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
