#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn rgb(hex: u32) -> Self {
        let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
        let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
        let b = (hex & 0xFF) as f32 / 255.0;
        Self { r, g, b, a: 1.0 }
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::rgba(0.0, 0.0, 0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradient {
    pub start_color: Color,
    pub end_color: Color,
    pub angle_radians: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackgroundFill {
    Solid(Color),
    LinearGradient(LinearGradient),
}

impl From<Color> for BackgroundFill {
    fn from(value: Color) -> Self {
        Self::Solid(value)
    }
}

pub fn gradient(start_color: Color, end_color: Color, angle_radians: f32) -> BackgroundFill {
    BackgroundFill::LinearGradient(LinearGradient {
        start_color,
        end_color,
        angle_radians,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Length {
    #[default]
    Auto,
    Px(f32),
    Fill,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub width: Length,
    pub height: Length,
}

impl Size {
    pub const fn new(width: Length, height: Length) -> Self {
        Self { width, height }
    }
}

impl Default for Size {
    fn default() -> Self {
        Self::new(Length::Auto, Length::Auto)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeInsets {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl EdgeInsets {
    pub const fn all(value: f32) -> Self {
        Self {
            top: value,
            right: value,
            bottom: value,
            left: value,
        }
    }
}

impl Default for EdgeInsets {
    fn default() -> Self {
        Self::all(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CornerRadii {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_right: f32,
    pub bottom_left: f32,
}

impl CornerRadii {
    pub const fn all(radius: f32) -> Self {
        Self {
            top_left: radius,
            top_right: radius,
            bottom_right: radius,
            bottom_left: radius,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EdgeWidths {
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
    pub left: f32,
}

impl EdgeWidths {
    pub const fn all(width: f32) -> Self {
        Self {
            top: width,
            right: width,
            bottom: width,
            left: width,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Direction {
    #[default]
    Row,
    Column,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JustifyContent {
    #[default]
    Start,
    Center,
    End,
    SpaceBetween,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlignItems {
    #[default]
    Start,
    Center,
    End,
    Stretch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayoutStyle {
    pub size: Size,
    pub padding: EdgeInsets,
    pub margin: EdgeInsets,
    pub direction: Direction,
    pub gap: f32,
    pub justify_content: JustifyContent,
    pub align_items: AlignItems,
}

impl Default for LayoutStyle {
    fn default() -> Self {
        Self {
            size: Size::default(),
            padding: EdgeInsets::default(),
            margin: EdgeInsets::default(),
            direction: Direction::default(),
            gap: 0.0,
            justify_content: JustifyContent::default(),
            align_items: AlignItems::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PaintStyle {
    pub background: Option<BackgroundFill>,
    pub corner_radii: CornerRadii,
    pub border_widths: EdgeWidths,
    pub border_color: Option<Color>,
    pub opacity: f32,
    pub clip_children: bool,
}

impl Default for PaintStyle {
    fn default() -> Self {
        Self {
            background: None,
            corner_radii: CornerRadii::default(),
            border_widths: EdgeWidths::default(),
            border_color: None,
            opacity: 1.0,
            clip_children: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextStyle {
    pub font_family: Option<String>,
    pub font_size: f32,
    pub line_height: Option<f32>,
    pub color: Color,
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_family: None,
            font_size: 14.0,
            line_height: None,
            color: Color::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Style {
    pub layout: LayoutStyle,
    pub paint: PaintStyle,
    pub text: TextStyle,
}
