use crate::error::{NekoError, NekoResult};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayoutSize {
    width: f32,
    height: f32,
}

impl LayoutSize {
    pub const ZERO: Self = Self {
        width: 0.0,
        height: 0.0,
    };

    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    pub fn width(self) -> f32 {
        self.width
    }

    pub fn height(self) -> f32 {
        self.height
    }

    pub(crate) fn validate_viewport(self) -> NekoResult<()> {
        if self.width.is_finite()
            && self.width >= 0.0
            && self.height.is_finite()
            && self.height >= 0.0
        {
            Ok(())
        } else {
            Err(NekoError::invalid_input(
                "viewport logical size must be finite and non-negative",
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayoutPoint {
    x: f32,
    y: f32,
}

impl LayoutPoint {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn x(self) -> f32 {
        self.x
    }

    pub fn y(self) -> f32 {
        self.y
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayoutRect {
    origin: LayoutPoint,
    size: LayoutSize,
}

impl LayoutRect {
    pub const ZERO: Self = Self {
        origin: LayoutPoint::ZERO,
        size: LayoutSize::ZERO,
    };

    pub const fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            origin: LayoutPoint::new(x, y),
            size: LayoutSize::new(width, height),
        }
    }

    pub fn origin(self) -> LayoutPoint {
        self.origin
    }

    pub fn size(self) -> LayoutSize {
        self.size
    }

    pub fn x(self) -> f32 {
        self.origin.x()
    }

    pub fn y(self) -> f32 {
        self.origin.y()
    }

    pub fn width(self) -> f32 {
        self.size.width()
    }

    pub fn height(self) -> f32 {
        self.size.height()
    }

    pub fn contains(self, point: LayoutPoint) -> bool {
        point.x() >= self.x()
            && point.y() >= self.y()
            && point.x() < self.x() + self.width()
            && point.y() < self.y() + self.height()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ViewportGeneration(u64);

impl ViewportGeneration {
    pub const INITIAL: Self = Self(1);

    pub fn raw(self) -> u64 {
        self.0
    }

    pub(crate) fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    logical_size: LayoutSize,
    scale_factor: f32,
    generation: ViewportGeneration,
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new(LayoutSize::new(800.0, 600.0), 1.0)
    }
}

impl Viewport {
    pub const fn new(logical_size: LayoutSize, scale_factor: f32) -> Self {
        Self {
            logical_size,
            scale_factor,
            generation: ViewportGeneration::INITIAL,
        }
    }

    pub fn logical_size(self) -> LayoutSize {
        self.logical_size
    }

    pub fn scale_factor(self) -> f32 {
        self.scale_factor
    }

    pub fn generation(self) -> ViewportGeneration {
        self.generation
    }

    pub(crate) fn validate(self) -> NekoResult<()> {
        self.logical_size.validate_viewport()?;
        if self.scale_factor.is_finite() && self.scale_factor > 0.0 {
            Ok(())
        } else {
            Err(NekoError::invalid_input(
                "viewport scale factor must be finite and positive",
            ))
        }
    }

    pub(crate) fn next_generation(self, logical_size: LayoutSize, scale_factor: f32) -> Self {
        Self {
            logical_size,
            scale_factor,
            generation: self.generation.next(),
        }
    }
}
