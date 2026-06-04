use crate::layout::LayoutRect;
use crate::platform::PhysicalSize;
use crate::render::PreparedFrameContext;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ActiveClip {
    Unclipped,
    Rect(LayoutRect),
    Empty,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PhysicalScissorRect {
    pub(super) left: i32,
    pub(super) top: i32,
    pub(super) right: i32,
    pub(super) bottom: i32,
}

impl PhysicalScissorRect {
    pub(super) fn full(surface: PhysicalSize) -> Option<Self> {
        if surface.is_zero() {
            return None;
        }
        Some(Self {
            left: 0,
            top: 0,
            right: u32_to_i32_saturating(surface.width()),
            bottom: u32_to_i32_saturating(surface.height()),
        })
    }

    pub(super) fn from_logical_clip(
        clip: LayoutRect,
        context: PreparedFrameContext,
    ) -> Option<Self> {
        let scale = context.scale_factor();
        if rect_is_empty(clip) || !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let surface = context.physical_surface_size();
        if surface.is_zero() {
            return None;
        }
        let surface_width = u32_to_i32_saturating(surface.width());
        let surface_height = u32_to_i32_saturating(surface.height());
        let left = logical_floor_to_i32(clip.x(), scale)?.clamp(0, surface_width);
        let top = logical_floor_to_i32(clip.y(), scale)?.clamp(0, surface_height);
        let right = logical_ceil_to_i32(clip.x() + clip.width(), scale)?.clamp(0, surface_width);
        let bottom = logical_ceil_to_i32(clip.y() + clip.height(), scale)?.clamp(0, surface_height);
        (right > left && bottom > top).then_some(Self {
            left,
            top,
            right,
            bottom,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct ClipStack {
    stack: Vec<Option<LayoutRect>>,
}

impl ClipStack {
    pub(super) fn push(&mut self, clip: LayoutRect) {
        let active = if rect_is_empty(clip) {
            None
        } else {
            match self.active_clip() {
                ActiveClip::Unclipped => Some(clip),
                ActiveClip::Rect(parent) => parent.intersect(clip),
                ActiveClip::Empty => None,
            }
        };
        self.stack.push(active);
    }

    pub(super) fn pop(&mut self) {
        debug_assert!(
            !self.stack.is_empty(),
            "clip stack pop without matching push"
        );
        let _ = self.stack.pop();
    }

    pub(super) fn active_clip(&self) -> ActiveClip {
        match self.stack.last().copied() {
            None => ActiveClip::Unclipped,
            Some(Some(clip)) => ActiveClip::Rect(clip),
            Some(None) => ActiveClip::Empty,
        }
    }

    pub(super) fn active_scissor(
        &self,
        context: PreparedFrameContext,
    ) -> Option<PhysicalScissorRect> {
        match self.active_clip() {
            ActiveClip::Unclipped => PhysicalScissorRect::full(context.physical_surface_size()),
            ActiveClip::Rect(clip) => PhysicalScissorRect::from_logical_clip(clip, context),
            ActiveClip::Empty => None,
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

fn rect_is_empty(rect: LayoutRect) -> bool {
    rect.width() <= 0.0 || rect.height() <= 0.0
}

fn logical_floor_to_i32(value: f32, scale: f32) -> Option<i32> {
    physical_float_to_i32((value * scale).floor())
}

fn logical_ceil_to_i32(value: f32, scale: f32) -> Option<i32> {
    physical_float_to_i32((value * scale).ceil())
}

fn physical_float_to_i32(value: f32) -> Option<i32> {
    if value.is_finite() && value >= i32::MIN as f32 && value <= i32::MAX as f32 {
        Some(value as i32)
    } else {
        None
    }
}

fn u32_to_i32_saturating(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::{LayoutSize, Viewport};

    #[test]
    fn logical_clip_converts_to_outward_rounded_physical_scissor() {
        let context = context(2.0, PhysicalSize::new(200, 100));

        assert_eq!(
            PhysicalScissorRect::from_logical_clip(LayoutRect::new(1.2, 2.3, 3.4, 4.5), context,),
            Some(PhysicalScissorRect {
                left: 2,
                top: 4,
                right: 10,
                bottom: 14,
            })
        );
    }

    #[test]
    fn logical_clip_scissor_clamps_to_framebuffer() {
        let context = context(2.0, PhysicalSize::new(200, 100));

        assert_eq!(
            PhysicalScissorRect::from_logical_clip(
                LayoutRect::new(98.2, 48.2, 10.0, 10.0),
                context,
            ),
            Some(PhysicalScissorRect {
                left: 196,
                top: 96,
                right: 200,
                bottom: 100,
            })
        );
    }

    #[test]
    fn nested_clip_push_intersects_and_pop_restores_parent() {
        let context = context(1.0, PhysicalSize::new(100, 100));
        let mut stack = ClipStack::default();

        stack.push(LayoutRect::new(0.0, 0.0, 10.0, 10.0));
        stack.push(LayoutRect::new(5.0, 5.0, 10.0, 10.0));

        assert_eq!(
            stack.active_clip(),
            ActiveClip::Rect(LayoutRect::new(5.0, 5.0, 5.0, 5.0))
        );
        assert_eq!(
            stack.active_scissor(context),
            Some(PhysicalScissorRect {
                left: 5,
                top: 5,
                right: 10,
                bottom: 10,
            })
        );

        stack.pop();

        assert_eq!(
            stack.active_clip(),
            ActiveClip::Rect(LayoutRect::new(0.0, 0.0, 10.0, 10.0))
        );

        stack.pop();

        assert_eq!(stack.active_clip(), ActiveClip::Unclipped);
        assert_eq!(
            stack.active_scissor(context),
            Some(PhysicalScissorRect {
                left: 0,
                top: 0,
                right: 100,
                bottom: 100,
            })
        );
    }

    #[test]
    fn empty_clip_intersection_skips_draws_without_poisoning_restoration() {
        let context = context(1.0, PhysicalSize::new(100, 100));
        let mut stack = ClipStack::default();

        stack.push(LayoutRect::new(0.0, 0.0, 10.0, 10.0));
        stack.push(LayoutRect::new(20.0, 20.0, 5.0, 5.0));

        assert_eq!(stack.active_clip(), ActiveClip::Empty);
        assert_eq!(stack.active_scissor(context), None);

        stack.pop();

        assert_eq!(
            stack.active_clip(),
            ActiveClip::Rect(LayoutRect::new(0.0, 0.0, 10.0, 10.0))
        );
    }

    fn context(scale_factor: f32, physical_size: PhysicalSize) -> PreparedFrameContext {
        PreparedFrameContext::for_surface(
            Viewport::new(LayoutSize::new(100.0, 50.0), scale_factor),
            physical_size,
            1,
        )
    }
}
