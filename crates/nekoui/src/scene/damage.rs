use crate::layout::LayoutRect;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DamageReason {
    Initial,
    ConservativeInputChange,
    StaleDrop,
    Unchanged,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DamageRegion {
    rects: Vec<LayoutRect>,
    reason: DamageReason,
}

impl DamageRegion {
    pub(crate) fn new(rects: Vec<LayoutRect>, reason: DamageReason) -> Self {
        Self { rects, reason }
    }

    pub(crate) fn initial(rect: LayoutRect) -> Self {
        Self::new(vec![rect], DamageReason::Initial)
    }

    pub(crate) fn unchanged() -> Self {
        Self::new(Vec::new(), DamageReason::Unchanged)
    }

    pub fn rects(&self) -> &[LayoutRect] {
        &self.rects
    }

    pub fn reason(&self) -> DamageReason {
        self.reason
    }

    pub fn region_count(&self) -> usize {
        self.rects.len()
    }

    pub fn total_area(&self) -> f32 {
        self.rects
            .iter()
            .map(|rect| rect.width().max(0.0) * rect.height().max(0.0))
            .sum()
    }
}
