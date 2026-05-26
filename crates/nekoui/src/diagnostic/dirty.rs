use bitflags::bitflags;

use crate::window::WindowId;

bitflags! {
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
    pub struct DirtyLanes: u32 {
        const BUILD = 1 << 0;
        const STYLE = 1 << 1;
        const LAYOUT = 1 << 2;
        const TEXT = 1 << 3;
        const SEMANTICS = 1 << 4;
        const PAINT = 1 << 5;
        const RESOURCE = 1 << 6;
        const CANVAS = 1 << 7;
        const SURFACE = 1 << 8;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DirtyLane {
    Build,
    Style,
    Layout,
    Text,
    Semantics,
    Paint,
    Resource,
    Canvas,
    Surface,
}

impl DirtyLane {
    pub fn all() -> &'static [DirtyLane] {
        &[
            DirtyLane::Build,
            DirtyLane::Style,
            DirtyLane::Layout,
            DirtyLane::Text,
            DirtyLane::Semantics,
            DirtyLane::Paint,
            DirtyLane::Resource,
            DirtyLane::Canvas,
            DirtyLane::Surface,
        ]
    }

    pub fn flag(self) -> DirtyLanes {
        match self {
            DirtyLane::Build => DirtyLanes::BUILD,
            DirtyLane::Style => DirtyLanes::STYLE,
            DirtyLane::Layout => DirtyLanes::LAYOUT,
            DirtyLane::Text => DirtyLanes::TEXT,
            DirtyLane::Semantics => DirtyLanes::SEMANTICS,
            DirtyLane::Paint => DirtyLanes::PAINT,
            DirtyLane::Resource => DirtyLanes::RESOURCE,
            DirtyLane::Canvas => DirtyLanes::CANVAS,
            DirtyLane::Surface => DirtyLanes::SURFACE,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            DirtyLane::Build => "build",
            DirtyLane::Style => "style",
            DirtyLane::Layout => "layout",
            DirtyLane::Text => "text",
            DirtyLane::Semantics => "semantics",
            DirtyLane::Paint => "paint",
            DirtyLane::Resource => "resource",
            DirtyLane::Canvas => "canvas",
            DirtyLane::Surface => "surface",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirtyLaneReport {
    pub window: WindowId,
    pub lanes: DirtyLanes,
    pub lane_names: Vec<&'static str>,
}

impl DirtyLaneReport {
    pub fn new(window: WindowId, lanes: DirtyLanes) -> Self {
        let lane_names = DirtyLane::all()
            .iter()
            .copied()
            .filter(|lane| lanes.contains(lane.flag()))
            .map(DirtyLane::name)
            .collect();

        Self {
            window,
            lanes,
            lane_names,
        }
    }
}
