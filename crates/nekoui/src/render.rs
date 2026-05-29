mod frame;
mod prepare;
#[cfg(any(test, target_os = "windows"))]
mod shaders;

pub(crate) use frame::{
    DrawItem, DrawItemKind, FrameGraphStats, PreparedFrame, PreparedFrameContext,
    PreparedFrameGeneration, PreparedPass, RenderPass, UploadIntent, UploadPlan,
};
#[cfg(test)]
pub(crate) use prepare::prepare_frame_graph;
pub(crate) use prepare::prepare_frame_graph_for_surface;
#[cfg(any(test, target_os = "windows"))]
pub(crate) use shaders::*;

#[cfg(test)]
mod tests;
