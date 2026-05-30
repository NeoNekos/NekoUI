mod build;
mod generation;
mod snapshot;
mod stats;

pub(crate) use build::{
    SemanticBuildInput, build_semantic_snapshot, semantic_publish_is_current_with_interaction,
};
pub(crate) use snapshot::SemanticTreeSnapshot;
pub(crate) use stats::SemanticBuildStats;

#[cfg(test)]
pub(crate) use build::{semantic_generation_for_inputs, semantic_publish_is_current};

#[cfg(test)]
mod tests;
