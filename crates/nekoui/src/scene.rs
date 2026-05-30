mod compile;
mod damage;
mod diagnostic;
mod fragment;
mod generation;
mod hit_test;
mod paint;

pub use damage::{DamageReason, DamageRegion};
pub use diagnostic::{ResourceDemandKind, SceneCompileStats, SceneDiagnostic, SceneResourceDemand};
pub use fragment::{PaintFragment, PaintFragmentKind, SceneOrder};
pub use generation::{SceneGeneration, SceneInputSignature, SceneSignatureFact};
pub(crate) use hit_test::HitTestPathNode;
pub use hit_test::{HitTestEntry, HitTestScene};
pub use paint::PaintScene;

pub(crate) use compile::{
    SceneCompileInput, compile_scene, scene_generation_for_inputs_with_interaction,
    scene_publish_is_current_with_interaction,
};
#[cfg(test)]
pub(crate) use compile::{scene_generation_for_inputs, scene_publish_is_current};

#[cfg(test)]
mod tests;
