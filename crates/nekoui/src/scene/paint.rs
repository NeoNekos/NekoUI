use super::damage::DamageRegion;
use super::diagnostic::{SceneCompileStats, SceneDiagnostic, SceneResourceDemand};
use super::fragment::PaintFragment;
use super::generation::SceneGeneration;
use super::hit_test::HitTestScene;

#[derive(Clone, Debug, PartialEq)]
pub struct PaintScene {
    generation: SceneGeneration,
    fragments: Vec<PaintFragment>,
    hit_test: HitTestScene,
    damage: DamageRegion,
    resource_demands: Vec<SceneResourceDemand>,
    diagnostics: Vec<SceneDiagnostic>,
    stats: SceneCompileStats,
}

impl PaintScene {
    pub(crate) fn new(
        generation: SceneGeneration,
        fragments: Vec<PaintFragment>,
        hit_test: HitTestScene,
        damage: DamageRegion,
        resource_demands: Vec<SceneResourceDemand>,
        diagnostics: Vec<SceneDiagnostic>,
        stats: SceneCompileStats,
    ) -> Self {
        Self {
            generation,
            fragments,
            hit_test,
            damage,
            resource_demands,
            diagnostics,
            stats,
        }
    }

    pub fn generation(&self) -> SceneGeneration {
        self.generation.clone()
    }

    pub fn fragments(&self) -> &[PaintFragment] {
        &self.fragments
    }

    pub fn hit_test(&self) -> &HitTestScene {
        &self.hit_test
    }

    pub fn damage(&self) -> &DamageRegion {
        &self.damage
    }

    pub fn resource_demands(&self) -> &[SceneResourceDemand] {
        &self.resource_demands
    }

    pub fn diagnostics(&self) -> &[SceneDiagnostic] {
        &self.diagnostics
    }

    pub fn stats(&self) -> &SceneCompileStats {
        &self.stats
    }
}
