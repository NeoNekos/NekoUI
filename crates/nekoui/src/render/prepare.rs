use std::collections::BTreeMap;
use std::time::Instant;

use crate::render::{
    DrawItem, DrawItemKind, FrameGraphStats, PreparedFrame, PreparedFrameContext,
    PreparedFrameGeneration, PreparedPass, RenderPass, UploadIntent, UploadPlan,
};
use crate::scene::{PaintFragmentKind, PaintScene, SceneOrder};

#[cfg(test)]
pub(crate) fn prepare_frame_graph(scene: &PaintScene) -> PreparedFrame {
    prepare_frame_graph_inner(scene, PreparedFrameContext::scene_only())
}

pub(crate) fn prepare_frame_graph_for_surface(
    scene: &PaintScene,
    context: PreparedFrameContext,
) -> PreparedFrame {
    prepare_frame_graph_inner(scene, context)
}

fn prepare_frame_graph_inner(scene: &PaintScene, context: PreparedFrameContext) -> PreparedFrame {
    let started = Instant::now();
    let mut draw_items = scene
        .fragments()
        .iter()
        .map(|fragment| {
            let kind = match fragment.kind() {
                PaintFragmentKind::Rect { color } => DrawItemKind::Rect { color: *color },
                PaintFragmentKind::Text {
                    text_generation,
                    text_metrics_generation,
                    color,
                    ..
                } => DrawItemKind::Text {
                    text_generation: text_generation.clone(),
                    text_metrics_generation: *text_metrics_generation,
                    layout: fragment
                        .text_layout()
                        .expect("text fragments carry a private layout ref")
                        .clone(),
                    clip: fragment.clip(),
                    color: *color,
                },
                PaintFragmentKind::ClipPush { .. } => DrawItemKind::ClipPush,
                PaintFragmentKind::ClipPop => DrawItemKind::ClipPop,
                PaintFragmentKind::Unsupported { capability } => {
                    DrawItemKind::Unsupported { capability }
                }
            };
            DrawItem::new(
                fragment.order(),
                fragment.node_id().raw(),
                fragment.rect(),
                kind,
            )
        })
        .collect::<Vec<_>>();
    draw_items.sort_by_key(DrawItem::order);

    let draw_orders_by_node = draw_orders_by_node(&draw_items);
    let upload_intents = scene
        .resource_demands()
        .iter()
        .map(|demand| {
            let dependent_draw_orders = draw_orders_by_node
                .get(&demand.owner_node_id())
                .cloned()
                .unwrap_or_default();
            if let Some(glyphs) = demand.glyphs() {
                UploadIntent::glyph(
                    demand.owner_node_id(),
                    demand.expected_generation().clone(),
                    dependent_draw_orders,
                    glyphs.clone(),
                )
            } else {
                UploadIntent::new(
                    demand.kind(),
                    demand.owner_node_id(),
                    demand.expected_generation().clone(),
                    dependent_draw_orders,
                )
            }
        })
        .collect::<Vec<_>>();

    let upload_plan = UploadPlan::new(upload_intents);
    let passes = prepared_passes(&draw_items, upload_plan.intents().len());
    let stats = FrameGraphStats {
        surface_generation: context.surface_generation(),
        pass_count: passes.len(),
        draw_item_count: draw_items.len(),
        upload_intent_count: upload_plan.intents().len(),
        layer_count: 0,
        unsupported_fragment_count: draw_items
            .iter()
            .filter(|item| matches!(item.kind(), DrawItemKind::Unsupported { .. }))
            .count(),
        stale_drop_count: 0,
        duration: started.elapsed(),
    };

    PreparedFrame::new(
        prepared_frame_generation(scene, context.surface_generation()),
        context,
        upload_plan,
        passes,
        draw_items,
        stats,
    )
}

fn prepared_frame_generation(
    scene: &PaintScene,
    surface_generation: Option<u64>,
) -> PreparedFrameGeneration {
    match surface_generation {
        Some(surface_generation) => {
            PreparedFrameGeneration::with_surface(scene.generation(), surface_generation)
        }
        None => PreparedFrameGeneration::new(scene.generation()),
    }
}

fn draw_orders_by_node(draw_items: &[DrawItem]) -> BTreeMap<u64, Vec<SceneOrder>> {
    let mut draw_orders = BTreeMap::<u64, Vec<SceneOrder>>::new();
    for item in draw_items {
        if matches!(
            item.kind(),
            DrawItemKind::Text { .. } | DrawItemKind::Unsupported { .. }
        ) {
            draw_orders
                .entry(item.node_id())
                .or_default()
                .push(item.order());
        }
    }
    draw_orders
}

fn prepared_passes(draw_items: &[DrawItem], upload_count: usize) -> Vec<PreparedPass> {
    let mut passes = Vec::new();
    if upload_count > 0 {
        passes.push(PreparedPass::new(
            RenderPass::Upload,
            Vec::new(),
            upload_count,
        ));
    }
    passes.push(PreparedPass::new(
        RenderPass::MainColor,
        draw_items.iter().map(DrawItem::order).collect(),
        0,
    ));
    passes
}
