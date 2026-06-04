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
    let mut draw_items = Vec::with_capacity(scene.fragments().len());
    let mut box_shape_count = 0;
    let mut unsupported_fragment_count = 0;
    for fragment in scene.fragments() {
        let kind = match fragment.kind() {
            PaintFragmentKind::BoxShape { shape } => {
                box_shape_count += 1;
                DrawItemKind::BoxShape { shape: *shape }
            }
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
            PaintFragmentKind::ClipPush { clip } => DrawItemKind::ClipPush { clip: *clip },
            PaintFragmentKind::ClipPop => DrawItemKind::ClipPop,
            PaintFragmentKind::Unsupported { capability } => {
                unsupported_fragment_count += 1;
                DrawItemKind::Unsupported { capability }
            }
        };
        draw_items.push(DrawItem::new(
            fragment.order(),
            fragment.node_id().raw(),
            fragment.rect(),
            kind,
        ));
    }
    draw_items.sort_by_key(DrawItem::order);

    let draw_orders_by_node = draw_orders_by_node(&draw_items);
    let mut upload_intents = Vec::with_capacity(scene.resource_demands().len());
    for demand in scene.resource_demands() {
        let dependent_draw_orders =
            dependent_draw_orders(&draw_orders_by_node, demand.owner_node_id());
        if let Some(glyphs) = demand.glyphs() {
            upload_intents.push(UploadIntent::glyph(
                demand.owner_node_id(),
                demand.expected_generation().clone(),
                dependent_draw_orders,
                glyphs.clone(),
            ));
        } else {
            upload_intents.push(UploadIntent::new(
                demand.kind(),
                demand.owner_node_id(),
                demand.expected_generation().clone(),
                dependent_draw_orders,
            ));
        }
    }

    let upload_plan = UploadPlan::new(upload_intents);
    let passes = prepared_passes(&draw_items, upload_plan.intents().len());
    let stats = FrameGraphStats {
        surface_generation: context.surface_generation(),
        pass_count: passes.len(),
        draw_item_count: draw_items.len(),
        upload_intent_count: upload_plan.intents().len(),
        layer_count: 0,
        box_shape_count,
        unsupported_fragment_count,
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

struct NodeDrawOrders {
    node_id: u64,
    orders: Vec<SceneOrder>,
}

fn draw_orders_by_node(draw_items: &[DrawItem]) -> Vec<NodeDrawOrders> {
    let mut draw_orders = Vec::<NodeDrawOrders>::new();
    for item in draw_items {
        if matches!(
            item.kind(),
            DrawItemKind::Text { .. } | DrawItemKind::Unsupported { .. }
        ) {
            match draw_orders.binary_search_by_key(&item.node_id(), |entry| entry.node_id) {
                Ok(index) => draw_orders[index].orders.push(item.order()),
                Err(index) => draw_orders.insert(
                    index,
                    NodeDrawOrders {
                        node_id: item.node_id(),
                        orders: vec![item.order()],
                    },
                ),
            }
        }
    }
    draw_orders
}

fn dependent_draw_orders(draw_orders_by_node: &[NodeDrawOrders], node_id: u64) -> Vec<SceneOrder> {
    draw_orders_by_node
        .binary_search_by_key(&node_id, |entry| entry.node_id)
        .map(|index| draw_orders_by_node[index].orders.clone())
        .unwrap_or_default()
}

fn prepared_passes(draw_items: &[DrawItem], upload_count: usize) -> Vec<PreparedPass> {
    let mut passes = Vec::with_capacity(if upload_count > 0 { 2 } else { 1 });
    if upload_count > 0 {
        passes.push(PreparedPass::new(
            RenderPass::Upload,
            Vec::new(),
            upload_count,
        ));
    }
    let mut draw_orders = Vec::with_capacity(draw_items.len());
    draw_orders.extend(draw_items.iter().map(DrawItem::order));
    passes.push(PreparedPass::new(RenderPass::MainColor, draw_orders, 0));
    passes
}
