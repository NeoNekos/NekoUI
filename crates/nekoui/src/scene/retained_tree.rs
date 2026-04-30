use smallvec::SmallVec;
use taffy::prelude::NodeId as TaffyNodeId;

use crate::SharedString;
use crate::element::{SpecArena, SpecKind, SpecNodeId, SpecPayload, WindowFrameArea};
use crate::style::{ResolvedStyle, ResolvedTextStyle};
use crate::text_system::{SharedTextLayout, TextMeasureKey};

use super::dirty::DirtyLaneMask;
use super::retained::{
    MeasureContext, NodeClass, NodeId, NodeKind, RetainedNode, RetainedTree, TextMeasureContext,
};

pub(super) fn build_node(
    tree: &mut RetainedTree,
    arena: &SpecArena,
    spec_id: SpecNodeId,
    parent: Option<NodeId>,
) -> NodeId {
    match arena.node(spec_id).kind {
        SpecKind::Div => build_div(tree, arena, spec_id, parent),
        SpecKind::Text => build_text(tree, arena, spec_id, parent),
    }
}

fn build_div(
    tree: &mut RetainedTree,
    arena: &SpecArena,
    spec_id: SpecNodeId,
    parent: Option<NodeId>,
) -> NodeId {
    let spec = arena.node(spec_id);
    let child_ids = arena
        .child_ids(spec_id)
        .into_iter()
        .map(|child| build_node(tree, arena, child, None))
        .collect::<SmallVec<[NodeId; 4]>>();
    let child_taffy_nodes = child_ids
        .iter()
        .map(|child_id| tree.nodes[*child_id].taffy_node)
        .collect::<SmallVec<[TaffyNodeId; 4]>>();

    let taffy_node = tree
        .taffy
        .new_with_children(
            crate::style::div_style_to_taffy(&spec.style),
            &child_taffy_nodes,
        )
        .expect("div node creation must succeed");

    let node_id = tree.nodes.insert_with_key(|id| RetainedNode {
        id,
        parent,
        children: child_ids.clone(),
        kind: NodeKind::Div,
        key: spec.key,
        owner_view_id: spec.owner_view_id,
        style: spec.style.clone(),
        window_frame_area: spec.window_frame_area,
        interaction: spec.interaction.clone(),
        semantics: spec.semantics.clone(),
        layout: crate::scene::LayoutBox::default(),
        dirty: DirtyLaneMask::BUILD.normalized(),
        compiled_fragment: None,
        taffy_node,
    });

    for child_id in child_ids {
        tree.nodes[child_id].parent = Some(node_id);
    }

    node_id
}

fn build_text(
    tree: &mut RetainedTree,
    arena: &SpecArena,
    spec_id: SpecNodeId,
    parent: Option<NodeId>,
) -> NodeId {
    let spec = arena.node(spec_id);
    let text_content = match &spec.payload {
        SpecPayload::Text(content) => content,
        SpecPayload::None => unreachable!("text node creation requires text payload"),
    };
    let taffy_node = tree
        .taffy
        .new_leaf_with_context(
            crate::style::text_style_to_taffy(&spec.style),
            MeasureContext::Text(TextMeasureContext::from_spec(spec)),
        )
        .expect("text node creation must succeed");

    tree.nodes.insert_with_key(|id| RetainedNode {
        id,
        parent,
        children: SmallVec::new(),
        kind: NodeKind::Text {
            content: text_content.clone(),
            block: Box::new(None),
            layout: None,
        },
        key: spec.key,
        owner_view_id: spec.owner_view_id,
        style: spec.style.clone(),
        window_frame_area: spec.window_frame_area,
        interaction: spec.interaction.clone(),
        semantics: spec.semantics.clone(),
        layout: crate::scene::LayoutBox::default(),
        dirty: DirtyLaneMask::BUILD.normalized(),
        compiled_fragment: None,
        taffy_node,
    })
}

pub(super) fn remove_subtree(tree: &mut RetainedTree, node_id: NodeId) {
    let children = tree.nodes[node_id].children.clone();
    for child_id in children {
        remove_subtree(tree, child_id);
    }

    let taffy_node = tree.nodes[node_id].taffy_node;
    tree.taffy
        .remove(taffy_node)
        .expect("retained subtree removal must succeed");
    tree.nodes.remove(node_id);
}

#[allow(dead_code)]
fn _keep_types_used(
    _a: SharedString,
    _b: ResolvedStyle,
    _c: ResolvedTextStyle,
    _d: TextMeasureKey,
    _e: Option<SharedTextLayout>,
    _f: Option<WindowFrameArea>,
    _g: NodeClass,
) {
}
