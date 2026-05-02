use smallvec::SmallVec;
use taffy::prelude::NodeId as TaffyNodeId;

use crate::element::{SpecArena, SpecKind, SpecNode, SpecNodeId, SpecPayload};
use crate::style::{
    StyleChange, diff_div_style, diff_text_style, div_style_to_taffy, text_style_to_taffy,
};

use super::dirty::DirtyLaneMask;
use super::retained::{
    MeasureContext, NodeClass, NodeId, NodeKind, RetainedTree, TextMeasureContext,
};

pub(super) fn update_from_spec(
    tree: &mut RetainedTree,
    arena: &SpecArena,
    root: SpecNodeId,
) -> DirtyLaneMask {
    clear_dirty_marks(tree);

    if !can_reuse_node(tree, tree.root, arena.node(root)) {
        *tree = RetainedTree::from_spec(arena, root);
        return DirtyLaneMask::BUILD.normalized();
    }

    diff_node(tree, tree.root, arena, root).normalized()
}

fn clear_dirty_marks(tree: &mut RetainedTree) {
    for (_, node) in &mut tree.nodes {
        node.dirty = DirtyLaneMask::empty();
    }
}

fn diff_node(
    tree: &mut RetainedTree,
    node_id: NodeId,
    arena: &SpecArena,
    spec_id: SpecNodeId,
) -> DirtyLaneMask {
    match arena.node(spec_id).kind {
        SpecKind::Div => diff_div(tree, node_id, arena, spec_id),
        SpecKind::Text => diff_text(tree, node_id, arena, spec_id),
    }
}

fn diff_div(
    tree: &mut RetainedTree,
    node_id: NodeId,
    arena: &SpecArena,
    spec_id: SpecNodeId,
) -> DirtyLaneMask {
    let spec = arena.node(spec_id);
    let style_change = diff_div_style(&tree.nodes[node_id].style, &spec.style);
    let mut dirty = dirty_from_style_change(style_change);
    if tree.nodes[node_id].style != spec.style {
        tree.nodes[node_id].style = spec.style.clone();
        tree.taffy
            .set_style(
                tree.nodes[node_id].taffy_node,
                div_style_to_taffy(&spec.style),
            )
            .expect("div style patch must succeed");
    }
    tree.nodes[node_id].key = spec.key;
    tree.nodes[node_id].owner_view_id = spec.owner_view_id;
    tree.nodes[node_id].window_frame_area = spec.window_frame_area;
    let previous_input_id = tree.nodes[node_id].input_id;
    tree.nodes[node_id].interaction = spec.interaction.clone();
    tree.nodes[node_id].input_id = if tree.nodes[node_id].interaction.text_input.is_some() {
        previous_input_id.or(Some(crate::input::InputNodeId::new()))
    } else {
        None
    };
    tree.nodes[node_id].semantics = spec.semantics.clone();

    let child_dirty = sync_children(tree, node_id, arena, arena.child_ids(spec_id).as_slice());
    dirty |= child_dirty;
    tree.nodes[node_id].dirty |= dirty;
    dirty
}

fn diff_text(
    tree: &mut RetainedTree,
    node_id: NodeId,
    arena: &SpecArena,
    spec_id: SpecNodeId,
) -> DirtyLaneMask {
    let spec = arena.node(spec_id);
    let text_content = match &spec.payload {
        SpecPayload::Text(content) => content,
        SpecPayload::None => unreachable!("text diff requires text payload"),
    };
    let style_change = diff_text_style(&tree.nodes[node_id].style, &spec.style);
    let mut dirty = dirty_from_style_change(style_change);
    let content_changed = match &tree.nodes[node_id].kind {
        NodeKind::Text { content, .. } => content.as_ref() != text_content.as_ref(),
        NodeKind::Div => unreachable!("text diff called for non-text node"),
    };

    if content_changed {
        dirty |= DirtyLaneMask::LAYOUT | DirtyLaneMask::PAINT;
    }

    if tree.nodes[node_id].style != spec.style {
        tree.nodes[node_id].style = spec.style.clone();
        tree.taffy
            .set_style(
                tree.nodes[node_id].taffy_node,
                text_style_to_taffy(&spec.style),
            )
            .expect("text style patch must succeed");
    }

    if let NodeKind::Text {
        content,
        block,
        layout,
    } = &mut tree.nodes[node_id].kind
    {
        *content = text_content.clone();
        if style_change.layout || style_change.text_shape || content_changed {
            **block = None;
            *layout = None;
        }
    }
    tree.nodes[node_id].key = spec.key;
    tree.nodes[node_id].owner_view_id = spec.owner_view_id;
    tree.nodes[node_id].window_frame_area = spec.window_frame_area;
    let previous_input_id = tree.nodes[node_id].input_id;
    tree.nodes[node_id].interaction = spec.interaction.clone();
    tree.nodes[node_id].input_id = if tree.nodes[node_id].interaction.text_input.is_some() {
        previous_input_id.or(Some(crate::input::InputNodeId::new()))
    } else {
        None
    };
    tree.nodes[node_id].semantics = spec.semantics.clone();

    if style_change.layout || style_change.text_shape || content_changed {
        tree.taffy
            .set_node_context(
                tree.nodes[node_id].taffy_node,
                Some(MeasureContext::Text(TextMeasureContext::from_spec(spec))),
            )
            .expect("text node context patch must succeed");
    }

    tree.nodes[node_id].dirty |= dirty;
    dirty
}

fn sync_children(
    tree: &mut RetainedTree,
    parent_id: NodeId,
    arena: &SpecArena,
    new_children: &[SpecNodeId],
) -> DirtyLaneMask {
    let old_children = tree.nodes[parent_id].children.clone();
    let keyed_path = uses_keyed_path(tree, arena, &old_children, new_children);
    let mut next_children = SmallVec::<[NodeId; 4]>::with_capacity(new_children.len());
    let mut dirty = DirtyLaneMask::empty();
    let mut rebuild_required = false;
    let mut child_list_changed = old_children.len() != new_children.len();
    let mut reused = rustc_hash::FxHashSet::<NodeId>::default();
    let mut keyed_old = rustc_hash::FxHashMap::<(u64, NodeClass), NodeId>::default();

    if keyed_path {
        for child_id in &old_children {
            if let Some(key) = tree.nodes[*child_id].key {
                keyed_old.insert((key, node_class(tree, *child_id)), *child_id);
            }
        }
    }

    for (index, spec_id) in new_children.iter().copied().enumerate() {
        let spec = arena.node(spec_id);
        let positional = old_children
            .get(index)
            .copied()
            .filter(|child_id| !reused.contains(child_id));
        let reused_child = if keyed_path {
            if let Some(key) = spec.key {
                keyed_old
                    .get(&(key, spec_class(spec)))
                    .copied()
                    .filter(|child_id| !reused.contains(child_id))
                    .filter(|child_id| can_reuse_node(tree, *child_id, spec))
                    .or_else(|| positional.filter(|child_id| can_reuse_node(tree, *child_id, spec)))
            } else {
                positional.filter(|child_id| can_reuse_node(tree, *child_id, spec))
            }
        } else {
            positional.filter(|child_id| can_reuse_node(tree, *child_id, spec))
        };

        let child_id = match reused_child {
            Some(existing_child) => {
                reused.insert(existing_child);
                if old_children.get(index).copied() != Some(existing_child) {
                    child_list_changed = true;
                    dirty |= DirtyLaneMask::LAYOUT;
                }
                dirty |= diff_node(tree, existing_child, arena, spec_id);
                tree.nodes[existing_child].parent = Some(parent_id);
                existing_child
            }
            None => {
                child_list_changed = true;
                rebuild_required = true;
                dirty |= DirtyLaneMask::BUILD;
                super::retained_tree::build_node(tree, arena, spec_id, Some(parent_id))
            }
        };

        next_children.push(child_id);
    }

    for old_child in old_children.iter().copied() {
        if !reused.contains(&old_child) {
            child_list_changed = true;
            rebuild_required = true;
            dirty |= DirtyLaneMask::BUILD;
            super::retained_tree::remove_subtree(tree, old_child);
        }
    }

    if child_list_changed {
        tree.nodes[parent_id].children = next_children.clone();
        let taffy_children = next_children
            .iter()
            .map(|child_id| tree.nodes[*child_id].taffy_node)
            .collect::<SmallVec<[TaffyNodeId; 4]>>();
        tree.taffy
            .set_children(tree.nodes[parent_id].taffy_node, &taffy_children)
            .expect("children patch must succeed");

        if !rebuild_required {
            dirty |= DirtyLaneMask::LAYOUT;
        }
    }

    dirty
}

fn can_reuse_node(tree: &RetainedTree, node_id: NodeId, spec: &SpecNode) -> bool {
    let node = &tree.nodes[node_id];
    let same_kind = matches!(
        (&node.kind, spec.kind),
        (NodeKind::Div, SpecKind::Div) | (NodeKind::Text { .. }, SpecKind::Text)
    );
    if !same_kind {
        return false;
    }

    match (node.key, spec.key) {
        (Some(existing), Some(new)) => existing == new,
        (None, None) => true,
        _ => false,
    }
}

fn uses_keyed_path(
    tree: &RetainedTree,
    arena: &SpecArena,
    old_children: &[NodeId],
    new_children: &[SpecNodeId],
) -> bool {
    old_children
        .iter()
        .any(|child_id| tree.nodes[*child_id].key.is_some())
        || new_children
            .iter()
            .any(|spec_id| arena.node(*spec_id).key.is_some())
}

fn node_class(tree: &RetainedTree, node_id: NodeId) -> NodeClass {
    match tree.nodes[node_id].kind {
        NodeKind::Div => NodeClass::Div,
        NodeKind::Text { .. } => NodeClass::Text,
    }
}

fn spec_class(spec: &SpecNode) -> NodeClass {
    match spec.kind {
        SpecKind::Div => NodeClass::Div,
        SpecKind::Text => NodeClass::Text,
    }
}

fn dirty_from_style_change(change: StyleChange) -> DirtyLaneMask {
    let mut dirty = DirtyLaneMask::empty();
    if change.layout {
        dirty |= DirtyLaneMask::LAYOUT;
    }
    if change.paint {
        dirty |= DirtyLaneMask::PAINT;
    }
    if change.text_shape {
        dirty |= DirtyLaneMask::LAYOUT | DirtyLaneMask::PAINT;
    }
    dirty
}
