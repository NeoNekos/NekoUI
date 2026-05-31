use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use crate::diagnostic::{
    DiagnosticArea, DiagnosticRecord, DiagnosticSeverity, DirtyLane, DirtyLanes,
};
use crate::element::{Element, ElementKind, ElementParts};
use crate::error::ErrorKind;
use crate::interaction::{InteractionHandlers, InteractionTarget, TextRange};
use crate::retained::{
    DirtyCause, IdentitySeed, RetainedDiffStats, RetainedDirty, RetainedIdentity,
    RetainedNodeSnapshot, RetainedTreeGeneration, RetainedTreeSnapshot,
};
use crate::style::{OutputParticipation, ResolvedStyle, StyleNodeSnapshot, StyleTreeSnapshot};
use crate::text::{EditableTextState, TextEditOutcome, TextRangeError};

#[derive(Clone, Debug, PartialEq)]
struct RetainedNode {
    identity: RetainedIdentity,
    kind: ElementKind,
    key: Option<crate::element::ElementKey>,
    style: crate::style::StyleDeclaration,
    focusable: bool,
    handlers: InteractionHandlers,
    resolved_style: ResolvedStyle,
    participation: OutputParticipation,
    text: Option<Cow<'static, str>>,
    editable: Option<EditableTextState>,
    children: Vec<RetainedNode>,
}

impl RetainedNode {
    fn count(&self) -> usize {
        1 + self.children.iter().map(Self::count).sum::<usize>()
    }

    fn snapshot(&self) -> RetainedNodeSnapshot {
        RetainedNodeSnapshot {
            id: self.identity.id(),
            generation: self.identity.generation(),
            kind: self.kind,
            key: self.key.clone(),
            style: self.style.clone(),
            focusable: self.focusable,
            handlers: self.handlers.clone(),
            resolved_style: self.resolved_style.clone(),
            participation: self.participation,
            text: self.text.clone(),
            editable: self.editable.clone(),
            children: self.children.iter().map(Self::snapshot).collect(),
        }
    }

    fn style_snapshot(&self) -> StyleNodeSnapshot {
        StyleNodeSnapshot::new(
            self.identity.id().raw(),
            self.identity.generation().raw(),
            self.resolved_style.clone(),
            self.participation,
            self.children.iter().map(Self::style_snapshot).collect(),
        )
    }

    fn layout_node(&self) -> RetainedLayoutNode<'_> {
        RetainedLayoutNode { node: self }
    }
}

#[derive(Debug, Default)]
pub(crate) struct RetainedTree {
    seed: IdentitySeed,
    root: Option<RetainedNode>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RetainedTreeDiff {
    pub stats: RetainedDiffStats,
    pub dirty: Vec<RetainedDirty>,
    pub duration: Duration,
    pub style_duration: Duration,
    pub records: Vec<DiagnosticRecord>,
}

impl RetainedTree {
    pub(crate) fn diff_root(&mut self, root: Element) -> RetainedTreeDiff {
        let started = Instant::now();
        let mut context = DiffContext::new();
        let old_root = self.root.take();
        context.stats.old_node_count = old_root.as_ref().map_or(0, RetainedNode::count);
        let new_root = self.diff_node(
            old_root,
            root,
            None,
            OutputParticipation::included(),
            &mut context,
        );
        context.stats.new_node_count = new_root.count();
        self.root = Some(new_root);

        let structural_change = context.stats.created > 0
            || context.stats.replaced > 0
            || context.stats.destroyed > 0
            || context.stats.moved_nodes > 0
            || context.stats.old_node_count != context.stats.new_node_count;
        if structural_change {
            self.seed.mark_tree_changed();
        }

        RetainedTreeDiff {
            stats: context.stats,
            dirty: context.dirty,
            duration: started.elapsed(),
            style_duration: context.style_duration,
            records: context.records,
        }
    }

    pub(crate) fn snapshot(&self) -> RetainedTreeSnapshot {
        RetainedTreeSnapshot::new(
            self.root.as_ref().map(|_| self.seed.tree_generation()),
            self.root.as_ref().map_or(0, RetainedNode::count),
            self.root.as_ref().map(RetainedNode::snapshot),
        )
    }

    pub(crate) fn style_snapshot(&self) -> StyleTreeSnapshot {
        StyleTreeSnapshot::new(
            self.root.as_ref().map_or(0, RetainedNode::count),
            self.root.as_ref().map(RetainedNode::style_snapshot),
        )
    }

    pub(crate) fn layout_input(&self) -> RetainedLayoutInput<'_> {
        RetainedLayoutInput { tree: self }
    }

    pub(crate) fn insert_text_at_target(
        &mut self,
        target: InteractionTarget,
        text: &str,
        replace: Option<TextRange>,
    ) -> Result<Option<TextEditOutcome>, TextRangeError> {
        let Some(root) = self.root.as_mut() else {
            return Ok(None);
        };
        let Some(node) = find_node_by_target_mut(root, target) else {
            return Ok(None);
        };
        let Some(editable) = node.editable.as_mut() else {
            return Ok(None);
        };
        editable.block_mut().insert_text(text, replace).map(Some)
    }

    pub(crate) fn delete_backward_at_target(
        &mut self,
        target: InteractionTarget,
    ) -> Result<Option<TextEditOutcome>, TextRangeError> {
        let Some(root) = self.root.as_mut() else {
            return Ok(None);
        };
        let Some(node) = find_node_by_target_mut(root, target) else {
            return Ok(None);
        };
        let Some(editable) = node.editable.as_mut() else {
            return Ok(None);
        };
        editable.block_mut().delete_backward().map(Some)
    }

    pub(crate) fn set_composition_at_target(
        &mut self,
        target: InteractionTarget,
        text: &str,
        cursor: Option<TextRange>,
        replace: Option<TextRange>,
    ) -> Result<Option<TextEditOutcome>, TextRangeError> {
        let Some(root) = self.root.as_mut() else {
            return Ok(None);
        };
        let Some(node) = find_node_by_target_mut(root, target) else {
            return Ok(None);
        };
        let Some(editable) = node.editable.as_mut() else {
            return Ok(None);
        };
        editable
            .block_mut()
            .set_composition(text, cursor, replace)
            .map(Some)
    }

    pub(crate) fn clear_composition_at_target(
        &mut self,
        target: InteractionTarget,
    ) -> Option<TextEditOutcome> {
        let root = self.root.as_mut()?;
        let node = find_node_by_target_mut(root, target)?;
        let editable = node.editable.as_mut()?;
        Some(editable.block_mut().clear_composition())
    }

    fn diff_node(
        &mut self,
        old: Option<RetainedNode>,
        element: Element,
        parent_style: Option<&ResolvedStyle>,
        parent_participation: OutputParticipation,
        context: &mut DiffContext,
    ) -> RetainedNode {
        let parts = element.into_parts();
        match old {
            Some(old_node) if old_node.key != parts.key => {
                context.stats.replaced += 1;
                context.destroy_subtree(&old_node, DirtyCause::NodeReplaced);
                self.create_node_from_parts(parts, parent_style, parent_participation, context)
            }
            Some(mut old_node) if old_node.kind == parts.kind => {
                let identity = old_node.identity;
                context.stats.preserved += 1;
                let style_started = Instant::now();
                let resolved_style = ResolvedStyle::resolve(&parts.style, parent_style);
                let participation = OutputParticipation::resolve(
                    parent_participation,
                    resolved_style.layout().display(),
                );
                context.style_duration += style_started.elapsed();
                let style_lanes = resolved_style.dirty_lanes_since(&old_node.resolved_style);
                if !style_lanes.is_empty() {
                    context.emit_dirty(Some(identity), DirtyCause::StyleChanged, style_lanes);
                }
                if participation != old_node.participation {
                    context.emit_dirty(
                        Some(identity),
                        DirtyCause::StyleChanged,
                        participation_lanes(),
                    );
                }
                let editable = reconcile_editable_state(&old_node, parts.kind, &parts.text);
                let old_display_text = node_display_text(&old_node);
                let new_display_text = editable
                    .as_ref()
                    .map(|editable| editable.block().display_text())
                    .or_else(|| parts.text.as_deref().map(ToOwned::to_owned));
                if old_display_text != new_display_text {
                    let mut text_lanes = DirtyLanes::empty();
                    text_lanes.insert(DirtyLane::Text.flag());
                    text_lanes.insert(DirtyLane::Layout.flag());
                    text_lanes.insert(DirtyLane::Semantics.flag());
                    text_lanes.insert(DirtyLane::Paint.flag());
                    context.emit_dirty(Some(identity), DirtyCause::TextChanged, text_lanes);
                }
                if old_node.handlers != parts.handlers {
                    context.emit_dirty(
                        Some(identity),
                        DirtyCause::StyleChanged,
                        interaction_lanes(),
                    );
                }
                if old_node.focusable != parts.focusable {
                    context.emit_dirty(
                        Some(identity),
                        DirtyCause::RetainedChanged,
                        focusable_lanes(),
                    );
                }

                let children = self.diff_children(
                    std::mem::take(&mut old_node.children),
                    parts.children,
                    Some(&resolved_style),
                    participation,
                    context,
                );

                RetainedNode {
                    identity,
                    kind: parts.kind,
                    key: parts.key,
                    style: parts.style,
                    focusable: parts.focusable,
                    handlers: parts.handlers,
                    resolved_style,
                    participation,
                    text: parts.text,
                    editable,
                    children,
                }
            }
            Some(old_node) => {
                context.stats.kind_mismatches += 1;
                context.stats.replaced += 1;
                context.records.push(
                    DiagnosticRecord::new(
                        DiagnosticArea::Retained,
                        DiagnosticSeverity::Warning,
                        ErrorKind::InvalidInput,
                        "retained.kind_mismatch",
                        "same identity matched an incompatible element kind",
                    )
                    .with_field("old_kind", old_node.kind.name())
                    .with_field("new_kind", parts.kind.name()),
                );
                context.destroy_subtree(&old_node, DirtyCause::NodeReplaced);
                self.create_node_from_parts(parts, parent_style, parent_participation, context)
            }
            None => self.create_node_from_parts(parts, parent_style, parent_participation, context),
        }
    }

    fn create_node_from_parts(
        &mut self,
        parts: ElementParts,
        parent_style: Option<&ResolvedStyle>,
        parent_participation: OutputParticipation,
        context: &mut DiffContext,
    ) -> RetainedNode {
        let identity = self.seed.allocate();
        context.stats.created += 1;
        let mut lanes = DirtyLanes::empty();
        lanes.insert(DirtyLane::Build.flag());
        lanes.insert(DirtyLane::Style.flag());
        lanes.insert(DirtyLane::Layout.flag());
        lanes.insert(DirtyLane::Semantics.flag());
        lanes.insert(DirtyLane::Paint.flag());
        if text_capable_kind(parts.kind) {
            lanes.insert(DirtyLane::Text.flag());
        }
        context.emit_dirty(Some(identity), DirtyCause::NodeCreated, lanes);

        let style_started = Instant::now();
        let resolved_style = ResolvedStyle::resolve(&parts.style, parent_style);
        let participation =
            OutputParticipation::resolve(parent_participation, resolved_style.layout().display());
        context.style_duration += style_started.elapsed();
        let children = self.diff_children(
            Vec::new(),
            parts.children,
            Some(&resolved_style),
            participation,
            context,
        );

        RetainedNode {
            identity,
            kind: parts.kind,
            key: parts.key,
            style: parts.style,
            focusable: parts.focusable,
            handlers: parts.handlers,
            resolved_style,
            participation,
            text: parts.text.clone(),
            editable: initial_editable_state(parts.kind, parts.text.as_deref()),
            children,
        }
    }

    fn diff_children(
        &mut self,
        old_children: Vec<RetainedNode>,
        new_children: Vec<Element>,
        parent_style: Option<&ResolvedStyle>,
        parent_participation: OutputParticipation,
        context: &mut DiffContext,
    ) -> Vec<RetainedNode> {
        let mut old_by_key: BTreeMap<crate::element::ElementKey, VecDeque<(usize, RetainedNode)>> =
            BTreeMap::new();
        let mut old_unkeyed = VecDeque::new();
        for (index, child) in old_children.into_iter().enumerate() {
            if let Some(key) = child.key.clone() {
                old_by_key.entry(key).or_default().push_back((index, child));
            } else {
                old_unkeyed.push_back((index, child));
            }
        }

        let mut seen_keys = BTreeSet::new();
        let mut result = Vec::with_capacity(new_children.len());

        for (new_index, element) in new_children.into_iter().enumerate() {
            let key = element.key().cloned();
            let duplicate_key = key
                .as_ref()
                .is_some_and(|key| !seen_keys.insert(key.clone()));
            let old = if let Some(key) = key.as_ref() {
                if duplicate_key {
                    context.stats.duplicate_keys += 1;
                    context.records.push(
                        DiagnosticRecord::new(
                            DiagnosticArea::Retained,
                            DiagnosticSeverity::Warning,
                            ErrorKind::InvalidInput,
                            "retained.duplicate_key",
                            "duplicate sibling element key",
                        )
                        .with_field("key", key.as_str().to_owned())
                        .with_field("duplicate_index", new_index.to_string()),
                    );
                    None
                } else {
                    old_by_key.get_mut(key).and_then(VecDeque::pop_front)
                }
            } else {
                pop_compatible_unkeyed(&mut old_unkeyed, element.kind(), new_index, context)
            };

            let moved_identity = old.as_ref().and_then(|(old_index, child)| {
                if *old_index != new_index && child.kind == element.kind() {
                    Some(child.identity)
                } else {
                    None
                }
            });
            if let Some(identity) = moved_identity {
                context.stats.moved_nodes += 1;
                context.emit_dirty(Some(identity), DirtyCause::NodeMoved, structural_lanes());
            }

            result.push(self.diff_node(
                old.map(|(_, child)| child),
                element,
                parent_style,
                parent_participation,
                context,
            ));
        }

        for keyed_children in old_by_key.into_values() {
            for (_, child) in keyed_children {
                context.destroy_subtree(&child, DirtyCause::NodeDestroyed);
            }
        }
        for (_, child) in old_unkeyed {
            context.destroy_subtree(&child, DirtyCause::NodeDestroyed);
        }

        result
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RetainedLayoutInput<'a> {
    tree: &'a RetainedTree,
}

impl<'a> RetainedLayoutInput<'a> {
    pub(crate) fn generation(self) -> Option<RetainedTreeGeneration> {
        self.tree
            .root
            .as_ref()
            .map(|_| self.tree.seed.tree_generation())
    }

    pub(crate) fn node_count(self) -> usize {
        self.tree.root.as_ref().map_or(0, RetainedNode::count)
    }

    pub(crate) fn root(self) -> Option<RetainedLayoutNode<'a>> {
        self.tree.root.as_ref().map(RetainedNode::layout_node)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RetainedLayoutNode<'a> {
    node: &'a RetainedNode,
}

impl<'a> RetainedLayoutNode<'a> {
    pub(crate) fn identity(self) -> RetainedIdentity {
        self.node.identity
    }

    pub(crate) fn kind(self) -> ElementKind {
        self.node.kind
    }

    pub(crate) fn key(self) -> Option<&'a crate::element::ElementKey> {
        self.node.key.as_ref()
    }

    pub(crate) fn resolved_style(self) -> &'a ResolvedStyle {
        &self.node.resolved_style
    }

    pub(crate) fn participation(self) -> OutputParticipation {
        self.node.participation
    }

    pub(crate) fn display_text(self) -> Option<String> {
        self.node
            .editable
            .as_ref()
            .map(|editable| editable.block().display_text())
            .or_else(|| self.node.text.as_deref().map(ToOwned::to_owned))
    }

    pub(crate) fn text_generation(self) -> crate::text::TextGeneration {
        self.node
            .editable
            .as_ref()
            .map_or(crate::text::TextGeneration::INITIAL, |editable| {
                editable.block().generation()
            })
    }

    pub(crate) fn children_len(self) -> usize {
        self.node.children.len()
    }

    pub(crate) fn children(self) -> impl Iterator<Item = RetainedLayoutNode<'a>> {
        self.node.children.iter().map(RetainedNode::layout_node)
    }
}

fn text_capable_kind(kind: ElementKind) -> bool {
    matches!(kind, ElementKind::Text | ElementKind::Input)
}

fn editable_kind(kind: ElementKind) -> bool {
    matches!(kind, ElementKind::Input)
}

fn initial_editable_state(kind: ElementKind, text: Option<&str>) -> Option<EditableTextState> {
    editable_kind(kind).then(|| EditableTextState::new(text.unwrap_or_default()))
}

fn reconcile_editable_state(
    old_node: &RetainedNode,
    new_kind: ElementKind,
    declared_text: &Option<Cow<'static, str>>,
) -> Option<EditableTextState> {
    if !editable_kind(new_kind) {
        return None;
    }
    old_node.editable.clone().or_else(|| {
        Some(EditableTextState::new(
            declared_text.as_deref().unwrap_or_default(),
        ))
    })
}

fn node_display_text(node: &RetainedNode) -> Option<String> {
    node.editable
        .as_ref()
        .map(|editable| editable.block().display_text())
        .or_else(|| node.text.as_deref().map(ToOwned::to_owned))
}

fn find_node_by_target_mut(
    node: &mut RetainedNode,
    target: InteractionTarget,
) -> Option<&mut RetainedNode> {
    if node.identity.id() == target.node_id()
        && node.identity.generation() == target.node_generation()
    {
        return Some(node);
    }
    node.children
        .iter_mut()
        .find_map(|child| find_node_by_target_mut(child, target))
}

fn pop_compatible_unkeyed(
    old_unkeyed: &mut VecDeque<(usize, RetainedNode)>,
    kind: ElementKind,
    new_index: usize,
    context: &mut DiffContext,
) -> Option<(usize, RetainedNode)> {
    if old_unkeyed
        .front()
        .is_some_and(|(_, child)| child.kind == kind)
    {
        let (old_index, child) = old_unkeyed.pop_front().unwrap();
        if old_index != new_index {
            context.stats.moved_nodes += 1;
            context.emit_dirty(
                Some(child.identity),
                DirtyCause::NodeMoved,
                structural_lanes(),
            );
        }
        Some((old_index, child))
    } else {
        None
    }
}

fn structural_lanes() -> DirtyLanes {
    let mut lanes = DirtyLanes::empty();
    lanes.insert(DirtyLane::Build.flag());
    lanes.insert(DirtyLane::Style.flag());
    lanes.insert(DirtyLane::Layout.flag());
    lanes.insert(DirtyLane::Semantics.flag());
    lanes.insert(DirtyLane::Paint.flag());
    lanes
}

fn participation_lanes() -> DirtyLanes {
    let mut lanes = DirtyLanes::empty();
    lanes.insert(DirtyLane::Layout.flag());
    lanes.insert(DirtyLane::Semantics.flag());
    lanes.insert(DirtyLane::Paint.flag());
    lanes
}

fn interaction_lanes() -> DirtyLanes {
    let mut lanes = DirtyLanes::empty();
    lanes.insert(DirtyLane::Semantics.flag());
    lanes
}

fn focusable_lanes() -> DirtyLanes {
    let mut lanes = DirtyLanes::empty();
    lanes.insert(DirtyLane::Semantics.flag());
    lanes.insert(DirtyLane::Paint.flag());
    lanes
}

struct DiffContext {
    stats: RetainedDiffStats,
    dirty: Vec<RetainedDirty>,
    style_duration: Duration,
    records: Vec<DiagnosticRecord>,
}

impl DiffContext {
    fn new() -> Self {
        Self {
            stats: RetainedDiffStats::default(),
            dirty: Vec::new(),
            style_duration: Duration::ZERO,
            records: Vec::new(),
        }
    }

    fn emit_dirty(
        &mut self,
        identity: Option<RetainedIdentity>,
        cause: DirtyCause,
        lanes: DirtyLanes,
    ) {
        self.dirty.push(RetainedDirty::new(identity, cause, lanes));
    }

    fn destroy_subtree(&mut self, node: &RetainedNode, cause: DirtyCause) {
        self.stats.destroyed += 1;
        self.emit_dirty(Some(node.identity), cause, structural_lanes());
        for child in &node.children {
            self.destroy_subtree(child, DirtyCause::NodeDestroyed);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::{DiagnosticArea, DirtyLane};
    use crate::element::{ElementKind, IntoElement, div, text};
    use crate::retained::RetainedTree;
    use crate::style::{Color, Display, StyleExt, fill, px};

    #[test]
    fn keyed_compatible_node_preserves_identity() {
        let mut tree = RetainedTree::default();
        tree.diff_root(div().child(text("a").key("item")).into_element());
        let first = tree.snapshot().find_by_key("item").unwrap().id();

        let diff = tree.diff_root(div().child(text("b").key("item")).into_element());
        let second = tree.snapshot().find_by_key("item").unwrap().id();

        assert_eq!(first, second);
        assert_eq!(diff.stats.preserved, 2);
    }

    #[test]
    fn same_key_incompatible_kind_replaces_and_diagnoses() {
        let mut tree = RetainedTree::default();
        tree.diff_root(div().child(text("a").key("item")).into_element());
        let first = tree.snapshot().find_by_key("item").unwrap().id();

        let diff = tree.diff_root(div().child(div().key("item")).into_element());
        let second = tree.snapshot().find_by_key("item").unwrap().id();

        assert_ne!(first, second);
        assert_eq!(diff.stats.kind_mismatches, 1);
        assert!(
            diff.records
                .iter()
                .any(|record| record.operation == "retained.kind_mismatch")
        );
    }

    #[test]
    fn duplicate_keys_preserve_first_and_create_later_duplicate() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .child(text("a").key("dup"))
                .child(text("b").key("other"))
                .into_element(),
        );
        let first = tree.snapshot().find_by_key("dup").unwrap().id();

        let diff = tree.diff_root(
            div()
                .child(text("a").key("dup"))
                .child(text("b").key("dup"))
                .into_element(),
        );
        let snapshot = tree.snapshot();
        let root = snapshot.root().unwrap();

        assert_eq!(root.children()[0].id(), first);
        assert_ne!(root.children()[1].id(), first);
        assert_eq!(diff.stats.duplicate_keys, 1);
        assert!(
            diff.records
                .iter()
                .any(|record| record.area == DiagnosticArea::Retained
                    && record.operation == "retained.duplicate_key")
        );
    }

    #[test]
    fn old_duplicate_key_remainders_are_destroyed_deterministically() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .child(text("a").key("dup"))
                .child(text("b").key("dup"))
                .into_element(),
        );

        let diff = tree.diff_root(div().child(text("a").key("dup")).into_element());
        let snapshot = tree.snapshot();
        let root = snapshot.root().unwrap();

        assert_eq!(root.children().len(), 1);
        assert_eq!(diff.stats.destroyed, 1);
    }

    #[test]
    fn nested_subtree_deletion_counts_each_removed_node_once() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(
                    div()
                        .key("removed")
                        .child(text("a").key("a"))
                        .child(text("b").key("b")),
                )
                .child(text("kept").key("kept"))
                .into_element(),
        );

        let diff = tree.diff_root(
            div()
                .key("root")
                .child(text("kept").key("kept"))
                .into_element(),
        );

        assert_eq!(diff.stats.destroyed, 3);
        assert_eq!(tree.snapshot().node_count(), 2);
    }

    #[test]
    fn nested_kind_replacement_counts_old_descendants_once() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(
                    div()
                        .key("replace")
                        .child(text("a").key("a"))
                        .child(text("b").key("b")),
                )
                .into_element(),
        );

        let diff = tree.diff_root(
            div()
                .key("root")
                .child(text("replacement").key("replace"))
                .into_element(),
        );

        assert_eq!(diff.stats.replaced, 1);
        assert_eq!(diff.stats.destroyed, 3);
        assert_eq!(diff.stats.created, 1);
    }

    #[test]
    fn keyed_move_with_kind_replacement_does_not_count_as_moved() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(text("stable").key("stable"))
                .child(text("replace").key("replace"))
                .into_element(),
        );

        let diff = tree.diff_root(
            div()
                .key("root")
                .child(div().key("replace"))
                .child(text("stable").key("stable"))
                .into_element(),
        );

        assert_eq!(diff.stats.kind_mismatches, 1);
        assert_eq!(diff.stats.replaced, 1);
        assert_eq!(diff.stats.moved_nodes, 1);
    }

    #[test]
    fn unkeyed_same_position_compatible_kind_preserves_identity() {
        let mut tree = RetainedTree::default();
        tree.diff_root(div().child(text("a")).into_element());
        let first = tree.snapshot().root().unwrap().children()[0].id();

        tree.diff_root(div().child(text("b")).into_element());
        let second = tree.snapshot().root().unwrap().children()[0].id();

        assert_eq!(first, second);
    }

    #[test]
    fn retained_snapshot_is_read_only_and_contains_payloads() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(text("Hello").font_size(px(18.0)).key("label"))
                .into_element(),
        );
        let snapshot = tree.snapshot();
        let label = snapshot.find_by_key("label").unwrap();

        assert_eq!(snapshot.node_count(), 2);
        assert_eq!(label.kind(), ElementKind::Text);
        assert_eq!(label.text(), Some("Hello"));
        assert_eq!(label.resolved_style().text().font_size(), px(18.0));
    }

    #[test]
    fn dirty_lanes_are_not_over_broad_for_style_changes() {
        let mut tree = RetainedTree::default();
        tree.diff_root(div().key("root").into_element());

        let visual = tree.diff_root(
            div()
                .key("root")
                .background(Color::rgb(1, 2, 3))
                .into_element(),
        );
        let visual_lanes = visual.dirty.iter().fold(
            crate::diagnostic::DirtyLanes::empty(),
            |mut lanes, dirty| {
                lanes.insert(dirty.lanes);
                lanes
            },
        );
        assert!(visual_lanes.contains(DirtyLane::Paint.flag()));
        assert!(!visual_lanes.contains(DirtyLane::Layout.flag()));
        assert!(!visual_lanes.contains(DirtyLane::Text.flag()));

        let layout = tree.diff_root(div().key("root").width(fill()).into_element());
        let layout_lanes = layout.dirty.iter().fold(
            crate::diagnostic::DirtyLanes::empty(),
            |mut lanes, dirty| {
                lanes.insert(dirty.lanes);
                lanes
            },
        );
        assert!(layout_lanes.contains(DirtyLane::Layout.flag()));

        let text = tree.diff_root(
            div()
                .key("root")
                .child(text("Hello").font_size(px(20.0)))
                .into_element(),
        );
        let text_lanes = text.dirty.iter().fold(
            crate::diagnostic::DirtyLanes::empty(),
            |mut lanes, dirty| {
                lanes.insert(dirty.lanes);
                lanes
            },
        );
        assert!(text_lanes.contains(DirtyLane::Text.flag()));
        assert!(text_lanes.contains(DirtyLane::Layout.flag()));
        assert!(text_lanes.contains(DirtyLane::Semantics.flag()));
    }

    #[test]
    fn text_payload_changes_emit_semantic_dirty_lanes() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(text("Hello").key("label"))
                .into_element(),
        );

        let diff = tree.diff_root(
            div()
                .key("root")
                .child(text("Goodbye").key("label"))
                .into_element(),
        );
        let lanes = diff.dirty.iter().fold(
            crate::diagnostic::DirtyLanes::empty(),
            |mut lanes, dirty| {
                lanes.insert(dirty.lanes);
                lanes
            },
        );

        assert!(lanes.contains(DirtyLane::Text.flag()));
        assert!(lanes.contains(DirtyLane::Layout.flag()));
        assert!(lanes.contains(DirtyLane::Semantics.flag()));
        assert!(lanes.contains(DirtyLane::Paint.flag()));
    }

    #[test]
    fn display_none_preserves_identity_and_suppresses_output_participation() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(div().key("hidden").child(text("nested").key("nested")))
                .into_element(),
        );
        let first_hidden = tree.snapshot().find_by_key("hidden").unwrap().id();

        let diff = tree.diff_root(
            div()
                .key("root")
                .child(
                    div()
                        .key("hidden")
                        .display(Display::None)
                        .child(text("nested").key("nested")),
                )
                .into_element(),
        );
        let snapshot = tree.snapshot();
        let hidden = snapshot.find_by_key("hidden").unwrap();
        let nested = snapshot.find_by_key("nested").unwrap();
        let lanes = diff.dirty.iter().fold(
            crate::diagnostic::DirtyLanes::empty(),
            |mut lanes, dirty| {
                lanes.insert(dirty.lanes);
                lanes
            },
        );

        assert_eq!(hidden.id(), first_hidden);
        assert!(!hidden.participation().layout());
        assert!(!hidden.participation().paint());
        assert!(!hidden.participation().hit_test());
        assert!(!hidden.participation().semantics());
        assert!(!nested.participation().layout());
        assert!(!nested.participation().semantics());
        assert!(lanes.contains(DirtyLane::Layout.flag()));
        assert!(lanes.contains(DirtyLane::Semantics.flag()));
        assert!(lanes.contains(DirtyLane::Paint.flag()));
    }
}
