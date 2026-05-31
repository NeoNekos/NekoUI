use std::time::Instant;

use crate::diagnostic::{DiagnosticArea, DiagnosticRecord, DiagnosticSeverity};
use crate::element::ElementKind;
use crate::error::ErrorKind;
use crate::interaction::{InteractionState, InteractionTarget};
use crate::layout::{LayoutNodeSnapshot, LayoutPoint, LayoutRect, LayoutTreeSnapshot};
use crate::retained::{RetainedNodeSnapshot, RetainedTreeSnapshot};
use crate::semantics::generation::SemanticGeneration;
use crate::semantics::generation::{
    SemanticInputSignature, SemanticSignatureFact, element_kind_fact, interaction_signature,
    text_hash,
};
use crate::semantics::snapshot::{
    SemanticAction, SemanticNodeId, SemanticNodeParts, SemanticNodeSnapshot, SemanticRole,
    SemanticStateParts, SemanticStateSnapshot,
};
use crate::semantics::{SemanticBuildStats, SemanticTreeSnapshot};
use crate::style::StyleTreeSnapshot;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SemanticBuildInput<'a> {
    pub retained: &'a RetainedTreeSnapshot,
    pub style: &'a StyleTreeSnapshot,
    pub layout: &'a LayoutTreeSnapshot,
    pub interaction: Option<&'a InteractionState>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SemanticBuildOutput {
    pub snapshot: SemanticTreeSnapshot,
    pub stats: SemanticBuildStats,
    pub records: Vec<DiagnosticRecord>,
}

pub(crate) fn build_semantic_snapshot(input: SemanticBuildInput<'_>) -> SemanticBuildOutput {
    let started = Instant::now();
    let generation = semantic_generation_for_inputs_with_interaction(
        input.retained,
        input.style,
        input.layout,
        input.interaction,
    );
    let mut builder = SemanticBuilder::new(generation.clone(), input.interaction);
    let root = match (input.retained.root(), input.layout.root()) {
        (Some(retained), Some(layout)) => {
            builder.visit(retained, layout, true, VisitContext::new(input.interaction))
        }
        (Some(retained), None) if retained.participation().semantics() => {
            builder.record_missing_bounds(retained);
            None
        }
        _ => None,
    };
    builder.finish(started, root)
}

#[cfg(test)]
pub(crate) fn semantic_generation_for_inputs(
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
) -> SemanticGeneration {
    semantic_generation_for_inputs_with_interaction(retained, style, layout, None)
}

pub(crate) fn semantic_generation_for_inputs_with_interaction(
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
    interaction: Option<&InteractionState>,
) -> SemanticGeneration {
    SemanticGeneration::new(
        retained.generation(),
        layout.generation(),
        layout.viewport().generation().raw(),
        style_signature(style),
        retained_semantic_signature(retained),
        interaction_signature(interaction),
    )
}

#[cfg(test)]
pub(crate) fn semantic_publish_is_current(
    expected: &SemanticGeneration,
    output: &SemanticGeneration,
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
) -> bool {
    output == expected && expected == &semantic_generation_for_inputs(retained, style, layout)
}

pub(crate) fn semantic_publish_is_current_with_interaction(
    output: &SemanticGeneration,
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
    interaction: Option<&InteractionState>,
) -> bool {
    output == &semantic_generation_for_inputs_with_interaction(retained, style, layout, interaction)
}

struct SemanticBuilder<'a> {
    generation: SemanticGeneration,
    interaction: Option<&'a InteractionState>,
    node_count: usize,
    records: Vec<DiagnosticRecord>,
}

impl<'a> SemanticBuilder<'a> {
    fn new(generation: SemanticGeneration, interaction: Option<&'a InteractionState>) -> Self {
        Self {
            generation,
            interaction,
            node_count: 0,
            records: Vec::new(),
        }
    }

    fn visit(
        &mut self,
        retained: &RetainedNodeSnapshot,
        layout: &LayoutNodeSnapshot,
        is_root: bool,
        context: VisitContext<'a>,
    ) -> Option<SemanticNodeSnapshot> {
        if !retained.participation().semantics() {
            return None;
        }
        if retained.id() != layout.node_id() {
            self.record_missing_bounds(retained);
            return None;
        }

        let scroll_offset = context.scroll_offset(retained, layout);
        let bounds = context.map_rect(semantic_bounds(retained, layout));
        let role = semantic_role(retained, is_root);
        let (name, value) = semantic_name_and_value(retained);
        let scrollable = layout.scroll().scrollable();
        let state = SemanticStateSnapshot::new(SemanticStateParts {
            focusable: retained.focusable(),
            focused: self.is_focused(retained),
            window_focused: self
                .interaction
                .is_some_and(InteractionState::window_focused),
            scrollable,
            editable: retained.editable().is_some(),
            selection: retained.text_block().map(|block| block.selection()),
            composition: retained
                .text_block()
                .and_then(|block| block.composition().map(|composition| composition.range())),
            composition_cursor: retained.text_block().and_then(|block| {
                block
                    .composition()
                    .and_then(|composition| composition.cursor())
            }),
        });
        let actions = semantic_actions(retained, scrollable);
        if requires_accessible_name(retained) && name.as_deref().is_none_or(str::is_empty) {
            self.record_missing_name(retained, role);
        }

        let child_context = VisitContext {
            interaction: context.interaction,
            dx: context.dx - scroll_offset.x(),
            dy: context.dy - scroll_offset.y(),
        };
        let children = self.visit_children(retained, layout, child_context);
        self.node_count += 1;

        Some(SemanticNodeSnapshot::new(SemanticNodeParts {
            id: SemanticNodeId::new(retained.id(), retained.generation()),
            key: retained.key().cloned(),
            role,
            name,
            value,
            bounds,
            state,
            actions,
            children,
        }))
    }

    fn visit_children(
        &mut self,
        retained: &RetainedNodeSnapshot,
        layout: &LayoutNodeSnapshot,
        context: VisitContext<'a>,
    ) -> Vec<SemanticNodeSnapshot> {
        let mut children = Vec::with_capacity(retained.children().len());
        let layout_children = layout.children();
        let mut layout_index = 0;

        for retained_child in retained.children() {
            if !retained_child.participation().semantics() {
                continue;
            }

            let Some(layout_child) =
                next_layout_child(layout_children, &mut layout_index, retained_child)
            else {
                self.record_missing_bounds(retained_child);
                continue;
            };

            if let Some(child) = self.visit(retained_child, layout_child, false, context) {
                children.push(child);
            }
        }

        children
    }

    fn is_focused(&self, retained: &RetainedNodeSnapshot) -> bool {
        self.interaction
            .and_then(|state| {
                if retained.kind() == ElementKind::Input {
                    state.text_input_focus()
                } else {
                    state.keyboard_focus()
                }
            })
            .is_some_and(|target| {
                target == InteractionTarget::new(retained.id(), retained.generation())
            })
    }

    fn record_missing_name(&mut self, retained: &RetainedNodeSnapshot, role: SemanticRole) {
        self.records.push(
            DiagnosticRecord::new(
                DiagnosticArea::Semantics,
                DiagnosticSeverity::Warning,
                ErrorKind::Diagnostic,
                "semantics.diagnostic",
                "semantic node requires an explicit accessible name",
            )
            .with_field("reason", "missing_name")
            .with_field("node_id", retained.id().raw().to_string())
            .with_field("node_generation", retained.generation().raw().to_string())
            .with_field("role", role.name())
            .with_field("kind", retained.kind().name())
            .with_field("focusable", retained.focusable().to_string())
            .with_field("has_click", retained.handlers().has_click().to_string())
            .with_field(
                "has_pointer_handlers",
                retained.handlers().has_pointer_handlers().to_string(),
            )
            .with_field(
                "has_key_handlers",
                retained.handlers().has_key_handlers().to_string(),
            ),
        );
    }

    fn record_missing_bounds(&mut self, retained: &RetainedNodeSnapshot) {
        self.records.push(
            DiagnosticRecord::new(
                DiagnosticArea::Semantics,
                DiagnosticSeverity::Warning,
                ErrorKind::Diagnostic,
                "semantics.diagnostic",
                "semantic node was skipped because layout bounds were unavailable",
            )
            .with_field("reason", "bounds_unavailable")
            .with_field("node_id", retained.id().raw().to_string())
            .with_field("node_generation", retained.generation().raw().to_string())
            .with_field("kind", retained.kind().name()),
        );
    }

    fn finish(self, started: Instant, root: Option<SemanticNodeSnapshot>) -> SemanticBuildOutput {
        let stats = SemanticBuildStats {
            node_count: self.node_count,
            diagnostic_count: self.records.len(),
            stale_drop_count: 0,
            duration: started.elapsed(),
        };
        let snapshot =
            SemanticTreeSnapshot::new(self.generation, self.node_count, root, stats.clone());
        SemanticBuildOutput {
            snapshot,
            stats,
            records: self.records,
        }
    }
}

#[derive(Clone, Copy)]
struct VisitContext<'a> {
    interaction: Option<&'a InteractionState>,
    dx: f32,
    dy: f32,
}

impl<'a> VisitContext<'a> {
    fn new(interaction: Option<&'a InteractionState>) -> Self {
        Self {
            interaction,
            dx: 0.0,
            dy: 0.0,
        }
    }

    fn map_rect(self, rect: LayoutRect) -> LayoutRect {
        rect.translate(self.dx, self.dy)
    }

    fn scroll_offset(
        self,
        retained: &RetainedNodeSnapshot,
        layout: &LayoutNodeSnapshot,
    ) -> LayoutPoint {
        if !layout.scroll().scrollable() {
            return LayoutPoint::ZERO;
        }
        self.interaction.map_or(LayoutPoint::ZERO, |state| {
            state.scroll_offset(InteractionTarget::new(retained.id(), retained.generation()))
        })
    }
}

fn next_layout_child<'a>(
    layout_children: &'a [LayoutNodeSnapshot],
    layout_index: &mut usize,
    retained_child: &RetainedNodeSnapshot,
) -> Option<&'a LayoutNodeSnapshot> {
    let slice = layout_children.get(*layout_index..)?;
    let offset = slice
        .iter()
        .position(|layout_child| layout_child.node_id() == retained_child.id())?;
    *layout_index += offset + 1;
    Some(&slice[offset])
}

fn semantic_bounds(retained: &RetainedNodeSnapshot, layout: &LayoutNodeSnapshot) -> LayoutRect {
    if retained.kind() == ElementKind::Text {
        layout.content_rect()
    } else {
        layout.border_rect()
    }
}

fn semantic_role(retained: &RetainedNodeSnapshot, is_root: bool) -> SemanticRole {
    if is_root {
        SemanticRole::Window
    } else if retained.kind() == ElementKind::Input {
        SemanticRole::Textbox
    } else if retained.kind() == ElementKind::Text {
        SemanticRole::Text
    } else {
        SemanticRole::Generic
    }
}

fn semantic_name_and_value(retained: &RetainedNodeSnapshot) -> (Option<String>, Option<String>) {
    if retained.kind() == ElementKind::Input {
        let value = retained
            .text_block()
            .map(|block| block.committed().to_owned())
            .unwrap_or_default();
        return (Some(value.clone()), Some(value));
    }
    if retained.kind() == ElementKind::Text {
        let value = retained.text().unwrap_or_default().to_owned();
        return (Some(value.clone()), Some(value));
    }
    (None, None)
}

fn semantic_actions(retained: &RetainedNodeSnapshot, scrollable: bool) -> Vec<SemanticAction> {
    let mut actions = Vec::with_capacity(3);
    if retained.handlers().has_click() {
        actions.push(SemanticAction::Activate);
    }
    if retained.focusable() {
        actions.push(SemanticAction::Focus);
    }
    if scrollable {
        actions.push(SemanticAction::Scroll);
    }
    if retained.kind() == ElementKind::Input {
        actions.push(SemanticAction::Edit);
    }
    actions
}

fn requires_accessible_name(retained: &RetainedNodeSnapshot) -> bool {
    retained.focusable() || retained.handlers().has_any_handlers()
}

fn retained_semantic_signature(retained: &RetainedTreeSnapshot) -> SemanticInputSignature {
    let mut facts = Vec::new();
    if let Some(root) = retained.root() {
        collect_retained_semantic_node(root, &mut facts);
    }
    SemanticInputSignature::new(facts)
}

fn collect_retained_semantic_node(
    node: &RetainedNodeSnapshot,
    facts: &mut Vec<SemanticSignatureFact>,
) {
    facts.push(SemanticSignatureFact::Node {
        node_id: node.id().raw(),
        node_generation: node.generation().raw(),
        kind: element_kind_fact(node.kind()),
    });
    facts.push(SemanticSignatureFact::Participation {
        semantics: node.participation().semantics(),
    });
    if !node.participation().semantics() {
        return;
    }
    facts.push(SemanticSignatureFact::Focusable(node.focusable()));
    facts.push(SemanticSignatureFact::Handlers {
        pointer: node.handlers().has_pointer_handlers(),
        click: node.handlers().has_click(),
        key: node.handlers().has_key_handlers(),
    });
    if node.kind() == ElementKind::Input {
        if let Some(block) = node.text_block() {
            facts.push(SemanticSignatureFact::EditableValue {
                len: block.committed().len(),
                generation: block.generation().raw(),
                composing: block.has_composition(),
            });
        }
    } else if node.kind() == ElementKind::Text {
        let text = node.text().unwrap_or_default();
        facts.push(SemanticSignatureFact::TextValue {
            len: text.len(),
            hash: text_hash(text),
        });
    }
    for child in node.children() {
        collect_retained_semantic_node(child, facts);
    }
}

fn style_signature(style: &StyleTreeSnapshot) -> SemanticInputSignature {
    let mut facts = Vec::new();
    if let Some(root) = style.root() {
        collect_style_node(root, &mut facts);
    }
    SemanticInputSignature::new(facts)
}

fn collect_style_node(
    node: &crate::style::StyleNodeSnapshot,
    facts: &mut Vec<SemanticSignatureFact>,
) {
    facts.push(SemanticSignatureFact::Node {
        node_id: node.node_id(),
        node_generation: node.node_generation(),
        kind: 0,
    });
    facts.push(SemanticSignatureFact::Participation {
        semantics: node.participation().semantics(),
    });
    if !node.participation().semantics() {
        return;
    }
    for child in node.children() {
        collect_style_node(child, facts);
    }
}
