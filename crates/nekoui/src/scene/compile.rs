use std::collections::BTreeMap;
use std::time::Instant;

use crate::element::ElementKind;
use crate::interaction::InteractionState;
use crate::layout::{
    LayoutNodeSnapshot, LayoutPoint, LayoutRect, LayoutTreeSnapshot, text_viewport_placement,
};
use crate::retained::{RetainedNodeSnapshot, RetainedTreeSnapshot};
use crate::scene::{
    DamageRegion, HitTestEntry, HitTestPathNode, HitTestScene, PaintFragment, PaintFragmentKind,
    PaintScene, ResourceDemandKind, SceneCompileStats, SceneDiagnostic, SceneGeneration,
    SceneInputSignature, SceneOrder, SceneResourceDemand, SceneSignatureFact,
};
use crate::style::{Color, Display, Overflow, StyleTreeSnapshot, TextOverflow};
use crate::text::TextGlyphDemand;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SceneCompileInput<'a> {
    pub retained: &'a RetainedTreeSnapshot,
    pub style: &'a StyleTreeSnapshot,
    pub layout: &'a LayoutTreeSnapshot,
    pub interaction: Option<&'a InteractionState>,
    pub previous: Option<&'a PaintScene>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SceneCompileOutput {
    pub scene: PaintScene,
    pub stats: SceneCompileStats,
}

pub(crate) fn compile_scene(input: SceneCompileInput<'_>) -> SceneCompileOutput {
    let started = Instant::now();
    let generation = SceneGeneration::new_with_scroll(
        input.retained.generation(),
        input.layout.generation(),
        style_signature(input.style),
        input.layout.viewport().generation().raw(),
        text_signature(input.retained),
        scroll_signature(input.interaction),
    );
    let damage = input.previous.map_or_else(
        || DamageRegion::initial(viewport_rect(input.layout)),
        |previous| {
            if previous.generation() == generation {
                DamageRegion::unchanged()
            } else {
                DamageRegion::new(
                    vec![viewport_rect(input.layout)],
                    crate::scene::DamageReason::ConservativeInputChange,
                )
            }
        },
    );

    let mut compiler = SceneCompiler::new(generation, damage);
    if let (Some(retained), Some(layout)) = (input.retained.root(), input.layout.root()) {
        compiler.visit(retained, layout, true, VisitContext::new(input.interaction));
    }
    compiler.finish(started)
}

#[cfg(test)]
pub(crate) fn scene_generation_for_inputs(
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
) -> SceneGeneration {
    SceneGeneration::new_with_scroll(
        retained.generation(),
        layout.generation(),
        style_signature(style),
        layout.viewport().generation().raw(),
        text_signature(retained),
        scroll_signature(None),
    )
}

pub(crate) fn scene_generation_for_inputs_with_interaction(
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
    interaction: Option<&InteractionState>,
) -> SceneGeneration {
    SceneGeneration::new_with_scroll(
        retained.generation(),
        layout.generation(),
        style_signature(style),
        layout.viewport().generation().raw(),
        text_signature(retained),
        scroll_signature(interaction),
    )
}

#[cfg(test)]
pub(crate) fn scene_matches_inputs(
    generation: SceneGeneration,
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
) -> bool {
    generation == scene_generation_for_inputs(retained, style, layout)
}

#[cfg(test)]
pub(crate) fn scene_publish_is_current(
    expected: SceneGeneration,
    output: SceneGeneration,
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
) -> bool {
    output == expected && scene_matches_inputs(expected, retained, style, layout)
}

pub(crate) fn scene_publish_is_current_with_interaction(
    expected: SceneGeneration,
    output: SceneGeneration,
    retained: &RetainedTreeSnapshot,
    style: &StyleTreeSnapshot,
    layout: &LayoutTreeSnapshot,
    interaction: Option<&InteractionState>,
) -> bool {
    output == expected
        && expected
            == scene_generation_for_inputs_with_interaction(retained, style, layout, interaction)
}

struct SceneCompiler {
    generation: SceneGeneration,
    text_generation: SceneInputSignature,
    style_generation: SceneInputSignature,
    order: u64,
    fragments: Vec<PaintFragment>,
    hit_entries: Vec<HitTestEntry>,
    demands: Vec<SceneResourceDemand>,
    diagnostics: Vec<SceneDiagnostic>,
    damage: DamageRegion,
    unsupported_fragment_count: usize,
}

impl SceneCompiler {
    fn new(generation: SceneGeneration, damage: DamageRegion) -> Self {
        let text_generation = generation.text_generation().clone();
        let style_generation = generation.style_generation().clone();
        Self {
            generation,
            text_generation,
            style_generation,
            order: 0,
            fragments: Vec::new(),
            hit_entries: Vec::new(),
            demands: Vec::new(),
            diagnostics: Vec::new(),
            damage,
            unsupported_fragment_count: 0,
        }
    }

    fn next_order(&mut self) -> SceneOrder {
        let order = SceneOrder::new(self.order);
        self.order += 1;
        order
    }

    fn visit(
        &mut self,
        retained: &RetainedNodeSnapshot,
        layout: &LayoutNodeSnapshot,
        ancestors_visually_emitted: bool,
        context: VisitContext,
    ) {
        if retained.id() != layout.node_id() || !retained.participation().layout() {
            return;
        }

        let scroll_offset = context.scroll_offset(retained, layout);
        let local_clip = if layout.scroll().clips() {
            Some(context.map_rect(layout.scroll().viewport()))
        } else {
            None
        };
        let child_clip = combine_clip(context.clip, local_clip);
        let path = context.path_with(retained);

        let opacity = retained.resolved_style().visual().opacity().as_f32();
        let visual_allowed = ancestors_visually_emitted && opacity > 0.0;
        if opacity > 0.0 && opacity < 1.0 {
            self.emit_unsupported(retained, layout.border_rect(), "partial_opacity_layer");
        }

        if retained.participation().hit_test() {
            let order = self.next_order();
            self.hit_entries.push(HitTestEntry::new(
                retained.id(),
                retained.generation(),
                context.map_rect(layout.border_rect()),
                context.clip,
                path.clone(),
                order,
            ));
        }

        if visual_allowed && retained.participation().paint() {
            if let Some(color) = retained.resolved_style().visual().background() {
                let order = self.next_order();
                self.fragments.push(PaintFragment::new(
                    retained.id(),
                    retained.generation(),
                    order,
                    context.map_rect(layout.border_rect()),
                    PaintFragmentKind::Rect { color },
                ));
            }

            if text_capable_kind(retained.kind()) && retained.display_text().is_some() {
                let Some(text_layout) = layout.text_layout().cloned() else {
                    self.emit_unsupported(retained, layout.content_rect(), "text.layout_missing");
                    return;
                };
                let placement =
                    text_viewport_placement(retained.kind(), layout.content_rect(), &text_layout);
                let text_rect = placement.text_draw_rect();
                let clip = (retained.kind() == ElementKind::Input)
                    .then(|| context.map_rect(placement.viewport_rect()));
                let order = self.next_order();
                let mut fragment = PaintFragment::new(
                    retained.id(),
                    retained.generation(),
                    order,
                    context.map_rect(text_rect),
                    PaintFragmentKind::Text {
                        text_generation: self.text_generation.clone(),
                        text_metrics_generation: self
                            .generation
                            .layout_generation()
                            .map_or(0, |generation| generation.raw()),
                        color: retained.resolved_style().text().text_color(),
                    },
                )
                .with_text_layout(text_layout.clone());
                if let Some(clip) = clip {
                    fragment = fragment.with_clip(clip);
                }
                self.fragments.push(fragment);
                self.demands.push(SceneResourceDemand::glyph(
                    retained.id().raw(),
                    self.text_generation.clone(),
                    TextGlyphDemand::new(text_layout),
                ));
            }

            if retained.kind() == ElementKind::Input
                && context.text_input_focused(retained)
                && let Some(text_layout) = layout.text_layout()
            {
                let placement =
                    text_viewport_placement(retained.kind(), layout.content_rect(), text_layout);
                let caret = context.map_rect(placement.visible_caret_rect());
                let order = self.next_order();
                self.fragments.push(PaintFragment::new(
                    retained.id(),
                    retained.generation(),
                    order,
                    caret,
                    PaintFragmentKind::Rect {
                        color: retained.resolved_style().text().text_color(),
                    },
                ));
            }
        }

        let emits_clip_fragments =
            visual_allowed && retained.participation().paint() && layout.scroll().clips();
        if emits_clip_fragments {
            let order = self.next_order();
            let viewport = context.map_rect(layout.scroll().viewport());
            self.fragments.push(PaintFragment::new(
                retained.id(),
                retained.generation(),
                order,
                viewport,
                PaintFragmentKind::ClipPush { clip: viewport },
            ));
        }

        let child_context = VisitContext {
            interaction: context.interaction,
            clip: child_clip,
            path,
            dx: context.dx - scroll_offset.x(),
            dy: context.dy - scroll_offset.y(),
        };
        let mut children_by_id = BTreeMap::new();
        for child in layout.children() {
            children_by_id.insert(child.node_id(), child);
        }
        for child in retained.children() {
            if let Some(layout_child) = children_by_id.get(&child.id()) {
                self.visit(child, layout_child, visual_allowed, child_context.clone());
            }
        }

        if emits_clip_fragments {
            let order = self.next_order();
            let viewport = context.map_rect(layout.scroll().viewport());
            self.fragments.push(PaintFragment::new(
                retained.id(),
                retained.generation(),
                order,
                viewport,
                PaintFragmentKind::ClipPop,
            ));
        }
    }

    fn emit_unsupported(
        &mut self,
        retained: &RetainedNodeSnapshot,
        rect: LayoutRect,
        capability: &'static str,
    ) {
        let order = self.next_order();
        self.fragments.push(PaintFragment::new(
            retained.id(),
            retained.generation(),
            order,
            rect,
            PaintFragmentKind::Unsupported { capability },
        ));
        self.demands.push(SceneResourceDemand::new(
            ResourceDemandKind::Unsupported,
            retained.id().raw(),
            self.style_generation.clone(),
        ));
        self.unsupported_fragment_count += 1;
    }

    fn finish(self, started: Instant) -> SceneCompileOutput {
        let stats = SceneCompileStats {
            node_count: self.hit_entries.len(),
            fragment_count: self.fragments.len(),
            hit_test_entry_count: self.hit_entries.len(),
            damage_region_count: self.damage.region_count(),
            resource_demand_count: self.demands.len(),
            unsupported_fragment_count: self.unsupported_fragment_count,
            stale_drop_count: 0,
            duration: started.elapsed(),
        };
        let mut diagnostics = self.diagnostics;
        if self.unsupported_fragment_count > 0 {
            diagnostics.push(SceneDiagnostic::new(
                "unsupported scene fragment capability",
                self.unsupported_fragment_count as u64,
            ));
        }
        let scene = PaintScene::new(
            self.generation,
            self.fragments,
            HitTestScene::new(self.hit_entries),
            self.damage,
            self.demands,
            diagnostics,
            stats.clone(),
        );
        SceneCompileOutput { scene, stats }
    }
}

#[derive(Clone)]
struct VisitContext<'a> {
    interaction: Option<&'a InteractionState>,
    clip: Option<LayoutRect>,
    path: Vec<HitTestPathNode>,
    dx: f32,
    dy: f32,
}

impl<'a> VisitContext<'a> {
    fn new(interaction: Option<&'a InteractionState>) -> Self {
        Self {
            interaction,
            clip: None,
            path: Vec::new(),
            dx: 0.0,
            dy: 0.0,
        }
    }

    fn map_rect(&self, rect: LayoutRect) -> LayoutRect {
        rect.translate(self.dx, self.dy)
    }

    fn scroll_offset(
        &self,
        retained: &RetainedNodeSnapshot,
        layout: &LayoutNodeSnapshot,
    ) -> LayoutPoint {
        if layout.scroll().overflow() != Overflow::Scroll {
            return LayoutPoint::ZERO;
        }
        self.interaction.map_or(LayoutPoint::ZERO, |state| {
            state.scroll_offset(crate::interaction::InteractionTarget::new(
                retained.id(),
                retained.generation(),
            ))
        })
    }

    fn text_input_focused(&self, retained: &RetainedNodeSnapshot) -> bool {
        self.interaction.is_some_and(|state| {
            state.text_input_focus().is_some_and(|target| {
                target.node_id() == retained.id()
                    && target.node_generation() == retained.generation()
            })
        })
    }

    fn path_with(&self, retained: &RetainedNodeSnapshot) -> Vec<HitTestPathNode> {
        let mut path = self.path.clone();
        path.push(HitTestPathNode::new(retained.id(), retained.generation()));
        path
    }
}

fn combine_clip(parent: Option<LayoutRect>, local: Option<LayoutRect>) -> Option<LayoutRect> {
    match (parent, local) {
        (Some(parent), Some(local)) => parent.intersect(local).or(Some(LayoutRect::ZERO)),
        (Some(parent), None) => Some(parent),
        (None, Some(local)) => Some(local),
        (None, None) => None,
    }
}

fn viewport_rect(layout: &LayoutTreeSnapshot) -> LayoutRect {
    let size = layout.viewport().logical_size();
    LayoutRect::new(0.0, 0.0, size.width(), size.height())
}

fn text_capable_kind(kind: ElementKind) -> bool {
    matches!(kind, ElementKind::Text | ElementKind::Input)
}

fn style_signature(style: &StyleTreeSnapshot) -> SceneInputSignature {
    let mut facts = Vec::new();
    facts.push(SceneSignatureFact::MaxLines(Some(style.node_count())));
    if let Some(root) = style.root() {
        collect_style_node(root, &mut facts);
    }
    SceneInputSignature::new(facts)
}

fn text_signature(retained: &RetainedTreeSnapshot) -> SceneInputSignature {
    let mut facts = Vec::new();
    if let Some(root) = retained.root() {
        collect_text_node(root, &mut facts);
    }
    SceneInputSignature::new(facts)
}

fn scroll_signature(interaction: Option<&InteractionState>) -> SceneInputSignature {
    let mut facts = Vec::new();
    if let Some(interaction) = interaction {
        facts.extend(interaction.scroll_offsets().map(|(target, offset)| {
            SceneSignatureFact::ScrollOffset {
                target: scroll_target_signature(target),
                x: offset.x().to_bits(),
                y: offset.y().to_bits(),
            }
        }));
    }
    let text_input_focus = interaction.and_then(InteractionState::text_input_focus);
    facts.push(SceneSignatureFact::TextInputFocus {
        target_id: text_input_focus.map(|target| target.node_id().raw()),
        target_generation: text_input_focus.map(|target| target.node_generation().raw()),
    });
    SceneInputSignature::new(facts)
}

fn scroll_target_signature(target: crate::interaction::InteractionTarget) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for value in [target.node_id().raw(), target.node_generation().raw()] {
        hash ^= value;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
fn collect_text_node(node: &RetainedNodeSnapshot, facts: &mut Vec<SceneSignatureFact>) {
    facts.push(SceneSignatureFact::Node {
        node_id: node.id().raw(),
        node_generation: node.generation().raw(),
    });
    if text_capable_kind(node.kind()) {
        facts.push(SceneSignatureFact::TextPayload(node.display_text()));
    }
    for child in node.children() {
        collect_text_node(child, facts);
    }
}

fn collect_style_node(node: &crate::style::StyleNodeSnapshot, facts: &mut Vec<SceneSignatureFact>) {
    facts.push(SceneSignatureFact::Node {
        node_id: node.node_id(),
        node_generation: node.node_generation(),
    });
    let participation = node.participation();
    facts.push(SceneSignatureFact::Participation {
        layout: participation.layout(),
        paint: participation.paint(),
        hit_test: participation.hit_test(),
        semantics: participation.semantics(),
    });
    facts.push(SceneSignatureFact::Background(color_fact(
        node.resolved().visual().background(),
    )));
    facts.push(SceneSignatureFact::Opacity(
        node.resolved().visual().opacity().as_f32().to_bits(),
    ));
    facts.push(SceneSignatureFact::TextColor(color_signature(
        node.resolved().text().text_color(),
    )));
    facts.push(SceneSignatureFact::FontSize(
        node.resolved().text().font_size().as_px().to_bits(),
    ));
    facts.push(SceneSignatureFact::TextOverflow(text_overflow_fact(
        node.resolved().text().text_overflow(),
    )));
    facts.push(SceneSignatureFact::MaxLines(
        node.resolved().text().max_lines(),
    ));
    facts.push(SceneSignatureFact::Display(display_fact(
        node.resolved().layout().display(),
    )));
    facts.push(SceneSignatureFact::Overflow(overflow_fact(
        node.resolved().layout().overflow(),
    )));
    for child in node.children() {
        collect_style_node(child, facts);
    }
}

fn color_fact(color: Option<Color>) -> Option<u64> {
    color.map(color_signature)
}

fn color_signature(color: Color) -> u64 {
    match color.color_space() {
        crate::style::ColorSpace::Srgb {
            red,
            green,
            blue,
            alpha,
        } => u32::from_be_bytes([red, green, blue, alpha]) as u64,
        crate::style::ColorSpace::Oklch {
            lightness,
            chroma,
            hue,
            alpha,
        } => {
            let mut hash = 0xcbf29ce484222325_u64;
            for value in [
                lightness.to_bits(),
                chroma.to_bits(),
                hue.to_bits(),
                alpha.to_bits(),
            ] {
                hash ^= value as u64;
                hash = hash.wrapping_mul(0x100000001b3);
            }
            hash
        }
    }
}

fn display_fact(display: Display) -> u8 {
    match display {
        Display::None => 0_u8,
        Display::Block => 1,
        Display::Flex => 2,
    }
}

fn overflow_fact(overflow: Overflow) -> u8 {
    match overflow {
        Overflow::Visible => 0,
        Overflow::Hidden => 1,
        Overflow::Scroll => 2,
    }
}

fn text_overflow_fact(overflow: TextOverflow) -> u8 {
    match overflow {
        TextOverflow::Visible => 0_u8,
        TextOverflow::Clip => 1,
        TextOverflow::Ellipsis => 2,
    }
}
