use std::sync::Arc;

use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use taffy::prelude::{AvailableSpace, NodeId as TaffyNodeId, Size as TaffySize, TaffyTree};

use crate::SharedString;
use crate::element::{SpecArena, SpecNode, SpecNodeId, SpecPayload, WindowFrameArea};
use crate::input::FocusPolicy;
use crate::style::{Background, ResolvedStyle, ResolvedTextStyle};
use crate::text_system::{SharedTextLayout, TextBlock, TextMeasureKey, TextSystem, measure_key};
use crate::window::WindowSize;

use super::retained_compile::get_or_build_subtree_fragment;
use super::retained_diff::update_from_spec;
use super::retained_frame_areas::{collect_window_frame_areas, window_frame_area_at};
use super::retained_tree::build_node;
use super::{CompiledScene, DirtyLaneMask, LayoutBox, LogicalBatch, Primitive, SceneNode};

new_key_type! {
    pub(crate) struct NodeId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum NodeClass {
    Div,
    Text,
}

#[derive(Debug, Clone)]
pub(crate) struct RetainedNode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub children: SmallVec<[NodeId; 4]>,
    pub kind: NodeKind,
    pub key: Option<u64>,
    pub owner_view_id: Option<u64>,
    pub style: ResolvedStyle,
    pub window_frame_area: Option<WindowFrameArea>,
    pub interaction: crate::element::InteractionState,
    pub semantics: crate::semantics::SemanticsState,
    pub layout: LayoutBox,
    pub dirty: DirtyLaneMask,
    pub(super) compiled_fragment: Option<Arc<CompiledSubtreeFragment>>,
    pub taffy_node: TaffyNodeId,
}

#[derive(Debug, Clone)]
pub(crate) enum NodeKind {
    Div,
    Text {
        content: SharedString,
        block: Box<Option<TextBlock>>,
        layout: Option<SharedTextLayout>,
    },
}

#[derive(Debug)]
pub(crate) struct RetainedTree {
    pub(super) root: NodeId,
    pub(super) nodes: SlotMap<NodeId, RetainedNode>,
    pub(super) taffy: TaffyTree<MeasureContext>,
}

#[derive(Debug, Clone)]
pub(super) enum MeasureContext {
    Text(TextMeasureContext),
}

#[derive(Debug, Clone)]
pub(super) struct TextMeasureContext {
    pub(super) text: SharedString,
    pub(super) style: ResolvedTextStyle,
    pub(super) last_key: Option<TextMeasureKey>,
    pub(super) last_layout: Option<SharedTextLayout>,
}

impl TextMeasureContext {
    pub(super) fn from_spec(spec: &SpecNode) -> Self {
        let text = match &spec.payload {
            SpecPayload::Text(text) => text,
            SpecPayload::None => unreachable!("text measure context requires text payload"),
        };
        Self {
            text: text.clone(),
            style: spec.style.text.clone(),
            last_key: None,
            last_layout: None,
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct CompiledSubtreeFragment {
    pub(super) scene_nodes: Arc<[SceneNode]>,
    pub(super) primitives: Arc<[Primitive]>,
    pub(super) logical_batches: Arc<[LogicalBatch]>,
}

impl RetainedTree {
    pub fn from_spec(arena: &SpecArena, root: SpecNodeId) -> Self {
        let mut tree = Self {
            root: NodeId::default(),
            nodes: SlotMap::with_key(),
            taffy: TaffyTree::new(),
        };

        let root_id = build_node(&mut tree, arena, root, None);
        tree.root = root_id;
        tree
    }

    pub fn update_from_spec(&mut self, arena: &SpecArena, root: SpecNodeId) -> DirtyLaneMask {
        update_from_spec(self, arena, root)
    }

    pub fn compute_layout(&mut self, size: WindowSize, text_system: &mut TextSystem) {
        self.taffy
            .compute_layout_with_measure(
                self.nodes[self.root].taffy_node,
                TaffySize {
                    width: AvailableSpace::Definite(size.width as f32),
                    height: AvailableSpace::Definite(size.height as f32),
                },
                |known_dimensions, available_space, _node_id, context, _style| {
                    let Some(MeasureContext::Text(text_context)) = context else {
                        return taffy::geometry::Size::ZERO;
                    };

                    let width = known_dimensions.width.or(match available_space.width {
                        AvailableSpace::Definite(value) => Some(value),
                        AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
                    });
                    let key = measure_key(&text_context.text, &text_context.style, width);

                    let layout = if let Some(cached_layout) = text_context
                        .last_key
                        .as_ref()
                        .zip(text_context.last_layout.as_ref())
                        .filter(|(cached_key, _)| **cached_key == key)
                        .map(|(_, cached_layout)| cached_layout.clone())
                    {
                        cached_layout
                    } else {
                        text_system.measure(&text_context.text, &text_context.style, width)
                    };

                    text_context.last_key = Some(key);
                    text_context.last_layout = Some(layout.clone());

                    taffy::geometry::Size {
                        width: layout.width,
                        height: layout.height,
                    }
                },
            )
            .expect("taffy layout computation must succeed for retained nodes");

        let taffy = &self.taffy;
        for (node_id, node) in &mut self.nodes {
            debug_assert_eq!(node.id, node_id);

            let layout = *taffy
                .layout(node.taffy_node)
                .expect("layout must be available after compute_layout");
            let next_layout = LayoutBox {
                x: layout.location.x,
                y: layout.location.y,
                width: layout.size.width,
                height: layout.size.height,
            };
            if node.layout != next_layout {
                node.dirty |= DirtyLaneMask::LAYOUT;
            }
            node.layout = next_layout;

            if let NodeKind::Text {
                content,
                block,
                layout: text_layout,
            } = &mut node.kind
            {
                let measured =
                    taffy
                        .get_node_context(node.taffy_node)
                        .and_then(|context| match context {
                            MeasureContext::Text(text_context) => text_context
                                .last_key
                                .as_ref()
                                .zip(text_context.last_layout.as_ref())
                                .map(|(key, layout)| {
                                    (key.width_bits.map(f32::from_bits), layout.clone())
                                }),
                        });

                match block.as_mut() {
                    Some(block) => {
                        if let Some((measured_width, measured_layout)) = measured.clone() {
                            text_system.sync_text_block(
                                block,
                                content.clone(),
                                node.style.text.clone(),
                                measured_width,
                            );
                            block.layout = measured_layout.clone();
                            *text_layout = Some(measured_layout);
                        } else {
                            let width = node.layout.width.is_finite().then_some(node.layout.width);
                            text_system.sync_text_block(
                                block,
                                content.clone(),
                                node.style.text.clone(),
                                width,
                            );
                            *text_layout = Some(block.layout.clone());
                        }
                    }
                    None => {
                        if let Some((measured_width, measured_layout)) = measured {
                            let mut new_block = text_system.new_text_block(
                                content.clone(),
                                node.style.text.clone(),
                                measured_width,
                            );
                            new_block.layout = measured_layout.clone();
                            *text_layout = Some(measured_layout);
                            **block = Some(new_block);
                        } else {
                            let width = node.layout.width.is_finite().then_some(node.layout.width);
                            let new_block = text_system.new_text_block(
                                content.clone(),
                                node.style.text.clone(),
                                width,
                            );
                            *text_layout = Some(new_block.layout.clone());
                            **block = Some(new_block);
                        }
                    }
                };
            }
        }
    }

    #[cfg(test)]
    pub fn root_layout(&self) -> LayoutBox {
        self.nodes[self.root].layout
    }

    #[cfg(test)]
    pub fn root_id(&self) -> NodeId {
        self.root
    }

    #[cfg(test)]
    pub fn node(&self, node_id: NodeId) -> &RetainedNode {
        &self.nodes[node_id]
    }

    #[cfg(test)]
    pub fn children(&self, node_id: NodeId) -> &[NodeId] {
        &self.nodes[node_id].children
    }

    pub fn compile_scene(&mut self) -> CompiledScene {
        let root_fragment = get_or_build_subtree_fragment(self, self.root, false, false);
        let clear_color = self.nodes[self.root]
            .style
            .paint
            .background
            .map(|background| match background {
                Background::Solid(color) => color,
                Background::LinearGradient(gradient) => gradient.start_color,
            });
        CompiledScene {
            clear_color,
            scene_nodes: root_fragment.scene_nodes.clone(),
            primitives: root_fragment.primitives.clone(),
            logical_batches: root_fragment.logical_batches.clone(),
            effect_regions: Arc::from([]),
        }
    }

    pub fn window_frame_area_at(
        &self,
        point: crate::style::Point<crate::style::Px>,
    ) -> Option<WindowFrameArea> {
        window_frame_area_at(self, point)
    }

    pub fn collect_window_frame_areas(&self) -> Vec<(WindowFrameArea, LayoutBox)> {
        collect_window_frame_areas(self)
    }

    pub(crate) fn retained_node(&self, node_id: NodeId) -> &RetainedNode {
        &self.nodes[node_id]
    }

    pub(crate) fn try_retained_node(&self, node_id: NodeId) -> Option<&RetainedNode> {
        self.nodes.get(node_id)
    }

    pub(crate) fn absolute_layout_box(&self, node_id: NodeId) -> LayoutBox {
        let target = &self.nodes[node_id];
        let mut current = Some(node_id);
        let mut x = 0.0;
        let mut y = 0.0;
        let width = target.layout.width;
        let height = target.layout.height;

        while let Some(id) = current {
            let node = &self.nodes[id];
            x += node.layout.x;
            y += node.layout.y;
            current = node.parent;
        }

        LayoutBox {
            x,
            y,
            width,
            height,
        }
    }

    pub fn focusable_node_at(
        &self,
        point: crate::style::Point<crate::style::Px>,
    ) -> Option<NodeId> {
        focusable_node_at(self, self.root, point, [0.0, 0.0])
    }

    pub fn text_input_node_at(
        &self,
        point: crate::style::Point<crate::style::Px>,
    ) -> Option<NodeId> {
        text_input_node_at(self, self.root, point, [0.0, 0.0])
    }

    pub fn first_focusable_node(&self) -> Option<NodeId> {
        first_focusable_node(self, self.root)
    }
}

fn focusable_node_at(
    tree: &RetainedTree,
    node_id: NodeId,
    point: crate::style::Point<crate::style::Px>,
    offset: [f32; 2],
) -> Option<NodeId> {
    let node = &tree.nodes[node_id];
    if !node_is_rendered(node) {
        return None;
    }

    let absolute = LayoutBox {
        x: offset[0] + node.layout.x,
        y: offset[1] + node.layout.y,
        width: node.layout.width,
        height: node.layout.height,
    };
    if !layout_box_contains_point(absolute, point) {
        return None;
    }

    let child_offset = [absolute.x, absolute.y];
    for child_id in node.children.iter().rev().copied() {
        if let Some(hit) = focusable_node_at(tree, child_id, point, child_offset) {
            return Some(hit);
        }
    }

    match node.interaction.focus_policy {
        FocusPolicy::Keyboard | FocusPolicy::TextInput if !node.semantics.disabled => Some(node_id),
        FocusPolicy::None => None,
        _ => None,
    }
}

fn text_input_node_at(
    tree: &RetainedTree,
    node_id: NodeId,
    point: crate::style::Point<crate::style::Px>,
    offset: [f32; 2],
) -> Option<NodeId> {
    let node = &tree.nodes[node_id];
    if !node_is_rendered(node) {
        return None;
    }

    let absolute = LayoutBox {
        x: offset[0] + node.layout.x,
        y: offset[1] + node.layout.y,
        width: node.layout.width,
        height: node.layout.height,
    };
    if !layout_box_contains_point(absolute, point) {
        return None;
    }

    let child_offset = [absolute.x, absolute.y];
    for child_id in node.children.iter().rev().copied() {
        if let Some(hit) = text_input_node_at(tree, child_id, point, child_offset) {
            return Some(hit);
        }
    }

    matches!(node.interaction.focus_policy, FocusPolicy::TextInput)
        .then_some(node_id)
        .filter(|_| !node.semantics.disabled)
}

fn first_focusable_node(tree: &RetainedTree, node_id: NodeId) -> Option<NodeId> {
    let node = &tree.nodes[node_id];
    if !node_is_rendered(node) {
        return None;
    }

    if matches!(
        node.interaction.focus_policy,
        FocusPolicy::Keyboard | FocusPolicy::TextInput
    ) && !node.semantics.disabled
    {
        return Some(node_id);
    }

    for child_id in node.children.iter().copied() {
        if let Some(found) = first_focusable_node(tree, child_id) {
            return Some(found);
        }
    }

    None
}

fn layout_box_contains_point(
    layout: LayoutBox,
    point: crate::style::Point<crate::style::Px>,
) -> bool {
    let x = point.x.get();
    let y = point.y.get();
    x >= layout.x && x <= layout.x + layout.width && y >= layout.y && y <= layout.y + layout.height
}

fn node_is_rendered(node: &RetainedNode) -> bool {
    node.layout.width > 0.0
        && node.layout.height > 0.0
        && !matches!(node.style.layout.display, crate::style::Display::None)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::app::{App, Render};
    use crate::element::{BuildCx, IntoElement, ParentElement, SpecArena};
    use crate::platform::window::WindowInfoSeed;
    use crate::scene::{ClipClass, ClipShape, EffectClass, Primitive, RectFill};
    use crate::style::px;
    use crate::style::{
        Absolute, BoxSizing, Color, CornerRadii, EdgeInsets, EdgeWidths, FontFamily, TextAlign,
        gradient,
    };
    use crate::text_system::TextSystem;
    use crate::window::{Window, WindowId, WindowInfo, WindowOptions, WindowSize};

    use super::{DirtyLaneMask, NodeKind, RetainedTree};

    fn test_window(logical: WindowSize, physical: WindowSize, scale_factor: f64) -> Window {
        Window::from_options(
            WindowId::new(),
            &WindowOptions::default(),
            WindowInfoSeed {
                content_size: logical,
                frame_size: Some(logical),
                physical_size: physical,
                scale_factor,
                position: None,
                current_display: None,
            },
        )
    }

    fn build_static_tree(root: crate::AnyElement) -> RetainedTree {
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(root)
            .unwrap();
        RetainedTree::from_spec(&arena, built.root)
    }

    #[test]
    fn text_measurement_wraps_within_available_width() {
        let root = crate::div()
            .width(px(200.0))
            .padding(EdgeInsets::all(10.0))
            .child(crate::text("hello neko ui hello neko ui hello neko ui").font_size(px(16.0)))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(200, 120), &mut text_system);

        let root_layout = tree.root_layout();
        assert_eq!(root_layout.width, 200.0);

        let text_id = tree.children(tree.root_id())[0];
        let text_node = tree.node(text_id);
        assert!(text_node.layout.width <= 180.0 + 0.5);
        assert!(text_node.layout.height > 20.0);

        match &text_node.kind {
            NodeKind::Text { layout, .. } => {
                let layout = layout.as_ref().expect("text layout exists");
                assert!(layout.runs.len() >= 2);
            }
            NodeKind::Div => panic!("expected text node"),
        }
    }

    #[test]
    fn text_block_reuses_taffy_measured_wrap_layout() {
        let root = crate::div()
            .width(px(200.0))
            .padding(EdgeInsets::all(10.0))
            .child(crate::text("hello neko ui hello neko ui hello neko ui").font_size(px(16.0)))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(200, 120), &mut text_system);

        let text_id = tree.children(tree.root_id())[0];
        let text_node = tree.node(text_id);
        match &text_node.kind {
            NodeKind::Text {
                block,
                layout: Some(layout),
                ..
            } => {
                let block = block.as_ref().as_ref().expect("text block should exist");
                assert!(Arc::ptr_eq(&block.layout, layout));
                assert!(layout.width <= 180.0 + 0.5);
                assert!(layout.runs.len() >= 2);
            }
            _ => panic!("expected text node"),
        }
    }

    #[test]
    fn owner_view_id_and_semantics_propagate_into_retained_nodes() {
        struct AccessibleView;

        impl Render for AccessibleView {
            fn render(
                &mut self,
                _window: &WindowInfo,
                _cx: &mut crate::Context<'_, Self>,
            ) -> impl IntoElement {
                crate::div()
                    .focusable()
                    .semantics_role(crate::SemanticsRole::Button)
                    .semantics_label("action")
                    .child(crate::text("press"))
            }
        }

        let app = App::new(Vec::new());
        let view = app.insert_view(AccessibleView);
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let root = crate::div().child(view).into_any_element();
        let mut arena = SpecArena::new();
        let built = app.build_root_spec(&window, &root, &mut arena).unwrap();
        let mut tree = RetainedTree::from_spec(&arena, built.root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        let child = tree.children(tree.root_id())[0];
        let child_node = tree.node(child);
        assert_eq!(child_node.owner_view_id, Some(view.id()));
        assert_eq!(child_node.semantics.role, crate::SemanticsRole::Button);
        assert_eq!(child_node.semantics.label.as_deref(), Some("action"));
    }

    #[test]
    fn pointer_hit_testing_skips_non_rendered_focusable_nodes() {
        let root = crate::div()
            .child(
                crate::div()
                    .display(crate::style::Display::None)
                    .focusable()
                    .text_input(crate::TextInputPurpose::Normal),
            )
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        let hidden_hit = tree.focusable_node_at(crate::style::point(px(0.0), px(0.0)));
        let hidden_text_input = tree.text_input_node_at(crate::style::point(px(0.0), px(0.0)));
        assert!(hidden_hit.is_none());
        assert!(hidden_text_input.is_none());
    }

    #[test]
    fn focusable_and_text_input_hit_testing_follow_interaction_metadata() {
        let root = crate::div()
            .child(
                crate::div()
                    .w(px(100.0))
                    .h(px(40.0))
                    .focusable()
                    .semantics_role(crate::SemanticsRole::Button),
            )
            .child(
                crate::div()
                    .w(px(100.0))
                    .h(px(40.0))
                    .mt(px(50.0))
                    .text_input(crate::TextInputPurpose::Normal),
            )
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        let second = tree.children(tree.root_id())[1];
        let second_layout = tree.node(second).layout;
        let focusable = tree.focusable_node_at(crate::style::point(px(10.0), px(10.0)));
        let text_input = tree.text_input_node_at(crate::style::point(
            px(second_layout.x + 10.0),
            px(second_layout.y + second_layout.height * 0.5),
        ));
        assert!(focusable.is_some());
        assert!(text_input.is_some());
        assert_ne!(focusable, text_input);
    }

    #[test]
    fn text_style_inherits_from_parent_div_and_child_can_override() {
        let inherited_color = Color::rgb(0x112233);
        let override_color = Color::rgb(0x445566);
        let root = crate::div()
            .font_family(["Noto Sans SC", "Segoe UI Emoji"])
            .font_size(px(24.0))
            .text_color(inherited_color)
            .text_center()
            .child(crate::text("inherited"))
            .child(crate::text("override").color(override_color))
            .into_any_element();

        let tree = build_static_tree(root);
        let first = tree.children(tree.root_id())[0];
        let second = tree.children(tree.root_id())[1];

        assert_eq!(
            tree.node(first).style.text.font_families.as_ref(),
            &[
                FontFamily::from("Noto Sans SC"),
                FontFamily::from("Segoe UI Emoji")
            ]
        );
        assert_eq!(
            tree.node(first).style.text.font_size,
            Absolute::from(px(24.0))
        );
        assert_eq!(tree.node(first).style.text.color, inherited_color);
        assert_eq!(tree.node(first).style.text.text_align, TextAlign::Center);
        assert_eq!(tree.node(second).style.text.color, override_color);
    }

    #[test]
    fn box_and_paint_style_do_not_inherit_to_text_children() {
        let root = crate::div()
            .bg(Color::rgb(0x111111))
            .border(2.0, Color::rgb(0x222222))
            .opacity(0.25)
            .child(crate::text("child"))
            .into_any_element();

        let tree = build_static_tree(root);
        let child = tree.children(tree.root_id())[0];
        assert_eq!(tree.node(child).style.paint.background, None);
        assert_eq!(
            tree.node(child).style.paint.border.widths,
            EdgeWidths::default()
        );
        assert_eq!(tree.node(child).style.paint.border.color, None);
        assert_eq!(tree.node(child).style.paint.opacity, 1.0);
    }

    #[test]
    fn align_self_overrides_parent_cross_axis_alignment() {
        let root = crate::div()
            .size(crate::style::size(px(200.0).into(), px(100.0).into()))
            .items_start()
            .child(crate::div().w(px(20.0)).h(px(20.0)).self_end())
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        let child = tree.children(tree.root_id())[0];
        assert_eq!(tree.node(child).layout.y, 80.0);
    }

    #[test]
    fn box_sizing_controls_whether_border_and_padding_expand_layout_size() {
        fn root_width_for(box_sizing: BoxSizing) -> f32 {
            let root = crate::div()
                .box_sizing(box_sizing)
                .w(px(100.0))
                .p(px(10.0))
                .border(5.0, Color::rgb(0x222222))
                .into_any_element();
            let mut tree = build_static_tree(root);
            let mut text_system = TextSystem::new();
            tree.compute_layout(WindowSize::new(500, 200), &mut text_system);
            tree.root_layout().width
        }

        assert_eq!(root_width_for(BoxSizing::BorderBox), 100.0);
        assert_eq!(root_width_for(BoxSizing::ContentBox), 130.0);
    }

    #[test]
    fn diff_marks_paint_without_rebuilding_tree_for_text_color_change() {
        let root = crate::div().child(crate::text("hello").color(Color::rgb(0x111111)));
        let updated = crate::div().child(crate::text("hello").color(Color::rgb(0x222222)));

        let mut tree = build_static_tree(root.into_any_element());
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated.into_any_element())
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::PAINT);
    }

    #[test]
    fn text_block_owner_survives_paint_only_text_color_updates() {
        let root = crate::div().child(crate::text("hello").color(Color::rgb(0x111111)));
        let updated = crate::div().child(crate::text("hello").color(Color::rgb(0x222222)));

        let mut tree = build_static_tree(root.into_any_element());
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        let text_id = tree.children(tree.root_id())[0];
        let (original_layout, original_revision) = match &tree.node(text_id).kind {
            NodeKind::Text {
                block,
                layout: Some(layout),
                ..
            } => {
                let block = block.as_ref().as_ref().expect("text block should exist");
                (layout.clone(), block.revision)
            }
            _ => panic!("expected text node"),
        };

        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated.into_any_element())
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::PAINT);

        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        match &tree.node(text_id).kind {
            NodeKind::Text {
                block,
                layout: Some(layout),
                ..
            } => {
                let block = block
                    .as_ref()
                    .as_ref()
                    .expect("text block should still exist");
                assert!(Arc::ptr_eq(&original_layout, layout));
                assert_eq!(block.revision.layout, original_revision.layout);
                assert_eq!(block.revision.style, original_revision.style + 1);
            }
            _ => panic!("expected text node"),
        }
    }

    #[test]
    fn diff_marks_layout_for_div_size_change() {
        let root = crate::div().width(px(100.0));
        let updated = crate::div().width(px(140.0));

        let mut tree = build_static_tree(root.into_any_element());
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated.into_any_element())
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::LAYOUT);
    }

    #[test]
    fn diff_marks_build_for_child_structure_change() {
        let root = crate::div().child(crate::text("a"));
        let updated = crate::div().child(crate::text("a")).child(crate::text("b"));

        let mut tree = build_static_tree(root.into_any_element());
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated.into_any_element())
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::BUILD.normalized());
    }

    #[test]
    fn keyed_reorder_reuses_existing_nodes_without_build() {
        let root = crate::div()
            .child(crate::text("a").key(1))
            .child(crate::text("b").key(2));
        let updated = crate::div()
            .child(crate::text("b").key(2))
            .child(crate::text("a").key(1));

        let mut tree = build_static_tree(root.into_any_element());
        let first = tree.children(tree.root_id())[0];
        let second = tree.children(tree.root_id())[1];

        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated.into_any_element())
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert!(!dirty.contains(DirtyLaneMask::BUILD));
        assert!(dirty.contains(DirtyLaneMask::LAYOUT));
        assert_eq!(tree.children(tree.root_id()), &[second, first]);
    }

    #[test]
    fn compile_scene_emits_scene_nodes_and_logical_batches() {
        let root = crate::div()
            .bg(Color::rgb(0x111111))
            .child(crate::text("hello"))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        assert_eq!(compiled.scene_nodes.len(), 2);
        assert!(!compiled.primitives.is_empty());
        assert!(!compiled.logical_batches.is_empty());
        assert_eq!(compiled.logical_batches[0].primitive_range.start, 0);
        assert!(
            compiled.logical_batches.last().unwrap().primitive_range.end as usize
                <= compiled.primitives.len()
        );
    }

    #[test]
    fn compile_scene_carries_clip_and_opacity_metadata() {
        let root = crate::div()
            .clip()
            .opacity(0.5)
            .child(crate::text("hello").opacity(0.25))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        assert_eq!(compiled.scene_nodes[0].opacity, 0.5);
        assert!(compiled.scene_nodes[0].clip.shape.is_some());
        assert_eq!(compiled.scene_nodes[1].opacity, 0.25);
    }

    #[test]
    fn paint_only_update_preserves_logical_batch_structure() {
        let root = crate::div()
            .bg(Color::rgb(0x111111))
            .child(crate::text("hello").color(Color::rgb(0x222222)))
            .into_any_element();
        let updated = crate::div()
            .bg(Color::rgb(0x333333))
            .child(crate::text("hello").color(Color::rgb(0x444444)))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let original = tree.compile_scene();

        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated)
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::PAINT);
        let updated_scene = tree.compile_scene();

        assert_eq!(original.scene_nodes.len(), updated_scene.scene_nodes.len());
        assert_eq!(
            original.logical_batches.len(),
            updated_scene.logical_batches.len()
        );
        assert_eq!(
            original.logical_batches[0].primitive_range,
            updated_scene.logical_batches[0].primitive_range
        );
    }

    #[test]
    fn paint_only_update_reuses_root_scene_structure_cache() {
        let root = crate::div()
            .bg(Color::rgb(0x111111))
            .child(crate::text("hello").color(Color::rgb(0x222222)))
            .into_any_element();
        let updated = crate::div()
            .bg(Color::rgb(0x333333))
            .child(crate::text("hello").color(Color::rgb(0x444444)))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let original = tree.compile_scene();

        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated)
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::PAINT);

        let updated_scene = tree.compile_scene();
        assert!(Arc::ptr_eq(
            &original.scene_nodes,
            &updated_scene.scene_nodes
        ));
        assert!(Arc::ptr_eq(
            &original.logical_batches,
            &updated_scene.logical_batches
        ));
        assert!(!Arc::ptr_eq(
            &original.primitives,
            &updated_scene.primitives
        ));
    }

    #[test]
    fn logical_batches_reflect_clip_and_opacity_classes() {
        let root = crate::div()
            .clip()
            .opacity(0.5)
            .bg(Color::rgb(0x111111))
            .child(crate::text("hello"))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        assert!(
            compiled
                .logical_batches
                .iter()
                .any(|batch| batch.clip_class == ClipClass::Rect)
        );
        assert!(
            compiled
                .logical_batches
                .iter()
                .any(|batch| batch.effect_class == EffectClass::Opacity)
        );
    }

    #[test]
    fn rounded_overflow_hidden_compiles_to_rounded_clip_shape() {
        let root = crate::div()
            .clip()
            .corner_radii(CornerRadii::all(12.0))
            .child(crate::text("hello"))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        assert!(matches!(
            compiled.scene_nodes[0].clip.shape,
            Some(ClipShape::RoundedRect { .. })
        ));
        assert!(
            compiled
                .logical_batches
                .iter()
                .any(|batch| batch.clip_class == ClipClass::RoundedRect)
        );
    }

    #[test]
    fn compile_scene_rect_primitive_carries_corner_and_border_style() {
        let root = crate::div()
            .bg(Color::rgb(0x111111))
            .corner_radii(CornerRadii {
                top_left: 8.0,
                top_right: 12.0,
                bottom_right: 16.0,
                bottom_left: 20.0,
            })
            .border_widths(EdgeWidths {
                top: 1.0,
                right: 2.0,
                bottom: 3.0,
                left: 4.0,
            })
            .border_color(Color::rgb(0x222222))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        let Primitive::Rect(rect) = &compiled.primitives[0] else {
            panic!("expected rect primitive");
        };
        assert_eq!(rect.corner_radii.top_left, 8.0);
        assert_eq!(rect.corner_radii.top_right, 12.0);
        assert_eq!(rect.corner_radii.bottom_right, 16.0);
        assert_eq!(rect.corner_radii.bottom_left, 20.0);
        assert_eq!(rect.border_widths.top, 1.0);
        assert_eq!(rect.border_widths.right, 2.0);
        assert_eq!(rect.border_widths.bottom, 3.0);
        assert_eq!(rect.border_widths.left, 4.0);
        assert_eq!(rect.border_color, Some(Color::rgb(0x222222)));
    }

    #[test]
    fn compile_scene_emits_rect_for_border_only_div() {
        let root = crate::div()
            .border(2.0, Color::rgb(0x333333))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        assert!(!compiled.primitives.is_empty());
        let Primitive::Rect(rect) = &compiled.primitives[0] else {
            panic!("expected rect primitive");
        };
        assert_eq!(rect.fill, RectFill::Solid(Color::rgba(0.0, 0.0, 0.0, 0.0)));
    }

    #[test]
    fn compile_scene_rect_primitive_carries_gradient_fill() {
        let root = crate::div()
            .bg(gradient(Color::rgb(0x111111), Color::rgb(0x444444), 1.25))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let compiled = tree.compile_scene();

        let Primitive::Rect(rect) = &compiled.primitives[0] else {
            panic!("expected rect primitive");
        };
        let RectFill::LinearGradient(gradient) = rect.fill else {
            panic!("expected gradient fill");
        };
        assert_eq!(gradient.start_color, Color::rgb(0x111111));
        assert_eq!(gradient.end_color, Color::rgb(0x444444));
        assert_eq!(gradient.angle_radians, 1.25);
    }

    #[test]
    fn paint_only_update_reuses_clean_sibling_fragment_cache() {
        let root = crate::div()
            .child(crate::div().bg(Color::rgb(0x111111)).key(1))
            .child(crate::div().bg(Color::rgb(0x222222)).key(2))
            .into_any_element();
        let updated = crate::div()
            .child(crate::div().bg(Color::rgb(0x333333)).key(1))
            .child(crate::div().bg(Color::rgb(0x222222)).key(2))
            .into_any_element();

        let mut tree = build_static_tree(root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);
        let _ = tree.compile_scene();

        let sibling = tree.children(tree.root_id())[1];
        let original_fragment = tree.nodes[sibling]
            .compiled_fragment
            .clone()
            .expect("clean sibling fragment cached after compile");

        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(updated)
            .unwrap();
        let dirty = tree.update_from_spec(&arena, built.root);
        assert_eq!(dirty, DirtyLaneMask::PAINT);

        let _ = tree.compile_scene();
        let reused_fragment = tree.nodes[sibling]
            .compiled_fragment
            .clone()
            .expect("clean sibling fragment remains cached");
        assert!(Arc::ptr_eq(&original_fragment, &reused_fragment));
    }

    #[test]
    fn view_nodes_resolve_before_retained_layout() {
        struct LabelView;

        impl Render for LabelView {
            fn render(
                &mut self,
                _window: &WindowInfo,
                _cx: &mut crate::Context<'_, Self>,
            ) -> impl IntoElement {
                crate::text("resolved from view")
            }
        }

        let app = App::new(Vec::new());
        let view = app.insert_view(LabelView);
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(640, 400), 2.0);
        let root = crate::div().child(view).into_any_element();
        let mut arena = SpecArena::new();
        let built = app.build_root_spec(&window, &root, &mut arena).unwrap();
        let mut tree = RetainedTree::from_spec(&arena, built.root);
        let mut text_system = TextSystem::new();
        tree.compute_layout(window.content_size(), &mut text_system);

        let child = tree.children(tree.root_id())[0];
        assert!(matches!(tree.node(child).kind, NodeKind::Text { .. }));
    }
}
