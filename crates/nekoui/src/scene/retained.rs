use slotmap::{SlotMap, new_key_type};
use smallvec::SmallVec;
use taffy::prelude::{
    AlignItems as TaffyAlignItems, AvailableSpace, Dimension, Display,
    JustifyContent as TaffyJustifyContent, LengthPercentage, LengthPercentageAuto,
    NodeId as TaffyNodeId, Rect, Size as TaffySize, Style as TaffyStyle, TaffyAuto, TaffyTree,
};
use taffy::style::FlexDirection as TaffyFlexDirection;

use crate::SharedString;
use crate::element::{Div, Element, ElementKind, Text};
use crate::style::{AlignItems, Direction, JustifyContent, Length, Style};
use crate::text_system::{TextLayout, TextMeasureKey, TextSystem, measure_key};
use crate::window::WindowSize;

use super::{CompiledScene, DirtyLaneMask, LayoutBox, Primitive};

new_key_type! {
    pub struct NodeId;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum NodeClass {
    Div,
    Text,
}

#[derive(Debug, Clone)]
pub struct RetainedNode {
    pub id: NodeId,
    pub parent: Option<NodeId>,
    pub children: SmallVec<[NodeId; 4]>,
    pub kind: NodeKind,
    pub key: Option<u64>,
    pub style: Style,
    pub layout: LayoutBox,
    pub dirty: DirtyLaneMask,
    pub taffy_node: TaffyNodeId,
}

#[derive(Debug, Clone)]
pub enum NodeKind {
    Div,
    Text {
        content: SharedString,
        layout: Option<TextLayout>,
    },
}

#[derive(Debug)]
pub struct RetainedTree {
    root: NodeId,
    nodes: SlotMap<NodeId, RetainedNode>,
    taffy: TaffyTree<MeasureContext>,
}

#[derive(Debug, Clone)]
enum MeasureContext {
    Text(TextMeasureContext),
}

#[derive(Debug, Clone)]
struct TextMeasureContext {
    text: SharedString,
    style: crate::style::TextStyle,
    last_key: Option<TextMeasureKey>,
    last_layout: Option<TextLayout>,
}

impl TextMeasureContext {
    fn new(text: &Text) -> Self {
        Self {
            text: text.content.clone(),
            style: text.style.text.clone(),
            last_key: None,
            last_layout: None,
        }
    }
}

impl RetainedTree {
    pub fn from_element(root: &Element) -> Self {
        let mut tree = Self {
            root: NodeId::default(),
            nodes: SlotMap::with_key(),
            taffy: TaffyTree::new(),
        };

        let root_id = tree.build_node(root, None);
        tree.root = root_id;
        tree
    }

    pub fn update_from_element(&mut self, root: &Element) -> DirtyLaneMask {
        self.clear_dirty_marks();

        if !self.can_reuse_node(self.root, root) {
            *self = RetainedTree::from_element(root);
            return DirtyLaneMask::BUILD.normalized();
        }

        self.diff_node(self.root, root).normalized()
    }

    pub fn compute_layout(&mut self, size: WindowSize, text_system: &mut TextSystem) {
        let mut measured_layouts = rustc_hash::FxHashMap::<TextMeasureKey, TextLayout>::default();
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

                    let width = known_dimensions
                        .width
                        .or_else(|| definite_space(available_space.width));
                    let key = measure_key(&text_context.text, &text_context.style, width);

                    let layout = if let Some(cached_layout) = text_context
                        .last_key
                        .as_ref()
                        .zip(text_context.last_layout.as_ref())
                        .filter(|(cached_key, _)| **cached_key == key)
                        .map(|(_, cached_layout)| cached_layout.clone())
                    {
                        cached_layout
                    } else if let Some(cached_layout) = measured_layouts.get(&key).cloned() {
                        cached_layout
                    } else {
                        let measured =
                            text_system.measure(&text_context.text, &text_context.style, width);
                        measured_layouts.insert(key.clone(), measured.clone());
                        measured
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
            node.layout = LayoutBox {
                x: layout.location.x,
                y: layout.location.y,
                width: layout.size.width,
                height: layout.size.height,
            };

            if let NodeKind::Text {
                layout: text_layout,
                ..
            } = &mut node.kind
            {
                *text_layout =
                    taffy
                        .get_node_context(node.taffy_node)
                        .and_then(|context| match context {
                            MeasureContext::Text(text_context) => text_context.last_layout.clone(),
                        });
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

    pub fn compile_scene(&self) -> CompiledScene {
        let mut primitives = Vec::with_capacity(self.nodes.len());
        self.collect_primitives(self.root, 0.0, 0.0, &mut primitives);
        CompiledScene {
            clear_color: None,
            primitives,
        }
    }

    fn clear_dirty_marks(&mut self) {
        for (_, node) in &mut self.nodes {
            node.dirty = DirtyLaneMask::empty();
        }
    }

    fn diff_node(&mut self, node_id: NodeId, element: &Element) -> DirtyLaneMask {
        match element.kind() {
            ElementKind::Div(div) => self.diff_div(node_id, div),
            ElementKind::Text(text) => self.diff_text(node_id, text),
            ElementKind::View(_) => {
                unreachable!("view nodes must be resolved before retained diff")
            }
        }
    }

    fn diff_div(&mut self, node_id: NodeId, div: &Div) -> DirtyLaneMask {
        let mut dirty = diff_div_style(&self.nodes[node_id].style, &div.style);
        if self.nodes[node_id].style != div.style {
            self.nodes[node_id].style = div.style.clone();
            self.taffy
                .set_style(
                    self.nodes[node_id].taffy_node,
                    div_style_to_taffy(&div.style),
                )
                .expect("div style patch must succeed");
        }
        self.nodes[node_id].key = div.key;

        let child_dirty = self.sync_children(node_id, &div.children);
        dirty |= child_dirty;
        self.nodes[node_id].dirty |= dirty;
        dirty
    }

    fn diff_text(&mut self, node_id: NodeId, text: &Text) -> DirtyLaneMask {
        let mut dirty = diff_text_style(&self.nodes[node_id].style, &text.style);
        let content_changed = match &self.nodes[node_id].kind {
            NodeKind::Text { content, .. } => content.as_ref() != text.content.as_ref(),
            NodeKind::Div => unreachable!("text diff called for non-text node"),
        };

        if content_changed {
            dirty |= DirtyLaneMask::LAYOUT | DirtyLaneMask::PAINT;
        }

        if self.nodes[node_id].style != text.style {
            self.nodes[node_id].style = text.style.clone();
            self.taffy
                .set_style(
                    self.nodes[node_id].taffy_node,
                    text_style_to_taffy(&text.style),
                )
                .expect("text style patch must succeed");
        }

        if let NodeKind::Text { content, layout } = &mut self.nodes[node_id].kind {
            *content = text.content.clone();
            if dirty.needs_layout() || dirty.contains(DirtyLaneMask::PAINT) {
                *layout = None;
            }
        }
        self.nodes[node_id].key = text.key;

        if !dirty.is_empty() {
            self.taffy
                .set_node_context(
                    self.nodes[node_id].taffy_node,
                    Some(MeasureContext::Text(TextMeasureContext::new(text))),
                )
                .expect("text node context patch must succeed");
        }

        self.nodes[node_id].dirty |= dirty;
        dirty
    }

    fn sync_children(&mut self, parent_id: NodeId, new_children: &[Element]) -> DirtyLaneMask {
        let old_children = self.nodes[parent_id].children.clone();
        let keyed_path = self.uses_keyed_path(&old_children, new_children);
        let mut next_children = SmallVec::<[NodeId; 4]>::with_capacity(new_children.len());
        let mut dirty = DirtyLaneMask::empty();
        let mut rebuild_required = false;
        let mut child_list_changed = old_children.len() != new_children.len();
        let mut reused = rustc_hash::FxHashSet::<NodeId>::default();
        let mut keyed_old = rustc_hash::FxHashMap::<(u64, NodeClass), NodeId>::default();

        if keyed_path {
            for child_id in &old_children {
                if let Some(key) = self.nodes[*child_id].key {
                    keyed_old.insert((key, self.node_class(*child_id)), *child_id);
                }
            }
        }

        for (index, element) in new_children.iter().enumerate() {
            let positional = old_children
                .get(index)
                .copied()
                .filter(|child_id| !reused.contains(child_id));
            let reused_child = if keyed_path {
                if let Some(key) = element_key(element) {
                    keyed_old
                        .get(&(key, element_class(element)))
                        .copied()
                        .filter(|child_id| !reused.contains(child_id))
                        .filter(|child_id| self.can_reuse_node(*child_id, element))
                        .or_else(|| {
                            positional.filter(|child_id| self.can_reuse_node(*child_id, element))
                        })
                } else {
                    positional.filter(|child_id| self.can_reuse_node(*child_id, element))
                }
            } else {
                positional.filter(|child_id| self.can_reuse_node(*child_id, element))
            };

            let child_id = match reused_child {
                Some(existing_child) => {
                    reused.insert(existing_child);
                    if old_children.get(index).copied() != Some(existing_child) {
                        child_list_changed = true;
                        dirty |= DirtyLaneMask::LAYOUT;
                    }
                    dirty |= self.diff_node(existing_child, element);
                    self.nodes[existing_child].parent = Some(parent_id);
                    existing_child
                }
                None => {
                    child_list_changed = true;
                    rebuild_required = true;
                    dirty |= DirtyLaneMask::BUILD;
                    self.build_node(element, Some(parent_id))
                }
            };

            next_children.push(child_id);
        }

        for old_child in old_children.iter().copied() {
            if !reused.contains(&old_child) {
                child_list_changed = true;
                rebuild_required = true;
                dirty |= DirtyLaneMask::BUILD;
                self.remove_subtree(old_child);
            }
        }

        if child_list_changed {
            self.nodes[parent_id].children = next_children.clone();
            let taffy_children = next_children
                .iter()
                .map(|child_id| self.nodes[*child_id].taffy_node)
                .collect::<SmallVec<[TaffyNodeId; 4]>>();
            self.taffy
                .set_children(self.nodes[parent_id].taffy_node, &taffy_children)
                .expect("children patch must succeed");

            if !rebuild_required {
                dirty |= DirtyLaneMask::LAYOUT;
            }
        }

        dirty
    }

    fn can_reuse_node(&self, node_id: NodeId, element: &Element) -> bool {
        let node = &self.nodes[node_id];
        let new_key = element_key(element);
        let same_kind = matches!(
            (&node.kind, element.kind()),
            (NodeKind::Div, ElementKind::Div(_)) | (NodeKind::Text { .. }, ElementKind::Text(_))
        );
        if !same_kind {
            return false;
        }

        match (node.key, new_key) {
            (Some(existing), Some(new)) => existing == new,
            (None, None) => true,
            _ => false,
        }
    }

    fn uses_keyed_path(&self, old_children: &[NodeId], new_children: &[Element]) -> bool {
        old_children
            .iter()
            .any(|child_id| self.nodes[*child_id].key.is_some())
            || new_children
                .iter()
                .any(|element| element_key(element).is_some())
    }

    fn node_class(&self, node_id: NodeId) -> NodeClass {
        match self.nodes[node_id].kind {
            NodeKind::Div => NodeClass::Div,
            NodeKind::Text { .. } => NodeClass::Text,
        }
    }

    fn build_node(&mut self, element: &Element, parent: Option<NodeId>) -> NodeId {
        match element.kind() {
            ElementKind::Div(div) => self.build_div(div, parent),
            ElementKind::Text(text) => self.build_text(text, parent),
            ElementKind::View(_) => {
                unreachable!("view nodes must be resolved before building the retained tree")
            }
        }
    }

    fn build_div(&mut self, div: &Div, parent: Option<NodeId>) -> NodeId {
        let child_ids = div
            .children
            .iter()
            .map(|child| self.build_node(child, None))
            .collect::<SmallVec<[NodeId; 4]>>();
        let child_taffy_nodes = child_ids
            .iter()
            .map(|child_id| self.nodes[*child_id].taffy_node)
            .collect::<SmallVec<[TaffyNodeId; 4]>>();

        let taffy_node = self
            .taffy
            .new_with_children(div_style_to_taffy(&div.style), &child_taffy_nodes)
            .expect("div node creation must succeed");

        let node_id = self.nodes.insert_with_key(|id| RetainedNode {
            id,
            parent,
            children: child_ids.clone(),
            kind: NodeKind::Div,
            key: div.key,
            style: div.style.clone(),
            layout: LayoutBox::default(),
            dirty: DirtyLaneMask::BUILD.normalized(),
            taffy_node,
        });

        for child_id in child_ids {
            self.nodes[child_id].parent = Some(node_id);
        }

        node_id
    }

    fn build_text(&mut self, text: &Text, parent: Option<NodeId>) -> NodeId {
        let taffy_node = self
            .taffy
            .new_leaf_with_context(
                text_style_to_taffy(&text.style),
                MeasureContext::Text(TextMeasureContext::new(text)),
            )
            .expect("text node creation must succeed");

        self.nodes.insert_with_key(|id| RetainedNode {
            id,
            parent,
            children: SmallVec::new(),
            kind: NodeKind::Text {
                content: text.content.clone(),
                layout: None,
            },
            key: text.key,
            style: text.style.clone(),
            layout: LayoutBox::default(),
            dirty: DirtyLaneMask::BUILD.normalized(),
            taffy_node,
        })
    }

    fn remove_subtree(&mut self, node_id: NodeId) {
        let children = self.nodes[node_id].children.clone();
        for child_id in children {
            self.remove_subtree(child_id);
        }

        let taffy_node = self.nodes[node_id].taffy_node;
        self.taffy
            .remove(taffy_node)
            .expect("retained subtree removal must succeed");
        self.nodes.remove(node_id);
    }

    fn collect_primitives(
        &self,
        node_id: NodeId,
        offset_x: f32,
        offset_y: f32,
        primitives: &mut Vec<Primitive>,
    ) {
        let node = &self.nodes[node_id];
        debug_assert_eq!(node.id, node_id);
        let bounds = LayoutBox {
            x: offset_x + node.layout.x,
            y: offset_y + node.layout.y,
            width: node.layout.width,
            height: node.layout.height,
        };

        match &node.kind {
            NodeKind::Div => {
                if let Some(background) = node.style.paint.background {
                    primitives.push(Primitive::Quad {
                        bounds,
                        color: background,
                    });
                }
            }
            NodeKind::Text { layout, .. } => {
                if let Some(layout) = layout.clone() {
                    primitives.push(Primitive::Text {
                        bounds,
                        layout,
                        color: node.style.text.color,
                    });
                }
            }
        }

        for child_id in &node.children {
            self.collect_primitives(*child_id, bounds.x, bounds.y, primitives);
        }
    }
}

fn definite_space(space: AvailableSpace) -> Option<f32> {
    match space {
        AvailableSpace::Definite(value) => Some(value),
        AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
    }
}

fn diff_div_style(old: &Style, new: &Style) -> DirtyLaneMask {
    let mut dirty = DirtyLaneMask::empty();
    if old.layout != new.layout {
        dirty |= DirtyLaneMask::LAYOUT;
    }
    if old.paint != new.paint {
        dirty |= DirtyLaneMask::PAINT;
    }
    dirty
}

fn diff_text_style(old: &Style, new: &Style) -> DirtyLaneMask {
    let mut dirty = DirtyLaneMask::empty();
    if old.layout != new.layout {
        dirty |= DirtyLaneMask::LAYOUT;
    }
    if old.paint != new.paint || old.text.color != new.text.color {
        dirty |= DirtyLaneMask::PAINT;
    }
    if old.text.font_family != new.text.font_family
        || old.text.font_size != new.text.font_size
        || old.text.line_height != new.text.line_height
    {
        dirty |= DirtyLaneMask::LAYOUT | DirtyLaneMask::PAINT;
    }
    dirty
}

fn element_key(element: &Element) -> Option<u64> {
    match element.kind() {
        ElementKind::Div(div) => div.key,
        ElementKind::Text(text) => text.key,
        ElementKind::View(_) => None,
    }
}

fn element_class(element: &Element) -> NodeClass {
    match element.kind() {
        ElementKind::Div(_) => NodeClass::Div,
        ElementKind::Text(_) => NodeClass::Text,
        ElementKind::View(_) => unreachable!("view nodes must be resolved before retained diff"),
    }
}

fn div_style_to_taffy(style: &Style) -> TaffyStyle {
    TaffyStyle {
        display: Display::Flex,
        flex_direction: match style.layout.direction {
            Direction::Row => TaffyFlexDirection::Row,
            Direction::Column => TaffyFlexDirection::Column,
        },
        size: TaffySize {
            width: length_to_dimension(style.layout.size.width),
            height: length_to_dimension(style.layout.size.height),
        },
        margin: Rect {
            left: edge_to_auto(style.layout.margin.left),
            right: edge_to_auto(style.layout.margin.right),
            top: edge_to_auto(style.layout.margin.top),
            bottom: edge_to_auto(style.layout.margin.bottom),
        },
        padding: Rect {
            left: edge_to_length(style.layout.padding.left),
            right: edge_to_length(style.layout.padding.right),
            top: edge_to_length(style.layout.padding.top),
            bottom: edge_to_length(style.layout.padding.bottom),
        },
        gap: TaffySize {
            width: LengthPercentage::length(style.layout.gap),
            height: LengthPercentage::length(style.layout.gap),
        },
        align_items: Some(match style.layout.align_items {
            AlignItems::Start => TaffyAlignItems::Start,
            AlignItems::Center => TaffyAlignItems::Center,
            AlignItems::End => TaffyAlignItems::End,
            AlignItems::Stretch => TaffyAlignItems::Stretch,
        }),
        justify_content: Some(match style.layout.justify_content {
            JustifyContent::Start => TaffyJustifyContent::Start,
            JustifyContent::Center => TaffyJustifyContent::Center,
            JustifyContent::End => TaffyJustifyContent::End,
            JustifyContent::SpaceBetween => TaffyJustifyContent::SpaceBetween,
        }),
        ..Default::default()
    }
}

fn text_style_to_taffy(style: &Style) -> TaffyStyle {
    TaffyStyle {
        display: Display::Block,
        size: TaffySize {
            width: length_to_dimension(style.layout.size.width),
            height: length_to_dimension(style.layout.size.height),
        },
        min_size: TaffySize {
            width: Dimension::length(0.0),
            height: Dimension::AUTO,
        },
        margin: Rect {
            left: edge_to_auto(style.layout.margin.left),
            right: edge_to_auto(style.layout.margin.right),
            top: edge_to_auto(style.layout.margin.top),
            bottom: edge_to_auto(style.layout.margin.bottom),
        },
        ..Default::default()
    }
}

fn length_to_dimension(length: Length) -> Dimension {
    match length {
        Length::Auto => Dimension::AUTO,
        Length::Px(value) => Dimension::length(value),
        Length::Fill => Dimension::percent(1.0),
    }
}

fn edge_to_length(value: f32) -> LengthPercentage {
    LengthPercentage::length(value)
}

fn edge_to_auto(value: f32) -> LengthPercentageAuto {
    LengthPercentageAuto::length(value)
}

#[cfg(test)]
mod tests {
    use crate::app::{App, Render};
    use crate::element::{IntoElement, ParentElement};
    use crate::style::{Color, EdgeInsets, Length};
    use crate::text_system::TextSystem;
    use crate::window::{Window, WindowId, WindowSize};

    use super::{DirtyLaneMask, NodeKind, RetainedTree};

    #[test]
    fn text_measurement_wraps_within_available_width() {
        let root = crate::div()
            .width(Length::Px(200.0))
            .padding(EdgeInsets::all(10.0))
            .child(crate::text("hello neko ui hello neko ui hello neko ui").font_size(16.0))
            .into_element();

        let mut tree = RetainedTree::from_element(&root);
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
    fn diff_marks_paint_without_rebuilding_tree_for_text_color_change() {
        let root = crate::div().child(crate::text("hello").color(Color::rgb(0x111111)));
        let updated = crate::div().child(crate::text("hello").color(Color::rgb(0x222222)));

        let mut tree = RetainedTree::from_element(&root.into_element());
        let dirty = tree.update_from_element(&updated.into_element());
        assert_eq!(dirty, DirtyLaneMask::PAINT);
    }

    #[test]
    fn diff_marks_layout_for_div_size_change() {
        let root = crate::div().width(Length::Px(100.0));
        let updated = crate::div().width(Length::Px(140.0));

        let mut tree = RetainedTree::from_element(&root.into_element());
        let dirty = tree.update_from_element(&updated.into_element());
        assert_eq!(dirty, DirtyLaneMask::LAYOUT);
    }

    #[test]
    fn diff_marks_build_for_child_structure_change() {
        let root = crate::div().child(crate::text("a"));
        let updated = crate::div().child(crate::text("a")).child(crate::text("b"));

        let mut tree = RetainedTree::from_element(&root.into_element());
        let dirty = tree.update_from_element(&updated.into_element());
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

        let mut tree = RetainedTree::from_element(&root.into_element());
        let first = tree.children(tree.root_id())[0];
        let second = tree.children(tree.root_id())[1];

        let dirty = tree.update_from_element(&updated.into_element());
        assert!(!dirty.contains(DirtyLaneMask::BUILD));
        assert!(dirty.contains(DirtyLaneMask::LAYOUT));
        assert_eq!(tree.children(tree.root_id()), &[second, first]);
    }

    #[test]
    fn view_nodes_resolve_before_retained_layout() {
        struct LabelView;

        impl Render for LabelView {
            fn render(
                &mut self,
                _window: &mut Window,
                _cx: &mut crate::Context<'_, Self>,
            ) -> impl IntoElement<Element = crate::Element> {
                crate::text("resolved from view")
            }
        }

        let app = App::new();
        let view = app.insert_view(LabelView);
        let mut window = Window::new_with_metrics(
            WindowId::new(),
            String::from("test"),
            WindowSize::new(320, 200),
            WindowSize::new(640, 400),
            2.0,
        );
        let root = crate::div().child(view).into_element();

        let (resolved, _) = app
            .resolve_root_element_with_views(&mut window, &root)
            .unwrap();
        let mut tree = RetainedTree::from_element(&resolved);
        let mut text_system = TextSystem::new();
        tree.compute_layout(window.size(), &mut text_system);

        let child = tree.children(tree.root_id())[0];
        assert!(matches!(tree.node(child).kind, NodeKind::Text { .. }));
    }
}
