use taffy::geometry::{Rect as TaffyRect, Size as TaffySize};
use taffy::style::{
    AvailableSpace, BoxSizing as TaffyBoxSizing, Dimension as TaffyDimension,
    Display as TaffyDisplay, FlexDirection, LengthPercentage, LengthPercentageAuto,
    Style as TaffyStyle,
};
use taffy::{NodeId, TaffyTree};

use crate::element::{ElementKey, ElementKind};
use crate::error::{NekoError, NekoResult};
use crate::layout::{LayoutNodeSnapshot, LayoutRect, LayoutSize, ScrollGeometry, Viewport};
use crate::retained::{RetainedIdentity, RetainedLayoutInput, RetainedLayoutNode};
use crate::style::{Dimension, Display, Length, ResolvedLayoutStyle, ResolvedTextStyle};
use crate::text::{
    FontManager, TextGeneration, TextLayoutRef, TextLayoutResult, TextMeasureQuery,
    TextMeasureSession, TextMeasureStats,
};

use super::snapshot::LayoutBoxes;

pub(crate) struct RawLayoutOutput {
    pub root: Option<LayoutNodeSnapshot>,
    pub node_count: usize,
    pub text_measure: TextMeasureStats,
}

#[derive(Debug)]
pub(crate) enum RawLayoutError {
    Plain(NekoError),
    WithTextStats {
        error: NekoError,
        text_measure: Box<TextMeasureStats>,
    },
}

#[derive(Clone, Debug, PartialEq)]
enum MeasureContext {
    Text {
        node_id: crate::retained::RetainedNodeId,
        node_generation: crate::retained::NodeGeneration,
        text: std::sync::Arc<str>,
        style: std::sync::Arc<ResolvedTextStyle>,
        scale_generation: u64,
        scale_factor: f32,
        text_layout: Option<TextLayoutRef>,
    },
}

struct BuiltNode {
    identity: RetainedIdentity,
    kind: ElementKind,
    key: Option<ElementKey>,
    overflow: crate::style::Overflow,
    taffy_id: NodeId,
    children: Vec<BuiltNode>,
}

pub(crate) fn compute(
    input: RetainedLayoutInput<'_>,
    viewport: Viewport,
    font_manager: &FontManager,
) -> Result<RawLayoutOutput, RawLayoutError> {
    viewport.validate().map_err(RawLayoutError::Plain)?;
    let Some(root) = input.root() else {
        return Ok(RawLayoutOutput {
            root: None,
            node_count: 0,
            text_measure: TextMeasureStats::default(),
        });
    };

    let mut tree = TaffyTree::<MeasureContext>::with_capacity(input.node_count());
    let mut text_session = TextMeasureSession::new(font_manager);
    let Some(built_root) =
        build_node(&mut tree, root, viewport, true).map_err(RawLayoutError::Plain)?
    else {
        return Ok(RawLayoutOutput {
            root: None,
            node_count: 0,
            text_measure: TextMeasureStats::default(),
        });
    };
    let mut text_measure_error = None;
    tree.compute_layout_with_measure(
        built_root.taffy_id,
        TaffySize {
            width: AvailableSpace::Definite(viewport.logical_size().width()),
            height: AvailableSpace::Definite(viewport.logical_size().height()),
        },
        |known_dimensions, available_space, node_id, node_context, style| {
            measure_content(
                known_dimensions,
                available_space,
                node_id,
                node_context,
                style,
                &mut text_session,
                &mut text_measure_error,
            )
        },
    )
    .map_err(|error| {
        RawLayoutError::Plain(NekoError::diagnostic(format!(
            "layout solver failed: {error}"
        )))
    })?;
    if let Some(error) = text_measure_error {
        return Err(RawLayoutError::WithTextStats {
            error,
            text_measure: Box::new(text_session.stats()),
        });
    }

    let root_snapshot =
        materialize_node(&tree, &built_root, LayoutOrigin::ZERO).map_err(RawLayoutError::Plain)?;
    let node_count = count_snapshot(&root_snapshot);
    Ok(RawLayoutOutput {
        root: Some(root_snapshot),
        node_count,
        text_measure: text_session.stats(),
    })
}

fn build_node(
    tree: &mut TaffyTree<MeasureContext>,
    node: RetainedLayoutNode<'_>,
    viewport: Viewport,
    is_root: bool,
) -> NekoResult<Option<BuiltNode>> {
    if !node.participation().layout() {
        return Ok(None);
    }
    validate_layout_style(node.resolved_style().layout())?;
    if node.kind() == ElementKind::Text {
        validate_text_style(node)?;
    }

    let mut child_ids = Vec::with_capacity(node.children_len());
    let mut children = Vec::with_capacity(node.children_len());
    for child in node.children() {
        if let Some(built_child) = build_node(tree, child, viewport, false)? {
            child_ids.push(built_child.taffy_id);
            children.push(built_child);
        }
    }

    let style = to_taffy_style(node.resolved_style().layout(), is_root, viewport);
    let taffy_id = if children.is_empty() {
        if node.kind() == ElementKind::Text {
            tree.new_leaf_with_context(style, text_measure_context(node, viewport))
        } else {
            tree.new_leaf(style)
        }
    } else {
        tree.new_with_children(style, &child_ids)
    }
    .map_err(|error| NekoError::diagnostic(format!("layout tree build failed: {error}")))?;

    Ok(Some(BuiltNode {
        identity: node.identity(),
        kind: node.kind(),
        key: node.key().cloned(),
        overflow: node.resolved_style().layout().overflow(),
        taffy_id,
        children,
    }))
}

fn to_taffy_style(style: &ResolvedLayoutStyle, is_root: bool, viewport: Viewport) -> TaffyStyle {
    let display = match style.display() {
        Display::None => TaffyDisplay::None,
        Display::Block | Display::Flex => TaffyDisplay::Flex,
    };
    let flex_direction = match style.display() {
        Display::Flex => FlexDirection::Row,
        Display::None | Display::Block => FlexDirection::Column,
    };
    let gap = style.gap().as_px();

    TaffyStyle {
        display,
        box_sizing: TaffyBoxSizing::BorderBox,
        size: TaffySize {
            width: to_taffy_dimension(style.width(), is_root, viewport.logical_size().width()),
            height: to_taffy_dimension(style.height(), false, viewport.logical_size().height()),
        },
        padding: to_taffy_padding(style.padding()),
        margin: to_taffy_margin(style.margin()),
        gap: TaffySize {
            width: LengthPercentage::length(gap),
            height: LengthPercentage::length(gap),
        },
        flex_shrink: 0.0,
        flex_direction,
        ..Default::default()
    }
}

fn validate_layout_style(style: &ResolvedLayoutStyle) -> NekoResult<()> {
    style.width().validate_for_layout("width")?;
    style.height().validate_for_layout("height")?;
    validate_edges(style.padding(), "padding")?;
    validate_edges(style.margin(), "margin")?;
    style.gap().validate_non_negative("gap")?;
    Ok(())
}

fn validate_text_style(node: RetainedLayoutNode<'_>) -> NekoResult<()> {
    node.resolved_style()
        .text()
        .font_size()
        .validate_non_negative("font size")
}

fn validate_edges(edges: crate::style::Edges<Length>, label: &'static str) -> NekoResult<()> {
    edges.top.validate_non_negative(label)?;
    edges.right.validate_non_negative(label)?;
    edges.bottom.validate_non_negative(label)?;
    edges.left.validate_non_negative(label)?;
    Ok(())
}

fn to_taffy_dimension(value: Dimension, fill_auto_at_root: bool, root_axis: f32) -> TaffyDimension {
    match value {
        Dimension::Auto if fill_auto_at_root => TaffyDimension::length(root_axis),
        Dimension::Auto => TaffyDimension::auto(),
        Dimension::Fill if fill_auto_at_root => TaffyDimension::length(root_axis),
        Dimension::Fill => TaffyDimension::percent(1.0),
        Dimension::Length(length) => TaffyDimension::length(length.as_px()),
    }
}

fn to_taffy_padding(edges: crate::style::Edges<Length>) -> TaffyRect<LengthPercentage> {
    TaffyRect {
        left: LengthPercentage::length(edges.left.as_px()),
        right: LengthPercentage::length(edges.right.as_px()),
        top: LengthPercentage::length(edges.top.as_px()),
        bottom: LengthPercentage::length(edges.bottom.as_px()),
    }
}

fn to_taffy_margin(edges: crate::style::Edges<Length>) -> TaffyRect<LengthPercentageAuto> {
    TaffyRect {
        left: LengthPercentageAuto::length(edges.left.as_px()),
        right: LengthPercentageAuto::length(edges.right.as_px()),
        top: LengthPercentageAuto::length(edges.top.as_px()),
        bottom: LengthPercentageAuto::length(edges.bottom.as_px()),
    }
}

fn text_measure_context(node: RetainedLayoutNode<'_>, viewport: Viewport) -> MeasureContext {
    MeasureContext::Text {
        node_id: node.identity().id(),
        node_generation: node.identity().generation(),
        text: std::sync::Arc::<str>::from(node.text().unwrap_or_default()),
        style: std::sync::Arc::new(node.resolved_style().text().clone()),
        scale_generation: viewport.generation().raw(),
        scale_factor: viewport.scale_factor(),
        text_layout: None,
    }
}

fn measure_content(
    known_dimensions: TaffySize<Option<f32>>,
    available_space: TaffySize<AvailableSpace>,
    _node_id: NodeId,
    node_context: Option<&mut MeasureContext>,
    _style: &TaffyStyle,
    text_session: &mut TextMeasureSession<'_>,
    text_measure_error: &mut Option<NekoError>,
) -> TaffySize<f32> {
    if let TaffySize {
        width: Some(width),
        height: Some(height),
    } = known_dimensions
    {
        return TaffySize { width, height };
    }

    match node_context {
        Some(MeasureContext::Text {
            node_id,
            node_generation,
            text,
            style,
            scale_generation,
            scale_factor,
            text_layout,
        }) => {
            let available_inline_width = match available_space.width {
                AvailableSpace::Definite(available) => Some(available),
                AvailableSpace::MinContent | AvailableSpace::MaxContent => None,
            };
            let font_generation = text_session.font_generation();
            let query = TextMeasureQuery {
                node_id: *node_id,
                node_generation: *node_generation,
                text_generation: TextGeneration::INITIAL,
                style_generation: TextGeneration::INITIAL,
                text,
                style,
                available_inline_width,
                font_generation,
                scale_generation: *scale_generation,
                scale_factor: *scale_factor,
            };
            match text_session.layout(query) {
                TextLayoutResult::Ready(layout) => {
                    let metrics = layout.metrics();
                    *text_layout = Some(layout.clone());
                    let width =
                        known_dimensions
                            .width
                            .unwrap_or_else(|| match available_space.width {
                                AvailableSpace::Definite(available) => metrics.width.min(available),
                                AvailableSpace::MinContent => metrics.min_content_width,
                                AvailableSpace::MaxContent => metrics.max_content_width,
                            });
                    let height = known_dimensions.height.unwrap_or(metrics.height);
                    TaffySize { width, height }
                }
                TextLayoutResult::Deferred(dependency) => {
                    text_measure_error.get_or_insert_with(|| {
                        NekoError::diagnostic(format!(
                            "text measurement deferred during synchronous layout: {} ({})",
                            dependency.kind().as_str(),
                            dependency.reason()
                        ))
                    });
                    TaffySize {
                        width: known_dimensions.width.unwrap_or(0.0),
                        height: known_dimensions.height.unwrap_or(0.0),
                    }
                }
                TextLayoutResult::Failed(error) => {
                    text_measure_error.get_or_insert_with(|| {
                        NekoError::diagnostic(format!(
                            "text measurement failed during synchronous layout: {} ({})",
                            error.kind().as_str(),
                            error.message()
                        ))
                    });
                    TaffySize {
                        width: known_dimensions.width.unwrap_or(0.0),
                        height: known_dimensions.height.unwrap_or(0.0),
                    }
                }
            }
        }
        None => TaffySize {
            width: known_dimensions.width.unwrap_or(0.0),
            height: known_dimensions.height.unwrap_or(0.0),
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LayoutOrigin {
    x: f32,
    y: f32,
}

impl LayoutOrigin {
    const ZERO: Self = Self { x: 0.0, y: 0.0 };
}

fn materialize_node(
    tree: &TaffyTree<MeasureContext>,
    built: &BuiltNode,
    parent_origin: LayoutOrigin,
) -> NekoResult<LayoutNodeSnapshot> {
    let layout = tree
        .layout(built.taffy_id)
        .map_err(|error| NekoError::diagnostic(format!("layout result missing: {error}")))?;
    let border_x = parent_origin.x + layout.location.x;
    let border_y = parent_origin.y + layout.location.y;
    let border_rect = LayoutRect::new(border_x, border_y, layout.size.width, layout.size.height);
    let padding_rect = LayoutRect::new(
        border_x + layout.border.left,
        border_y + layout.border.top,
        (layout.size.width - layout.border.left - layout.border.right).max(0.0),
        (layout.size.height - layout.border.top - layout.border.bottom).max(0.0),
    );
    let content_rect = LayoutRect::new(
        border_x + layout.border.left + layout.padding.left,
        border_y + layout.border.top + layout.padding.top,
        (layout.size.width
            - layout.border.left
            - layout.border.right
            - layout.padding.left
            - layout.padding.right)
            .max(0.0),
        (layout.size.height
            - layout.border.top
            - layout.border.bottom
            - layout.padding.top
            - layout.padding.bottom)
            .max(0.0),
    );
    let margin_rect = LayoutRect::new(
        border_x - layout.margin.left,
        border_y - layout.margin.top,
        layout.size.width + layout.margin.left + layout.margin.right,
        layout.size.height + layout.margin.top + layout.margin.bottom,
    );
    let children = built
        .children
        .iter()
        .map(|child| {
            materialize_node(
                tree,
                child,
                LayoutOrigin {
                    x: border_x,
                    y: border_y,
                },
            )
        })
        .collect::<NekoResult<Vec<_>>>()?;
    let content_size = LayoutSize::new(layout.content_size.width, layout.content_size.height);
    let scroll = ScrollGeometry::new(
        built.overflow,
        content_rect,
        conservative_content_extent(content_rect, content_size, &children),
    );
    let text_layout = tree
        .get_node_context(built.taffy_id)
        .and_then(|context| match context {
            MeasureContext::Text { text_layout, .. } => text_layout.clone(),
        });

    Ok(LayoutNodeSnapshot::new(
        built.identity.id(),
        built.kind,
        built.key.clone(),
        LayoutBoxes {
            margin_rect,
            border_rect,
            padding_rect,
            content_rect,
            content_size,
            scroll,
            text_layout,
        },
        children,
    ))
}

fn conservative_content_extent(
    viewport: LayoutRect,
    content_size: LayoutSize,
    children: &[LayoutNodeSnapshot],
) -> LayoutSize {
    let mut right = viewport.width().max(content_size.width());
    let mut bottom = viewport.height().max(content_size.height());
    for child in children {
        let rect = child.margin_rect();
        right = right.max(rect.x() + rect.width() - viewport.x());
        bottom = bottom.max(rect.y() + rect.height() - viewport.y());
    }
    LayoutSize::new(right.max(0.0), bottom.max(0.0))
}

fn count_snapshot(node: &LayoutNodeSnapshot) -> usize {
    1 + node.children().iter().map(count_snapshot).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use crate::element::{IntoElement, div, text};
    use crate::error::ErrorKind;
    use crate::layout::{LayoutSize, Viewport, compute_layout};
    use crate::retained::RetainedTree;
    use crate::style::{Display, StyleExt, px};
    use crate::text::FontManager;

    #[test]
    fn display_none_subtrees_are_not_sent_to_taffy_output() {
        let mut tree = RetainedTree::default();
        tree.diff_root(
            div()
                .key("root")
                .child(text("visible").key("visible"))
                .child(text("hidden").key("hidden").display(Display::None))
                .into_element(),
        );

        let output = compute_layout(
            tree.layout_input(),
            Viewport::new(LayoutSize::new(200.0, 100.0), 1.0),
            None,
            &FontManager::default(),
        )
        .unwrap();

        assert_eq!(output.snapshot.node_count(), 2);
        assert!(output.snapshot.find_by_key("visible").is_some());
        assert!(output.snapshot.find_by_key("hidden").is_none());
    }

    #[test]
    fn border_box_width_is_not_expanded_by_padding() {
        let mut tree = RetainedTree::default();
        tree.diff_root(div().key("root").w(px(120.0)).p(px(10.0)).into_element());

        let output = compute_layout(
            tree.layout_input(),
            Viewport::new(LayoutSize::new(300.0, 100.0), 1.0),
            None,
            &FontManager::default(),
        )
        .unwrap();
        let root = output.snapshot.root().unwrap();

        assert_eq!(root.border_rect().width(), 120.0);
        assert_eq!(root.content_rect().width(), 100.0);
    }

    #[test]
    fn invalid_length_is_rejected_before_taffy() {
        let mut tree = RetainedTree::default();
        tree.diff_root(div().key("root").w(px(f32::INFINITY)).into_element());

        let error = compute_layout(
            tree.layout_input(),
            Viewport::new(LayoutSize::new(300.0, 100.0), 1.0),
            None,
            &FontManager::default(),
        )
        .unwrap_err();

        assert_eq!(error.error().kind(), ErrorKind::InvalidInput);
    }
}
