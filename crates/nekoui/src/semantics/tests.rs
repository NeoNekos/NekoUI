use crate::diagnostic::DiagnosticArea;
use crate::diagnostic::signal::SignalId;
use crate::element::{Element, IntoElement, div, text};
use crate::interaction::{InteractionState, InteractionTarget};
use crate::layout::{LayoutPoint, LayoutRect, LayoutSize, Viewport, compute_layout};
use crate::retained::RetainedTree;
use crate::semantics::generation::SemanticSignatureFact;
use crate::semantics::snapshot::{SemanticAction, SemanticRole};
use crate::semantics::{
    SemanticBuildInput, build_semantic_snapshot, semantic_generation_for_inputs,
    semantic_publish_is_current,
};
use crate::style::{Color, Display, Opacity, Overflow, StyleExt, px};
use crate::text::FontManager;

fn semantic_snapshot_for(root: impl IntoElement) -> crate::semantics::SemanticTreeSnapshot {
    semantic_output_for(root).snapshot
}

fn semantic_output_for(root: impl IntoElement) -> crate::semantics::build::SemanticBuildOutput {
    semantic_output_for_element(root.into_element(), None)
}

fn semantic_output_for_element(
    root: Element,
    interaction: Option<&InteractionState>,
) -> crate::semantics::build::SemanticBuildOutput {
    let mut tree = RetainedTree::default();
    tree.diff_root(root);
    let retained = tree.snapshot();
    let style = tree.style_snapshot();
    let layout = compute_layout(
        tree.layout_input(),
        Viewport::new(LayoutSize::new(300.0, 200.0), 1.0),
        None,
        &FontManager::default(),
    )
    .unwrap()
    .snapshot;

    build_semantic_snapshot(SemanticBuildInput {
        retained: &retained,
        style: &style,
        layout: &layout,
        interaction,
    })
}

#[test]
fn text_nodes_use_text_role_name_and_value() {
    let snapshot = semantic_snapshot_for(div().key("root").child(text("Hello").key("label")));
    let root = snapshot.root().unwrap();
    let label = snapshot.find_by_key("label").unwrap();

    assert_eq!(snapshot.node_count(), 2);
    assert_eq!(snapshot.stats().node_count, 2);
    assert_eq!(root.role(), SemanticRole::Window);
    assert_eq!(root.key().unwrap().as_str(), "root");
    assert_eq!(root.children().len(), 1);
    assert_eq!(
        label.id().retained_id().raw(),
        root.children()[0].id().retained_id().raw()
    );
    assert_eq!(label.id().retained_generation().raw(), 1);
    assert_eq!(label.role(), SemanticRole::Text);
    assert_eq!(label.name(), Some("Hello"));
    assert_eq!(label.value(), Some("Hello"));
}

#[test]
fn display_none_excludes_semantic_subtree() {
    let snapshot = semantic_snapshot_for(
        div()
            .key("root")
            .child(text("visible").key("visible"))
            .child(
                div()
                    .key("hidden")
                    .display(Display::None)
                    .child(text("hidden")),
            ),
    );

    assert!(snapshot.find_by_key("visible").is_some());
    assert!(snapshot.find_by_key("hidden").is_none());
    assert_eq!(snapshot.node_count(), 2);
}

#[test]
fn opacity_zero_preserves_semantic_nodes() {
    let snapshot = semantic_snapshot_for(
        div()
            .key("root")
            .opacity(Opacity::new(0.0))
            .child(text("still semantic").key("label")),
    );

    assert!(snapshot.find_by_key("label").is_some());
    assert_eq!(snapshot.node_count(), 2);
}

#[test]
fn clickable_div_remains_generic_and_records_missing_name() {
    let output = semantic_output_for(
        div().key("root").child(
            div()
                .key("clickable")
                .w(px(10.0))
                .h(px(10.0))
                .on_click(|_| ()),
        ),
    );
    let clickable = output.snapshot.find_by_key("clickable").unwrap();

    assert_eq!(clickable.role(), SemanticRole::Generic);
    assert_eq!(clickable.name(), None);
    assert!(clickable.actions().contains(&SemanticAction::Activate));
    assert!(output.records.iter().any(|record| {
        record.area == DiagnosticArea::Semantics
            && record.operation == "semantics.diagnostic"
            && record
                .fields
                .get("reason")
                .is_some_and(|value| value == "missing_name")
    }));
}

#[test]
fn focusable_div_records_missing_name_and_focus_state() {
    let mut tree = RetainedTree::default();
    tree.diff_root(
        div()
            .key("root")
            .child(
                div()
                    .key("focusable")
                    .focusable(true)
                    .w(px(10.0))
                    .h(px(10.0)),
            )
            .into_element(),
    );
    let retained = tree.snapshot();
    let style = tree.style_snapshot();
    let layout = compute_layout(
        tree.layout_input(),
        Viewport::new(LayoutSize::new(300.0, 200.0), 1.0),
        None,
        &FontManager::default(),
    )
    .unwrap()
    .snapshot;
    let focusable = retained.find_by_key("focusable").unwrap();
    let mut interaction = InteractionState::default();
    interaction.set_window_focused(true);
    interaction.set_keyboard_focus(Some(InteractionTarget::new(
        focusable.id(),
        focusable.generation(),
    )));

    let output = build_semantic_snapshot(SemanticBuildInput {
        retained: &retained,
        style: &style,
        layout: &layout,
        interaction: Some(&interaction),
    });
    let node = output.snapshot.find_by_key("focusable").unwrap();

    assert!(node.state().focusable());
    assert!(node.state().focused());
    assert!(node.state().window_focused());
    assert!(node.actions().contains(&SemanticAction::Focus));
    assert_eq!(output.stats.diagnostic_count, 1);
}

#[test]
fn bounds_are_layout_derived_not_paint_derived() {
    let snapshot = semantic_snapshot_for(
        div()
            .key("root")
            .bg(Color::rgb(1, 2, 3))
            .p(px(10.0))
            .h(px(200.0))
            .child(text("Hello").key("label")),
    );
    let root = snapshot.root().unwrap();
    let label = snapshot.find_by_key("label").unwrap();

    assert_eq!(root.bounds(), LayoutRect::new(0.0, 0.0, 300.0, 200.0));
    assert_eq!(label.bounds().x(), 10.0);
    assert_eq!(label.bounds().y(), 10.0);
}

#[test]
fn scroll_offsets_map_child_coordinates() {
    let mut tree = RetainedTree::default();
    tree.diff_root(
        div()
            .key("root")
            .w(px(100.0))
            .h(px(100.0))
            .overflow(Overflow::Scroll)
            .child(text("top").key("top").h(px(100.0)))
            .child(text("content").key("content").h(px(100.0)))
            .into_element(),
    );
    let retained = tree.snapshot();
    let style = tree.style_snapshot();
    let layout = compute_layout(
        tree.layout_input(),
        Viewport::new(LayoutSize::new(300.0, 200.0), 1.0),
        None,
        &FontManager::default(),
    )
    .unwrap()
    .snapshot;
    let root = retained.find_by_key("root").unwrap();
    let mut interaction = InteractionState::default();
    interaction.set_scroll_offset(
        InteractionTarget::new(root.id(), root.generation()),
        LayoutPoint::new(0.0, 40.0),
    );

    let snapshot = build_semantic_snapshot(SemanticBuildInput {
        retained: &retained,
        style: &style,
        layout: &layout,
        interaction: Some(&interaction),
    })
    .snapshot;
    let content = snapshot.find_by_key("content").unwrap();

    assert_eq!(content.bounds().y(), 60.0);
    assert!(snapshot.find_by_key("root").unwrap().state().scrollable());
}

#[test]
fn nested_scroll_offsets_accumulate_for_semantic_bounds() {
    let mut tree = RetainedTree::default();
    tree.diff_root(
        div()
            .key("outer")
            .w(px(100.0))
            .h(px(100.0))
            .overflow(Overflow::Scroll)
            .child(text("top").key("top").h(px(40.0)))
            .child(
                div()
                    .key("inner")
                    .w(px(120.0))
                    .h(px(60.0))
                    .overflow(Overflow::Scroll)
                    .child(text("spacer").key("spacer").w(px(40.0)).h(px(30.0)))
                    .child(text("target").key("target").w(px(160.0)).h(px(100.0))),
            )
            .into_element(),
    );
    let retained = tree.snapshot();
    let style = tree.style_snapshot();
    let layout = compute_layout(
        tree.layout_input(),
        Viewport::new(LayoutSize::new(300.0, 200.0), 1.0),
        None,
        &FontManager::default(),
    )
    .unwrap()
    .snapshot;
    let outer = retained.find_by_key("outer").unwrap();
    let inner = retained.find_by_key("inner").unwrap();
    let target_layout = layout.find_by_key("target").unwrap();
    let mut interaction = InteractionState::default();
    interaction.set_scroll_offset(
        InteractionTarget::new(outer.id(), outer.generation()),
        LayoutPoint::new(0.0, 30.0),
    );
    interaction.set_scroll_offset(
        InteractionTarget::new(inner.id(), inner.generation()),
        LayoutPoint::new(12.0, 8.0),
    );

    let snapshot = build_semantic_snapshot(SemanticBuildInput {
        retained: &retained,
        style: &style,
        layout: &layout,
        interaction: Some(&interaction),
    })
    .snapshot;
    let target = snapshot.find_by_key("target").unwrap();

    assert_eq!(
        target.bounds(),
        target_layout.content_rect().translate(-12.0, -38.0)
    );
    assert!(snapshot.find_by_key("outer").unwrap().state().scrollable());
    assert!(snapshot.find_by_key("inner").unwrap().state().scrollable());
    assert!(
        snapshot
            .find_by_key("outer")
            .unwrap()
            .actions()
            .contains(&SemanticAction::Scroll)
    );
    assert!(
        snapshot
            .find_by_key("inner")
            .unwrap()
            .actions()
            .contains(&SemanticAction::Scroll)
    );
}

#[test]
fn generation_publish_validation_detects_stale_inputs() {
    let mut tree = RetainedTree::default();
    tree.diff_root(
        div()
            .key("root")
            .child(text("a").key("label"))
            .into_element(),
    );
    let retained = tree.snapshot();
    let style = tree.style_snapshot();
    let layout = compute_layout(
        tree.layout_input(),
        Viewport::new(LayoutSize::new(300.0, 200.0), 1.0),
        None,
        &FontManager::default(),
    )
    .unwrap()
    .snapshot;
    let expected = semantic_generation_for_inputs(&retained, &style, &layout);

    tree.diff_root(
        div()
            .key("root")
            .child(text("b").key("label"))
            .into_element(),
    );
    let current_retained = tree.snapshot();
    let current_style = tree.style_snapshot();
    let current_layout = compute_layout(
        tree.layout_input(),
        Viewport::new(LayoutSize::new(300.0, 200.0), 1.0),
        Some(&layout),
        &FontManager::default(),
    )
    .unwrap()
    .snapshot;

    assert!(!semantic_publish_is_current(
        &expected,
        &expected,
        &current_retained,
        &current_style,
        &current_layout,
    ));
}

#[test]
fn generation_signature_records_semantic_facts() {
    let snapshot = semantic_snapshot_for(div().key("root").child(text("Hello").key("label")));
    let generation = snapshot.generation();

    assert!(generation.retained_generation().is_some());
    assert!(generation.layout_generation().is_some());
    assert!(generation.viewport_generation() > 0);
    assert!(
        generation.semantic_signature().facts().iter().any(|fact| {
            matches!(fact, SemanticSignatureFact::TextValue { len, .. } if *len == 5)
        })
    );
    assert!(generation.style_signature().facts().iter().any(|fact| {
        matches!(
            fact,
            SemanticSignatureFact::Participation { semantics: true }
        )
    }));
    assert!(generation.interaction_signature().facts().is_empty());
}

#[test]
fn semantic_signal_names_are_stable() {
    assert_eq!(SignalId::SemanticsBuild.name(), "semantics.build");
    assert_eq!(SignalId::SemanticsNodeCount.name(), "semantics.node_count");
    assert_eq!(SignalId::SemanticsDiagnostic.name(), "semantics.diagnostic");
    assert_eq!(SignalId::SemanticsStaleDrop.name(), "semantics.stale_drop");
    assert_eq!(
        SignalId::SemanticsDurationMicros.name(),
        "semantics.duration_micros"
    );
}
