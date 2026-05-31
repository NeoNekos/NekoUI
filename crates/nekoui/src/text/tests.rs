use std::collections::BTreeSet;
use std::sync::Arc;

use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::style::{ResolvedStyle, StyleDeclaration, px};
use crate::text::measure::{
    TEXT_MEASURE_SAMPLE_LIMIT, TextMeasureDependency, TextMeasureDependencyKind, TextMeasureError,
    TextMeasureErrorKind,
};
use crate::text::{
    FontManager, TextGeneration, TextLayoutMode, TextLayoutResult, TextMeasureQuery,
    TextMeasureResult, TextMeasureSession,
};

#[test]
fn measurement_uses_shaped_layout_not_scalar_count_estimates() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(10.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let result = session.measure(TextMeasureQuery {
        node_id: RetainedNodeId::new(1),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "👨‍👩‍👧‍👦a",
        style: style.text(),
        available_inline_width: None,
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    });

    match result {
        TextMeasureResult::Ready(metrics) => {
            assert!(metrics.width > 10.0);
            assert_eq!(metrics.width, metrics.max_content_width);
            assert!(metrics.min_content_width > 0.0);
            assert!(metrics.min_content_width <= metrics.max_content_width);
            assert_eq!(metrics.line_count, 1);
        }
        TextMeasureResult::Deferred(_) | TextMeasureResult::Failed(_) => unreachable!(),
    }
}

#[test]
fn layout_uses_shaped_glyph_positions_and_stable_generation() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(18.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let query = TextMeasureQuery {
        node_id: RetainedNodeId::new(31),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "AV",
        style: style.text(),
        available_inline_width: None,
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    };

    let first = match session.layout(query.clone()) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };
    let second = match session.layout(query) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };

    assert_eq!(first.generation(), second.generation());
    assert_eq!(first.key(), second.key());
    assert!(first.glyph_demands().len() <= first.glyphs().len());
    assert!(!first.glyphs().is_empty());
    assert!(
        first
            .glyphs()
            .iter()
            .any(|glyph| glyph.x() != 0 || glyph.y() != 0)
    );
    assert_eq!(session.stats().cache_hits, 1);
}

#[test]
fn layout_deduplicates_glyph_demands_by_key_while_preserving_instances() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(18.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let layout = match session.layout(TextMeasureQuery {
        node_id: RetainedNodeId::new(32),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "AAAAAA",
        style: style.text(),
        available_inline_width: None,
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    }) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };

    let glyph_keys = layout
        .glyphs()
        .iter()
        .copied()
        .map(|glyph| glyph.key())
        .collect::<BTreeSet<_>>();
    let demand_keys = layout
        .glyph_demands()
        .iter()
        .copied()
        .map(|demand| demand.key())
        .collect::<BTreeSet<_>>();

    assert_eq!(demand_keys, glyph_keys);
    assert_eq!(layout.glyph_demands().len(), demand_keys.len());
    assert!(layout.glyphs().len() >= layout.glyph_demands().len());
}

#[test]
fn scale_factor_participates_in_layout_key_and_glyph_demands() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(18.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let query = TextMeasureQuery {
        node_id: RetainedNodeId::new(33),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "AA",
        style: style.text(),
        available_inline_width: None,
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    };

    let scale_one = match session.layout(query.clone()) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };
    let scale_two = match session.layout(TextMeasureQuery {
        scale_factor: 2.0,
        ..query
    }) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };

    let scale_one_keys = scale_one
        .glyph_demands()
        .iter()
        .copied()
        .map(|demand| demand.key())
        .collect::<BTreeSet<_>>();
    let scale_two_keys = scale_two
        .glyph_demands()
        .iter()
        .copied()
        .map(|demand| demand.key())
        .collect::<BTreeSet<_>>();

    assert_ne!(scale_one.key(), scale_two.key());
    assert_eq!(scale_one.scale_factor(), 1.0);
    assert_eq!(scale_two.scale_factor(), 2.0);
    assert!(!scale_one_keys.is_empty());
    assert_ne!(scale_one_keys, scale_two_keys);
    assert_eq!(session.stats().cache_misses, 2);
}

#[test]
fn pass_local_memo_hits_repeated_queries() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);

    for _ in 0..2 {
        let font_generation = session.font_generation();
        session.measure(TextMeasureQuery {
            node_id: RetainedNodeId::new(1),
            node_generation: NodeGeneration::INITIAL,
            text_generation: TextGeneration::INITIAL,
            style_generation: TextGeneration::INITIAL,
            text: "memo",
            style: style.text(),
            available_inline_width: Some(50.0),
            layout_mode: TextLayoutMode::SoftWrap,
            font_generation,
            scale_generation: 1,
            scale_factor: 1.0,
        });
    }

    let stats = session.stats();
    assert_eq!(stats.query_count, 2);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.events.len(), 2);
}

#[test]
fn font_generation_participates_in_memo_key() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let mut font_manager = FontManager::default();
    font_manager.bump_generation_for_test();
    let mut session = TextMeasureSession::new(&font_manager);
    let generation = session.font_generation();

    session.measure(TextMeasureQuery {
        node_id: RetainedNodeId::new(7),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "font",
        style: style.text(),
        available_inline_width: None,
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: generation,
        scale_generation: 1,
        scale_factor: 1.0,
    });

    assert_eq!(generation.raw(), 2);
    assert_eq!(session.stats().cache_misses, 1);
}

#[test]
fn non_ready_results_carry_private_payloads() {
    let deferred = TextMeasureResult::Deferred(TextMeasureDependency::new(
        TextMeasureDependencyKind::FontFallbackUnavailable,
        "font load",
    ));
    let failed = TextMeasureResult::Failed(TextMeasureError::new(
        TextMeasureErrorKind::ShapingFailed,
        "font failure",
    ));

    match deferred {
        TextMeasureResult::Deferred(dependency) => {
            assert_eq!(
                dependency.kind(),
                TextMeasureDependencyKind::FontFallbackUnavailable
            );
            assert_eq!(dependency.reason(), "font load");
        }
        TextMeasureResult::Ready(_) | TextMeasureResult::Failed(_) => unreachable!(),
    }
    match failed {
        TextMeasureResult::Failed(error) => {
            assert_eq!(error.kind(), TextMeasureErrorKind::ShapingFailed);
            assert_eq!(error.message(), "font failure");
        }
        TextMeasureResult::Ready(_) | TextMeasureResult::Deferred(_) => unreachable!(),
    }
}

#[test]
fn stale_font_generation_defers_and_counts_blocker() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);

    let result = session.measure(TextMeasureQuery {
        node_id: RetainedNodeId::new(3),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "defer",
        style: style.text(),
        available_inline_width: None,
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: crate::text::FontGeneration::INITIAL.next(),
        scale_generation: 1,
        scale_factor: 1.0,
    });

    match result {
        TextMeasureResult::Deferred(dependency) => {
            assert_eq!(
                dependency.kind(),
                TextMeasureDependencyKind::FontGenerationStale
            );
        }
        TextMeasureResult::Ready(_) | TextMeasureResult::Failed(_) => unreachable!(),
    }
    let stats = session.stats();
    assert_eq!(stats.deferred_count, 1);
    assert_eq!(stats.blocked_count, 1);
    assert_eq!(stats.blockers[0].node_id, RetainedNodeId::new(3));
}

#[test]
fn memo_key_tracks_policy_dimensions() {
    let base = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let clipped = ResolvedStyle::resolve(
        &StyleDeclaration::default()
            .font_size(px(12.0))
            .line_clamp(1),
        None,
    );
    let larger = ResolvedStyle::resolve(
        &StyleDeclaration::default()
            .font_size(px(14.0))
            .line_clamp(1),
        None,
    );
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);

    for style in [base.text(), clipped.text(), larger.text()] {
        session.measure(TextMeasureQuery {
            node_id: RetainedNodeId::new(8),
            node_generation: NodeGeneration::INITIAL,
            text_generation: TextGeneration::INITIAL,
            style_generation: TextGeneration::INITIAL,
            text: "policy",
            style,
            available_inline_width: Some(18.0),
            layout_mode: TextLayoutMode::SoftWrap,
            font_generation: session.font_generation(),
            scale_generation: 1,
            scale_factor: 1.0,
        });
    }

    assert_eq!(session.stats().cache_misses, 3);
}

#[test]
fn trailing_caret_rect_uses_final_visible_shaped_line() {
    let style = ResolvedStyle::resolve(
        &StyleDeclaration::default()
            .font_size(px(12.0))
            .line_clamp(2),
        None,
    );
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let layout = match session.layout(TextMeasureQuery {
        node_id: RetainedNodeId::new(41),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "AAAA AAAA A",
        style: style.text(),
        available_inline_width: Some(36.0),
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    }) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };
    let caret = layout.trailing_caret_rect();
    let metrics = layout.metrics();

    assert_eq!(metrics.line_count, 2);
    assert!(caret.y() > 0.0);
    assert!(caret.height() < metrics.height);
    assert!(caret.x() <= metrics.width);
}

#[test]
fn single_line_input_mode_does_not_soft_wrap_at_available_width() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let available_inline_width = 36.0;
    let expected_line_height = style.text().font_size().as_px() * 1.2;
    let layout = match session.layout(TextMeasureQuery {
        node_id: RetainedNodeId::new(42),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "AAAA AAAA AAAA AAAA",
        style: style.text(),
        available_inline_width: Some(available_inline_width),
        layout_mode: TextLayoutMode::SingleLineInput,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    }) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };
    let metrics = layout.metrics();

    assert_eq!(metrics.line_count, 1);
    assert!((metrics.height - expected_line_height).abs() < 0.01);
    assert!(layout.trailing_caret_rect().x() > available_inline_width);
    assert_eq!(layout.trailing_caret_rect().y(), 0.0);
    assert_eq!(layout.key().layout_mode, TextLayoutMode::SingleLineInput);
}

#[test]
fn soft_wrap_text_mode_still_wraps_at_available_width() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);
    let expected_line_height = style.text().font_size().as_px() * 1.2;
    let layout = match session.layout(TextMeasureQuery {
        node_id: RetainedNodeId::new(43),
        node_generation: NodeGeneration::INITIAL,
        text_generation: TextGeneration::INITIAL,
        style_generation: TextGeneration::INITIAL,
        text: "AAAA AAAA AAAA AAAA",
        style: style.text(),
        available_inline_width: Some(36.0),
        layout_mode: TextLayoutMode::SoftWrap,
        font_generation: session.font_generation(),
        scale_generation: 1,
        scale_factor: 1.0,
    }) {
        TextLayoutResult::Ready(layout) => layout,
        TextLayoutResult::Deferred(_) | TextLayoutResult::Failed(_) => unreachable!(),
    };
    let metrics = layout.metrics();

    assert!(metrics.line_count > 1);
    assert!(metrics.height > expected_line_height);
    assert!(layout.trailing_caret_rect().y() > 0.0);
    assert_eq!(layout.key().layout_mode, TextLayoutMode::SoftWrap);
}

#[test]
fn font_snapshot_is_generation_stamped_and_carries_blob_metadata() {
    let mut font_manager = FontManager::default();
    let initial = font_manager.fallback_snapshot();
    font_manager.install_test_blob(crate::text::font::FontBlobRef::new_for_test(
        10,
        Arc::from([1_u8, 2, 3]),
        2,
    ));
    let updated = font_manager.fallback_snapshot();

    assert_ne!(initial.generation(), updated.generation());
    assert_eq!(updated.metadata_count(), 1);
    assert_eq!(updated.blob_count(), 1);
}

#[test]
fn ready_event_samples_are_bounded_but_counters_are_exact() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);

    for _ in 0..(TEXT_MEASURE_SAMPLE_LIMIT + 17) {
        session.measure(TextMeasureQuery {
            node_id: RetainedNodeId::new(11),
            node_generation: NodeGeneration::INITIAL,
            text_generation: TextGeneration::INITIAL,
            style_generation: TextGeneration::INITIAL,
            text: "bounded ready samples",
            style: style.text(),
            available_inline_width: Some(80.0),
            layout_mode: TextLayoutMode::SoftWrap,
            font_generation: session.font_generation(),
            scale_generation: 1,
            scale_factor: 1.0,
        });
    }

    let stats = session.stats();
    assert_eq!(stats.query_count, (TEXT_MEASURE_SAMPLE_LIMIT + 17) as u64);
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, (TEXT_MEASURE_SAMPLE_LIMIT + 16) as u64);
    assert_eq!(stats.events.len(), TEXT_MEASURE_SAMPLE_LIMIT);
    assert_eq!(stats.blockers.len(), 0);
    assert!(stats.total_duration.as_nanos() > 0);
}

#[test]
fn blocker_samples_are_bounded_but_deferred_counters_are_exact() {
    let style = ResolvedStyle::resolve(&StyleDeclaration::default().font_size(px(12.0)), None);
    let font_manager = FontManager::default();
    let mut session = TextMeasureSession::new(&font_manager);

    for index in 0..(TEXT_MEASURE_SAMPLE_LIMIT + 9) {
        session.measure(TextMeasureQuery {
            node_id: RetainedNodeId::new(index as u64 + 20),
            node_generation: NodeGeneration::INITIAL,
            text_generation: TextGeneration::INITIAL,
            style_generation: TextGeneration::INITIAL,
            text: "bounded deferred samples",
            style: style.text(),
            available_inline_width: None,
            layout_mode: TextLayoutMode::SoftWrap,
            font_generation: crate::text::FontGeneration::INITIAL.next(),
            scale_generation: 1,
            scale_factor: 1.0,
        });
    }

    let stats = session.stats();
    assert_eq!(stats.query_count, (TEXT_MEASURE_SAMPLE_LIMIT + 9) as u64);
    assert_eq!(stats.deferred_count, (TEXT_MEASURE_SAMPLE_LIMIT + 9) as u64);
    assert_eq!(stats.blocked_count, (TEXT_MEASURE_SAMPLE_LIMIT + 9) as u64);
    assert_eq!(stats.events.len(), TEXT_MEASURE_SAMPLE_LIMIT);
    assert_eq!(stats.blockers.len(), TEXT_MEASURE_SAMPLE_LIMIT);
    assert_eq!(stats.cache_hits, 0);
    assert_eq!(stats.cache_misses, 0);
    assert!(stats.total_duration.as_nanos() > 0);
}
