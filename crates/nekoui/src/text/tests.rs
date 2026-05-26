use std::sync::Arc;

use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::style::{ResolvedStyle, StyleDeclaration, px};
use crate::text::measure::{
    TEXT_MEASURE_SAMPLE_LIMIT, TextMeasureDependency, TextMeasureDependencyKind, TextMeasureError,
    TextMeasureErrorKind,
};
use crate::text::{
    FontManager, TextGeneration, TextMeasureQuery, TextMeasureResult, TextMeasureSession,
};

#[test]
fn measurement_uses_grapheme_clusters_not_scalar_count() {
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
        font_generation: session.font_generation(),
        scale_generation: 1,
    });

    match result {
        TextMeasureResult::Ready(metrics) => {
            assert_eq!(metrics.width, 10.0);
            assert_eq!(metrics.min_content_width, 5.0);
            assert_eq!(metrics.max_content_width, 10.0);
            assert_eq!(metrics.line_count, 1);
        }
        TextMeasureResult::Deferred(_) | TextMeasureResult::Failed(_) => unreachable!(),
    }
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
            font_generation,
            scale_generation: 1,
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
        font_generation: generation,
        scale_generation: 1,
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
        font_generation: crate::text::FontGeneration::INITIAL.next(),
        scale_generation: 1,
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
            font_generation: session.font_generation(),
            scale_generation: 1,
        });
    }

    assert_eq!(session.stats().cache_misses, 3);
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
            font_generation: session.font_generation(),
            scale_generation: 1,
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
            font_generation: crate::text::FontGeneration::INITIAL.next(),
            scale_generation: 1,
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
