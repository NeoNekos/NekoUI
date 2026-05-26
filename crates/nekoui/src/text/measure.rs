use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use unicode_segmentation::UnicodeSegmentation;

use crate::retained::{NodeGeneration, RetainedNodeId};
use crate::style::{ResolvedTextStyle, TextOverflow};
use crate::text::{FontFallbackSnapshot, FontGeneration, FontManager};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) struct TextGeneration(u64);

impl TextGeneration {
    pub(crate) const INITIAL: Self = Self(1);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextMeasureDependencyKind {
    FontGenerationStale,
    FontFallbackUnavailable,
    ShapingUnavailable,
}

impl TextMeasureDependencyKind {
    const ALL: [Self; 3] = [
        Self::FontGenerationStale,
        Self::FontFallbackUnavailable,
        Self::ShapingUnavailable,
    ];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::FontGenerationStale => "font_generation_stale",
            Self::FontFallbackUnavailable => "font_fallback_unavailable",
            Self::ShapingUnavailable => "shaping_unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextMeasureErrorKind {
    InvalidStyle,
    ShapingFailed,
}

impl TextMeasureErrorKind {
    const ALL: [Self; 2] = [Self::InvalidStyle, Self::ShapingFailed];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidStyle => "invalid_style",
            Self::ShapingFailed => "shaping_failed",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextWrapPolicy {
    SoftWrap,
}

impl TextWrapPolicy {
    const DEFAULT: Self = Self::SoftWrap;

    const fn rank(self) -> u8 {
        match self {
            Self::SoftWrap => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextWhiteSpacePolicy {
    Normal,
}

impl TextWhiteSpacePolicy {
    const DEFAULT: Self = Self::Normal;

    const fn rank(self) -> u8 {
        match self {
            Self::Normal => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub(crate) enum TextLineHeightPolicy {
    Default,
}

impl TextLineHeightPolicy {
    const DEFAULT: Self = Self::Default;

    const fn rank(self) -> u8 {
        match self {
            Self::Default => 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TextMetrics {
    pub(crate) width: f32,
    pub(crate) min_content_width: f32,
    pub(crate) max_content_width: f32,
    pub(crate) height: f32,
    pub(crate) baseline: f32,
    pub(crate) line_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureDependency {
    kind: TextMeasureDependencyKind,
    reason: Cow<'static, str>,
}

impl TextMeasureDependency {
    pub(crate) fn new(
        kind: TextMeasureDependencyKind,
        reason: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }

    pub(crate) fn kind(&self) -> TextMeasureDependencyKind {
        self.kind
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureError {
    kind: TextMeasureErrorKind,
    message: Cow<'static, str>,
}

impl TextMeasureError {
    pub(crate) fn new(kind: TextMeasureErrorKind, message: impl Into<Cow<'static, str>>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn kind(&self) -> TextMeasureErrorKind {
        self.kind
    }

    pub(crate) fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TextMeasureResult {
    Ready(TextMetrics),
    Deferred(TextMeasureDependency),
    Failed(TextMeasureError),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TextMeasureQuery<'a> {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) node_generation: NodeGeneration,
    pub(crate) text_generation: TextGeneration,
    pub(crate) style_generation: TextGeneration,
    pub(crate) text: &'a str,
    pub(crate) style: &'a ResolvedTextStyle,
    pub(crate) available_inline_width: Option<f32>,
    pub(crate) font_generation: FontGeneration,
    pub(crate) scale_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureBlocker {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) result: &'static str,
    pub(crate) kind: &'static str,
    pub(crate) reason: Cow<'static, str>,
    pub(crate) duration: Duration,
}

pub(crate) const TEXT_MEASURE_SAMPLE_LIMIT: usize = 128;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextMeasureEvent {
    pub(crate) node_id: RetainedNodeId,
    pub(crate) result: &'static str,
    pub(crate) cache: &'static str,
    pub(crate) duration: Duration,
    pub(crate) line_count: Option<usize>,
    pub(crate) min_content_width_bits: Option<u32>,
    pub(crate) max_content_width_bits: Option<u32>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TextMeasureStats {
    pub(crate) query_count: u64,
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) measured_count: u64,
    pub(crate) deferred_count: u64,
    pub(crate) failed_count: u64,
    pub(crate) blocked_count: u64,
    pub(crate) total_duration: Duration,
    pub(crate) blockers: Vec<TextMeasureBlocker>,
    pub(crate) events: Vec<TextMeasureEvent>,
}

#[derive(Clone, Debug, PartialEq)]
struct MemoKey {
    node_id: RetainedNodeId,
    node_generation: NodeGeneration,
    text_generation: TextGeneration,
    style_generation: TextGeneration,
    available_inline_width_bits: Option<u32>,
    font_size_bits: u32,
    line_height_policy: TextLineHeightPolicy,
    wrap_policy: TextWrapPolicy,
    white_space_policy: TextWhiteSpacePolicy,
    max_lines: Option<usize>,
    text_overflow: TextOverflow,
    font_generation: FontGeneration,
    scale_generation: u64,
}

impl Eq for MemoKey {}

impl Ord for MemoKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.node_id
            .cmp(&other.node_id)
            .then_with(|| self.node_generation.cmp(&other.node_generation))
            .then_with(|| self.text_generation.cmp(&other.text_generation))
            .then_with(|| self.style_generation.cmp(&other.style_generation))
            .then_with(|| {
                self.available_inline_width_bits
                    .cmp(&other.available_inline_width_bits)
            })
            .then_with(|| self.font_size_bits.cmp(&other.font_size_bits))
            .then_with(|| {
                self.line_height_policy
                    .rank()
                    .cmp(&other.line_height_policy.rank())
            })
            .then_with(|| self.wrap_policy.rank().cmp(&other.wrap_policy.rank()))
            .then_with(|| {
                self.white_space_policy
                    .rank()
                    .cmp(&other.white_space_policy.rank())
            })
            .then_with(|| self.max_lines.cmp(&other.max_lines))
            .then_with(|| {
                text_overflow_rank(self.text_overflow).cmp(&text_overflow_rank(other.text_overflow))
            })
            .then_with(|| self.font_generation.cmp(&other.font_generation))
            .then_with(|| self.scale_generation.cmp(&other.scale_generation))
    }
}

impl PartialOrd for MemoKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub(crate) struct TextMeasureSession<'a> {
    font_manager: &'a FontManager,
    fallback_snapshot: FontFallbackSnapshot,
    memo: BTreeMap<MemoKey, TextMetrics>,
    stats: TextMeasureStats,
}

impl<'a> TextMeasureSession<'a> {
    pub(crate) fn new(font_manager: &'a FontManager) -> Self {
        let _ = TextMeasureDependencyKind::ALL;
        let _ = TextMeasureErrorKind::ALL;
        Self {
            font_manager,
            fallback_snapshot: font_manager.fallback_snapshot(),
            memo: BTreeMap::new(),
            stats: TextMeasureStats::default(),
        }
    }

    pub(crate) fn font_generation(&self) -> FontGeneration {
        self.fallback_snapshot.generation()
    }

    pub(crate) fn measure(&mut self, query: TextMeasureQuery<'_>) -> TextMeasureResult {
        let started = std::time::Instant::now();
        self.stats.query_count += 1;
        if query.font_generation != self.font_manager.generation()
            || query.font_generation != self.fallback_snapshot.generation()
        {
            let dependency = TextMeasureDependency::new(
                TextMeasureDependencyKind::FontGenerationStale,
                "font generation changed before measurement",
            );
            return self.record_deferred(query.node_id, dependency, started.elapsed());
        }
        if self.fallback_snapshot.metadata_count() == 0 {
            let dependency = TextMeasureDependency::new(
                TextMeasureDependencyKind::FontFallbackUnavailable,
                "font fallback snapshot has no entries",
            );
            return self.record_deferred(query.node_id, dependency, started.elapsed());
        }
        if !query.style.font_size().as_px().is_finite() {
            let error = TextMeasureError::new(
                TextMeasureErrorKind::InvalidStyle,
                "font size must be finite for text measurement",
            );
            return self.record_failed(query.node_id, error, started.elapsed());
        }
        let key = MemoKey::from_query(&query);
        if let Some(metrics) = self.memo.get(&key).copied() {
            let duration = started.elapsed();
            self.stats.cache_hits += 1;
            self.stats.total_duration += duration;
            self.record_event(query.node_id, "ready", "hit", duration, Some(metrics));
            return TextMeasureResult::Ready(metrics);
        }

        self.stats.cache_misses += 1;
        let metrics = measure_uncached(&query, &self.fallback_snapshot);
        self.memo.insert(key, metrics);
        self.stats.measured_count += 1;
        let duration = started.elapsed();
        self.stats.total_duration += duration;
        self.record_event(query.node_id, "ready", "miss", duration, Some(metrics));
        TextMeasureResult::Ready(metrics)
    }

    pub(crate) fn stats(&self) -> TextMeasureStats {
        self.stats.clone()
    }

    fn record_deferred(
        &mut self,
        node_id: RetainedNodeId,
        dependency: TextMeasureDependency,
        duration: Duration,
    ) -> TextMeasureResult {
        self.stats.deferred_count += 1;
        self.stats.blocked_count += 1;
        self.stats.total_duration += duration;
        self.record_blocker(TextMeasureBlocker {
            node_id,
            result: "deferred",
            kind: dependency.kind().as_str(),
            reason: Cow::Owned(dependency.reason().to_owned()),
            duration,
        });
        self.record_event(node_id, "deferred", "none", duration, None);
        TextMeasureResult::Deferred(dependency)
    }

    fn record_failed(
        &mut self,
        node_id: RetainedNodeId,
        error: TextMeasureError,
        duration: Duration,
    ) -> TextMeasureResult {
        self.stats.failed_count += 1;
        self.stats.blocked_count += 1;
        self.stats.total_duration += duration;
        self.record_blocker(TextMeasureBlocker {
            node_id,
            result: "failed",
            kind: error.kind().as_str(),
            reason: Cow::Owned(error.message().to_owned()),
            duration,
        });
        self.record_event(node_id, "failed", "none", duration, None);
        TextMeasureResult::Failed(error)
    }

    fn record_blocker(&mut self, blocker: TextMeasureBlocker) {
        if self.stats.blockers.len() < TEXT_MEASURE_SAMPLE_LIMIT {
            self.stats.blockers.push(blocker);
        }
    }

    fn record_event(
        &mut self,
        node_id: RetainedNodeId,
        result: &'static str,
        cache: &'static str,
        duration: Duration,
        metrics: Option<TextMetrics>,
    ) {
        if self.stats.events.len() >= TEXT_MEASURE_SAMPLE_LIMIT {
            return;
        }
        self.stats.events.push(TextMeasureEvent {
            node_id,
            result,
            cache,
            duration,
            line_count: metrics.map(|metrics| metrics.line_count),
            min_content_width_bits: metrics.map(|metrics| metrics.min_content_width.to_bits()),
            max_content_width_bits: metrics.map(|metrics| metrics.max_content_width.to_bits()),
        });
    }
}

impl MemoKey {
    fn from_query(query: &TextMeasureQuery<'_>) -> Self {
        Self {
            node_id: query.node_id,
            node_generation: query.node_generation,
            text_generation: query.text_generation,
            style_generation: query.style_generation,
            available_inline_width_bits: query.available_inline_width.map(f32::to_bits),
            font_size_bits: query.style.font_size().as_px().to_bits(),
            line_height_policy: TextLineHeightPolicy::DEFAULT,
            wrap_policy: TextWrapPolicy::DEFAULT,
            white_space_policy: TextWhiteSpacePolicy::DEFAULT,
            max_lines: query.style.max_lines(),
            text_overflow: query.style.text_overflow(),
            font_generation: query.font_generation,
            scale_generation: query.scale_generation,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct TextAnalysis {
    clusters: Arc<[TextCluster]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TextCluster {
    advance: f32,
    break_after: bool,
}

#[derive(Clone, Debug, PartialEq)]
struct ShapedParagraph {
    clusters: Arc<[TextCluster]>,
    natural_width: f32,
    min_content_width: f32,
    line_height: f32,
    baseline: f32,
}

#[derive(Clone, Debug, PartialEq)]
struct LineLayout {
    width: f32,
    height: f32,
    baseline: f32,
    line_count: usize,
}

fn measure_uncached(query: &TextMeasureQuery<'_>, snapshot: &FontFallbackSnapshot) -> TextMetrics {
    let analysis = analyze_text(query);
    let shaped = shape_paragraph(query, snapshot, analysis);
    let lines = layout_lines(query, &shaped);
    TextMetrics {
        width: lines.width,
        min_content_width: shaped.min_content_width,
        max_content_width: shaped.natural_width,
        height: lines.height,
        baseline: lines.baseline,
        line_count: lines.line_count,
    }
}

fn analyze_text(query: &TextMeasureQuery<'_>) -> TextAnalysis {
    let font_size = query.style.font_size().as_px();
    let advance = font_size * 0.5;
    let clusters = UnicodeSegmentation::graphemes(query.text, true)
        .map(|cluster| TextCluster {
            advance,
            break_after: cluster.chars().all(char::is_whitespace),
        })
        .collect::<Vec<_>>();
    TextAnalysis {
        clusters: Arc::from(clusters),
    }
}

fn shape_paragraph(
    query: &TextMeasureQuery<'_>,
    snapshot: &FontFallbackSnapshot,
    analysis: TextAnalysis,
) -> ShapedParagraph {
    let font_size = query.style.font_size().as_px();
    let line_height = font_size * 1.2;
    let baseline = font_size * 0.8;
    let natural_width = analysis
        .clusters
        .iter()
        .map(|cluster| cluster.advance)
        .sum::<f32>();
    let fallback_marker = snapshot.default_family().len() as f32 * 0.0;
    let min_content_width = analysis
        .clusters
        .iter()
        .map(|cluster| cluster.advance)
        .fold(0.0_f32, f32::max)
        + fallback_marker;
    ShapedParagraph {
        clusters: analysis.clusters,
        natural_width,
        min_content_width,
        line_height,
        baseline,
    }
}

fn layout_lines(query: &TextMeasureQuery<'_>, shaped: &ShapedParagraph) -> LineLayout {
    let measured_width = match query.available_inline_width {
        Some(width) if width.is_finite() && width > 0.0 => shaped.natural_width.min(width),
        Some(_) => 0.0,
        None => shaped.natural_width,
    };
    let mut line_count = line_count_for_width(shaped, measured_width);
    if let Some(max_lines) = query.style.max_lines() {
        line_count = line_count.min(max_lines.max(1));
    }
    LineLayout {
        width: measured_width,
        height: shaped.line_height * line_count as f32,
        baseline: shaped.baseline,
        line_count,
    }
}

fn line_count_for_width(shaped: &ShapedParagraph, width: f32) -> usize {
    if width <= 0.0 || shaped.natural_width <= width {
        return 1;
    }
    let mut line_count = 1;
    let mut line_width = 0.0;
    for cluster in shaped.clusters.iter() {
        if line_width > 0.0 && line_width + cluster.advance > width {
            line_count += 1;
            line_width = 0.0;
        }
        line_width += cluster.advance;
        if cluster.break_after && line_width > width {
            line_count += 1;
            line_width = 0.0;
        }
    }
    line_count
}

const fn text_overflow_rank(value: TextOverflow) -> u8 {
    match value {
        TextOverflow::Visible => 0,
        TextOverflow::Clip => 1,
        TextOverflow::Ellipsis => 2,
    }
}
