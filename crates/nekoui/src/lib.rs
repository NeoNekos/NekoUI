mod app;
mod element;
pub mod error;
pub mod input;
mod platform;
mod scene;
pub mod semantics;
pub mod style;
pub mod text_system;
pub mod window;

pub type SharedString = std::sync::Arc<str>;

pub use app::{
    AppContext, Application, BackgroundExecutor, Context, Entity, EventEmitter, LastWindowBehavior,
    Render, Subscription, Task, TaskResult, UiExecutor, View, WeakEntity,
};
pub use element::{
    AnyElement, Div, Fragment, IntoElement, IntoElements, ParentElement, Text, div, text,
};
pub use error::{Error, PlatformError, RuntimeError};
pub use input::{
    CaretRect, FocusPolicy, InputNodeId, PointerButton, PointerEvent, PointerPhase,
    TextInputEvent, TextInputPurpose, TextInputState,
};
pub use semantics::{SemanticsRole, SemanticsState};
pub use style::{
    Absolute, AlignItems, AlignSelf, Background, BackgroundFill, Border, Bounds, BoxSizing, Color,
    CornerRadii, Corners, Definite, Direction, Display, EdgeInsets, EdgeWidths, Edges,
    FlexDirection, FlexWrap, FontFamily, FontStyle, FontWeight, Gap, IntoFontFamilies,
    JustifyContent, LayoutSize, LayoutStyle, Length, LinearGradient, Overflow, PaintStyle, Percent,
    Point, Px, Rem, ResolvedStyle, ResolvedTextStyle, Size, Style, TextAlign, TextOverflow,
    TextStyle, WhiteSpace, bounds, gradient, percent, point, px, rem, size,
};
pub use text_system::{TextCacheStats, TextLayout, TextMeasureKey, TextRun, TextSystem};
pub use window::{
    DisplayId, DisplayInfo, DisplaySelector, WindowAppearance, WindowBehavior, WindowGeometry,
    WindowGeometryPatch, WindowHandle, WindowId, WindowInfo, WindowOptions, WindowPlacement,
    WindowSize, WindowStartPosition,
};
