mod app;
mod element;
pub mod error;
mod platform;
mod scene;
pub mod style;
mod text_system;
pub mod window;

pub type SharedString = std::sync::Arc<str>;

pub use app::{
    App, Application, BackgroundExecutor, Context, Entity, EventEmitter, LastWindowBehavior,
    Render, Subscription, Task, TaskResult, UiExecutor, View, WeakEntity,
};
pub use element::{Div, Element, ElementKind, IntoElement, ParentElement, Text, div, text};
pub use error::{Error, PlatformError, RuntimeError};
pub use scene::DirtyLaneMask;
pub use style::{
    AlignItems, Color, Direction, EdgeInsets, JustifyContent, LayoutStyle, Length, PaintStyle,
    Size, Style, TextStyle,
};
pub use window::{Window, WindowHandle, WindowId, WindowOptions, WindowSize};
