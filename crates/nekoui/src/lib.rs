#![forbid(unsafe_code)]

pub mod app;
pub mod diagnostic;
pub mod element;
pub mod error;
pub mod interaction;
pub mod layout;
pub mod prelude;
pub mod retained;
mod runtime;
pub mod scene;
pub mod style;
mod text;
pub mod window;

pub use app::{
    AppContext, Application, Context, Entity, Render, Subscription, TestRun, WeakEntity,
};
pub use element::{Div, Element, ElementKey, ElementKind, IntoElement, Text, div, text};
pub use error::{ErrorKind, NekoError, NekoResult};
pub use layout::{LayoutGeneration, LayoutNodeSnapshot, LayoutPoint, LayoutTreeSnapshot, Viewport};
pub use style::{
    Color, ColorSpace, Dimension, Display, Length, Opacity, StyleExt, TextOverflow, auto, fill,
    opacity, px,
};
pub use window::WindowHandle;
