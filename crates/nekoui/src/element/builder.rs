use std::borrow::Cow;
use std::rc::Rc;

use crate::element::ElementKey;
use crate::error::NekoResult;
use crate::interaction::{
    ClickEvent, InteractionHandlers, IntoHandlerResult, KeyEvent, PointerEvent,
};
use crate::style::StyleDeclaration;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum ElementKind {
    Div,
    Text,
    Input,
}

impl ElementKind {
    pub fn is_container(self) -> bool {
        matches!(self, ElementKind::Div)
    }

    pub fn name(self) -> &'static str {
        match self {
            ElementKind::Div => "div",
            ElementKind::Text => "text",
            ElementKind::Input => "input",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    kind: ElementKind,
    key: Option<ElementKey>,
    style: StyleDeclaration,
    focusable: bool,
    handlers: InteractionHandlers,
    text: Option<Cow<'static, str>>,
    children: Vec<Element>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ElementParts {
    pub kind: ElementKind,
    pub key: Option<ElementKey>,
    pub style: StyleDeclaration,
    pub focusable: bool,
    pub handlers: InteractionHandlers,
    pub text: Option<Cow<'static, str>>,
    pub children: Vec<Element>,
}

impl Element {
    pub(crate) fn new(kind: ElementKind) -> Self {
        Self {
            kind,
            key: None,
            style: StyleDeclaration::default(),
            focusable: false,
            handlers: InteractionHandlers::default(),
            text: None,
            children: Vec::new(),
        }
    }

    pub fn kind(&self) -> ElementKind {
        self.kind
    }

    pub fn key(&self) -> Option<&ElementKey> {
        self.key.as_ref()
    }

    pub fn style(&self) -> &StyleDeclaration {
        &self.style
    }

    pub fn focusable(&self) -> bool {
        self.focusable
    }

    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    pub fn children(&self) -> &[Element] {
        &self.children
    }

    pub(crate) fn validate_inputs(&self) -> NekoResult<()> {
        self.style.validate_inputs()?;
        for child in &self.children {
            child.validate_inputs()?;
        }
        Ok(())
    }

    pub(crate) fn style_mut(&mut self) -> &mut StyleDeclaration {
        &mut self.style
    }

    pub(crate) fn handlers_mut(&mut self) -> &mut InteractionHandlers {
        &mut self.handlers
    }

    pub(crate) fn into_parts(self) -> ElementParts {
        ElementParts {
            kind: self.kind,
            key: self.key,
            style: self.style,
            focusable: self.focusable,
            handlers: self.handlers,
            text: self.text,
            children: self.children,
        }
    }

    fn with_key(mut self, key: impl Into<ElementKey>) -> Self {
        self.key = Some(key.into());
        self
    }
}

pub trait IntoElement {
    fn into_element(self) -> Element;
}

impl IntoElement for Element {
    fn into_element(self) -> Element {
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Div {
    element: Element,
}

impl Div {
    pub fn key(mut self, key: impl Into<ElementKey>) -> Self {
        self.element = self.element.with_key(key);
        self
    }

    pub fn child(mut self, child: impl IntoElement) -> Self {
        self.element.children.push(child.into_element());
        self
    }

    pub fn focusable(mut self, focusable: bool) -> Self {
        self.element.focusable = focusable;
        self
    }

    pub fn on_pointer_down<R>(mut self, handler: impl Fn(&PointerEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_down(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_pointer_up<R>(mut self, handler: impl Fn(&PointerEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_up(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_pointer_move<R>(mut self, handler: impl Fn(&PointerEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_move(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_click<R>(mut self, handler: impl Fn(&ClickEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_click(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_key_down<R>(mut self, handler: impl Fn(&KeyEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_down(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_key_up<R>(mut self, handler: impl Fn(&KeyEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_up(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_pointer_down_with<R>(
        mut self,
        handler: impl for<'a> Fn(&PointerEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_down(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_pointer_up_with<R>(
        mut self,
        handler: impl for<'a> Fn(&PointerEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_up(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_pointer_move_with<R>(
        mut self,
        handler: impl for<'a> Fn(&PointerEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_move(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_click_with<R>(
        mut self,
        handler: impl for<'a> Fn(&ClickEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_click(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_key_down_with<R>(
        mut self,
        handler: impl for<'a> Fn(&KeyEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_down(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_key_up_with<R>(
        mut self,
        handler: impl for<'a> Fn(&KeyEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_up(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn children(mut self, children: impl IntoIterator<Item = impl IntoElement>) -> Self {
        self.element
            .children
            .extend(children.into_iter().map(IntoElement::into_element));
        self
    }

    pub fn as_element(&self) -> &Element {
        &self.element
    }

    pub(crate) fn style_mut(&mut self) -> &mut StyleDeclaration {
        self.element.style_mut()
    }
}

impl IntoElement for Div {
    fn into_element(self) -> Element {
        self.element
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    element: Element,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    element: Element,
}

impl Text {
    pub fn key(mut self, key: impl Into<ElementKey>) -> Self {
        self.element = self.element.with_key(key);
        self
    }

    pub fn as_element(&self) -> &Element {
        &self.element
    }

    pub fn focusable(mut self, focusable: bool) -> Self {
        self.element.focusable = focusable;
        self
    }

    pub fn on_pointer_down<R>(mut self, handler: impl Fn(&PointerEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_down(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_pointer_up<R>(mut self, handler: impl Fn(&PointerEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_up(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_pointer_move<R>(mut self, handler: impl Fn(&PointerEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_move(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_click<R>(mut self, handler: impl Fn(&ClickEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_click(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_key_down<R>(mut self, handler: impl Fn(&KeyEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_down(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_key_up<R>(mut self, handler: impl Fn(&KeyEvent) -> R + 'static) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_up(Rc::new(move |event, _cx| handler(event).into_result()));
        self
    }

    pub fn on_pointer_down_with<R>(
        mut self,
        handler: impl for<'a> Fn(&PointerEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_down(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_pointer_up_with<R>(
        mut self,
        handler: impl for<'a> Fn(&PointerEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_up(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_pointer_move_with<R>(
        mut self,
        handler: impl for<'a> Fn(&PointerEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_pointer_move(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_click_with<R>(
        mut self,
        handler: impl for<'a> Fn(&ClickEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_click(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_key_down_with<R>(
        mut self,
        handler: impl for<'a> Fn(&KeyEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_down(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub fn on_key_up_with<R>(
        mut self,
        handler: impl for<'a> Fn(&KeyEvent, &mut crate::app::AppContext<'a>) -> R + 'static,
    ) -> Self
    where
        R: IntoHandlerResult + 'static,
    {
        self.element
            .handlers_mut()
            .set_key_up(Rc::new(move |event, cx| handler(event, cx).into_result()));
        self
    }

    pub(crate) fn style_mut(&mut self) -> &mut StyleDeclaration {
        self.element.style_mut()
    }
}

impl IntoElement for Text {
    fn into_element(self) -> Element {
        self.element
    }
}

impl Input {
    pub fn key(mut self, key: impl Into<ElementKey>) -> Self {
        self.element = self.element.with_key(key);
        self
    }

    pub fn as_element(&self) -> &Element {
        &self.element
    }

    pub fn focusable(mut self, focusable: bool) -> Self {
        self.element.focusable = focusable;
        self
    }

    pub(crate) fn style_mut(&mut self) -> &mut StyleDeclaration {
        self.element.style_mut()
    }
}

impl IntoElement for Input {
    fn into_element(self) -> Element {
        self.element
    }
}

pub fn div() -> Div {
    Div {
        element: Element::new(ElementKind::Div),
    }
}

pub fn text(value: impl Into<Cow<'static, str>>) -> Text {
    let mut element = Element::new(ElementKind::Text);
    element.text = Some(value.into());
    Text { element }
}

pub fn input(value: impl Into<Cow<'static, str>>) -> Input {
    let mut element = Element::new(ElementKind::Input);
    element.text = Some(value.into());
    element.focusable = true;
    Input { element }
}

#[cfg(test)]
mod tests {
    use crate::element::{ElementKind, IntoElement, div, text};

    #[test]
    fn div_is_container_and_text_is_leaf() {
        let root = div().child(text("Hello")).into_element();

        assert_eq!(root.kind(), ElementKind::Div);
        assert_eq!(root.children().len(), 1);
        assert_eq!(root.children()[0].kind(), ElementKind::Text);
        assert_eq!(root.children()[0].text(), Some("Hello"));
    }

    #[test]
    fn keys_are_pure_declarations() {
        let root = div().key("root").child(text("Hello").key("greeting"));
        let element = root.into_element();

        assert_eq!(element.key().unwrap().as_str(), "root");
        assert_eq!(element.children()[0].key().unwrap().as_str(), "greeting");
    }
}
