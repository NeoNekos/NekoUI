use smallvec::SmallVec;

use crate::app::View;
use crate::input::{FocusPolicy, TextInputState};

use super::div::Div;
use super::text::Text;

#[derive(Debug, Clone, PartialEq)]
pub struct AnyElement(Box<AnyElementKind>);

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AnyElementKind {
    Div(Div),
    Text(Text),
    View(ViewSpec),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Fragment {
    items: SmallVec<[Box<AnyElement>; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ViewSpec {
    pub(crate) entity_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum WindowFrameArea {
    Drag,
    Close,
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct InteractionState {
    pub(crate) focus_policy: FocusPolicy,
    pub(crate) text_input: Option<TextInputState>,
}

pub trait IntoElement: Sized {
    fn into_any_element(self) -> AnyElement;
}

pub trait IntoElements {
    fn extend_into(self, fragment: &mut Fragment);
}

pub trait ParentElement {
    fn child(self, child: impl IntoElement) -> Self
    where
        Self: Sized;

    fn children(self, children: impl IntoElements) -> Self
    where
        Self: Sized;
}

impl AnyElement {
    pub(crate) fn div(div: Div) -> Self {
        Self(Box::new(AnyElementKind::Div(div)))
    }

    pub(crate) fn text(text: Text) -> Self {
        Self(Box::new(AnyElementKind::Text(text)))
    }

    pub(crate) fn view(entity_id: u64) -> Self {
        Self(Box::new(AnyElementKind::View(ViewSpec { entity_id })))
    }

    pub(crate) fn kind(&self) -> &AnyElementKind {
        self.0.as_ref()
    }
}

impl Fragment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, element: AnyElement) {
        self.items.push(Box::new(element));
    }

    pub fn iter(&self) -> impl Iterator<Item = &AnyElement> {
        self.items.iter().map(Box::as_ref)
    }
}

impl IntoIterator for Fragment {
    type Item = AnyElement;
    type IntoIter = FragmentIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        FragmentIntoIter {
            inner: self.items.into_iter(),
        }
    }
}

pub struct FragmentIntoIter {
    inner: smallvec::IntoIter<[Box<AnyElement>; 4]>,
}

impl Iterator for FragmentIntoIter {
    type Item = AnyElement;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|element| *element)
    }
}

impl IntoElement for AnyElement {
    fn into_any_element(self) -> AnyElement {
        self
    }
}

impl IntoElements for AnyElement {
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.push(self);
    }
}

impl IntoElements for Fragment {
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.items.extend(self.items);
    }
}

impl<T> IntoElements for Option<T>
where
    T: IntoElement,
{
    fn extend_into(self, fragment: &mut Fragment) {
        if let Some(element) = self {
            fragment.push(element.into_any_element());
        }
    }
}

impl<T> IntoElements for Vec<T>
where
    T: IntoElement,
{
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.items.extend(
            self.into_iter()
                .map(|element| Box::new(element.into_any_element())),
        );
    }
}

impl<T, const N: usize> IntoElements for [T; N]
where
    T: IntoElement,
{
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.items.extend(
            self.into_iter()
                .map(|element| Box::new(element.into_any_element())),
        );
    }
}

impl<A, B> IntoElements for (A, B)
where
    A: IntoElement,
    B: IntoElement,
{
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.push(self.0.into_any_element());
        fragment.push(self.1.into_any_element());
    }
}

impl<A, B, C> IntoElements for (A, B, C)
where
    A: IntoElement,
    B: IntoElement,
    C: IntoElement,
{
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.push(self.0.into_any_element());
        fragment.push(self.1.into_any_element());
        fragment.push(self.2.into_any_element());
    }
}

impl<A, B, C, D> IntoElements for (A, B, C, D)
where
    A: IntoElement,
    B: IntoElement,
    C: IntoElement,
    D: IntoElement,
{
    fn extend_into(self, fragment: &mut Fragment) {
        fragment.push(self.0.into_any_element());
        fragment.push(self.1.into_any_element());
        fragment.push(self.2.into_any_element());
        fragment.push(self.3.into_any_element());
    }
}

impl<T> IntoElement for View<T> {
    fn into_any_element(self) -> AnyElement {
        AnyElement::view(self.id())
    }
}

impl<T> FromIterator<T> for Fragment
where
    T: IntoElement,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut fragment = Fragment::new();
        fragment.items.extend(
            iter.into_iter()
                .map(|element| Box::new(element.into_any_element())),
        );
        fragment
    }
}
