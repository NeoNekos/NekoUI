use hashbrown::HashSet;
use smallvec::SmallVec;

use crate::SharedString;
use crate::error::RuntimeError;
use crate::semantics::SemanticsState;
use crate::style::{ResolvedStyle, ResolvedTextStyle};
use crate::window::WindowInfo;

use super::core::{AnyElement, AnyElementKind, Fragment, InteractionState, WindowFrameArea};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SpecNodeId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SpecKind {
    Div,
    Text,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum SpecPayload {
    None,
    Text(SharedString),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpecNode {
    pub kind: SpecKind,
    pub key: Option<u64>,
    pub style: ResolvedStyle,
    pub window_frame_area: Option<WindowFrameArea>,
    pub owner_view_id: Option<u64>,
    pub interaction: InteractionState,
    pub semantics: SemanticsState,
    pub payload: SpecPayload,
    pub first_child: Option<SpecNodeId>,
    pub next_sibling: Option<SpecNodeId>,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct SpecArena {
    nodes: Vec<SpecNode>,
}

#[derive(Debug)]
pub(crate) struct BuildResult {
    pub root: SpecNodeId,
    pub referenced_views: HashSet<u64>,
}

type ViewResolver<'a> = dyn FnMut(u64, &WindowInfo) -> Result<AnyElement, RuntimeError> + 'a;

pub(crate) struct BuildCx<'a> {
    arena: &'a mut SpecArena,
    window: &'a WindowInfo,
    view_resolver: &'a mut ViewResolver<'a>,
    referenced_views: HashSet<u64>,
    current_owner_view: Option<u64>,
}

impl SpecArena {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn reset(&mut self) {
        self.nodes.clear();
    }

    pub(crate) fn alloc(&mut self, node: SpecNode) -> SpecNodeId {
        let id = SpecNodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub(crate) fn node(&self, id: SpecNodeId) -> &SpecNode {
        &self.nodes[id.0]
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn child_ids(&self, id: SpecNodeId) -> SmallVec<[SpecNodeId; 4]> {
        let mut out = SmallVec::new();
        let mut cursor = self.node(id).first_child;
        while let Some(child_id) = cursor {
            out.push(child_id);
            cursor = self.node(child_id).next_sibling;
        }
        out
    }
}

impl<'a> BuildCx<'a> {
    pub(crate) fn new(
        window: &'a WindowInfo,
        view_resolver: &'a mut ViewResolver<'a>,
        arena: &'a mut SpecArena,
    ) -> Self {
        arena.reset();
        Self {
            arena,
            window,
            view_resolver,
            referenced_views: HashSet::new(),
            current_owner_view: None,
        }
    }

    pub(crate) fn build_root(mut self, root: AnyElement) -> Result<BuildResult, RuntimeError> {
        let root = self.lower_any(root, &ResolvedTextStyle::default())?;
        Ok(BuildResult {
            root,
            referenced_views: self.referenced_views,
        })
    }

    fn lower_any(
        &mut self,
        element: AnyElement,
        inherited_text: &ResolvedTextStyle,
    ) -> Result<SpecNodeId, RuntimeError> {
        match element.kind() {
            AnyElementKind::Div(div) => self.lower_div(div, inherited_text),
            AnyElementKind::Text(text) => self.lower_text(text, inherited_text),
            AnyElementKind::View(view) => {
                self.referenced_views.insert(view.entity_id);
                let previous_owner = self.current_owner_view;
                self.current_owner_view = Some(view.entity_id);
                let rendered = (self.view_resolver)(view.entity_id, self.window)?;
                let lowered = self.lower_any(rendered, inherited_text);
                self.current_owner_view = previous_owner;
                lowered
            }
        }
    }

    fn lower_div(
        &mut self,
        div: &crate::element::Div,
        inherited_text: &ResolvedTextStyle,
    ) -> Result<SpecNodeId, RuntimeError> {
        let resolved_style = div.style.resolve_with_parent(inherited_text);
        let child_ids = self.lower_fragment(&div.children, &resolved_style.text)?;
        let first_child = link_siblings(self.arena, child_ids);
        Ok(self.arena.alloc(SpecNode {
            kind: SpecKind::Div,
            key: div.key,
            style: resolved_style,
            window_frame_area: div.window_frame_area,
            owner_view_id: self.current_owner_view,
            interaction: div.interaction.clone(),
            semantics: div.semantics.clone(),
            payload: SpecPayload::None,
            first_child,
            next_sibling: None,
        }))
    }

    fn lower_text(
        &mut self,
        text: &crate::element::Text,
        inherited_text: &ResolvedTextStyle,
    ) -> Result<SpecNodeId, RuntimeError> {
        Ok(self.arena.alloc(SpecNode {
            kind: SpecKind::Text,
            key: text.key,
            style: text.style.resolve_with_parent(inherited_text),
            window_frame_area: text.window_frame_area,
            owner_view_id: self.current_owner_view,
            interaction: text.interaction.clone(),
            semantics: if text.semantics.label.is_none() {
                let mut semantics = text.semantics.clone();
                semantics.label = Some(text.content.clone());
                semantics
            } else {
                text.semantics.clone()
            },
            payload: SpecPayload::Text(text.content.clone()),
            first_child: None,
            next_sibling: None,
        }))
    }

    fn lower_fragment(
        &mut self,
        fragment: &Fragment,
        inherited_text: &ResolvedTextStyle,
    ) -> Result<SmallVec<[SpecNodeId; 4]>, RuntimeError> {
        let mut ids = SmallVec::new();
        for child in fragment.iter() {
            ids.push(self.lower_any(child.clone(), inherited_text)?);
        }
        Ok(ids)
    }
}

fn link_siblings(
    arena: &mut SpecArena,
    child_ids: SmallVec<[SpecNodeId; 4]>,
) -> Option<SpecNodeId> {
    let mut iter = child_ids.into_iter();
    let first = iter.next()?;
    let mut previous = first;
    for child in iter {
        arena.nodes[previous.0].next_sibling = Some(child);
        previous = child;
    }
    Some(first)
}
