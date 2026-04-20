use hashbrown::HashSet;
use smallvec::SmallVec;

use crate::SharedString;
use crate::error::RuntimeError;
use crate::style::Style;
use crate::window::Window;

use super::core::{AnyElement, AnyElementKind, Fragment};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpecNodeId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecKind {
    Div,
    Text,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpecPayload {
    None,
    Text(SharedString),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpecNode {
    pub kind: SpecKind,
    pub key: Option<u64>,
    pub style: Style,
    pub payload: SpecPayload,
    pub first_child: Option<SpecNodeId>,
    pub next_sibling: Option<SpecNodeId>,
}

#[derive(Debug, Default, Clone)]
pub struct SpecArena {
    nodes: Vec<SpecNode>,
}

#[derive(Debug)]
pub struct BuildResult {
    pub root: SpecNodeId,
    pub referenced_views: HashSet<u64>,
}

type ViewResolver<'a> = dyn FnMut(u64, &mut Window) -> Result<AnyElement, RuntimeError> + 'a;

pub struct BuildCx<'a> {
    arena: &'a mut SpecArena,
    window: &'a mut Window,
    view_resolver: &'a mut ViewResolver<'a>,
    referenced_views: HashSet<u64>,
}

impl SpecArena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
    }

    pub fn alloc(&mut self, node: SpecNode) -> SpecNodeId {
        let id = SpecNodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn node(&self, id: SpecNodeId) -> &SpecNode {
        &self.nodes[id.0]
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn child_ids(&self, id: SpecNodeId) -> SmallVec<[SpecNodeId; 4]> {
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
    pub fn new(
        window: &'a mut Window,
        view_resolver: &'a mut ViewResolver<'a>,
        arena: &'a mut SpecArena,
    ) -> Self {
        arena.reset();
        Self {
            arena,
            window,
            view_resolver,
            referenced_views: HashSet::new(),
        }
    }

    pub fn build_root(mut self, root: AnyElement) -> Result<BuildResult, RuntimeError> {
        let root = self.lower_any(root)?;
        Ok(BuildResult {
            root,
            referenced_views: self.referenced_views,
        })
    }

    fn lower_any(&mut self, element: AnyElement) -> Result<SpecNodeId, RuntimeError> {
        match element.kind() {
            AnyElementKind::Div(div) => self.lower_div(div),
            AnyElementKind::Text(text) => self.lower_text(text),
            AnyElementKind::View(view) => {
                self.referenced_views.insert(view.entity_id);
                let rendered = (self.view_resolver)(view.entity_id, self.window)?;
                self.lower_any(rendered)
            }
        }
    }

    fn lower_div(&mut self, div: &crate::element::Div) -> Result<SpecNodeId, RuntimeError> {
        let child_ids = self.lower_fragment(&div.children)?;
        let first_child = link_siblings(self.arena, child_ids);
        Ok(self.arena.alloc(SpecNode {
            kind: SpecKind::Div,
            key: div.key,
            style: div.style.clone(),
            payload: SpecPayload::None,
            first_child,
            next_sibling: None,
        }))
    }

    fn lower_text(&mut self, text: &crate::element::Text) -> Result<SpecNodeId, RuntimeError> {
        Ok(self.arena.alloc(SpecNode {
            kind: SpecKind::Text,
            key: text.key,
            style: text.style.clone(),
            payload: SpecPayload::Text(text.content.clone()),
            first_child: None,
            next_sibling: None,
        }))
    }

    fn lower_fragment(
        &mut self,
        fragment: &Fragment,
    ) -> Result<SmallVec<[SpecNodeId; 4]>, RuntimeError> {
        let mut ids = SmallVec::new();
        for child in fragment.iter() {
            ids.push(self.lower_any(child.clone())?);
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
