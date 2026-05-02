use crate::input::{
    CaretRect, InputNodeId, PointerButton, PointerEvent, PointerPhase, TextInputPurpose,
};
use crate::scene::{NodeId, RetainedTree};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FocusManager {
    focused: Option<NodeId>,
}

impl FocusManager {
    pub(crate) const fn new() -> Self {
        Self { focused: None }
    }

    pub(crate) fn focused(self) -> Option<NodeId> {
        self.focused
    }

    pub(crate) fn set_focused(&mut self, focused: Option<NodeId>) -> bool {
        if self.focused == focused {
            return false;
        }
        self.focused = focused;
        true
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct TextInputTarget {
    pub(crate) node: Option<NodeId>,
    pub(crate) input_id: Option<InputNodeId>,
    pub(crate) ime_allowed: bool,
    pub(crate) purpose: TextInputPurpose,
    pub(crate) caret_rect: Option<CaretRect>,
    pub(crate) preedit: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct HitTestSnapshot {
    pub(crate) focusable: Option<NodeId>,
    pub(crate) text_input: Option<NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct PointerButtons {
    pub(crate) primary: bool,
    pub(crate) secondary: bool,
    pub(crate) middle: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct InputRouter {
    pub(crate) last_hit: Option<NodeId>,
    pub(crate) buttons: PointerButtons,
}

impl InputRouter {
    pub(crate) fn hit_test(
        &mut self,
        tree: &RetainedTree,
        position: crate::style::Point<crate::style::Px>,
    ) -> HitTestSnapshot {
        let focusable = tree.focusable_node_at(position);
        let text_input = tree.text_input_node_at(position);
        self.last_hit = text_input.or(focusable);
        HitTestSnapshot {
            focusable,
            text_input,
        }
    }

    pub(crate) fn pointer_event(
        &mut self,
        tree: &RetainedTree,
        event: PointerEvent,
    ) -> HitTestSnapshot {
        if let Some(button) = event.button {
            let pressed = matches!(event.phase, PointerPhase::Down);
            match button {
                PointerButton::Primary => self.buttons.primary = pressed,
                PointerButton::Secondary => self.buttons.secondary = pressed,
                PointerButton::Middle => self.buttons.middle = pressed,
                _ => {}
            }
        }

        self.hit_test(tree, event.position)
    }
}

#[cfg(test)]
mod tests {
    use crate::element::{BuildCx, IntoElement, ParentElement, SpecArena};
    use crate::platform::window::WindowInfoSeed;
    use crate::scene::RetainedTree;
    use crate::window::{Window, WindowId, WindowInfo, WindowOptions, WindowSize};

    use super::{FocusManager, InputRouter};

    fn test_window(logical: WindowSize, physical: WindowSize, scale_factor: f64) -> Window {
        Window::from_options(
            WindowId::new(),
            &WindowOptions::default(),
            WindowInfoSeed {
                content_size: logical,
                frame_size: Some(logical),
                physical_size: physical,
                scale_factor,
                position: None,
                current_display: None,
            },
        )
    }

    fn build_static_tree(root: crate::AnyElement) -> RetainedTree {
        let window = test_window(WindowSize::new(320, 200), WindowSize::new(320, 200), 1.0);
        let mut resolver = |_view_id: u64,
                            _window: &WindowInfo|
         -> Result<crate::AnyElement, crate::RuntimeError> {
            unreachable!("static test tree should not resolve nested views")
        };
        let mut arena = SpecArena::new();
        let built = BuildCx::new(&window, &mut resolver, &mut arena)
            .build_root(root)
            .unwrap();
        RetainedTree::from_spec(&arena, built.root)
    }

    #[test]
    fn focus_manager_tracks_changes() {
        let mut manager = FocusManager::new();
        assert!(!manager.set_focused(None));
    }

    #[test]
    fn input_router_hit_test_prefers_text_input() {
        let root = crate::div()
            .child(
                crate::div()
                    .w(crate::style::px(120.0))
                    .h(crate::style::px(40.0))
                    .text_input(crate::TextInputPurpose::Normal),
            )
            .into_any_element();
        let mut tree = build_static_tree(root);
        let mut text_system = crate::text_system::TextSystem::new();
        tree.compute_layout(WindowSize::new(320, 200), &mut text_system);

        let mut router = InputRouter::default();
        let hit = router.hit_test(
            &tree,
            crate::style::Point::new(crate::style::px(10.0), crate::style::px(10.0)),
        );
        assert!(hit.text_input.is_some());
        assert_eq!(hit.focusable, hit.text_input);
    }
}
