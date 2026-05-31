use std::ops::Range;

use crate::interaction::TextRange;

use super::measure::TextGeneration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TextRangeError {
    Reversed,
    OutOfBounds,
    NotBoundary,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextComposition {
    text: String,
    range: TextRange,
    cursor: Option<TextRange>,
}

impl TextComposition {
    fn new(text: String, range: TextRange, cursor: Option<TextRange>) -> Self {
        Self {
            text,
            range,
            cursor,
        }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn range(&self) -> TextRange {
        self.range
    }

    pub(crate) fn cursor(&self) -> Option<TextRange> {
        self.cursor
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TextBlock {
    committed: String,
    selection: TextRange,
    composition: Option<TextComposition>,
    generation: TextGeneration,
}

impl TextBlock {
    pub(crate) fn new(initial: impl Into<String>) -> Self {
        let committed = initial.into();
        let end = committed.len();
        Self {
            committed,
            selection: TextRange::collapsed(end),
            composition: None,
            generation: TextGeneration::INITIAL,
        }
    }

    pub(crate) fn committed(&self) -> &str {
        &self.committed
    }

    pub(crate) fn display_text(&self) -> String {
        let Some(composition) = &self.composition else {
            return self.committed.clone();
        };
        let mut value = self.committed.clone();
        value.replace_range(to_range(composition.range), composition.text());
        value
    }

    pub(crate) fn generation(&self) -> TextGeneration {
        self.generation
    }

    pub(crate) fn selection(&self) -> TextRange {
        self.selection
    }

    pub(crate) fn composition(&self) -> Option<&TextComposition> {
        self.composition.as_ref()
    }

    pub(crate) fn has_composition(&self) -> bool {
        self.composition.is_some()
    }

    pub(crate) fn set_selection(&mut self, selection: TextRange) -> Result<(), TextRangeError> {
        validate_range(self.committed(), selection)?;
        self.selection = selection;
        Ok(())
    }

    pub(crate) fn insert_text(
        &mut self,
        text: &str,
        replace: Option<TextRange>,
    ) -> Result<TextEditOutcome, TextRangeError> {
        let range = if let Some(composition) = self.composition.take() {
            composition.range()
        } else {
            replace.unwrap_or(self.selection)
        };
        validate_range(self.committed(), range)?;
        self.committed.replace_range(to_range(range), text);
        let caret = range.start() + text.len();
        self.set_selection(TextRange::collapsed(caret))?;
        self.bump_generation();
        Ok(TextEditOutcome::Mutated)
    }

    pub(crate) fn delete_backward(&mut self) -> Result<TextEditOutcome, TextRangeError> {
        if self.composition.is_some() {
            return Ok(self.clear_composition());
        }
        validate_range(self.committed(), self.selection)?;
        if !self.selection.is_collapsed() {
            self.committed.replace_range(to_range(self.selection), "");
            self.set_selection(TextRange::collapsed(self.selection.start()))?;
            self.bump_generation();
            return Ok(TextEditOutcome::Mutated);
        }
        let caret = self.selection.start();
        if caret == 0 {
            return Ok(TextEditOutcome::Unchanged);
        }
        let previous = self.committed[..caret]
            .char_indices()
            .next_back()
            .map_or(0, |(index, _)| index);
        self.committed.replace_range(previous..caret, "");
        self.set_selection(TextRange::collapsed(previous))?;
        self.bump_generation();
        Ok(TextEditOutcome::Mutated)
    }

    pub(crate) fn set_composition(
        &mut self,
        text: &str,
        cursor: Option<TextRange>,
        replace: Option<TextRange>,
    ) -> Result<TextEditOutcome, TextRangeError> {
        if text.is_empty() {
            let cleared = self.composition.take().is_some();
            if cleared {
                self.bump_generation();
                return Ok(TextEditOutcome::Mutated);
            }
            return Ok(TextEditOutcome::Unchanged);
        }

        validate_composition_text(text, cursor)?;
        let range = if let Some(composition) = &self.composition {
            composition.range()
        } else {
            replace.unwrap_or(self.selection)
        };
        validate_range(self.committed(), range)?;
        self.composition = Some(TextComposition::new(text.to_owned(), range, cursor));
        self.bump_generation();
        Ok(TextEditOutcome::Mutated)
    }

    pub(crate) fn clear_composition(&mut self) -> TextEditOutcome {
        if self.composition.take().is_some() {
            self.bump_generation();
            TextEditOutcome::Mutated
        } else {
            TextEditOutcome::Unchanged
        }
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.next();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct EditableTextState {
    block: TextBlock,
}

impl EditableTextState {
    pub(crate) fn new(initial: impl Into<String>) -> Self {
        Self {
            block: TextBlock::new(initial),
        }
    }

    pub(crate) fn block(&self) -> &TextBlock {
        &self.block
    }

    pub(crate) fn block_mut(&mut self) -> &mut TextBlock {
        &mut self.block
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TextEditOutcome {
    Mutated,
    Unchanged,
}

pub(crate) fn validate_range(text: &str, range: TextRange) -> Result<(), TextRangeError> {
    if range.validate_for_text(text) {
        return Ok(());
    }
    if range.start() > range.end() {
        return Err(TextRangeError::Reversed);
    }
    if range.end() > text.len() {
        return Err(TextRangeError::OutOfBounds);
    }
    if !text.is_char_boundary(range.start()) || !text.is_char_boundary(range.end()) {
        return Err(TextRangeError::NotBoundary);
    }
    Ok(())
}

fn validate_composition_text(text: &str, cursor: Option<TextRange>) -> Result<(), TextRangeError> {
    if let Some(cursor) = cursor {
        validate_range(text, cursor)?;
    }
    Ok(())
}

fn to_range(range: TextRange) -> Range<usize> {
    range.start()..range.end()
}

#[cfg(test)]
mod tests {
    use crate::interaction::TextRange;
    use crate::text::{TextBlock, TextEditOutcome, TextRangeError};

    #[test]
    fn commit_replaces_selection_and_advances_generation() {
        let mut block = TextBlock::new("hello");
        let initial_generation = block.generation();
        block.set_selection(TextRange::new(1, 4)).unwrap();

        assert_eq!(
            block.insert_text("i", None).unwrap(),
            TextEditOutcome::Mutated
        );

        assert_eq!(block.committed(), "hio");
        assert_eq!(block.selection(), TextRange::collapsed(2));
        assert!(block.generation() > initial_generation);
    }

    #[test]
    fn preedit_is_temporary_and_commit_replaces_it() {
        let mut block = TextBlock::new("ab");
        block.set_selection(TextRange::collapsed(1)).unwrap();

        block
            .set_composition("文", Some(TextRange::collapsed(3)), None)
            .unwrap();
        assert_eq!(block.committed(), "ab");
        assert_eq!(block.display_text(), "a文b");
        assert!(block.has_composition());

        block.insert_text("字", None).unwrap();
        assert_eq!(block.committed(), "a字b");
        assert_eq!(block.display_text(), "a字b");
        assert!(!block.has_composition());
    }

    #[test]
    fn delete_backward_removes_previous_utf8_char_boundary_unit() {
        let mut block = TextBlock::new("aé🙂");

        assert_eq!(block.delete_backward().unwrap(), TextEditOutcome::Mutated);
        assert_eq!(block.committed(), "aé");
        assert_eq!(block.selection(), TextRange::collapsed("aé".len()));

        assert_eq!(block.delete_backward().unwrap(), TextEditOutcome::Mutated);
        assert_eq!(block.committed(), "a");
        assert_eq!(block.selection(), TextRange::collapsed(1));
    }

    #[test]
    fn delete_backward_deletes_selection_when_selection_is_not_collapsed() {
        let mut block = TextBlock::new("aéx");
        block.set_selection(TextRange::new(1, "aé".len())).unwrap();

        assert_eq!(block.delete_backward().unwrap(), TextEditOutcome::Mutated);

        assert_eq!(block.committed(), "ax");
        assert_eq!(block.selection(), TextRange::collapsed(1));
    }

    #[test]
    fn delete_backward_clears_composition_without_committing_it() {
        let mut block = TextBlock::new("ab");
        block.set_selection(TextRange::collapsed(1)).unwrap();
        block.set_composition("文", None, None).unwrap();

        assert_eq!(block.delete_backward().unwrap(), TextEditOutcome::Mutated);

        assert_eq!(block.committed(), "ab");
        assert_eq!(block.display_text(), "ab");
        assert_eq!(block.selection(), TextRange::collapsed(1));
        assert!(!block.has_composition());
    }

    #[test]
    fn delete_backward_at_start_is_unchanged() {
        let mut block = TextBlock::new("ab");
        block.set_selection(TextRange::collapsed(0)).unwrap();

        assert_eq!(block.delete_backward().unwrap(), TextEditOutcome::Unchanged);

        assert_eq!(block.committed(), "ab");
        assert_eq!(block.selection(), TextRange::collapsed(0));
    }

    #[test]
    fn empty_preedit_without_composition_does_not_delete_selection() {
        let mut block = TextBlock::new("hello");
        block.set_selection(TextRange::new(1, 4)).unwrap();

        assert_eq!(
            block.set_composition("", None, None).unwrap(),
            TextEditOutcome::Unchanged
        );

        assert_eq!(block.committed(), "hello");
        assert_eq!(block.selection(), TextRange::new(1, 4));
    }

    #[test]
    fn invalid_ranges_are_rejected_without_mutation() {
        let mut block = TextBlock::new("éx");

        assert_eq!(
            block
                .insert_text("a", Some(TextRange::new(1, 1)))
                .unwrap_err(),
            TextRangeError::NotBoundary
        );
        assert_eq!(
            block
                .insert_text("a", Some(TextRange::new(0, 20)))
                .unwrap_err(),
            TextRangeError::OutOfBounds
        );
        assert_eq!(
            block
                .insert_text("a", Some(TextRange::new(2, 1)))
                .unwrap_err(),
            TextRangeError::Reversed
        );
        assert_eq!(block.committed(), "éx");
    }
}
