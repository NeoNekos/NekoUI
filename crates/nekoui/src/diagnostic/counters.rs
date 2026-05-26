use std::collections::BTreeMap;

use crate::diagnostic::DiagnosticRecord;
use crate::diagnostic::signal::SignalId;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticCounters {
    typed: [u64; SignalId::COUNT],
    custom: BTreeMap<&'static str, u64>,
}

impl Default for DiagnosticCounters {
    fn default() -> Self {
        Self {
            typed: [0; SignalId::COUNT],
            custom: BTreeMap::new(),
        }
    }
}

impl DiagnosticCounters {
    pub fn increment(&mut self, name: &'static str) {
        self.add(name, 1);
    }

    pub fn add(&mut self, name: &'static str, amount: u64) {
        if let Some(signal) = SignalId::from_name(name) {
            self.add_signal(signal, amount);
        } else {
            *self.custom.entry(name).or_insert(0) += amount;
        }
    }

    pub(crate) fn increment_signal(&mut self, signal: SignalId) {
        self.add_signal(signal, 1);
    }

    pub(crate) fn add_signal(&mut self, signal: SignalId, amount: u64) {
        self.typed[signal.index()] += amount;
    }

    pub fn get(&self, name: &'static str) -> u64 {
        if let Some(signal) = SignalId::from_name(name) {
            self.signal(signal)
        } else {
            self.custom.get(name).copied().unwrap_or(0)
        }
    }

    pub(crate) fn signal(&self, signal: SignalId) -> u64 {
        self.typed[signal.index()]
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        SignalId::ALL
            .into_iter()
            .map(|signal| (signal.name(), self.signal(signal)))
            .filter(|(_, value)| *value != 0)
            .chain(self.custom.iter().map(|(name, value)| (*name, *value)))
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Diagnostics {
    counters: DiagnosticCounters,
    records: Vec<DiagnosticRecord>,
}

impl Diagnostics {
    pub fn increment(&mut self, name: &'static str) {
        self.counters.increment(name);
    }

    pub fn add(&mut self, name: &'static str, amount: u64) {
        self.counters.add(name, amount);
    }

    pub(crate) fn increment_signal(&mut self, signal: SignalId) {
        self.counters.increment_signal(signal);
    }

    pub(crate) fn add_signal(&mut self, signal: SignalId, amount: u64) {
        self.counters.add_signal(signal, amount);
    }

    pub fn record(&mut self, record: DiagnosticRecord) {
        self.records.push(record);
    }

    pub(crate) fn extend_records(&mut self, records: impl IntoIterator<Item = DiagnosticRecord>) {
        self.records.extend(records);
    }

    pub fn snapshot(&self) -> DiagnosticSnapshot {
        DiagnosticSnapshot {
            counters: self.counters.clone(),
            records: self.records.clone(),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiagnosticSnapshot {
    counters: DiagnosticCounters,
    records: Vec<DiagnosticRecord>,
}

impl DiagnosticSnapshot {
    pub fn counter(&self, name: &'static str) -> u64 {
        self.counters.get(name)
    }

    pub fn counters(&self) -> &DiagnosticCounters {
        &self.counters
    }

    pub fn records(&self) -> &[DiagnosticRecord] {
        &self.records
    }
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::signal::SignalId;
    use crate::diagnostic::{DiagnosticSnapshot, Diagnostics};

    #[test]
    fn counter_snapshot_is_structured_and_stable() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.increment("runtime.command_queued");
        let first = diagnostics.snapshot();
        diagnostics.increment_signal(SignalId::RuntimeCommandQueued);

        assert_eq!(first.counter("runtime.command_queued"), 1);
        assert_eq!(diagnostics.snapshot().counter("runtime.command_queued"), 2);
    }

    #[test]
    fn typed_signals_project_to_stable_dotted_names() {
        let mut diagnostics = Diagnostics::default();

        diagnostics.increment_signal(SignalId::RetainedDiff);
        diagnostics.add_signal(SignalId::LayoutNodesTotal, 3);

        let snapshot = diagnostics.snapshot();
        assert_eq!(snapshot.counter("retained.diff"), 1);
        assert_eq!(snapshot.counter("layout.nodes_total"), 3);
        assert!(
            snapshot
                .counters()
                .iter()
                .any(|(name, value)| name == "retained.diff" && value == 1)
        );
    }

    #[test]
    fn default_snapshot_has_no_string_log_dependency() {
        let snapshot = DiagnosticSnapshot::default();

        assert_eq!(snapshot.records(), []);
    }
}
