use std::collections::BTreeMap;

use crate::diagnostic::DiagnosticRecord;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiagnosticCounters {
    values: BTreeMap<&'static str, u64>,
}

impl DiagnosticCounters {
    pub fn increment(&mut self, name: &'static str) {
        *self.values.entry(name).or_insert(0) += 1;
    }

    pub fn add(&mut self, name: &'static str, amount: u64) {
        *self.values.entry(name).or_insert(0) += amount;
    }

    pub fn get(&self, name: &'static str) -> u64 {
        self.values.get(name).copied().unwrap_or(0)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, u64)> + '_ {
        self.values.iter().map(|(name, value)| (*name, *value))
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

    pub fn record(&mut self, record: DiagnosticRecord) {
        self.records.push(record);
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
    use crate::diagnostic::{DiagnosticSnapshot, Diagnostics};

    #[test]
    fn counter_snapshot_is_structured_and_stable() {
        let mut diagnostics = Diagnostics::default();
        diagnostics.increment("runtime.command_queued");
        let first = diagnostics.snapshot();
        diagnostics.increment("runtime.command_queued");

        assert_eq!(first.counter("runtime.command_queued"), 1);
        assert_eq!(diagnostics.snapshot().counter("runtime.command_queued"), 2);
    }

    #[test]
    fn default_snapshot_has_no_string_log_dependency() {
        let snapshot = DiagnosticSnapshot::default();

        assert_eq!(snapshot.records(), []);
    }
}
