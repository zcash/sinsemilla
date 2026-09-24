# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `sinsemilla::CommitDomain::new_with_separate_domains`
- The `std` feature, which is not enabled by default. It makes
  `HashDomain::hash_to_point`, `HashDomain::hash`, `CommitDomain::commit` and
  `CommitDomain::short_commit` faster for the personalizations listed below, by
  evaluating them against lookup tables built on first use. The budget for those
  tables is sized to the private L2 cache of the target: 4 MiB on macOS, 128 KiB
  on Android, and 1 MiB everywhere else. Set `SINSEMILLA_TABLE_LIMIT`, in bytes,
  at build time to change it.

  The budget decides how long a message the tables cover, at 64 KiB per word, and
  a message longer than that is hashed as before. Orchard hashes 51 words for
  `CommitIvk`, 52 for a Merkle parent hash and 109 for a note commitment, so at
  the default budgets macOS covers the first two and no target covers the note
  commitment. To change that, set `SINSEMILLA_TABLE_LIMIT` to at least 3.31 MiB
  for a Merkle parent hash, or 6.875 MiB for every message Orchard hashes.

### Changed
- MSRV is now 1.88.
- Migrated to `ff 0.14` and `group 0.14`.
- The following set of personalizations now have internal pre-computations,
  giving performance improvements to the `HashDomain` and `CommitDomain`
  constructors:
  - `z.cash:Orchard-MerkleCRH`
  - `z.cash:Orchard-NoteCommit`
  - `z.cash:Orchard-CommitIvk`
- `HashDomain::hash_to_point` and `HashDomain::hash` are faster and no longer
  allocate internally.
- `sinsemilla::SINSEMILLA_S` is now a `static` instead of a `const`. It can no
  longer be used in a const context.

## [0.1.0] - 2024-12-13
Initial release, extracted from `halo2_gadgets 0.3.0`. Includes minor changes
for `no-std` support.
