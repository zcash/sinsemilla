# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-11-26
### Added
- `sinsemilla::HashDomain::hash_many_vartime` for public inputs.
- Table precomputations for known personalisations (for their expected message
  lengths), behind the new `precompute-tables` feature flag. It supports
  `no_std` with `alloc` on targets with pointer-sized atomics. It adds the
  following new APIs:
  - `sinsemilla::precompute`
  - `sinsemilla::HashDomain::precomputed_message_words`, behind the
    `test-dependencies` feature flag.
  - The generator table's default size is 4 MiB on macOS, 128 KiB on Android,
    and 1 MiB everywhere else. Set `SINSEMILLA_TABLE_LIMIT`, in bytes, at build
    time to change it. Once initialized, the additional per-known-domain tables
    add up to about 2.3% of the generator table's size.
- The new `merkle-table` feature enables `precompute-tables` and guarantees
  table coverage of the 52-word Orchard Merkle hash, even with a smaller target
  default or `SINSEMILLA_TABLE_LIMIT`.

### Changed
- The following set of personalizations now have internal pre-computations,
  giving performance improvements to the `HashDomain` and `CommitDomain`
  constructors:
  - `z.cash:Orchard-MerkleCRH`
  - `z.cash:Orchard-NoteCommit`
  - `z.cash:Orchard-CommitIvk`
- `HashDomain::hash_to_point` and `HashDomain::hash` are faster and no longer
  allocate internally. With `precompute-tables`, the first eligible hash may
  still allocate to initialize tables; use `sinsemilla::precompute` to control
  when that happens.

## [0.2.0] - 2026-09-25
### Added
- `sinsemilla::CommitDomain::new_with_separate_domains`

### Changed
- MSRV is now 1.88.
- Migrated to `ff 0.14`, `group 0.14`, `pasta_curves 0.6`.
- `sinsemilla::SINSEMILLA_S` is now a `static` instead of a `const`. It can no
  longer be used in a const context.

## [0.1.0] - 2024-12-13
Initial release, extracted from `halo2_gadgets 0.3.0`. Includes minor changes
for `no-std` support.
