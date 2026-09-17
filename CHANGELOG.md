# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `sinsemilla::table`, position-weighted tables for the hash domains Zcash
  specifies, generated at build time into read-only data. One generator table
  serves every domain and every message length, because the coefficient of a
  word counted from the end of the message depends on neither; a start table
  holds `[2^n] Q` per known personalization and per length, up to 128 words.
  `TableDomain::hash` evaluates a hash with one mixed addition per word and
  `STRIDE - 1` doublings in total, against one doubling per word for
  `HashDomain`.

  It is a separate entry point, not a faster path inside `HashDomain`, because
  it does NOT reproduce the specification's bottom: the specification uses
  incomplete addition and is bottom on exceptional inputs, and reassociating the
  sum never forms those intermediate accumulators. The two agree on every input
  where the specification is not bottom. A caller that needs bottom, or a
  message longer than the tables cover, must use `HashDomain`.

  The table size follows a budget sized to the private L2 it shares with the
  caller: 128 KiB on Android, 4 MiB on macOS, 1 MiB elsewhere. Set
  `SINSEMILLA_TABLE_LIMIT` (bytes) at build time to choose another; it selects
  the stride, which changes the cost and never the result.

### Changed
- MSRV is now 1.88.
- `HashDomain::hash_to_point` now computes the accumulator step `[2] A + S` as
  one doubling and one mixed addition rather than as two incomplete additions,
  which is about 1.3x faster on a 52-word message. This is not an observable
  change: it returns the bottom element on exactly the same inputs as before.
- `HashDomain::new` and `CommitDomain::new` (and `new_with_separate_domains`)
  return stored points for the personalizations Zcash specifies instead of
  hashing to the curve: `z.cash:Orchard-MerkleCRH`, `z.cash:Orchard-NoteCommit`,
  `z.cash:Orchard-CommitIvk`, and the OrchardZSA hash domain
  `z.cash:ZSA-NoteCommit`. Orchard constructs a domain on every Merkle hash and
  note commitment, so this removes most of their cost: on an Apple M-series
  laptop, domain construction plus a 520-bit Merkle hash goes from 89 us to
  30 us, and plus a note commitment from 189 us to 73 us. Every other
  personalization is still hashed to the curve, and the results are unchanged.
- The Sinsemilla `S` generators and the stored `Q` and `R` points are now kept
  as affine points built at compile time with `pasta_curves`'
  `from_xy_unchecked`, instead of being converted from coordinates with
  `from_xy` (which re-checks the curve equation) on every message word and every
  domain construction. The tests check each point against a fresh hash to the
  curve. On an Apple M-series laptop, a 520-bit hash goes from 30.0 us to
  27.3 us, a 1086-bit short commitment from 95.6 us to 88.9 us, and constructing
  the `z.cash:Orchard-MerkleCRH` and `z.cash:Orchard-NoteCommit` domains from
  76 ns to 11 ns and from 248 ns to 117 ns. Results are unchanged.

  The affine `S` table duplicates the 64 KiB of coordinates in `SINSEMILLA_S`,
  but only a consumer that uses that public table pays for both: one that only
  hashes never names it, so the linker drops it and the binary ends up slightly
  smaller. Measured on aarch64-apple-darwin with a release example binary,
  640 608 -> 640 048 bytes when only hashing, and 642 512 -> 708 112 bytes when
  also using `SINSEMILLA_S`. The table is built at compile time into read-only
  data, so it costs no RAM and no start-up time.
- `HashDomain::hash_to_point` reads the message a `K`-bit word at a time
  instead of first collecting it into a `Vec<bool>`. The result, the inputs
  that return bottom, and the panic on messages longer than `K * C` bits are
  unchanged.
- Migrated to `ff 0.14` and `group 0.14`. This is a breaking change for
  consumers: `group 0.14` moves the affine operations onto the new
  `group::CurveAffine` supertrait, so a caller that imported
  `group::cofactor::CofactorCurveAffine` or `group::prime::PrimeCurveAffine`
  to reach `to_curve`, `identity` or `generator` on `pasta_curves::pallas::Affine`
  now imports `group::CurveAffine` instead.
- `sinsemilla::SINSEMILLA_S` is now a `static` instead of a `const`. It can no
  longer be used in a const context.

## [0.1.0] - 2024-12-13
Initial release, extracted from `halo2_gadgets 0.3.0`. Includes minor changes
for `no-std` support.
