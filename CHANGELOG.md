# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Rust's notion of
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- MSRV is now 1.88.
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
