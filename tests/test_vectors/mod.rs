//! Test vectors vendored from [`zcash-test-vectors`].
//!
//! # Regenerating
//!
//! ```text
//! ztv=$(mktemp -d)
//! git clone --depth 1 https://github.com/zcash/zcash-test-vectors.git "$ztv"
//!
//! # Generate, into the layout the upstream repository commits.
//! (cd "$ztv" && python3 -m zcash_test_vectors.orchard.sinsemilla -t rust) \
//!     | rustfmt --edition 2021 \
//!     > "$ztv/test-vectors/rust/orchard_sinsemilla.rs"
//!
//! # Copy into this directory, rewriting the visibility on the way.
//! sed 's/pub(crate) /pub /' "$ztv/test-vectors/rust/orchard_sinsemilla.rs" \
//!     > tests/test_vectors/orchard_sinsemilla.rs
//!
//! rm -rf "$ztv"
//! cargo test --test kat
//! ```
//!
//! [`zcash-test-vectors`]: https://github.com/zcash/zcash-test-vectors

pub mod orchard_sinsemilla;
