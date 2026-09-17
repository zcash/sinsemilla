//! Generates the position-weighted tables the crate hashes with.
//!
//! The entries are `[2^w] S_j` for the Sinsemilla generators and `[2^n] Q_d` for the
//! known domain points, so they cannot be written as `const` expressions: the curve
//! arithmetic that produces them is not `const fn`. They are computed here and written
//! out as `from_xy_unchecked` literals, which the crate includes. Every entry is
//! recomputed from the curve arithmetic again in the crate's tests, so this generator is
//! only a producer; the test is the guarantee.
//!
//! Two tables, and the split is what lets one of them serve every domain and length:
//!
//! - the generator table `T[q][w] = [2^(q * stride)] S_w`, weighted by the position
//!   counted FROM THE END of the message. Those weights depend neither on the domain nor
//!   on the message length, so one table serves both;
//! - the start table `START[d][n] = [2^n] Q_d`, one row per known personalization and one
//!   column per message length in words. This is what the domain and the length are
//!   absorbed into.
//!
//! See `src/table.rs` for the algebra and for what the budget means.

use std::{env, fmt::Write as _, fs, path::PathBuf};

use ff::PrimeField;
use group::{Curve, CurveAffine as _, Group};
use pasta_curves::{arithmetic::CurveAffine, arithmetic::CurveExt, pallas};

/// Number of Sinsemilla `S` generators, which is the number of entries in a table row.
const S_LEN: usize = 1 << 10;

/// Bytes per affine point: two 32-byte base field elements.
const POINT_BYTES: usize = 64;

/// Longest message the tables cover, in `K`-bit words.
///
/// The Orchard messages are a 51-word `CommitIvk` input, a 52-word Merkle parent hash and
/// a 109-word note commitment, and ZIP 226 adds an asset base to the last of those. 128
/// words (1280 bits) covers all of them; a longer message falls back to the generic
/// evaluator rather than being hashed wrongly.
const MAX_WORDS: usize = 128;

/// Personalizations, duplicated from the crate because a build script cannot use it.
const S_PERSONALIZATION: &str = "z.cash:SinsemillaS";
const Q_PERSONALIZATION: &str = "z.cash:SinsemillaQ";

/// The hash domains Zcash specifies, in the order the generated table stores them. These
/// are the same personalizations whose `Q` the crate already stores.
const DOMAINS: [&str; 4] = [
    "z.cash:Orchard-MerkleCRH",
    "z.cash:Orchard-NoteCommit-M",
    "z.cash:Orchard-CommitIvk-M",
    "z.cash:ZSA-NoteCommit-M",
];

/// Default budget for the generator table, in bytes, sized to the private L2 the table
/// has to share with the caller's own working set.
///
/// The hardware figures describe the market as of September 2026 and will date.
///
/// - Android: the phones sold in the largest numbers run most of their cores as
///   Cortex-A55, whose private L2 is typically 128 KiB (Arm allows 64 to 256 KiB), and
///   the scheduler may place a wallet on one of them.
/// - iOS: every iPhone chip since the A13 has 2 performance and 4 efficiency cores, none
///   with private L2. The efficiency cores share 4 to 8 MiB between four of them, about
///   1 MiB each, and a wallet syncing in the background is likely to run there. So iOS
///   takes the same budget as the default rather than the full table macOS gets.
/// - macOS: Apple silicon from M2 onwards shares 16 MiB of L2 between four performance
///   cores, so one core's share is 4 MiB.
/// - Everything else: private per-core L2 is 512 KiB on AMD Zen 3, 1 MiB on Zen 4 and Zen
///   5 and on Arm Neoverse N1, V1 and N2, and 1.25 to 2 MiB on recent Intel. 1 MiB is not
///   the smallest of those, so this is a trade-off rather than a guarantee.
fn default_budget(target_os: &str) -> usize {
    match target_os {
        "android" => 128 << 10,
        "macos" => 4 << 20,
        _ => 1 << 20,
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SINSEMILLA_TABLE_LIMIT");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let budget = match env::var("SINSEMILLA_TABLE_LIMIT") {
        Ok(v) => v
            .parse::<usize>()
            .expect("SINSEMILLA_TABLE_LIMIT must be a number of bytes"),
        Err(_) => default_budget(&target_os),
    };

    // One row covers `stride` positions and `rows * stride` must reach MAX_WORDS, so the
    // largest table within the budget is the one with the most rows, and never fewer
    // than one.
    let rows = (budget / (S_LEN * POINT_BYTES)).clamp(1, MAX_WORDS);
    let stride = MAX_WORDS.div_ceil(rows);
    let rows = MAX_WORDS.div_ceil(stride);

    // The generator table, weighted by position from the end: row q carries
    // 2^(q * stride). Doubling a whole row at a time shares the work across its entries.
    let s_hasher = pallas::Point::hash_to_curve(S_PERSONALIZATION);
    let mut weighted: Vec<pallas::Point> = (0..S_LEN as u32)
        .map(|j| s_hasher(&j.to_le_bytes()))
        .collect();

    let mut table = vec![pallas::Affine::identity(); rows * S_LEN];
    for q in 0..rows {
        pallas::Point::batch_normalize(&weighted, &mut table[q * S_LEN..(q + 1) * S_LEN]);
        if q + 1 < rows {
            for p in weighted.iter_mut() {
                for _ in 0..stride {
                    *p = p.double();
                }
            }
        }
    }

    // The start table: [2^n] Q_d for every known domain and every covered length, with
    // column n - 1 holding [2^n] Q_d.
    let q_hasher = pallas::Point::hash_to_curve(Q_PERSONALIZATION);
    let mut starts = Vec::with_capacity(DOMAINS.len() * MAX_WORDS);
    for domain in DOMAINS {
        let mut point = q_hasher(domain.as_bytes());
        let mut row = Vec::with_capacity(MAX_WORDS);
        for _ in 0..MAX_WORDS {
            point = point.double();
            row.push(point);
        }
        let mut affine = vec![pallas::Affine::identity(); MAX_WORDS];
        pallas::Point::batch_normalize(&row, &mut affine);
        starts.extend(affine);
    }

    let mut out = String::with_capacity(rows * S_LEN * 160);
    out.push_str("// @generated by build.rs. Every entry is checked in the tests.\n");
    writeln!(out, "pub(crate) const STRIDE: usize = {stride};").unwrap();
    writeln!(out, "pub(crate) const ROWS: usize = {rows};").unwrap();
    writeln!(out, "pub(crate) const MAX_WORDS: usize = {MAX_WORDS};").unwrap();
    writeln!(out, "pub(crate) const BUDGET_BYTES: usize = {budget};").unwrap();
    writeln!(
        out,
        "pub(crate) const DOMAINS: [&str; {}] = {DOMAINS:?};",
        DOMAINS.len()
    )
    .unwrap();

    writeln!(
        out,
        "pub(crate) static STARTS: [[pallas::Affine; MAX_WORDS]; {}] = [",
        DOMAINS.len()
    )
    .unwrap();
    for d in 0..DOMAINS.len() {
        out.push_str("    [\n");
        for point in &starts[d * MAX_WORDS..(d + 1) * MAX_WORDS] {
            writeln!(out, "        {},", affine_literal(point)).unwrap();
        }
        out.push_str("    ],\n");
    }
    out.push_str("];\n");

    writeln!(
        out,
        "pub(crate) static TABLE: [[pallas::Affine; {S_LEN}]; ROWS] = ["
    )
    .unwrap();
    for q in 0..rows {
        out.push_str("    [\n");
        for point in &table[q * S_LEN..(q + 1) * S_LEN] {
            writeln!(out, "        {},", affine_literal(point)).unwrap();
        }
        out.push_str("    ],\n");
    }
    out.push_str("];\n");

    let path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("tables.rs");
    fs::write(&path, out).expect("writing the generated tables");
}

/// One affine point as a `from_xy_unchecked` call over two `from_raw` coordinates.
fn affine_literal(point: &pallas::Affine) -> String {
    let coords = point
        .coordinates()
        .expect("no entry is the identity: the generators have prime order");
    format!(
        "pallas::Affine::from_xy_unchecked({}, {})",
        base_literal(coords.x()),
        base_literal(coords.y()),
    )
}

/// One base field element as `Fp::from_raw`, which reads four little-endian limbs.
fn base_literal(value: &pallas::Base) -> String {
    let bytes = value.to_repr();
    let limbs: Vec<String> = bytes
        .chunks(8)
        .map(|c| {
            let mut limb = [0u8; 8];
            limb.copy_from_slice(c);
            format!("0x{:016x}", u64::from_le_bytes(limb))
        })
        .collect();
    format!("pallas::Base::from_raw([{}])", limbs.join(", "))
}
