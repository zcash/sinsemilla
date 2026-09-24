//! Chooses the default table budget for the target, and makes a change to
//! `SINSEMILLA_TABLE_LIMIT` rebuild the crate.
//!
//! The budget has to be decided here rather than with `cfg!` in the crate, because it is
//! a byte count per target operating system and a build script is where the target is
//! available as a value.

fn main() {
    println!("cargo::rerun-if-env-changed=SINSEMILLA_TABLE_LIMIT");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!(
        "cargo::rustc-env=SINSEMILLA_DEFAULT_TABLE_LIMIT={}",
        default_budget(&target_os),
    );
}

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
///
/// A budget covers `budget / 64 KiB - 1` words. The tabulated domains hash 51 words for
/// `CommitIvk`, 52 for a Merkle parent hash and 109 for a note commitment, so these
/// budgets cover 63 words on macOS, 15 elsewhere and 1 on Android: macOS reaches the
/// first two, and no target reaches the note commitment. Raising the macOS budget to
/// 6.875 MiB, 110 rows, would cover all three; that is two of the four to five cores'
/// share of the 16 MiB an Apple silicon cluster shares, rather than one.
fn default_budget(target_os: &str) -> usize {
    match target_os {
        "android" => 128 << 10,
        "macos" => 4 << 20,
        _ => 1 << 20,
    }
}
