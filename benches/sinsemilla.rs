//! Benchmarks for Sinsemilla hashing.
//!
//! The 520-bit case is the one that matters for wallet syncing and for block validation:
//! a fixed domain and exactly 52 words.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use group::ff::Field;
use pasta_curves::pallas;
use sinsemilla::{CommitDomain, HashDomain, C, K};

/// Length in bits of a Merkle parent hash: 52 [`K`]-bit words.
const MERKLE_HASH_BITS: usize = 52 * K;

/// Length in bits of an Orchard note commitment message: `g_d`, `pk_d`, `v`, `rho`, `psi`.
const NOTE_COMMIT_BITS: usize = 256 + 256 + 64 + 255 + 255;

/// Length in bits of an Orchard `CommitIvk` message: `ak` and `nk`, each `L_ORCHARD_BASE`.
///
/// Only the table benchmarks hash at this length, and they need the `std` feature.
#[cfg(feature = "std")]
const COMMIT_IVK_BITS: usize = 255 + 255;

/// Number of messages in the batched benchmark.
const BATCH: usize = 1 << 10;

/// Returns `n` random bits.
fn random_bits(n: usize) -> Vec<bool> {
    rand::random_iter().take(n).collect()
}

fn bits_label(bits: usize) -> String {
    format!("{bits}-bits")
}

fn batch_label(bits: usize) -> String {
    format!("{}/{BATCH}-messages", bits_label(bits))
}

fn hash(c: &mut Criterion) {
    let domain = HashDomain::new("z.cash:test-Sinsemilla");

    let mut group = c.benchmark_group("hash");
    for bits in [K, MERKLE_HASH_BITS, K * C] {
        let msg = random_bits(bits);
        group.throughput(Throughput::Elements((bits / K) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(bits_label(bits)),
            &msg,
            |b, msg| b.iter(|| domain.hash(black_box(msg).iter().copied())),
        );
    }
    group.finish();
}

fn hash_batch(c: &mut Criterion) {
    let domain = HashDomain::new("z.cash:test-Sinsemilla");
    let msgs: Vec<_> = (0..BATCH).map(|_| random_bits(MERKLE_HASH_BITS)).collect();

    let mut group = c.benchmark_group("hash/batch");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function(
        BenchmarkId::from_parameter(batch_label(MERKLE_HASH_BITS)),
        |b| {
            b.iter(|| {
                for msg in &msgs {
                    black_box(domain.hash(msg.iter().copied()));
                }
            })
        },
    );
    group.finish();
}

/// Domain construction, which Orchard pays on every Merkle hash and note commitment.
///
/// The Zcash personalizations return a stored point; any other one is hashed to the curve,
/// so both are measured.
fn domain(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain");
    for (label, name) in [
        ("orchard-merkle-crh", "z.cash:Orchard-MerkleCRH"),
        ("unknown", "z.cash:test-Sinsemilla"),
    ] {
        group.bench_function(BenchmarkId::new("hash", label), |b| {
            b.iter(|| HashDomain::new(black_box(name)))
        });
    }
    for (label, name) in [
        ("orchard-note-commit", "z.cash:Orchard-NoteCommit"),
        ("orchard-commit-ivk", "z.cash:Orchard-CommitIvk"),
        ("unknown", "z.cash:test-NoteCommit"),
    ] {
        group.bench_function(BenchmarkId::new("commit", label), |b| {
            b.iter(|| CommitDomain::new(black_box(name)))
        });
    }
    group.finish();
}

fn commit(c: &mut Criterion) {
    let domain = CommitDomain::new("z.cash:test-NoteCommit");

    let msg = random_bits(NOTE_COMMIT_BITS);
    let r = pallas::Scalar::random(&mut rand::rng());

    let mut group = c.benchmark_group("short_commit");
    group.bench_function(
        BenchmarkId::from_parameter(bits_label(NOTE_COMMIT_BITS)),
        |b| b.iter(|| domain.short_commit(black_box(&msg).iter().copied(), black_box(&r))),
    );
    group.finish();
}

/// The position-weighted tables against the specification-shaped evaluator, on the
/// messages Orchard hashes.
///
/// Both sides go through `HashDomain::hash`, because that is how a caller reaches either
/// one: a personalization the specification fixes dispatches to the tables, and any other
/// personalization of the same length does not. So the pair differs in the evaluator and
/// in nothing else.
///
/// Each is measured twice, and the difference between the two is the whole question of
/// where the table lives:
///
/// - `one-message` hashes a single message over and over, so the handful of cache lines
///   its words select stay in L1 for the whole run. This is the warm-L1 best case, and it
///   exercises only the fraction of the table that one message touches.
/// - `many-messages` walks `BATCH` distinct messages, so each hash reads a fresh scatter
///   of lines and the working set is the whole table. This is what a wallet sees.
#[cfg(feature = "std")]
fn table(c: &mut Criterion) {
    let mut group = c.benchmark_group("table");
    for (label, personalization, bits) in [
        ("merkle-crh", "z.cash:Orchard-MerkleCRH", MERKLE_HASH_BITS),
        (
            "note-commit",
            "z.cash:Orchard-NoteCommit-M",
            NOTE_COMMIT_BITS,
        ),
        ("commit-ivk", "z.cash:Orchard-CommitIvk-M", COMMIT_IVK_BITS),
    ] {
        // Same length, same code path, but no start table, so it takes the accumulator.
        let untabled = HashDomain::new("z.cash:test-Sinsemilla");
        let tabled = HashDomain::new(personalization);
        let msgs: Vec<_> = (0..BATCH).map(|_| random_bits(bits)).collect();

        for (domain, evaluator) in [(&untabled, "generic"), (&tabled, "table")] {
            group.bench_function(
                BenchmarkId::new(format!("{evaluator}/one-message"), label),
                |b| b.iter(|| domain.hash(black_box(&msgs[0]).iter().copied())),
            );

            group.throughput(Throughput::Elements(BATCH as u64));
            group.bench_function(
                BenchmarkId::new(format!("{evaluator}/many-messages"), label),
                |b| {
                    b.iter(|| {
                        for msg in &msgs {
                            black_box(domain.hash(msg.iter().copied()));
                        }
                    })
                },
            );
        }
    }
    group.finish();
}

#[cfg(feature = "std")]
criterion_group!(benches, hash, hash_batch, domain, commit, table);
#[cfg(not(feature = "std"))]
criterion_group!(benches, hash, hash_batch, domain, commit);
criterion_main!(benches);
