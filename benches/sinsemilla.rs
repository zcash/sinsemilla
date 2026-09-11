//! Benchmarks for Sinsemilla hashing.
//!
//! The 520-bit case is the one that matters for wallet syncing and for block validation:
//! a fixed domain and exactly 52 words. `hash/batch` measures it separately from a single
//! evaluation because a specialisation that trades table size for arithmetic only pays
//! for itself when the table stays resident across many of them.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use group::ff::Field;
use pasta_curves::pallas;
use sinsemilla::{CommitDomain, HashDomain, C, K};

/// Length in bits of a Merkle parent hash: 52 [`K`]-bit words.
const MERKLE_HASH_BITS: usize = 52 * K;

/// Length in bits of an Orchard note commitment message: `g_d`, `pk_d`, `v`, `rho`, `psi`.
const NOTE_COMMIT_BITS: usize = 256 + 256 + 64 + 255 + 255;

/// Number of messages in the batched benchmark.
///
/// Large enough that a precomputed table cannot stay in L1, so the batch measures the
/// steady state a wallet or full node sees rather than a warm-cache best case.
const BATCH: usize = 1 << 10;

/// Returns `n` random bits.
fn random_bits(n: usize) -> Vec<bool> {
    rand::random_iter().take(n).collect()
}

fn hash(c: &mut Criterion) {
    let domain = HashDomain::new("z.cash:test-Sinsemilla");

    let mut group = c.benchmark_group("hash");
    for bits in [K, MERKLE_HASH_BITS, K * C] {
        let msg = random_bits(bits);
        group.throughput(Throughput::Elements((bits / K) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bits), &msg, |b, msg| {
            b.iter(|| domain.hash(black_box(msg).iter().copied()))
        });
    }
    group.finish();
}

fn hash_batch(c: &mut Criterion) {
    let domain = HashDomain::new("z.cash:test-Sinsemilla");
    let msgs: Vec<_> = (0..BATCH).map(|_| random_bits(MERKLE_HASH_BITS)).collect();

    let mut group = c.benchmark_group("hash/batch");
    group.throughput(Throughput::Elements(BATCH as u64));
    group.bench_function(BenchmarkId::from_parameter(BATCH), |b| {
        b.iter(|| {
            for msg in &msgs {
                black_box(domain.hash(msg.iter().copied()));
            }
        })
    });
    group.finish();
}

fn commit(c: &mut Criterion) {
    let domain = CommitDomain::new("z.cash:test-NoteCommit");

    let msg = random_bits(NOTE_COMMIT_BITS);
    let r = pallas::Scalar::random(&mut rand::rng());

    c.bench_function("short_commit", |b| {
        b.iter(|| domain.short_commit(black_box(&msg).iter().copied(), black_box(&r)))
    });
}

criterion_group!(benches, hash, hash_batch, commit);
criterion_main!(benches);
