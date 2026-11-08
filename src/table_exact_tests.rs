use alloc::vec::Vec;

use group::{Curve, CurveAffine as _, Group, ff::Field};
use pasta_curves::pallas;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use super::{LONGEST_MESSAGE_WORDS, TableDomain, evaluate};
use crate::{HashDomain, K, s_generator};

fn check(q: pallas::Point, words: &[u16]) {
    let initial = (0..words.len()).fold(q, |point, _| point.double());
    let bits = words
        .iter()
        .flat_map(|word| (0..K).map(move |bit| word & (1 << bit) != 0));
    let expected = HashDomain::from_Q(q).hash_to_point(bits);
    let actual = evaluate(initial, words);
    assert_eq!(
        Option::<pallas::Point>::from(actual),
        Option::<pallas::Point>::from(expected),
    );
}

#[test]
fn weighted_checks_match_spec_for_forced_exceptions() {
    let half = pallas::Scalar::from(2).invert().unwrap();
    let n = LONGEST_MESSAGE_WORDS.min(52);
    let words: Vec<u16> = (0..n).map(|i| u16::try_from(i % 1024).unwrap()).collect();
    for position in [0, n / 2, n - 1] {
        let s = s_generator(usize::from(words[position])).to_curve();
        // The first three are exactly the three exceptional relations. Identity
        // checks also cover non-protocol test domains. S = 2A is not exceptional.
        for target in [s, -s, -s * half, pallas::Point::identity(), s * half] {
            // Solve backwards for Q so the recurrence reaches target at this position.
            let q = words[..position].iter().rev().fold(target, |acc, word| {
                (acc - s_generator(usize::from(*word))) * half
            });
            check(q, &words);
        }
    }
}

#[test]
fn weighted_checks_match_spec_for_random_domains() {
    let mut rng = StdRng::seed_from_u64(0x7765696768746564);
    for _ in 0..128 {
        let n = rng.random_range(1..=LONGEST_MESSAGE_WORDS.min(52));
        let words: Vec<_> = (0..n).map(|_| rng.random_range(0..1 << K)).collect();
        check(pallas::Point::random(&mut rng), &words);
    }
}

#[test]
fn precompute_initializes_shared_tables_and_covered_first_steps() {
    for _ in 0..2 {
        super::precompute();
        assert_eq!(super::precomputed_message_words(), LONGEST_MESSAGE_WORDS);
        for (initialized, &words) in super::FIRST_STEPS_INITIALIZED
            .iter()
            .zip(&super::KNOWN_DOMAIN_WORDS)
        {
            assert_eq!(
                initialized.load(core::sync::atomic::Ordering::Acquire),
                words <= LONGEST_MESSAGE_WORDS,
            );
        }
    }
}

#[test]
fn first_steps_preserve_forced_exceptions() {
    let n = LONGEST_MESSAGE_WORDS.min(52);
    let tables = super::tables();
    let addends = &tables.generators[n - 1];
    let doubled = &tables.generators[n];
    let t = addends[7].affine().to_curve();
    for start in [t.double(), -t.double(), -t] {
        let start = super::Entry::from(&start.to_affine());
        assert!(super::build_first_steps(start, addends, doubled).is_none());
    }
    let row = super::build_first_steps(addends[7], addends, doubled)
        .expect("an ordinary doubling in the weighted representation is not exceptional");
    assert_eq!(row[7], doubled[7]);
}

#[test]
fn prewarmed_batches_match_the_specification() {
    super::precompute();
    let mut rng = StdRng::seed_from_u64(0x707265636f6d7075);
    for (domain, n) in super::KNOWN_DOMAINS
        .into_iter()
        .zip(super::KNOWN_DOMAIN_WORDS)
    {
        if n > LONGEST_MESSAGE_WORDS {
            continue;
        }
        let tabled = TableDomain::new(domain).unwrap();
        let reference = HashDomain::from_Q(HashDomain::new(domain).Q());
        for width in [1, super::LOCKSTEP_LANES] {
            let words: Vec<u16> = (0..n * width)
                .map(|_| rng.random_range(0..1 << K))
                .collect();
            let expected: Vec<Option<pallas::Base>> = words
                .chunks_exact(n)
                .map(|msg| {
                    reference
                        .hash(
                            msg.iter()
                                .flat_map(|word| (0..K).map(move |bit| word & (1 << bit) != 0)),
                        )
                        .into()
                })
                .collect();
            let actual: Vec<Option<pallas::Base>> = tabled
                .hash_many_vartime(&words, n)
                .into_iter()
                .map(Option::from)
                .collect();
            assert_eq!(actual, expected, "{domain}, {width} messages");
        }
    }
}

#[cfg(feature = "merkle-table")]
#[test]
fn merkle_table_covers_the_full_orchard_message() {
    super::precompute();
    assert!(super::tables().generators.len() >= 53);
    let domain = HashDomain::new("z.cash:Orchard-MerkleCRH");
    assert!(domain.precomputed_message_words() >= 52);
    assert_eq!(
        HashDomain::from_Q(domain.Q()).precomputed_message_words(),
        0
    );
}
