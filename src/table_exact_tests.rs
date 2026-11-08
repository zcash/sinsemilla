use alloc::vec::Vec;

use group::{CurveAffine as _, Group, ff::Field};
use pasta_curves::pallas;
use rand::{RngExt, SeedableRng, rngs::StdRng};

use super::{LONGEST_MESSAGE_WORDS, TableDomain};
use crate::{HashDomain, K, s_generator};

fn check(q: pallas::Point, words: &[u16]) {
    let initial = (0..words.len()).fold(q, |point, _| point.double());
    let bits = words
        .iter()
        .flat_map(|word| (0..K).map(move |bit| word & (1 << bit) != 0));
    let expected = HashDomain::from_Q(q).hash_to_point(bits);
    let actual = TableDomain::evaluate(words, initial);
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
fn precompute_initializes_shared_tables() {
    super::precompute();
    assert_eq!(super::precomputed_message_words(), LONGEST_MESSAGE_WORDS);
    super::precompute();
    assert_eq!(super::precomputed_message_words(), LONGEST_MESSAGE_WORDS);
}

#[cfg(feature = "merkle-table")]
#[test]
fn merkle_table_covers_the_full_orchard_message() {
    super::precompute();
    assert!(super::TABLES.generators.len() >= 53);
    let domain = HashDomain::new("z.cash:Orchard-MerkleCRH");
    assert!(domain.precomputed_message_words() >= 52);
    assert_eq!(
        HashDomain::from_Q(domain.Q()).precomputed_message_words(),
        0
    );
}
