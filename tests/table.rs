//! Differential and structural tests for the compiled-in position-weighted tables.
//!
//! The tables are produced by `build.rs`, which is only a producer: nothing about the
//! generated literals is obvious from reading them, and a wrong entry would still hash
//! *something*. Two properties make them safe to ship, and both are checked here:
//!
//! 1. every entry equals what the curve arithmetic says it should be, recomputed from
//!    the Sinsemilla generators and the domain points in this test rather than trusted
//!    from the generator;
//! 2. the table evaluator agrees with the specification-shaped `HashDomain` on messages
//!    where the specification is not bottom.
//!
//! The second would catch a mistake in the weights, the row order, or the Horner
//! combination; the first says *where* such a mistake is.

use group::{CurveAffine as _, Group, GroupEncoding};
use pasta_curves::{arithmetic::CurveExt, pallas};
use sinsemilla::{table, HashDomain, K, Q_PERSONALIZATION, S_PERSONALIZATION};

fn random_bits(n: usize) -> Vec<bool> {
    rand::random_iter().take(n).collect()
}

/// The specification's answer, which the tables must reproduce.
fn reference(domain: &HashDomain, msg: &[bool]) -> pallas::Point {
    Option::from(domain.hash_to_point(msg.iter().copied())).expect("not an exceptional input")
}

/// The table agrees with the generic evaluator, for every known domain, at the message
/// lengths those domains are actually used at and at the boundaries of what is covered.
#[test]
fn agrees_with_generic() {
    // 51 words is CommitIvk, 52 a Merkle parent hash, 109 a note commitment; 1 and
    // LONGEST_MESSAGE_WORDS are the ends of the covered range, and the two around 10
    // exercise a partly-filled final word.
    let lengths = [
        1,
        9,
        11,
        51 * K,
        52 * K,
        109 * K,
        table::LONGEST_MESSAGE_WORDS * K,
    ];

    for personalization in table::KNOWN_DOMAINS {
        let domain = HashDomain::new(personalization);
        let tabled = table::TableDomain::new(personalization).expect("a known domain");

        for bits in lengths {
            let msg = random_bits(bits);
            let words = table::to_words(msg.iter().copied()).expect("a covered length");

            assert_eq!(
                tabled.hash_to_point(&words).to_bytes(),
                reference(&domain, &msg).to_bytes(),
                "{personalization}, {bits} bits",
            );
            assert_eq!(
                tabled.hash(&words),
                Option::<pallas::Base>::from(domain.hash(msg.iter().copied()))
                    .expect("not an exceptional input"),
                "hash, {personalization}, {bits} bits",
            );
        }
    }
}

/// It also agrees on the all-zero and all-one messages, which the random trials will
/// never draw and which exercise word 0 and word 2^K - 1 in every position.
#[test]
fn agrees_on_extreme_messages() {
    let personalization = table::KNOWN_DOMAINS[0];
    let domain = HashDomain::new(personalization);
    let tabled = table::TableDomain::new(personalization).expect("a known domain");

    for bit in [false, true] {
        let msg = vec![bit; 52 * K];
        let words = table::to_words(msg.iter().copied()).expect("a covered length");
        assert_eq!(
            tabled.hash_to_point(&words).to_bytes(),
            reference(&domain, &msg).to_bytes(),
            "all-{bit} message",
        );
    }
}

/// Every generator-table entry is `[2^(q * STRIDE)] S_w`, recomputed here. This is what
/// makes the build script a producer rather than a trusted input.
#[test]
fn generator_entries_are_the_weighted_generators() {
    let hasher = pallas::Point::hash_to_curve(S_PERSONALIZATION);

    for (q, row) in table::rows().enumerate() {
        let weight = q * table::STRIDE_USED;

        for (w, entry) in row.iter().enumerate() {
            let mut expected = hasher(&(w as u32).to_le_bytes());
            for _ in 0..weight {
                expected = expected.double();
            }
            assert_eq!(
                entry.to_curve().to_bytes(),
                expected.to_bytes(),
                "row {q}, entry {w}, weight 2^{weight}",
            );
        }
    }
}

/// Every start-table entry is `[2^n] Q_d`, recomputed here from the personalization.
#[test]
fn start_entries_are_the_weighted_domain_points() {
    let hasher = pallas::Point::hash_to_curve(Q_PERSONALIZATION);

    for personalization in table::KNOWN_DOMAINS {
        let mut expected = hasher(personalization.as_bytes());

        for n in 1..=table::LONGEST_MESSAGE_WORDS {
            expected = expected.double();
            assert_eq!(
                table::start(personalization, n)
                    .expect("a known domain and a covered length")
                    .to_bytes(),
                expected.to_bytes(),
                "{personalization}, [2^{n}] Q",
            );
        }
    }
}

/// The rows cover every position the tables claim to cover.
#[test]
fn rows_cover_the_longest_message() {
    assert!(table::rows().count() * table::STRIDE_USED >= table::LONGEST_MESSAGE_WORDS);
    assert!(
        (table::rows().count() - 1) * table::STRIDE_USED < table::LONGEST_MESSAGE_WORDS,
        "a row is unused, so the budget bought nothing",
    );
}

/// An unknown personalization has no start table, so it has no table domain.
#[test]
fn unknown_personalizations_have_no_table() {
    assert!(table::TableDomain::new("z.cash:test-Sinsemilla").is_none());
}

/// Messages the tables do not cover are reported, not hashed wrongly.
#[test]
fn uncovered_lengths_are_rejected() {
    assert!(table::to_words(core::iter::empty()).is_none());
    assert!(table::to_words(core::iter::repeat_n(
        false,
        table::LONGEST_MESSAGE_WORDS * K + 1
    ))
    .is_none());
}
