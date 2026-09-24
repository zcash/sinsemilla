//! Position-weighted tables for the Zcash hash domains, built on first use.
//!
//! # What they compute
//!
//! The specification's recurrence is $A_0 = \mathcal{Q}$,
//! $A_i = \[2\] A_{i-1} + S_{m_i}$, one doubling per word. For a message of $N$ words it
//! expands to
//!
//! $$A_N = \[2^N\] \mathcal{Q} + \sum_{i=1}^{N} \[2^{N-i}\] S_{m_i},$$
//!
//! which is Horner's rule. Horner's rule is normally attractive *because* it avoids
//! storing the powers; here the coefficients come from a fixed set of $2^K$ points and
//! the same hashes run over and over, so the opposite trade is the right one: store the
//! powers. Substituting $B_i = \[2^{N-i}\] A_i$ gives
//! $B_i = B_{i-1} + \[2^{N-i}\] S_{m_i}$, with the doubling gone from the recurrence and
//! the scale unwinding on its own, since $B_0 = \[2^N\] \mathcal{Q}$ and $B_N = A_N$.
//!
//! The table is not merely a cache: with the one-hot encoding $x$ of the message,
//! $H(m) = \[2^N\] \mathcal{Q} + \langle x, T \rangle$, so $T$ is the vector Pedersen
//! generator matrix and this evaluator works in the representation Sinsemilla's
//! collision-resistance proof is stated in, rather than the sequential form the
//! specification writes for the circuit. [`C`] is the matching bound: it stops the
//! coefficients wrapping modulo the group order, which is what keeps a collision a
//! nontrivial relation.
//!
//! # Why one table serves every domain and every length
//!
//! Counting positions from the end of the message, $j = N - i$, the coefficient of
//! $S_{m_{N-j}}$ is $2^j$, which mentions neither the domain nor $N$. So the generator
//! table is shared, and only the term $\[2^N\] \mathcal{Q}$ knows which domain and which
//! length this is. That term is a single point, so it is tabulated per domain and per
//! length instead of being recomputed.
//!
//! # Row and weight
//!
//! A weight is the exponent on the power of two multiplying a generator:
//! `generators[w][j]` is $\[2^w\] S_j$. The weight of a position is its distance from the
//! end of the message, so it does not depend on the domain or on the length.
//!
//! A row is one weight of the table: all $2^K$ generators at that weight, [`ROW_BYTES`]
//! bytes. Rows and weights are one to one, and [`WEIGHTS`] is the count of both.
//!
//! A word is [`K`] = 10 bits, so it takes $2^K$ = 1024 values and a row holds 1024
//! generators. A generator is an affine Pallas point, two 32-byte field elements, so a
//! row is 64 KiB.
//!
//! The messages the tabulated domains hash:
//!
//! | message                | bits                | words |
//! |------------------------|---------------------|-------|
//! | `CommitIvk`            | 255 + 255 = 510     | 51    |
//! | Merkle parent hash     | 10 + 255 + 255 = 520| 52    |
//! | note commitment        | 256 + 256 + 64 + 255 + 255 = 1086, padded to 1090 | 109 |
//!
//! Hashing $N$ words reads weights $0$ to $N-1$ to add and $1$ to $N$ to test, so it
//! needs $N+1$ rows: [`WEIGHTS`] is [`LONGEST_MESSAGE_WORDS`] plus one.
//!
//! # Reproducing $\bot$
//!
//! The specification uses incomplete addition and is $\bot$ when an exceptional case
//! arises, which is a statement about its sequential accumulators $\mathsf{Acc}_i$. This
//! evaluator never forms them. It does not have to: the running position-weighted prefix
//!
//! $$P_0 = \[2^N\] \mathcal{Q}, \qquad
//!   P_i = P_{i-1} + T_i, \qquad T_i = \[2^{N-i}\] S_{m_i}$$
//!
//! satisfies $P_i = \[2^{N-i}\] \mathsf{Acc}_i$, and multiplication by a power of two is
//! injective on a group of odd prime order, so scaling each of
//! [Theorem 5.4.4][theorem544]'s conditions at position $i$ by $\[2^{N-i+1}\]$ is an
//! equivalence rather than an approximation:
//!
//! | specification                                        | rescaled               |
//! |------------------------------------------------------|------------------------|
//! | $\mathsf{Acc}_{i-1} = S_{m_i}$ ($\alpha = -1$)       | $P_{i-1} = \[2\] T_i$  |
//! | $\mathsf{Acc}_{i-1} = -S_{m_i}$ ($\alpha = 1$)       | $P_{i-1} = -\[2\] T_i$ |
//! | $S_{m_i} = -\[2\] \mathsf{Acc}_{i-1}$ ($\alpha = 2$) | $P_{i-1} = -T_i$       |
//!
//! The first two collapse into one test, because $P = \pm X$ exactly when
//! $x(P) = x(X)$: compare $x(P_{i-1})$ against the entry one weight up, which is a table
//! lookup, one squaring and one multiplication (see [`has_x_of`]). The third says
//! $P_{i-1} + T_i = \mathcal{O}$, so it is read off the addition that was happening
//! anyway. The disjunction of the per-position flags is the specification's $\bot$,
//! exactly; it does not matter that the completed sum keeps going after a flag fires, for
//! the same reason the simulated incomplete addition computes $p + q$ and then masks it.
//!
//! The test reads $P_{i-1}$, the prefix through position $i-1$, which the evaluation only
//! holds if it adds one position at a time. That is what requires one row per weight.
//!
//! [theorem544]: https://zips.z.cash/protocol/protocol.pdf#concretesinsemillahash
//! [`HashDomain::from_Q`]: crate::HashDomain::from_Q
//!
//! # Cost
//!
//! The tables are built once, on first use, and live for the rest of the process. The
//! generator table is [`WEIGHTS`] rows of $2^K$ affine points, 64 KiB per row, and the
//! start table is one point per known domain per covered length, so the budget is what
//! decides the memory and the build cost. Whichever call touches the tables first pays
//! for building them, and every other thread waits on it. See [`TABLE_LIMIT`] for the
//! per-target budgets and how to override them.
//!
//! Building is one doubling and one batch inversion per row over $2^K$ points, and no
//! hashes to the curve at all: both the generators and the domain points are constants
//! the crate already stores, so the build only weights them.
//!
//! One evaluation is one mixed addition, one squaring and one multiplication per word,
//! against one doubling and one mixed addition per word for [`HashDomain`]. The
//! benchmarks measure both; see `benches/sinsemilla.rs`.
//!
//! This module needs the `std` feature, for [`std::sync::LazyLock`]. A `no_std` build
//! has no tables and uses [`HashDomain`].
//!
//! [`C`]: crate::C
//! [`HashDomain`]: crate::HashDomain
//! [`HashDomain::hash_to_point`]: crate::HashDomain::hash_to_point

use alloc::{vec, vec::Vec};
use std::sync::LazyLock;

use group::{Curve, CurveAffine as _, Group};
use pasta_curves::{
    arithmetic::{CurveAffine as _, CurveExt},
    pallas,
};
use subtle::{Choice, ConstantTimeEq, CtOption};

use crate::{known_domains, sinsemilla_s, K, Q_PERSONALIZATION};

/// Number of Sinsemilla generators, which is the number of entries in a table row.
const S_LEN: usize = 1 << K;

/// Bytes in one weight row of the generator table.
const ROW_BYTES: usize = S_LEN * core::mem::size_of::<pallas::Affine>();

/// Longest message worth covering, in [`K`]-bit words.
///
/// The longest message the tabulated domains hash is a note commitment, at 109 words.
/// [`WEIGHTS`] is capped here, so a budget above 6.875 MiB buys no further coverage. A
/// longer message belongs on [`HashDomain`], which has no bound below [`C`].
///
/// [`HashDomain`]: crate::HashDomain
/// [`C`]: crate::C
const LONGEST_WORTH_COVERING: usize = 109;

/// Default budget for this target, in bytes, chosen by the build script.
///
/// Reproducing $\bot$ forces one row per weight (see the module documentation), so the
/// budget buys message LENGTH directly: `budget / 64 KiB` rows cover one word fewer than
/// that. The lengths that matter are 51 words for `CommitIvk`, 52 for a Merkle parent
/// hash, and 109 for a note commitment. See `build.rs` for the per-target figures and
/// what they are sized against.
const DEFAULT_TABLE_LIMIT: usize = parse_usize(env!("SINSEMILLA_DEFAULT_TABLE_LIMIT"));

/// The table budget for this build, in bytes.
pub const TABLE_LIMIT: usize = match option_env!("SINSEMILLA_TABLE_LIMIT") {
    // An exported-but-empty variable reads as unset rather than as a build failure, since
    // that is what a shell leaves behind after `SINSEMILLA_TABLE_LIMIT=`.
    Some(limit) if !limit.is_empty() => parse_usize(limit),
    _ => DEFAULT_TABLE_LIMIT,
};

/// Parses a decimal `usize` at compile time, for [`TABLE_LIMIT`].
///
/// # Panics
///
/// Panics at compile time if `s` is not a non-empty run of ASCII digits, which is what
/// reports a mistyped `SINSEMILLA_TABLE_LIMIT` as a build failure rather than as a
/// silently different table.
const fn parse_usize(s: &str) -> usize {
    let bytes = s.as_bytes();
    assert!(!bytes.is_empty(), "SINSEMILLA_TABLE_LIMIT is empty");

    let mut value = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        let digit = bytes[i];
        assert!(
            digit.is_ascii_digit(),
            "SINSEMILLA_TABLE_LIMIT is not a decimal byte count",
        );
        value = value * 10 + (digit - b'0') as usize;
        i += 1;
    }
    value
}

/// Weight rows in the generator table, one per power of two the evaluation can need.
///
/// A message of `n` words adds $\[2^{n-1-i}\] S$ at position `i` and tests against
/// $\[2^{n-i}\] S$ one weight up, so covering `n` words takes `n + 1` rows.
pub const WEIGHTS: usize = {
    let affordable = TABLE_LIMIT / ROW_BYTES;
    if affordable > LONGEST_WORTH_COVERING + 1 {
        LONGEST_WORTH_COVERING + 1
    } else {
        affordable
    }
};

/// Longest message the tables cover, in [`K`]-bit words, or zero if the budget affords no
/// table at all.
pub const LONGEST_MESSAGE_WORDS: usize = WEIGHTS.saturating_sub(1);

/// The personalizations that have a start table, and so can be hashed with the tables.
///
/// These are the personalizations whose $\mathcal{Q}$ this crate already stores; see
/// [`crate::known_domains`] for what each one is used for and where.
pub const KNOWN_DOMAINS: [&str; 3] = [
    // A parent of the Orchard note commitment tree, hashed once per Merkle path layer.
    "z.cash:Orchard-MerkleCRH",
    // The Orchard note commitment, hashed once per output.
    "z.cash:Orchard-NoteCommit-M",
    // The commitment to (ak, nk) that derives an incoming viewing key.
    "z.cash:Orchard-CommitIvk-M",
];

/// The generator table and the per-domain start points.
struct Tables {
    /// `generators[w][j]` is $\[2^w\] S_j$.
    generators: Vec<Vec<pallas::Affine>>,
    /// `starts[d][n - 1]` is $\[2^n\] \mathcal{Q}_d$ for the `d`th of [`KNOWN_DOMAINS`].
    starts: Vec<Vec<pallas::Affine>>,
}

impl Tables {
    /// Builds both tables.
    ///
    /// Neither the generators nor the domain points are hashed to the curve here. Both
    /// are already stored by the crate, and re-deriving them would be $2^K + 3$ hashes
    /// to the curve, which is most of what building the tables costs. The two tests
    /// below still check every entry against a fresh hash to the curve, so the stored
    /// constants are pinned to the specification rather than trusted.
    fn build() -> Self {
        // One row per weight, each the row below doubled. Doubling a whole row at a time
        // shares the work across its entries, and one batch inversion normalizes each row.
        // Row 0 is the generator set itself, which is stored affine already.
        let mut weighted: Vec<pallas::Point> = sinsemilla_s::S_AFFINE
            .iter()
            .map(|point| point.to_curve())
            .collect();

        let mut generators = Vec::with_capacity(WEIGHTS);
        for w in 0..WEIGHTS {
            generators.push(if w == 0 {
                sinsemilla_s::S_AFFINE.to_vec()
            } else {
                for point in weighted.iter_mut() {
                    *point = point.double();
                }
                let mut row = vec![pallas::Affine::default(); S_LEN];
                pallas::Point::batch_normalize(&weighted, &mut row);
                row
            });
        }

        // The start table: [2^n] Q for every known domain and every covered length,
        // with column n - 1 holding [2^n] Q.
        let starts = KNOWN_DOMAINS
            .iter()
            .map(|domain| {
                // Every entry of `KNOWN_DOMAINS` is stored, so the fallback is
                // unreachable; it is a fallback rather than an `expect` so that adding a
                // personalization to one list and not the other stays correct.
                let mut point = known_domains::q(domain).unwrap_or_else(|| {
                    pallas::Point::hash_to_curve(Q_PERSONALIZATION)(domain.as_bytes())
                });
                let mut projective = Vec::with_capacity(LONGEST_MESSAGE_WORDS);
                for _ in 0..LONGEST_MESSAGE_WORDS {
                    point = point.double();
                    projective.push(point);
                }
                let mut affine = vec![pallas::Affine::default(); LONGEST_MESSAGE_WORDS];
                pallas::Point::batch_normalize(&projective, &mut affine);
                affine
            })
            .collect();

        Tables { generators, starts }
    }
}

static TABLES: LazyLock<Tables> = LazyLock::new(Tables::build);

/// Whether `p` has the affine $x$-coordinate of `target`, equivalently whether
/// $p = \pm \mathsf{target}$.
///
/// Tests $X = x Z^2$ on the Jacobian coordinates rather than normalizing either side, so
/// it costs one squaring and one multiplication and no inversion.
///
/// `target` must not be the identity, which holds for every table entry. If `p` IS the
/// identity the answer is meaningless, which is harmless here: `p` is the accumulator, and
/// the accumulator can only reach the identity at a position whose own test already fired
/// (see [`TableDomain::hash_to_point`]).
fn has_x_of(p: &pallas::Point, target: &pallas::Affine) -> Choice {
    let (x, _, z) = p.jacobian_coordinates();
    let target_x = *target.coordinates().unwrap().x();

    x.ct_eq(&(target_x * z.square()))
}

/// A hash domain that has a start table, and so can be hashed with the tables.
///
/// Construct one once and hash many messages with it. The first construction builds the
/// tables; see the module documentation for what that costs.
#[derive(Clone, Copy, Debug)]
pub struct TableDomain {
    /// Index into [`KNOWN_DOMAINS`], and so into the start table.
    domain: usize,
}

impl TableDomain {
    /// The table domain for `personalization`, or `None` if it is not one of
    /// [`KNOWN_DOMAINS`].
    ///
    /// Building the tables is deferred to the first hash, not done here, so this is
    /// cheap to call per message.
    pub fn new(personalization: &str) -> Option<Self> {
        if LONGEST_MESSAGE_WORDS == 0 {
            // The budget affords no usable table. Returning here is what keeps the
            // `LazyLock` from ever being forced, so such a build has no tables at all
            // rather than empty ones.
            return None;
        }

        KNOWN_DOMAINS
            .iter()
            .position(|d| *d == personalization)
            .map(|domain| TableDomain { domain })
    }

    /// $\mathsf{SinsemillaHashToPoint}$ over a message already split into [`K`]-bit
    /// words, least-significant bit first within each word.
    ///
    /// Returns $\bot$ on exactly the inputs the specification does; see
    /// "Reproducing $\bot$" in the module documentation.
    ///
    /// # Panics
    ///
    /// Panics if `words` is empty, longer than [`LONGEST_MESSAGE_WORDS`], or contains a
    /// word that is not a [`K`]-bit value.
    pub fn hash_to_point(&self, words: &[u16]) -> CtOption<pallas::Point> {
        assert!(!words.is_empty(), "the message has no words");
        assert!(
            words.len() <= LONGEST_MESSAGE_WORDS,
            "the message is longer than the tables cover",
        );

        let n = words.len();
        // The position-weighted prefix P_i, which is [2^(n-i)] Acc_i.
        let mut acc = TABLES.starts[self.domain][n - 1].to_curve();
        let mut exceptional = Choice::from(0u8);

        for (i, &word) in words.iter().enumerate() {
            let word = usize::from(word);
            assert!(word < S_LEN, "word is not a K-bit value");

            // Position i adds T = [2^(n-1-i)] S, and the specification's three
            // exceptional conditions on Acc_i rescale onto P_i by the factor [2^(n-i)]:
            //   Acc = S    and    Acc = -S     become    P = +-[2] T
            //   S = -[2] Acc                   becomes   P = -T
            // The first pair is one x-coordinate test against the entry one weight up.
            // The second says the addition below lands on the identity, so it needs no
            // test of its own.
            let addend = &TABLES.generators[n - 1 - i][word];
            let doubled = &TABLES.generators[n - i][word];

            exceptional |= has_x_of(&acc, doubled);
            acc += addend;
            exceptional |= acc.is_identity();
        }

        CtOption::new(acc, !exceptional)
    }
}

#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};

    use group::{Curve, CurveAffine as _, Group, GroupEncoding};
    use pasta_curves::{arithmetic::CurveExt, pallas};

    use super::{
        has_x_of, TableDomain, Tables, KNOWN_DOMAINS, LONGEST_MESSAGE_WORDS, ROW_BYTES, TABLES,
        TABLE_LIMIT, WEIGHTS,
    };
    use crate::{HashDomain, K, Q_PERSONALIZATION, S_PERSONALIZATION};

    /// Packs a bit string into [`K`]-bit words, zero-padding the final word exactly as the
    /// dispatch in [`crate::HashDomain`] pads it.
    fn words(msg: &[bool]) -> Vec<u16> {
        let mut words = vec![0u16; msg.len().div_ceil(K)];
        for (i, bit) in msg.iter().enumerate() {
            words[i / K] |= u16::from(*bit) << (i % K);
        }
        words
    }

    /// Asserts the evaluator did not report an exceptional case, and returns the point.
    fn unwrap(point: subtle::CtOption<pallas::Point>) -> pallas::Point {
        Option::from(point).expect("not an exceptional input")
    }

    fn random_bits(n: usize) -> Vec<bool> {
        rand::random_iter().take(n).collect()
    }

    /// The specification's answer, from a domain that cannot reach the tables.
    ///
    /// Going through [`HashDomain::from_Q`] rather than [`HashDomain::new`] is the whole
    /// point: `new` on a known personalization now dispatches *here*, so it would compare
    /// the tables against themselves.
    fn reference(personalization: &str, msg: &[bool]) -> pallas::Point {
        let q = pallas::Point::hash_to_curve(Q_PERSONALIZATION)(personalization.as_bytes());
        let domain = HashDomain::from_Q(q);
        Option::from(domain.hash_to_point(msg.iter().copied())).expect("not an exceptional input")
    }

    /// The tables agree with the specification-shaped evaluator, for every known domain,
    /// at the lengths those domains are used at and at the ends of the covered range.
    #[test]
    fn agrees_with_generic() {
        // 51 words is CommitIvk, 52 a Merkle parent hash, 109 a note commitment; 1 and
        // LONGEST_MESSAGE_WORDS are the ends of the covered range, and 9 and 11 exercise
        // a partly-filled final word.
        let lengths = [1, 9, 11, 51 * K, 52 * K, 109 * K, LONGEST_MESSAGE_WORDS * K];

        for personalization in KNOWN_DOMAINS {
            let tabled = TableDomain::new(personalization).expect("a known domain");

            // A smaller budget covers fewer lengths, so it has fewer to check rather than
            // a failing test.
            for bits in lengths
                .into_iter()
                .filter(|bits| bits.div_ceil(K) <= LONGEST_MESSAGE_WORDS)
            {
                let msg = random_bits(bits);
                assert_eq!(
                    unwrap(tabled.hash_to_point(&words(&msg))).to_bytes(),
                    reference(personalization, &msg).to_bytes(),
                    "{personalization}, {bits} bits",
                );
            }
        }
    }

    /// It also agrees on the all-zero and all-one messages, which the random trials will
    /// never draw and which exercise word 0 and word 2^K - 1 in every position.
    #[test]
    fn agrees_on_extreme_messages() {
        let personalization = KNOWN_DOMAINS[0];

        for bit in [false, true] {
            let msg = vec![bit; TEST_WORDS * K];
            assert_eq!(
                unwrap(
                    TableDomain::new(personalization)
                        .expect("a known domain")
                        .hash_to_point(&words(&msg)),
                )
                .to_bytes(),
                reference(personalization, &msg).to_bytes(),
                "all-{bit} message",
            );
        }
    }

    /// Dispatch reaches the tables: a known personalization built with
    /// [`HashDomain::new`] agrees with the table evaluator called directly.
    ///
    /// Without this, every test above would still pass if `new` had quietly stopped
    /// wiring the tables in.
    #[test]
    fn hash_domain_dispatches_to_the_tables() {
        for personalization in KNOWN_DOMAINS {
            let domain = HashDomain::new(personalization);
            assert!(domain.tabled.is_some(), "{personalization} has no tables");

            let msg = random_bits(TEST_WORDS * K);
            let expected = unwrap(
                TableDomain::new(personalization)
                    .expect("a known domain")
                    .hash_to_point(&words(&msg)),
            );
            assert_eq!(
                Option::<pallas::Point>::from(domain.hash_to_point(msg.iter().copied()))
                    .expect("the table path never returns bottom")
                    .to_bytes(),
                expected.to_bytes(),
                "{personalization}",
            );
        }
    }

    /// A message past the covered range falls back to the specification's path, which is
    /// the branch that keeps `LONGEST_MESSAGE_WORDS` from being a correctness bound.
    #[test]
    fn uncovered_lengths_fall_back() {
        let personalization = KNOWN_DOMAINS[0];
        let domain = HashDomain::new(personalization);

        for words in [LONGEST_MESSAGE_WORDS + 1, crate::C] {
            let msg = random_bits(words * K);
            assert_eq!(
                Option::<pallas::Point>::from(domain.hash_to_point(msg.iter().copied()))
                    .expect("not an exceptional input")
                    .to_bytes(),
                reference(personalization, &msg).to_bytes(),
                "{words} words",
            );
        }
    }

    /// An empty message is not a covered length either, and hashes to $\mathcal{Q}$.
    #[test]
    fn empty_messages_fall_back() {
        let personalization = KNOWN_DOMAINS[0];
        let domain = HashDomain::new(personalization);

        assert_eq!(
            Option::<pallas::Point>::from(domain.hash_to_point(core::iter::empty()))
                .expect("not an exceptional input")
                .to_bytes(),
            domain.Q().to_bytes(),
        );
    }

    /// Every generator-table entry is `[2^w] S_j`, recomputed here rather than trusted
    /// from the builder.
    ///
    /// The generators are hashed to the curve once and then doubled row by row, which is
    /// the same shape as the builder but arrived at independently; hashing to the curve
    /// per entry instead would be `WEIGHTS * 2^K` hash-to-curves and dominate the suite.
    #[test]
    fn generator_entries_are_the_weighted_generators() {
        let hasher = pallas::Point::hash_to_curve(S_PERSONALIZATION);
        let mut expected: Vec<pallas::Point> =
            (0..1u32 << K).map(|j| hasher(&j.to_le_bytes())).collect();

        for (weight, row) in TABLES.generators.iter().enumerate() {
            for (w, (entry, expected)) in row.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    entry.to_curve().to_bytes(),
                    expected.to_bytes(),
                    "entry {w}, weight 2^{weight}",
                );
            }
            for point in expected.iter_mut() {
                *point = point.double();
            }
        }
    }

    /// Every start-table entry is `[2^n] Q_d`, recomputed here from the personalization.
    #[test]
    fn start_entries_are_the_weighted_domain_points() {
        let hasher = pallas::Point::hash_to_curve(Q_PERSONALIZATION);

        for (d, personalization) in KNOWN_DOMAINS.iter().enumerate() {
            let mut expected = hasher(personalization.as_bytes());

            for n in 1..=LONGEST_MESSAGE_WORDS {
                expected = expected.double();
                assert_eq!(
                    TABLES.starts[d][n - 1].to_curve().to_bytes(),
                    expected.to_bytes(),
                    "{personalization}, [2^{n}] Q",
                );
            }
        }
    }

    /// The table has exactly the row the deepest test reads, and no row beyond it, and
    /// this build covers the lengths the tests below assume something about.
    ///
    /// These are facts about the constants, so they hold at compile time rather than at
    /// run time; a `#[test]` over them would be `assert!(true)`.
    const _: () = assert!(WEIGHTS == LONGEST_MESSAGE_WORDS + 1);
    const _: () = assert!(
        LONGEST_MESSAGE_WORDS >= 1,
        "SINSEMILLA_TABLE_LIMIT affords no table, so there is nothing here to test",
    );

    /// The longest message these tests use, which is a Merkle parent hash where the budget
    /// covers one and the whole coverage where it does not.
    const TEST_WORDS: usize = if LONGEST_MESSAGE_WORDS < 52 {
        LONGEST_MESSAGE_WORDS
    } else {
        52
    };

    /// `has_x_of` accepts a point and its negation and nothing else, which is the whole
    /// content of collapsing the two `alpha = +-1` conditions into one x-coordinate test.
    #[test]
    fn has_x_of_accepts_exactly_a_point_and_its_negation() {
        let s = pallas::Point::hash_to_curve(S_PERSONALIZATION);
        let p = s(&7u32.to_le_bytes());
        let other = s(&8u32.to_le_bytes());

        assert!(bool::from(has_x_of(&p, &p.to_affine())));
        assert!(bool::from(has_x_of(&(-p), &p.to_affine())));
        assert!(!bool::from(has_x_of(&other, &p.to_affine())));
        assert!(!bool::from(has_x_of(&p.double(), &p.to_affine())));

        // The accumulator is Jacobian with a non-trivial Z, which is the case the
        // `X == x * Z^2` form exists for; a test that only ever passed Z = 1 would not
        // exercise it.
        let jacobian = p + other - other;
        assert_ne!(jacobian.jacobian_coordinates().2, pallas::Base::one());
        assert!(bool::from(has_x_of(&jacobian, &p.to_affine())));
    }

    /// The table built is the shape the budget bought.
    ///
    /// The budget arithmetic itself is a fact about constants and holds at compile time,
    /// just above; this checks that the builder honours it.
    const _: () = assert!(WEIGHTS * ROW_BYTES <= TABLE_LIMIT, "over the table budget");

    #[test]
    fn the_table_fits_the_budget() {
        assert_eq!(TABLES.generators.len(), WEIGHTS);
        assert!(TABLES.generators.iter().all(|row| row.len() == 1 << K));
    }

    /// An unknown personalization has no start table, so it has no table domain and its
    /// `HashDomain` keeps the specification's bottom.
    #[test]
    fn unknown_personalizations_have_no_table() {
        assert!(TableDomain::new("z.cash:test-Sinsemilla").is_none());
        assert!(HashDomain::new("z.cash:test-Sinsemilla").tabled.is_none());
    }

    /// Building the tables twice gives the same tables, which is what makes the
    /// process-lifetime `LazyLock` safe to share.
    #[test]
    fn building_is_deterministic() {
        let built = Tables::build();
        assert_eq!(built.generators, TABLES.generators);
        assert_eq!(built.starts, TABLES.starts);
    }
}
