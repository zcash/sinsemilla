//! Position-weighted tables for the Zcash hash domains, compiled in.
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
//! Counting positions from the END of the message, $j = N - i$, the coefficient of
//! $S_{m_{N-j}}$ is $2^j$, which mentions neither the domain nor $N$. So the generator
//! table is shared, and only the term $\[2^N\] \mathcal{Q}$ knows which domain and which
//! length this is. That term is a single point, so it is tabulated per domain and per
//! length instead of being recomputed.
//!
//! With stride $k$ the generator table keeps one row per $k$ positions and the
//! evaluation pays $k - 1$ doublings in TOTAL, not per word, by grouping the sum on
//! $j \bmod k$:
//!
//! $$\sum_{j=0}^{N-1} \[2^j\] S_{m_{N-j}}
//!   = \sum_{s=0}^{k-1} \[2^s\] \underbrace{\sum_{q} \[2^{qk}\] S_{m_{N-(qk+s)}}}_{U_s},$$
//!
//! where every $U_s$ reads the same rows.
//!
//! # This does NOT return bottom
//!
//! [`HashDomain::hash_to_point`] follows the specification, which uses incomplete
//! addition and is $\bot$ when an exceptional case arises. This evaluator reassociates
//! the sum and adds completely, so it never forms the specification's intermediate
//! accumulators and cannot report their exceptional cases; it returns a point for every
//! input. On the inputs where the specification is not $\bot$, which is everything but a
//! negligible set, the two agree, and the tests pin that. **A caller that needs the
//! specification's $\bot$ must use [`HashDomain`].** That difference, not performance, is
//! why this is a separate entry point rather than the implementation of
//! [`HashDomain::hash_to_point`].
//!
//! # Cost
//!
//! The tables are generated at build time into read-only data, so they cost binary size,
//! and source size and compile time in the build that generates them, but no RAM and no
//! start-up work.
//!
//! One row of the generator table is $2^K$ affine points, 64 KiB, and covers [`STRIDE_USED`]
//! positions, so the table is `ceil(MAX_WORDS / STRIDE)` rows. A larger stride is a
//! smaller table and more doublings: one evaluation pays `STRIDE - 1` doublings and one
//! mixed addition per word, against one doubling and one mixed addition per word for
//! [`HashDomain`]. The start table is one point per known domain per covered length,
//! `4 * MAX_WORDS * 64` bytes, and does not vary with the stride.
//!
//! A table competes for cache with the caller's own working set: one that does not fit
//! in private L2 evicts whatever else was there, and that cost lands on the caller's
//! work rather than on this crate. That is what the budget below is sized against.
//!
//! # The budget
//!
//! [`STRIDE_USED`] follows from a budget on the generator table, sized to the private L2 it
//! has to share with the caller: 128 KiB on Android, 4 MiB on macOS, and 1 MiB elsewhere
//! (iOS included, whose efficiency cores get about that much). Set
//! `SINSEMILLA_TABLE_LIMIT` (bytes) at build time to choose another; the stride changes
//! the cost and never the result.
//!
//! An environment variable rather than a Cargo feature ladder is deliberate: Cargo
//! features are additive and are unified across the dependency graph, so two dependents
//! asking for different sizes would silently get both, and a `cfg` ladder written as if
//! they were exclusive would pick whichever arm came first.
//!
//! [`C`]: crate::C
//! [`HashDomain`]: crate::HashDomain
//! [`HashDomain::hash_to_point`]: crate::HashDomain::hash_to_point

use group::Group;
use pasta_curves::pallas;

use crate::{extract_p, K};

include!(concat!(env!("OUT_DIR"), "/tables.rs"));

/// Longest message the tables cover, in [`K`]-bit words.
///
/// Covers every Orchard message: 51 words for `CommitIvk`, 52 for a Merkle parent hash,
/// 109 for a note commitment, and the ZIP 226 note commitment above that.
pub const LONGEST_MESSAGE_WORDS: usize = MAX_WORDS;

/// Positions covered by one row of the generator table, and so one more than the number
/// of doublings an evaluation pays.
pub const STRIDE_USED: usize = STRIDE;

/// Size of the generator table in bytes.
pub const TABLE_BYTES: usize = ROWS * (1 << K) * 64;

/// The budget the stride was chosen against: `SINSEMILLA_TABLE_LIMIT` if it was set at
/// build time, else the per-target default.
pub const BUDGET_BYTES_USED: usize = BUDGET_BYTES;

/// The personalizations that have a start table, and so can be hashed with the table.
pub const KNOWN_DOMAINS: [&str; 4] = DOMAINS;

/// The generator-table rows, for tests that recompute them.
///
/// Row `q` holds $\[2^{qk}\] S_w$ at index `w`, for stride `k` = [`STRIDE_USED`].
pub fn rows() -> impl Iterator<Item = &'static [pallas::Affine; 1 << K]> {
    TABLE.iter()
}

/// The stored $\[2^n\] \mathcal{Q}$ for a known domain, for tests that recompute it.
///
/// `None` if the personalization is unknown or `n` is not a covered length.
pub fn start(personalization: &str, n: usize) -> Option<pallas::Point> {
    let d = DOMAINS.iter().position(|d| *d == personalization)?;
    (1..=MAX_WORDS)
        .contains(&n)
        .then(|| STARTS[d][n - 1].into())
}

/// A hash domain that has a compiled-in start table.
///
/// Construct one once and hash many messages with it; it holds no allocation and is
/// `Copy`, since everything it needs is in read-only data.
#[derive(Clone, Copy, Debug)]
pub struct TableDomain {
    /// `starts[n - 1]` is $\[2^n\] \mathcal{Q}$ for this domain.
    starts: &'static [pallas::Affine; MAX_WORDS],
}

impl TableDomain {
    /// The table domain for `personalization`, or `None` if it is not one of
    /// [`KNOWN_DOMAINS`].
    pub fn new(personalization: &str) -> Option<Self> {
        DOMAINS
            .iter()
            .position(|d| *d == personalization)
            .map(|d| TableDomain { starts: &STARTS[d] })
    }

    /// $\mathsf{SinsemillaHashToPoint}$ over a message already split into [`K`]-bit
    /// words, least-significant bit first within each word.
    ///
    /// Never returns $\bot$; see the module documentation.
    ///
    /// # Panics
    ///
    /// Panics if `words` is empty, longer than [`LONGEST_MESSAGE_WORDS`], or contains a
    /// word that is not a [`K`]-bit value.
    pub fn hash_to_point(&self, words: &[u16]) -> pallas::Point {
        assert!(!words.is_empty(), "the message has no words");
        assert!(
            words.len() <= MAX_WORDS,
            "the message is longer than the tables cover",
        );

        // One partial sum per residue class of the position, counted from the end of the
        // message, modulo the stride. The [2^N] Q term has weight 2^0, so it joins the
        // class the Horner combination does not scale.
        let mut partial = [pallas::Point::identity(); STRIDE];
        partial[0] = self.starts[words.len() - 1].into();

        for (j, &word) in words.iter().rev().enumerate() {
            let word = usize::from(word);
            assert!(word < 1 << K, "word is not a K-bit value");
            partial[j % STRIDE] += TABLE[j / STRIDE][word];
        }

        // Horner over the partial sums: stride - 1 doublings, whatever the stride.
        partial
            .iter()
            .rev()
            .copied()
            .reduce(|acc, class| acc.double() + class)
            .expect("the stride is nonzero")
    }

    /// $\mathsf{SinsemillaHash}$ over a message already split into [`K`]-bit words.
    ///
    /// Never returns $\bot$; see the module documentation.
    ///
    /// # Panics
    ///
    /// As [`TableDomain::hash_to_point`].
    pub fn hash(&self, words: &[u16]) -> pallas::Base {
        extract_p(self.hash_to_point(words))
    }
}

/// Packs a bit string into [`K`]-bit words, zero-padding the final word exactly as the
/// generic evaluator pads it.
///
/// Returns `None` if the message is empty or longer than [`LONGEST_MESSAGE_WORDS`]
/// words, which are the cases the tables do not cover and where the caller should use
/// [`HashDomain`] instead.
///
/// [`HashDomain`]: crate::HashDomain
pub fn to_words(msg: impl Iterator<Item = bool>) -> Option<alloc::vec::Vec<u16>> {
    let mut words = alloc::vec::Vec::with_capacity(MAX_WORDS);

    for (len, bit) in msg.enumerate() {
        if len.is_multiple_of(K) {
            if words.len() == MAX_WORDS {
                return None;
            }
            words.push(0);
        }
        if bit {
            let last = words.len() - 1;
            words[last] |= 1 << (len % K);
        }
    }

    (!words.is_empty()).then_some(words)
}
