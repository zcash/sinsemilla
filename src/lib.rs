//! Implementation of Sinsemilla outside the circuit.

#![no_std]

// We require `alloc` for now.
#[macro_use]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use group::{Curve, Wnaf};
use pasta_curves::{
    arithmetic::{CurveAffine, CurveExt},
    pallas,
};
use subtle::CtOption;

mod addition;
use self::addition::IncompletePoint;
mod known_domains;
mod sinsemilla_s;
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
mod table;
pub use sinsemilla_s::SINSEMILLA_S;

/// Number of bits of each message piece in $\mathsf{SinsemillaHashToPoint}$
pub const K: usize = 10;

/// $\frac{1}{2^K}$
pub const INV_TWO_POW_K: [u8; 32] = [
    1, 0, 192, 196, 160, 229, 70, 82, 221, 165, 74, 202, 85, 7, 62, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 240, 63,
];

/// The largest integer such that $2^c \leq (r_P - 1) / 2$, where $r_P$ is the order
/// of Pallas.
pub const C: usize = 253;

// Sinsemilla Q generators

/// SWU hash-to-curve personalization for Sinsemilla $Q$ generators.
pub const Q_PERSONALIZATION: &str = "z.cash:SinsemillaQ";

// Sinsemilla S generators

/// SWU hash-to-curve personalization for Sinsemilla $S$ generators.
pub const S_PERSONALIZATION: &str = "z.cash:SinsemillaS";

/// Converts a little-endian [`K`]-bit string into an integer.
pub fn lebs2ip_k(bits: [bool; K]) -> u32 {
    bits.iter()
        .enumerate()
        .fold(0u32, |acc, (i, b)| acc + if *b { 1 << i } else { 0 })
}

/// Coordinate extractor for Pallas.
///
/// Defined in [Zcash Protocol Spec § 5.4.9.7: Coordinate Extractor for Pallas][concreteextractorpallas].
///
/// [concreteextractorpallas]: https://zips.z.cash/protocol/nu5.pdf#concreteextractorpallas
fn extract_p_bottom(point: CtOption<pallas::Point>) -> CtOption<pallas::Base> {
    point.map(extract_p)
}

/// Coordinate extractor for Pallas, on a point that is known to exist.
///
/// Defined in [Zcash Protocol Spec § 5.4.9.7: Coordinate Extractor for Pallas][concreteextractorpallas].
///
/// [concreteextractorpallas]: https://zips.z.cash/protocol/nu5.pdf#concreteextractorpallas
fn extract_p(point: pallas::Point) -> pallas::Base {
    point
        .to_affine()
        .coordinates()
        .map(|c| *c.x())
        .unwrap_or_else(pallas::Base::zero)
}

/// The Sinsemilla generator $\mathcal{S}(j)$ as an affine point, for `j` < $2^K$.
fn s_generator(j: usize) -> pallas::Affine {
    sinsemilla_s::S_AFFINE[j]
}

/// A domain in which $\mathsf{SinsemillaHashToPoint}$ and $\mathsf{SinsemillaHash}$ can
/// be used.
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct HashDomain {
    /// $\mathcal{Q}(D)$, the point the accumulator starts from.
    ///
    /// Hashed to the curve from the personalization under [`Q_PERSONALIZATION`], except
    /// for the personalizations the specification fixes, which are read from stored
    /// constants instead. Both routes give the same point; the stored ones only skip the
    /// hash-to-curve, which costs more than the hash it sets up.
    Q: pallas::Point,
    /// The position-weighted tables for this personalization, when it has them.
    ///
    /// `None` for every other personalization, and for a domain built with
    /// [`HashDomain::from_Q`] from a bare $\mathcal{Q}$, which has no personalization to
    /// look up. Those evaluate the specification's recurrence directly.
    #[cfg(feature = "std")]
    tabled: Option<table::TableDomain>,
}

impl HashDomain {
    /// Constructs a new `HashDomain` with a specific prefix string.
    pub fn new(domain: &str) -> Self {
        HashDomain {
            Q: known_domains::q(domain).unwrap_or_else(|| {
                pallas::Point::hash_to_curve(Q_PERSONALIZATION)(domain.as_bytes())
            }),
            #[cfg(feature = "std")]
            tabled: table::TableDomain::new(domain),
        }
    }

    /// $\mathsf{SinsemillaHashToPoint}$ from [§ 5.4.1.9][concretesinsemillahash].
    ///
    /// For a personalization the specification fixes, and a message the position-weighted
    /// tables cover, this evaluates against those tables instead of the specification's
    /// recurrence. The result, including which inputs give $\bot$, is unchanged: the
    /// tables reproduce the exceptional cases rather than skipping them, by rescaling
    /// [Theorem 5.4.4][theorem544]'s conditions onto the weighted prefix.
    ///
    /// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
    /// [theorem544]: https://zips.z.cash/protocol/protocol.pdf#concretesinsemillahash
    pub fn hash_to_point(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Point> {
        self.hash_to_point_impl(msg)
    }

    /// Splits `msg` into [`K`]-bit words and hashes it, with the tables where they apply.
    ///
    /// The message is packed into words before the choice is made, because the choice
    /// depends on the length and the iterator can only be read once. [`C`] words is the
    /// longest message the specification allows, so the buffer is a fixed 506 bytes of
    /// stack and neither path allocates.
    ///
    /// # Panics
    ///
    /// Panics if the message is longer than [`K`] * [`C`] bits.
    fn hash_to_point_impl(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Point> {
        let mut words = [0u16; C];
        let mut len = 0usize;

        for bit in msg {
            assert!(len < K * C, "message is longer than K * C bits");
            words[len / K] |= u16::from(bit) << (len % K);
            len += 1;
        }
        let words = &words[..len.div_ceil(K)];

        #[cfg(feature = "std")]
        if let Some(tabled) = self.tabled {
            if (1..=table::LONGEST_MESSAGE_WORDS).contains(&words.len()) {
                return tabled.hash_to_point(words);
            }
        }

        self.accumulate(words).into()
    }

    /// The specification's recurrence, $A_i = \[2\] A_{i-1} \;⸭\; S_{m_i}$, over `words`.
    fn accumulate(&self, words: &[u16]) -> IncompletePoint {
        words
            .iter()
            .fold(IncompletePoint::from(self.Q), |acc, word| {
                acc.double_and_add(s_generator(usize::from(*word)))
            })
    }

    /// $\mathsf{SinsemillaHash}$ from [§ 5.4.1.9][concretesinsemillahash].
    ///
    /// [concretesinsemillahash]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillahash
    ///
    /// # Panics
    ///
    /// This panics if the message length is greater than [`K`] * [`C`]
    pub fn hash(&self, msg: impl Iterator<Item = bool>) -> CtOption<pallas::Base> {
        extract_p_bottom(self.hash_to_point(msg))
    }

    /// Constructs a new `HashDomain` from a given `Q`.
    ///
    /// This is only for testing use.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn from_Q(Q: pallas::Point) -> Self {
        HashDomain {
            Q,
            // A bare Q has no personalization, so it never takes the table path.
            #[cfg(feature = "std")]
            tabled: None,
        }
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn Q(&self) -> pallas::Point {
        self.Q
    }
}

/// A domain in which $\mathsf{SinsemillaCommit}$ and $\mathsf{SinsemillaShortCommit}$ can
/// be used.
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct CommitDomain {
    M: HashDomain,
    R: pallas::Point,
}

impl CommitDomain {
    /// Constructs a new `CommitDomain` with a specific prefix string.
    pub fn new(domain: &str) -> Self {
        Self::new_with_separate_domains(domain, domain)
    }

    /// Constructs a new `CommitDomain` from different values for `hash_domain` and `blind_domain`
    /// `new_with_separate_domains` is used in the OrchardZSA note commitment, where we use the
    /// OrchardZSA hash domain `z.cash:ZSA-NoteCommit` and reuse the Orchard blind domain
    /// `z.cash:Orchard-NoteCommit`, as specified in
    /// [ZIP 226](https://zips.z.cash/zip-0226#note-structure-commitment).
    pub fn new_with_separate_domains(hash_domain: &str, blind_domain: &str) -> Self {
        let m_prefix = format!("{hash_domain}-M");
        let r_prefix = format!("{blind_domain}-r");
        CommitDomain {
            M: HashDomain::new(&m_prefix),
            R: known_domains::r(&r_prefix)
                .unwrap_or_else(|| pallas::Point::hash_to_curve(&r_prefix)(&[])),
        }
    }

    /// $\mathsf{SinsemillaCommit}$ from [§ 5.4.8.4][concretesinsemillacommit].
    ///
    /// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
    #[allow(non_snake_case)]
    pub fn commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Point> {
        // We use complete addition for the blinding factor.
        self.M
            .hash_to_point_impl(msg)
            .map(|p| p + Wnaf::new().scalar(r).base(self.R))
    }

    /// $\mathsf{SinsemillaShortCommit}$ from [§ 5.4.8.4][concretesinsemillacommit].
    ///
    /// [concretesinsemillacommit]: https://zips.z.cash/protocol/nu5.pdf#concretesinsemillacommit
    pub fn short_commit(
        &self,
        msg: impl Iterator<Item = bool>,
        r: &pallas::Scalar,
    ) -> CtOption<pallas::Base> {
        extract_p_bottom(self.commit(msg, r))
    }

    /// Returns the Sinsemilla $R$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn R(&self) -> pallas::Point {
        self.R
    }

    /// Returns the Sinsemilla $Q$ constant for this domain.
    #[cfg(any(test, feature = "test-dependencies"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "test-dependencies")))]
    #[allow(non_snake_case)]
    pub fn Q(&self) -> pallas::Point {
        self.M.Q
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::sinsemilla_s::{SINSEMILLA_S, S_AFFINE};
    use super::{lebs2ip_k, s_generator, HashDomain, IncompletePoint, C, K};
    use group::{Curve, CurveAffine as _};
    use pasta_curves::{
        arithmetic::{CurveAffine, CurveExt},
        pallas,
    };
    use subtle::CtOption;

    /// The evaluator as it was before it streamed words: collect the message, zero-pad it
    /// to a multiple of `K` bits, then fold over the `K`-bit chunks.
    fn reference(domain: &HashDomain, msg: &[bool]) -> CtOption<pallas::Point> {
        assert!(msg.len() <= K * C);
        let mut padded = msg.to_vec();
        padded.resize(msg.len().div_ceil(K) * K, false);
        padded
            .chunks(K)
            .fold(IncompletePoint::from(domain.Q), |acc, chunk| {
                let word = lebs2ip_k(chunk.try_into().expect("K bits")) as usize;
                acc.double_and_add(s_generator(word))
            })
            .into()
    }

    /// A deterministic bit stream (xorshift64), so the test needs no RNG dependency.
    fn bits(seed: u64, n: usize) -> Vec<bool> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x & 1 == 1
            })
            .collect()
    }

    fn assert_same(domain: &HashDomain, msg: &[bool]) {
        let expected = reference(domain, msg);
        let actual = domain.hash_to_point(msg.iter().copied());
        assert_eq!(
            bool::from(actual.is_some()),
            bool::from(expected.is_some()),
            "bottom disagrees at {} bits",
            msg.len(),
        );
        if bool::from(expected.is_some()) {
            assert_eq!(actual.unwrap(), expected.unwrap(), "{} bits", msg.len());
        }
    }

    /// Streaming agrees with collect-then-chunk at every length the hash accepts, which
    /// covers every amount of padding and both the empty and the longest message.
    #[test]
    fn streaming_matches_reference_at_every_length() {
        let domain = HashDomain::new("z.cash:test-Sinsemilla");
        for len in 0..=K * C {
            assert_same(&domain, &bits(len as u64 + 1, len));
        }
    }

    /// Bottom is still reported on exactly the same inputs. With `Q = S(0)`, the first
    /// word 0 makes the first addition a doubling, one of the exceptional cases, whether
    /// that word is written out or produced entirely by padding.
    #[test]
    fn streaming_preserves_bottom() {
        let (x, y) = SINSEMILLA_S[0];
        let q = pallas::Affine::from_xy(x, y).unwrap().to_curve();
        let domain = HashDomain::from_Q(q);

        for len in [1, K - 1, K, K + 1, 5 * K] {
            let msg = vec![false; len];
            assert!(bool::from(
                domain.hash_to_point(msg.iter().copied()).is_none()
            ));
            assert_same(&domain, &msg);
        }
        // A first word other than 0 avoids that case.
        let mut msg = vec![false; K];
        msg[0] = true;
        assert_same(&domain, &msg);
    }

    #[test]
    #[should_panic(expected = "longer than K * C bits")]
    fn rejects_messages_longer_than_k_times_c() {
        let domain = HashDomain::new("z.cash:test-Sinsemilla");
        let _ = domain.hash_to_point(core::iter::repeat_n(false, K * C + 1));
    }

    #[test]
    fn sinsemilla_s() {
        let hasher = pallas::Point::hash_to_curve(super::S_PERSONALIZATION);

        for j in 0..(1u32 << K) {
            let computed = {
                let point = hasher(&j.to_le_bytes()).to_affine().coordinates().unwrap();
                (*point.x(), *point.y())
            };
            let actual = SINSEMILLA_S[j as usize];
            assert_eq!(computed, actual);

            let point = S_AFFINE[j as usize];
            assert!(bool::from(point.is_on_curve()));
            assert_eq!(
                (
                    *point.coordinates().unwrap().x(),
                    *point.coordinates().unwrap().y()
                ),
                actual,
            );
            assert_eq!(s_generator(j as usize), point);
        }
    }
}
