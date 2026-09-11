//! Known-answer tests against the vectors published in [`zcash-test-vectors`].
//!
//! [`zcash-test-vectors`]: https://github.com/zcash/zcash-test-vectors

mod test_vectors;

use group::{ff::PrimeField, GroupEncoding};
use pasta_curves::pallas;
use sinsemilla::{HashDomain, K};

/// $\mathsf{SinsemillaHashToPoint}$ and $\mathsf{SinsemillaHash}$ over messages of
/// varying length, including lengths that require padding.
#[test]
fn sinsemilla_test_vectors() {
    for (i, tv) in test_vectors::orchard_sinsemilla::TEST_VECTORS
        .iter()
        .enumerate()
    {
        let domain_str = core::str::from_utf8(tv.domain).expect("domain is UTF-8");
        let domain = HashDomain::new(domain_str);

        let point: pallas::Point = Option::from(domain.hash_to_point(tv.msg.iter().copied()))
            .unwrap_or_else(|| panic!("vector {i} is not exceptional"));
        assert_eq!(point.to_bytes(), tv.point, "hash_to_point, vector {i}");

        let hash: pallas::Base = Option::from(domain.hash(tv.msg.iter().copied()))
            .unwrap_or_else(|| panic!("vector {i} is not exceptional"));
        assert_eq!(hash.to_repr(), tv.hash, "hash, vector {i}");
    }
}

/// Personalization of $\mathsf{MerkleCRH^{Orchard}}$, from
/// [§ 5.4.1.9][orchardmerklecrh] of the Zcash protocol specification.
///
/// [orchardmerklecrh]: https://zips.z.cash/protocol/protocol.pdf#orchardmerklecrh
const MERKLE_CRH_PERSONALIZATION: &str = "z.cash:Orchard-MerkleCRH";

/// Number of bits of a Merkle node that are hashed, i.e. $\ell_{\mathsf{MerkleOrchard}}$.
const L_MERKLE: usize = 255;

/// Length in bits of a $\mathsf{MerkleCRH^{Orchard}}$ message: a [`K`]-bit layer index
/// followed by two [`L_MERKLE`]-bit nodes.
const MERKLE_CRH_BITS: usize = K + 2 * L_MERKLE;

/// Length in [`K`]-bit words of a $\mathsf{MerkleCRH^{Orchard}}$ message.
///
/// This is the fixed length that the position-weighted specialisation is built around,
/// so the assertion below is load-bearing rather than decorative.
const MERKLE_CRH_WORDS: usize = MERKLE_CRH_BITS / K;
const _: () = assert!(MERKLE_CRH_BITS == MERKLE_CRH_WORDS * K);
const _: () = assert!(MERKLE_CRH_WORDS == 52);

/// The hash domain of $\mathsf{MerkleCRH^{Orchard}}$.
///
/// Constructing this performs a hash-to-curve, so callers that measure or repeat the
/// hash should build it once and reuse it.
fn merkle_crh_domain() -> HashDomain {
    HashDomain::new(MERKLE_CRH_PERSONALIZATION)
}

/// Assembles the $\mathsf{MerkleCRH^{Orchard}}$ message for a parent of `left` and
/// `right` at `l`.
///
/// `l` is $\mathsf{MerkleDepth^{Orchard}} - 1 - \mathsf{layer}$, matching the
/// specification rather than the layer index.
///
/// `left` and `right` are consumed as little-endian bit strings truncated to
/// [`L_MERKLE`] bits, not as field elements: the upstream test vectors include node
/// values that are not canonical `pallas::Base` encodings.
fn merkle_crh_msg(l: u32, left: &[u8; 32], right: &[u8; 32]) -> [bool; MERKLE_CRH_BITS] {
    assert!(l < 1 << K);

    let mut msg = [false; MERKLE_CRH_BITS];
    for (i, bit) in msg[..K].iter_mut().enumerate() {
        *bit = (l >> i) & 1 == 1;
    }
    for (node, chunk) in [left, right].into_iter().zip(msg[K..].chunks_mut(L_MERKLE)) {
        for (i, bit) in chunk.iter_mut().enumerate() {
            *bit = (node[i / 8] >> (i % 8)) & 1 == 1;
        }
    }
    msg
}

/// $\mathsf{MerkleCRH^{Orchard}}$.
///
/// # Panics
///
/// Panics if the hash hits an exceptional case, which is infeasible to reach under the
/// discrete logarithm relation assumption.
fn merkle_crh(domain: &HashDomain, l: u32, left: &[u8; 32], right: &[u8; 32]) -> pallas::Base {
    let msg = merkle_crh_msg(l, left, right);
    Option::from(domain.hash(msg.into_iter())).expect("hash is not exceptional")
}

/// A $\mathsf{MerkleCRH^{Orchard}}$ parent of two distinct nodes.
#[test]
fn orchard_merkle_parent() {
    const L: u32 = 25;
    const LEFT: [u8; 32] = [
        0x05, 0x65, 0x53, 0x16, 0xa0, 0x7e, 0x6e, 0xc8, 0xc9, 0x76, 0x9a, 0xf5, 0x4e, 0xf9, 0x8b,
        0x30, 0x66, 0x7b, 0xfb, 0x63, 0x02, 0xb3, 0x29, 0x87, 0xd5, 0x52, 0x22, 0x7d, 0xae, 0x86,
        0xa0, 0x87,
    ];
    const RIGHT: [u8; 32] = [
        0x06, 0x04, 0x13, 0x57, 0xde, 0x59, 0xba, 0x64, 0x95, 0x9d, 0x1b, 0x60, 0xf9, 0x3d, 0xe2,
        0x4d, 0xfe, 0x5e, 0xa1, 0xe2, 0x6e, 0xd9, 0xe8, 0xa7, 0x3d, 0x35, 0xb2, 0x25, 0xa1, 0x84,
        0x5b, 0xa7,
    ];
    const PARENT: [u8; 32] = [
        0xb9, 0x2a, 0x4b, 0xae, 0xbb, 0x72, 0xc7, 0xa8, 0xa2, 0xa0, 0x0a, 0xa4, 0xdc, 0x16, 0x82,
        0xca, 0xd4, 0x7a, 0xb8, 0x34, 0xba, 0xa4, 0x5e, 0xd9, 0x4d, 0x6d, 0x9c, 0xde, 0x0a, 0x76,
        0x62, 0x01,
    ];

    let domain = merkle_crh_domain();
    assert_eq!(merkle_crh(&domain, L, &LEFT, &RIGHT).to_repr(), PARENT);

    // The layer index is part of the message, so a different layer is a different hash.
    assert_ne!(merkle_crh(&domain, L + 1, &LEFT, &RIGHT).to_repr(), PARENT);
    // The hash is not symmetric in its two nodes.
    assert_ne!(merkle_crh(&domain, L, &RIGHT, &LEFT).to_repr(), PARENT);
}
