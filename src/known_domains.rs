//! Precomputed Sinsemilla constants for the personalizations Zcash specifies.
//!
//! [`HashDomain::new`] and [`CommitDomain::new`] derive $Q$ (and $R$) with a hash to the
//! curve, which costs more than the hash itself. Orchard constructs a domain on every
//! Merkle hash and every note commitment, so returning the stored point for a known
//! personalization removes most of that per-call cost without changing the interface.
//! Every other personalization is still hashed to the curve.
//!
//! Each entry is checked against a fresh hash to the curve in the tests below.
//!
//! [`HashDomain::new`]: crate::HashDomain::new
//! [`CommitDomain::new`]: crate::CommitDomain::new

use group::CurveAffine as _;
use pasta_curves::pallas;

/// A personalization and its point.
///
/// The points are built with `from_xy_unchecked`, so a lookup does not re-check that
/// they are on the curve; the tests below check it once, and check each point against
/// a fresh hash to the curve.
type Entry = (&'static str, pallas::Affine);

/// $\mathcal{Q}(D) = \mathsf{GroupHash}^{\mathbb{P}}(\texttt{"z.cash:SinsemillaQ"}, D)$,
/// keyed by $D$.
///
/// The `-M` entries are the hash domains of the Orchard commitments. These are every
/// Sinsemilla domain Orchard uses: its other `z.cash:` personalizations
/// (`z.cash:Orchard`, `z.cash:Orchard-cv`, `z.cash:Orchard-gd`) are hash-to-curve
/// personalizations for fixed bases, not Sinsemilla domains, so they do not belong here.
static Q: [Entry; 3] = [
    (
        // MerkleCRH^Orchard, in `zcash/orchard` `src/tree.rs`: hashes one parent of the
        // note commitment tree, so a spend walks it once per layer of its Merkle path,
        // and a node adds every output of every block to it.
        "z.cash:Orchard-MerkleCRH",
        pallas::Affine::from_xy_unchecked(
            pallas::Base::from_raw([
                0xf8b9_c7f9_7f29_c6a0,
                0xc9be_b955_c08d_1070,
                0xa00f_365a_ef89_0e99,
                0x1616_d296_63a8_18b9,
            ]),
            pallas::Base::from_raw([
                0x86e9_aece_25f2_ea62,
                0xe21c_96ea_0574_1596,
                0x2dc4_f23e_4fa3_5979,
                0x3586_42a3_e3af_2099,
            ]),
        ),
    ),
    (
        // NoteCommit^Orchard, in `zcash/orchard` `src/note/commitment.rs`: commits to
        // (g_d, pk_d, v, rho, psi), giving the note commitment that becomes the leaf the
        // domain above hashes into the tree.
        "z.cash:Orchard-NoteCommit-M",
        pallas::Affine::from_xy_unchecked(
            pallas::Base::from_raw([
                0x320e_ba09_40a8_745d,
                0xc596_0f5a_fd46_dd2a,
                0xf79f_f2b4_79b0_ed5d,
                0x1780_07a0_56fb_cd0d,
            ]),
            pallas::Base::from_raw([
                0x8727_0a5a_7349_ac63,
                0x8822_1288_81db_5e9e,
                0x4ebe_c2d9_6ef4_c92c,
                0x32a0_5893_8ac6_7083,
            ]),
        ),
    ),
    (
        // CommitIvk^Orchard, in `zcash/orchard` `src/spec.rs`: commits to (ak, nk) to
        // derive the incoming viewing key, so it runs in key derivation rather than per
        // note or per block.
        "z.cash:Orchard-CommitIvk-M",
        pallas::Affine::from_xy_unchecked(
            pallas::Base::from_raw([
                0x6bcb_2f92_790f_82f2,
                0x421b_cc24_5128_a232,
                0x7dcc_81b8_5aa2_41fa,
                0x05bc_0cf1_4aa9_c811,
            ]),
            pallas::Base::from_raw([
                0xbe5a_e5ce_cfad_debe,
                0x46c4_351d_c96d_a5f1,
                0xef59_0746_20de_054b,
                0x1b01_4cf6_d41a_bee6,
            ]),
        ),
    ),
];

/// $\mathcal{R} = \mathsf{GroupHash}^{\mathbb{P}}(D \,||\, \texttt{"-r"}, \texttt{""})$,
/// keyed by $D \,||\, \texttt{"-r"}$.
static R: [Entry; 2] = [
    (
        // Blinding base of NoteCommit^Orchard, the R the commitment adds [rcm] R by.
        "z.cash:Orchard-NoteCommit-r",
        pallas::Affine::from_xy_unchecked(
            pallas::Base::from_raw([
                0x2c02_2c48_0ffc_6e13,
                0x239e_c55c_fc14_a47c,
                0xcd23_9fab_936f_3df2,
                0x26b2_06c3_28a9_4533,
            ]),
            pallas::Base::from_raw([
                0xc341_53c6_22cf_37e3,
                0x79c7_c078_23e0_dcf6,
                0xa954_1a2f_2366_e1c6,
                0x3cde_564b_686d_c04c,
            ]),
        ),
    ),
    (
        // Blinding base of CommitIvk^Orchard, the R the commitment adds [rivk] R by.
        "z.cash:Orchard-CommitIvk-r",
        pallas::Affine::from_xy_unchecked(
            pallas::Base::from_raw([
                0x9823_486e_5ff8_a118,
                0x0295_7fe2_d31a_edc7,
                0x1634_290a_4080_8948,
                0x25a2_2ccd_5070_134e,
            ]),
            pallas::Base::from_raw([
                0x3fe7_93b3_e37f_dda9,
                0x6b44_42fb_1b58_a6c7,
                0xc2c8_90c4_284b_5794,
                0x29cf_d299_66a2_faeb,
            ]),
        ),
    ),
];

fn lookup(table: &[Entry], key: &str) -> Option<pallas::Point> {
    table
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, point)| point.to_curve())
}

/// The stored $Q$ for the hash domain `domain`, if it is a known one.
pub(crate) fn q(domain: &str) -> Option<pallas::Point> {
    lookup(&Q, domain)
}

/// The stored $R$ for the blinding personalization `personalization`, which ends in
/// `-r`, if it is a known one.
pub(crate) fn r(personalization: &str) -> Option<pallas::Point> {
    lookup(&R, personalization)
}

#[cfg(test)]
mod tests {
    use pasta_curves::{
        arithmetic::{CurveAffine, CurveExt},
        pallas,
    };

    use super::{q, r, Q, R};
    use crate::Q_PERSONALIZATION;

    #[test]
    fn q_matches_hash_to_curve() {
        for (domain, point) in Q.iter() {
            assert!(bool::from(point.is_on_curve()), "{domain}");
            let expected = pallas::Point::hash_to_curve(Q_PERSONALIZATION)(domain.as_bytes());
            assert_eq!(q(domain), Some(expected), "{domain}");
        }
    }

    #[test]
    fn r_matches_hash_to_curve() {
        for (personalization, point) in R.iter() {
            assert!(bool::from(point.is_on_curve()), "{personalization}");
            let expected = pallas::Point::hash_to_curve(personalization)(&[]);
            assert_eq!(r(personalization), Some(expected), "{personalization}");
        }
    }

    #[test]
    fn unknown_personalizations_are_not_found() {
        assert_eq!(q("z.cash:test-Sinsemilla"), None);
        // Exact match only: a prefix or a different suffix is a different domain.
        assert_eq!(q("z.cash:Orchard-MerkleCRH-"), None);
        assert_eq!(q("z.cash:Orchard-NoteCommit-r"), None);
        assert_eq!(r("z.cash:Orchard-NoteCommit-M"), None);
    }
}
