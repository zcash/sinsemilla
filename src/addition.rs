#[cfg(test)]
use core::ops::Add;

use group::{CurveAffine as _, Group};
use pasta_curves::pallas;
use subtle::{ConstantTimeEq, CtOption};

/// P ∪ {⊥}
///
/// Simulated incomplete addition built over complete addition.
#[derive(Clone, Copy, Debug)]
pub(super) struct IncompletePoint(CtOption<pallas::Point>);

impl From<pallas::Point> for IncompletePoint {
    fn from(p: pallas::Point) -> Self {
        IncompletePoint(CtOption::new(p, 1.into()))
    }
}

impl From<IncompletePoint> for CtOption<pallas::Point> {
    fn from(p: IncompletePoint) -> Self {
        p.0
    }
}

impl IncompletePoint {
    /// Computes $\[2\] \, \mathsf{self} \;⸭\; \mathsf{rhs}$, the Sinsemilla accumulator
    /// step, as one doubling and one mixed addition instead of two incomplete additions.
    ///
    /// The specification writes the step as $(\mathsf{acc} ⸭ S) ⸭ \mathsf{acc}$. Writing
    /// $A$ for `self` and $S$ for `rhs`, [Theorem 5.4.4][theorem544] enumerates the
    /// exceptional cases as $[\alpha] A + S = \mathcal{O}$ for $\alpha \in \{-1, 1, 2\}$:
    ///
    /// - $S = A$ ($\alpha = -1$): the first addition is a doubling;
    /// - $S = -A$ ($\alpha = 1$): the first addition cancels;
    /// - $S = -\[2\] A$ ($\alpha = 2$): the *second* addition cancels, its left operand
    ///   being $A ⸭ S$ rather than $A$.
    ///
    /// That enumeration relies on the standing assumption that neither $\mathcal{Q}(D)$
    /// nor any $\mathcal{S}(j)$ is $\mathcal{O}$, from which no intermediate accumulator
    /// can be $\mathcal{O}$ either unless one of the three already occurred. The two
    /// identity checks below are therefore redundant with respect to the specification;
    /// they are kept because the literal implementation performed them, and reproducing
    /// all five conditions is what makes this a drop-in replacement that returns $\bot$
    /// on exactly the same inputs.
    ///
    /// [theorem544]: https://zips.z.cash/protocol/protocol.pdf#concretesinsemillahash
    pub(super) fn double_and_add(self, rhs: pallas::Affine) -> IncompletePoint {
        IncompletePoint(self.0.and_then(|p| {
            let q = rhs.to_curve();
            let double = p.double();

            CtOption::new(
                // Use mixed addition for efficiency.
                double + rhs,
                !(p.is_identity()
                    | q.is_identity()
                    | p.ct_eq(&q)
                    | p.ct_eq(&-q)
                    | double.ct_eq(&-q)),
            )
        }))
    }
}

// The two incomplete additions below are the literal specification form of the
// accumulator step, $(\mathsf{acc} ⸭ S) ⸭ \mathsf{acc}$. The hash no longer takes that
// path: it uses `double_and_add`, which is one doubling and one mixed addition instead.
//
// They are kept, compiled only under `cfg(test)`, because the equivalence of the two
// forms is the entire content of that rewrite, and an oracle to test it against is worth
// more than the lines it costs. `tests::double_and_add_matches_two_incomplete_additions`
// is the only caller. Deleting them would leave the rewrite pinned by nothing.
#[cfg(test)]
impl Add for IncompletePoint {
    type Output = IncompletePoint;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self::Output {
        // ⊥ ⸭ ⊥ = ⊥
        // ⊥ ⸭ P = ⊥
        IncompletePoint(self.0.and_then(|p| {
            // P ⸭ ⊥ = ⊥
            rhs.0.and_then(|q| {
                // 0 ⸭ 0 = ⊥
                // 0 ⸭ P = ⊥
                // P ⸭ 0 = ⊥
                // (x, y) ⸭ (x', y') = ⊥ if x == x'
                // (x, y) ⸭ (x', y') = (x, y) + (x', y') if x != x'
                CtOption::new(
                    p + q,
                    !(p.is_identity() | q.is_identity() | p.ct_eq(&q) | p.ct_eq(&-q)),
                )
            })
        }))
    }
}

#[cfg(test)]
impl Add<pallas::Affine> for IncompletePoint {
    type Output = IncompletePoint;

    /// Specialisation of incomplete addition for mixed addition.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: pallas::Affine) -> Self::Output {
        // ⊥ ⸭ ⊥ = ⊥
        // ⊥ ⸭ P = ⊥
        IncompletePoint(self.0.and_then(|p| {
            // P ⸭ ⊥ = ⊥ is satisfied by definition.
            let q = rhs.to_curve();

            // 0 ⸭ 0 = ⊥
            // 0 ⸭ P = ⊥
            // P ⸭ 0 = ⊥
            // (x, y) ⸭ (x', y') = ⊥ if x == x'
            // (x, y) ⸭ (x', y') = (x, y) + (x', y') if x != x'
            CtOption::new(
                // Use mixed addition for efficiency.
                p + rhs,
                !(p.is_identity() | q.is_identity() | p.ct_eq(&q) | p.ct_eq(&-q)),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use group::{ff::Field, Curve, CurveAffine as _, Group};
    use pasta_curves::{arithmetic::CurveExt, pallas};
    use subtle::CtOption;

    use super::IncompletePoint;
    use crate::S_PERSONALIZATION;

    /// The specification's form of the step, through the two incomplete additions.
    fn two_incomplete_additions(acc: pallas::Point, s: pallas::Affine) -> CtOption<pallas::Point> {
        let acc = IncompletePoint::from(acc);
        ((acc + s) + acc).into()
    }

    /// Asserts the two forms agree on the value and on which inputs are $\bot$.
    fn assert_same(acc: pallas::Point, s: pallas::Affine, case: &str) {
        let expected = two_incomplete_additions(acc, s);
        let actual: CtOption<pallas::Point> = IncompletePoint::from(acc).double_and_add(s).into();

        assert_eq!(
            bool::from(actual.is_some()),
            bool::from(expected.is_some()),
            "bottom disagrees: {case}",
        );
        if bool::from(expected.is_some()) {
            assert_eq!(
                actual.unwrap().to_affine(),
                expected.unwrap().to_affine(),
                "value disagrees: {case}",
            );
        }
    }

    /// [`IncompletePoint::double_and_add`] returns what the two incomplete additions it
    /// replaced return, on every case that distinguishes them.
    ///
    /// The three exceptional cases are the ones the doc comment on `double_and_add`
    /// enumerates, plus the two identity guards the literal implementation also
    /// performed. `S = [2] A` is the case to keep: it is NOT exceptional, so the
    /// replacement has to return a value rather than $\bot$, and it is exactly where
    /// the mixed addition that replaced the second incomplete addition is handed two
    /// equal points.
    #[test]
    fn double_and_add_matches_two_incomplete_additions() {
        let hash = pallas::Point::hash_to_curve(S_PERSONALIZATION);
        let half = pallas::Scalar::from(2).invert().unwrap();

        for i in 0u32..64 {
            let a = hash(&i.to_le_bytes());
            let s = hash(&(i + 1024).to_le_bytes());

            assert_same(a, s.to_affine(), "no exceptional case");
            // alpha = -1: the first addition is a doubling.
            assert_same(a, a.to_affine(), "S = A");
            // alpha = 1: the first addition cancels.
            assert_same(a, (-a).to_affine(), "S = -A");
            // alpha = 2: the second addition cancels.
            assert_same(a, (-a.double()).to_affine(), "S = -[2] A");
            // Not exceptional, and the one input that makes the mixed addition double.
            assert_same(s * half, s.to_affine(), "S = [2] A");
            // The identity guards, which the specification's assumptions make
            // unreachable but which the literal implementation still performed.
            assert_same(
                pallas::Point::identity(),
                s.to_affine(),
                "A is the identity",
            );
            assert_same(a, pallas::Affine::identity(), "S is the identity");
            assert_same(
                pallas::Point::identity(),
                pallas::Affine::identity(),
                "both are the identity",
            );
        }
    }
}
