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
