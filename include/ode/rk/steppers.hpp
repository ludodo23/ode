#pragma once

#include "problem.hpp"
#include "step_result.hpp"
#include "tableau.hpp"
#include <cmath>
#include <array>
#include <utility>

namespace ode {

// ════════════════════════════════════════════════════════════════════════════
// Méthodes FIXES
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Steppers génériques Runge-Kutta explicites
// - Évite la duplication
// - Sépare la "méthode" (tableau de Butcher) de l’algorithme
// ════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Base générique : Runge-Kutta explicite via tableau de Butcher
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Generic explicit Runge-Kutta stepper.
 * @tparam Problem Problem type.
 * @tparam Stages Number of RK stages.
 * @tparam Tableau Butcher tableau provider.
 */
// TODO mettre le Stages dans le tableau
template<typename Problem, std::size_t Stages, typename Tableau>
struct ExplicitRKStepper {
    using State = typename Problem::state_type;

    StepResult<State> step(const Problem& prob,
                           double t,
                           const State& y,
                           double dt) const
    {
        std::array<State, Stages> k;

        // Calcul des étapes intermédiaires
        for (std::size_t i = 0; i < Stages; ++i) {
            State yi = y;

            for (std::size_t j = 0; j < i; ++j) {
                yi = yi + (dt * Tableau::a[i][j]) * k[j];
            }

            k[i] = prob.f(t + Tableau::c[i] * dt, yi);
        }

        // Combinaison finale
        State result = y;
        for (std::size_t i = 0; i < Stages; ++i) {
            result = result + (dt * Tableau::b[i]) * k[i];
        }

        return { result };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Aliases lisibles
// ─────────────────────────────────────────────────────────────────────────────

template<typename Problem>
using EulerStepper = ExplicitRKStepper<Problem, 1, EulerTableau>;

template<typename Problem>
using RK2Stepper = ExplicitRKStepper<Problem, 2, RK2Tableau>;

template<typename Problem>
using RK4Stepper = ExplicitRKStepper<Problem, 4, RK4Tableau>;

// ════════════════════════════════════════════════════════════════════════════
// Méthodes ADAPTATIVES (embedded RK)
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Steppers adaptatifs génériques (Embedded Runge-Kutta)
// - Sépare tableau de Butcher + solution haute/basse précision
// - Mutualise RK23 / RK45 / futurs DOP853, Cash-Karp, Fehlberg...
// - Dense output spécialisé par méthode
// ════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Base générique Embedded RK
// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Generic embedded Runge-Kutta stepper for adaptive time-stepping.
 * @tparam Problem Problem type.
 * @tparam Stages Number of RK stages.
 * @tparam Tableau Butcher tableau provider (with b_high, b_low, fsal, etc.).
 * 
 * The step() method computes both the high-order and low-order solutions, estimates the error, and constructs
 * the dense output polynomial.
 * 
 */
// TODO: mettre le Stages dans le tableau.
template<typename Problem,
         std::size_t Stages,
         typename Tableau>
struct EmbeddedRKStepper {
    using State  = typename Problem::state_type;
    using Result = AdaptiveStepResult<State, typename Tableau::dense_type, typename Tableau::error_type>;

    /**
     * @brief Compute a single adaptive step.
     * 
     * This method performs the following steps:
     * 1. Compute the intermediate stages k[i] using the tableau coefficients a[i][j] following the standard equation: k[i] = f(t + c[i]*dt, y + dt * sum(a[i][j] * k[j], j=0..i-1)) where the sum is over the previous stages j < i.
     * 2. Compute the high-order solution y_high using b_high w.r.t. equation: y_high = y + dt * sum(b_high[i] * k[i], i=0..Stages-1).
     * 3. If FSAL is enabled, compute the FSAL stage k_fsal for the next step using the high-order solution: k_fsal = f(t + dt, y_high).
     * 4. Compute the low-order solution y_low using b_low and optionally the FSAL stage if fsal_embedded is enabled w.r.t. equation: y_low = y + dt * (sum(b_low[i] * k[i], i=0..Stages-1) + b_low_fsal * k_fsal).
     * 5. Construct the dense output polynomial using the provided DenseOutput type.
     * 
     * The returned AdaptiveStepResult contains:
     * - y: the high-order solution at the end of the step.
     * - error: the estimated error.
     * - dense: the dense output polynomial for interpolation within the step.
     * 
     * @param prob Problem instance.
     * @param t Current time.
     * @param y Current state.
     * @param dt Time step.
     * @return Adaptive step result.
     */
    Result step(const Problem& prob,
                double t,
                const State& y,
                double dt) const
    {
        std::array<State, Stages> k;

        // Calcul des stages
        for (std::size_t i = 0; i < Stages; ++i) {
            State yi = y;

            for (std::size_t j = 0; j < i; ++j) {
                yi = yi + (dt * Tableau::a[i][j]) * k[j];
            }

            k[i] = prob.f(t + Tableau::c[i] * dt, yi);
        }

        // Solution ordre haut
        State y_high = y;
        for (std::size_t i = 0; i < Stages; ++i) {
            y_high = y_high + (dt * Tableau::b_high[i]) * k[i];
        }

        // FSAL éventuel
        State k_fsal{};
        if constexpr (Tableau::fsal) {
            k_fsal = prob.f(t + dt, y_high);
        }

        // ---------------------------------------------------------------------
        // Method-specific error computation
        // ---------------------------------------------------------------------
        auto error = Tableau::compute_error(y, y_high, k, k_fsal, dt);

        // ---------------------------------------------------------------------
        // Dense output
        // ---------------------------------------------------------------------
        auto dense = Tableau::make_dense(prob, t, y, y_high, dt, k, k_fsal);

        return {
            y_high,
            error,
            dense
        };
    }
};

/** @brief RK23 stepper for adaptive time-stepping.
 * 
 * This struct implements the RK23 method, which is a 3-stage embedded Runge-Kutta method of order 2(3). It uses the RK23Tableau for the coefficients and the RK23DenseOutput for interpolation. The step() method computes both the high-order and low-order solutions, estimates the error, and constructs the dense output polynomial for interpolation within the step.
 */
template<typename Problem>
using RK23Stepper = EmbeddedRKStepper<Problem, 3, RK23Tableau<typename Problem::state_type>>;


/**
 * @brief RK45 Dormand–Prince stepper for adaptive time-stepping.
 * 
 * This struct implements the RK45 method, which is a 6-stage embedded Runge-Kutta method of order 4(5). It uses the RK45Tableau for the coefficients and the RK45DenseOutput for interpolation. The step() method computes both the high-order and low-order solutions, estimates the error, and constructs the dense output polynomial for interpolation within the step.
 */
template<typename Problem>
using RK45Stepper = EmbeddedRKStepper<Problem, 6, RK45Tableau<typename Problem::state_type>>;


} // namespace ode