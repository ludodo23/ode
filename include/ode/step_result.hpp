#pragma once

#include "ode/concept_api.hpp"

namespace ode {

// ─── Résultat d'un pas fixe ──────────────────────────────────────────────────

/**
 * @brief Result of a fixed-step integration.
 * @tparam S The state type.
 */
template<StateType S>
struct StepResult {
    S y;
};

// ─── Résultat d'un pas adaptatif avec dense output ───────────────────────────
// Dense : foncteur appelable comme Dense(double t) -> S
// Il encode le polynôme local sur [t, t+dt]

/**
 * @brief Result of an adaptive-step integration with dense output.
 * @tparam S The state type.
 * @tparam Dense The type of the dense output function.
 * @tparam Error The type of the error estimate.
 */
template<StateType S, DenseType<S> Dense, ErrorNormType<S> Error>
struct AdaptiveStepResult {
    S     y;      // solution au bout du pas
    Error error;  // estimée d'erreur pour le pas
    Dense dense;  // polynôme d'interpolation local
};

} // namespace ode
