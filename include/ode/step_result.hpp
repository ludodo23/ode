#pragma once

#include "state.hpp"

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
 */
template<StateType S, typename Dense>
struct AdaptiveStepResult {
    S     y;      // solution au bout du pas
    S     error;  // estimée d'erreur (y_haute_ordre - y_basse_ordre)
    Dense dense;  // polynôme d'interpolation local
};

} // namespace ode
