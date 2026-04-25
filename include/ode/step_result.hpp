#pragma once

#include "state.hpp"

namespace ode {

// ─── Résultat d'un pas fixe ──────────────────────────────────────────────────

template<StateType S>
struct StepResult {
    S y;
};

// ─── Résultat d'un pas adaptatif avec dense output ───────────────────────────
// Dense : foncteur appelable comme Dense(double t) -> S
// Il encode le polynôme local sur [t, t+dt]

template<StateType S, typename Dense>
struct AdaptiveStepResult {
    S     y;      // solution au bout du pas
    S     error;  // estimée d'erreur (y_haute_ordre - y_basse_ordre)
    Dense dense;  // polynôme d'interpolation local
};

} // namespace ode
