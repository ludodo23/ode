#pragma once

#include "state.hpp"
#include <vector>
#include <functional>

namespace ode {

// ─── Options d'intégration ───────────────────────────────────────────────────

/**
 * @brief Options for integration.
 */
struct Options {
    /** @brief The final time. */
    double t_end = 1.0;
    /** @brief The initial time step. */
    double dt    = 1e-3;   // pas initial (ou fixe pour les méthodes non-adaptatives)

    // Tolérances pour le contrôle adaptatif
    /** @brief The relative tolerance. */
    double rtol = 1e-6;
    /** @brief The absolute tolerance. */
    double atol = 1e-9;

    /**
     * @brief The evaluation times for dense output.
     * 
     * If non-empty, the integrator will provide dense output at these specific times.
     */
    std::vector<double> t_eval = {};

    // Limites de sécurité
    /** @brief The minimum time step. */
    double dt_min = 1e-12;
    /** @brief The maximum time step. */
    double dt_max = 1.0;

    /** @brief The maximum number of steps. */
    int max_steps = 1'000'000;
};

// ─── Solution ────────────────────────────────────────────────────────────────

/**
 * @brief The solution of the ODE integration.
 * @tparam S The state type.
 */
template<StateType S>
struct Solution {
    /** @brief The time points. */
    std::vector<double> t;
    /** @brief The state points. */
    std::vector<S>      y;

    // Dense output global (optionnel, rempli si t_eval fourni ou méthode adaptative)
    /** @brief The interpolation function for dense output. */
    std::function<S(double)> interp;

    /** @brief Flag indicating if the integration was successful. */
    bool success   = true;
    /** @brief The number of steps taken. */
    int  n_steps   = 0;
    /** @brief The number of rejected steps. */
    int  n_rejected = 0;
    /** @brief A message describing the result of the integration. */
    std::string message;
};

} // namespace ode
