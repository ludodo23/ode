#pragma once

#include "state.hpp"
#include <vector>
#include <functional>

namespace ode {

// ─── Options d'intégration ───────────────────────────────────────────────────

struct Options {
    double t_end = 1.0;
    double dt    = 1e-3;   // pas initial (ou fixe pour les méthodes non-adaptatives)

    // Tolérances pour le contrôle adaptatif
    double rtol = 1e-6;
    double atol = 1e-9;

    // Si non-vide : points d'évaluation souhaités (dense output)
    std::vector<double> t_eval = {};

    // Limites de sécurité
    double dt_min = 1e-12;
    double dt_max = 1.0;

    int max_steps = 1'000'000;
};

// ─── Solution ────────────────────────────────────────────────────────────────

template<StateType S>
struct Solution {
    std::vector<double> t;
    std::vector<S>      y;

    // Dense output global (optionnel, rempli si t_eval fourni ou méthode adaptative)
    std::function<S(double)> interp;

    bool success   = true;
    int  n_steps   = 0;
    int  n_rejected = 0;
    std::string message;
};

} // namespace ode
