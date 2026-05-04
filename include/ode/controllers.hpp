#pragma once

#include "state.hpp"
#include <algorithm>
#include <cmath>

namespace ode {

// ─── Contrôleur à pas fixe ───────────────────────────────────────────────────

/**
 * @brief Fixed step controller.
 * 
 * Basic controller that always accepts the step and does not change the time step size.
 */
class FixedController {
public:
    /**
     * @brief FixedController constructor.
     * @param dt The fixed time step size to use for all steps.
     */
    explicit FixedController(double dt) : dt_(dt) {}

    /**
     * @brief Get the fixed time step size.
     * @return The fixed time step size.
     */
    double dt() const { return dt_; }

    /**
     * @brief Accept the current step.
     * @param args Arguments for the step.
     * @return Always returns true.
     */
    template<typename... Args>
    bool accept(Args&&...) { return true; } // Toujours accepté, pas ne change pas

private:
    /**
     * @brief The fixed time step size.
     */
    double dt_;
};

// ─── Contrôleur adaptatif (I-controller standard) ────────────────────────────
// Basé sur la norme mixte relative/absolue par composante
// Facteur de sécurité : 0.9, exposant : 1/(q+1) avec q = ordre de l'erreur

/**
 * @brief Adaptive controller.
 * 
 * Adaptive controller that adjusts the time step size based on the estimated error.
 */
class AdaptiveController {
public:
    /**
     * @brief AdaptiveController constructor.
     * @param dt_init The initial time step size.
     * @param rtol The relative tolerance.
     * @param atol The absolute tolerance.
     * @param dt_min The minimum time step size.
     * @param dt_max The maximum time step size.
     * @param error_order The order of the error estimate.
     */
    AdaptiveController(double dt_init, double rtol, double atol,
                       double dt_min = 1e-12, double dt_max = 1.0,
                       int error_order = 3)
        : dt_(dt_init), rtol_(rtol), atol_(atol)
        , dt_min_(dt_min), dt_max_(dt_max)
        , exponent_(1.0 / error_order)
    {}

    // Norme scalaire : erreur normalisée sur un état scalaire
    bool accept(double err, double y)
    {
        double sc = atol_ + rtol_ * std::abs(y);
        double e  = std::abs(err) / sc;
        return update(e);
    }

    // Norme vectorielle : on passe la norme globale pre-calculée
    // L'appelant doit calculer norm(error / (atol + rtol * |y|))
    bool accept_normalized(double e_norm)
    {
        return update(e_norm);
    }

    double dt() const { return dt_; }

private:
    bool update(double e)
    {
        // Facteur de mise à l'échelle : safety * (1/e)^(1/(q+1))
        constexpr double safety   = 0.9;
        constexpr double min_fac  = 0.2;
        constexpr double max_fac  = 10.0;

        double factor = safety * std::pow(1.0 / std::max(e, 1e-10), exponent_);
        factor = std::clamp(factor, min_fac, max_fac);

        if (e <= 1.0) {
            // Pas accepté : on peut agrandir
            dt_ = std::clamp(dt_ * factor, dt_min_, dt_max_);
            return true;
        } else {
            // Pas rejeté : on réduit
            dt_ = std::clamp(dt_ * factor, dt_min_, dt_max_);
            return false;
        }
    }

    double dt_;
    double rtol_, atol_;
    double dt_min_, dt_max_;
    double exponent_;
};

} // namespace ode
