#pragma once

#include "problem.hpp"
#include "controllers.hpp"
#include "samplers.hpp"

namespace ode {

namespace detail {
    template<typename S>
    double adl_norm(const S& s) {
        using ode::norm;
        return norm(s);
    }
}

/**
 * @brief Integrate an ODE problem using a given stepper and controller.
 * 
 * @tparam Problem The type of the ODE problem.
 * @tparam Stepper The type of the stepper.
 * @tparam Controller The type of the controller.
 * @tparam Sampler The type of the sampler.
 * @param prob The ODE problem to integrate.
 * @param stepper The stepper to use for integration.
 * @param controller The controller to use for step size control.
 * @param sampler The sampler to use for collecting the solution.
 * @param t_end The end time of the integration.
 * @param max_steps The maximum number of steps to take.
 * @return The solution containing the integrated values.
 */
template<typename Problem, typename Stepper,
         typename Controller, typename Sampler>
auto integrate(const Problem& prob,
               Stepper&       stepper,
               Controller&    controller,
               Sampler&       sampler,
               double         t_end,
               int            max_steps = 1'000'000)
{
    using State = typename Problem::state_type;

    double t = prob.t0;
    State  y = prob.y0;
    int    n_steps    = 0;
    int    n_rejected = 0;

    sampler.init(t, y);

    constexpr double eps = 1e-12;  // garde contre dt ≈ 0 en fin d'intégration

    while (t < t_end) {
        if (n_steps >= max_steps && max_steps >= 0) {
            auto sol       = sampler.result();
            sol.success    = false;
            sol.message    = "Max steps reached";
            sol.n_steps    = n_steps;
            sol.n_rejected = n_rejected;
            return sol;
        }

        double dt = std::min(controller.dt(), t_end - t);
        if (dt < eps * (1.0 + std::abs(t_end)))
            break;  // on est arrivé à t_end à eps près

        auto res = stepper.step(prob, t, y, dt);

        // TODO à revoir
        if constexpr (requires { res.error; }) {
            double en     = detail::adl_norm(res.error);
            double yn     = detail::adl_norm(res.y);
            double e_norm = en / (1e-9 + 1e-6 * yn);

            if (!controller.accept_normalized(e_norm)) {
                ++n_rejected;
                continue;
            }
        }

        t += dt;
        y  = res.y;
        ++n_steps;

        if constexpr (requires { res.dense; }) {
            sampler.observe(t, y, res.dense);
        } else {
            sampler.observe(t, y);
        }
    }

    auto sol       = sampler.result();
    sol.n_steps    = n_steps;
    sol.n_rejected = n_rejected;
    return sol;
}

/**
 * @brief Integrate a separable ODE problem using a given stepper and sampler.
 * 
 * @tparam Problem The type of the separable ODE problem.
 * @tparam Stepper The type of the stepper.
 * @tparam Sampler The type of the sampler.
 * @param prob The separable ODE problem to integrate.
 * @param stepper The stepper to use for integration.
 * @param sampler The sampler to use for collecting the solution.
 * @param t_end The end time of the integration.
 * @param dt The time step to use for integration.
 * @param max_steps The maximum number of steps to take.
 * @return The solution containing the integrated values.
 */
template<typename Problem, typename Stepper, typename Sampler>
auto integrate_separable(const Problem& prob,
                         Stepper&       stepper,
                         Sampler&       sampler,
                         double         t_end,
                         double         dt,
                         int            max_steps = 1'000'000)
{
    using State = typename Problem::state_type;

    double t = prob.t0;
    State  x = prob.x0;
    State  v = prob.v0;
    int    n_steps = 0;

    sampler.init(t, x);

    constexpr double eps = 1e-12;

    while (t < t_end) {
        if (n_steps >= max_steps && max_steps >= 0) break;

        double step = std::min(dt, t_end - t);
        if (step < eps * (1.0 + std::abs(t_end))) break;

        auto res = stepper.step(prob, t, x, v, step);

        t += step;
        x  = res.x;
        v  = res.v;
        ++n_steps;

        sampler.observe(t, x);
    }

    auto sol    = sampler.result();
    sol.n_steps = n_steps;
    return sol;
}

} // namespace ode
