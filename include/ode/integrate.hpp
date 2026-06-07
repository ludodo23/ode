#pragma once

#include "ode/problem.hpp"
#include "ode/controllers.hpp"
#include "ode/samplers.hpp"

namespace ode {

/**
 * @brief Integrate an ODE problem using a given stepper and controller.
 *
 * Handles both standard first-order problems and separable problems (e.g. Verlet).
 * For separable problems, the state is split into (y, yp) = (x, v) and the stepper
 * receives an AugmentedState. The sampler always receives an AugmentedState for
 * separable problems, yielding a Solution<S> with both y and yp filled.
 *
 * @tparam Problem    The type of the ODE problem.
 * @tparam Stepper    The type of the stepper.
 * @tparam Controller The type of the controller (unused for fixed-step methods).
 * @tparam Sampler    The type of the sampler.
 * @param prob        The ODE problem to integrate.
 * @param stepper     The stepper to use for integration.
 * @param controller  The controller to use for step size control.
 * @param sampler     The sampler to use for collecting the solution.
 * @param t_end       The end time of the integration.
 * @param max_steps   The maximum number of steps to take.
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
    int    n_steps    = 0;
    int    n_rejected = 0;

    constexpr double eps = 1e-12;
    State y = prob.y0;
    sampler.init(t, y);

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
        if (dt < eps * (1.0 + std::abs(t_end))) break;

        auto res = stepper.step(prob, t, y, dt);

        if constexpr (requires { res.error; }) {
            double norm_error = res.error(controller.atol(), controller.rtol());
            if (!controller.accept_normalized(norm_error)) {
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

} // namespace ode