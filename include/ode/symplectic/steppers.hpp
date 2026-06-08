#pragma once


#include "ode/states.hpp"
#include "ode/step_result.hpp"

namespace ode {

// ─── Velocity Verlet (Störmer-Verlet, ordre 2 symplectique) ─────────────────
// Pour SeparableProblem : x'' = a(x)
// 2 évaluations d'accélération par pas.

/**
 * @brief Velocity Verlet (Störmer-Verlet) symplectic integrator for second-order ODEs.
 * This struct implements the Velocity Verlet method, which is a symplectic integrator designed for solving second-order ordinary differential equations of the form x'' = a(x). The method is particularly well-suited for problems in classical mechanics, such as planetary motion or molecular dynamics, where preserving the symplectic structure of the phase space is important for long-term stability and accuracy. The step() method takes the current time t, position x, velocity v, and time step dt, and returns the new position and velocity after one integration step. The method requires two evaluations of the acceleration function per step: one at the initial position to compute the intermediate velocity, and one at the new position to compute the final velocity.
 */
template<typename Problem>
struct VelocityVerletStepper {
    using State = typename Problem::state_type;

    StepResult<State>
    step(const Problem& prob,
         double t,
         const State& z,
         double dt) const
    {
        auto dz0 = prob.f(t, z);

        auto v_mid = z.yp + (dt * 0.5) * dz0.yp;
        auto x_new = z.y  + dt * v_mid;

        State z_tmp{x_new, v_mid};

        auto dz1 = prob.f(t + dt, z_tmp);

        auto v_new = v_mid + (dt * 0.5) * dz1.yp;

        return StepResult<State>{
            State{x_new, v_new}
        };
    }
};

} // namespace ode