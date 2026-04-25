#pragma once

#include "problem.hpp"
#include "step_result.hpp"
#include <cmath>

namespace ode {

// ════════════════════════════════════════════════════════════════════════════
// Méthodes FIXES
// ════════════════════════════════════════════════════════════════════════════

// ─── Euler explicite (ordre 1) ───────────────────────────────────────────────

template<typename Problem>
struct EulerStepper {
    using State = typename Problem::state_type;

    StepResult<State> step(const Problem& prob,
                           double t, const State& y, double dt) const
    {
        return { y + dt * prob.f(t, y) };
    }
};

// ─── RK2 Midpoint (ordre 2) ──────────────────────────────────────────────────

template<typename Problem>
struct RK2Stepper {
    using State = typename Problem::state_type;

    StepResult<State> step(const Problem& prob,
                           double t, const State& y, double dt) const
    {
        auto k1 = prob.f(t,          y);
        auto k2 = prob.f(t + dt/2.0, y + (dt/2.0) * k1);
        return { y + dt * k2 };
    }
};

// ─── RK4 classique (ordre 4) ─────────────────────────────────────────────────

template<typename Problem>
struct RK4Stepper {
    using State = typename Problem::state_type;

    StepResult<State> step(const Problem& prob,
                           double t, const State& y, double dt) const
    {
        auto k1 = prob.f(t,          y);
        auto k2 = prob.f(t + dt/2.0, y + (dt/2.0) * k1);
        auto k3 = prob.f(t + dt/2.0, y + (dt/2.0) * k2);
        auto k4 = prob.f(t + dt,     y + dt        * k3);
        return { y + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4) };
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Méthodes ADAPTATIVES (embedded RK)
// ════════════════════════════════════════════════════════════════════════════

// ─── RK23 Bogacki–Shampine (ordre 3, estimateur ordre 2) ────────────────────
// FSAL : k4 du pas courant = k1 du suivant → 3 éval/pas accepté
// Référence : Bogacki & Shampine, "A 3(2) pair of Runge-Kutta formulas", 1989

template<typename Problem>
struct RK23Stepper {
    using State = typename Problem::state_type;

    // Interpolant cubique certifié ordre 3, θ = (t-t0)/dt ∈ [0,1]
    struct Dense {
        State  y0, y1, k1, k4;
        double t0, dt;

        State operator()(double t) const {
            double th = (t - t0) / dt;
            return (1.0 - th)*y0 + th*y1
                 + th*(th - 1.0) * (
                       (1.0 - 2.0*th)*(y1 - y0)
                     + (th - 1.0)*dt*k1
                     + th*dt*k4
                   );
        }
    };

    using Result = AdaptiveStepResult<State, Dense>;

    Result step(const Problem& prob,
                double t, const State& y, double dt) const
    {
        auto k1 = prob.f(t,              y);
        auto k2 = prob.f(t + dt/2.0,     y + (dt/2.0)     * k1);
        auto k3 = prob.f(t + 3.0*dt/4.0, y + (3.0*dt/4.0) * k2);

        // Solution ordre 3
        auto y3 = y + dt*(2.0/9.0*k1 + 1.0/3.0*k2 + 4.0/9.0*k3);

        // FSAL : k4 = f(t+dt, y3)
        auto k4 = prob.f(t + dt, y3);

        // Solution ordre 2 (embedded)
        auto y2 = y + dt*(7.0/24.0*k1 + 1.0/4.0*k2
                        + 1.0/3.0*k3  + 1.0/8.0*k4);

        return { y3, y3 - y2, Dense{y, y3, k1, k4, t, dt} };
    }
};

// ─── RK45 Dormand–Prince (ordre 5, estimateur ordre 4) ──────────────────────
// FSAL : k7 = k1 du pas suivant → 5 éval/pas accepté
// Référence : Hairer, Nørsett, Wanner — "Solving ODEs I", Table 5.2 p.178

template<typename Problem>
struct RK45Stepper {
    using State = typename Problem::state_type;

    // Interpolant continu de Shampine (1986) — ordre 4 aux points de sortie,
    // ordre 3 uniforme sur [t0, t0+dt].
    // θ = (t-t0)/dt ∈ [0,1]
    struct Dense {
        State  y0, k1, k3, k4, k5, k6;
        double t0, dt;

        State operator()(double t) const {
            double th  = (t - t0) / dt;
            double th2 = th * th;
            double th3 = th2 * th;

            // Coefficients de Shampine (1986), repris dans Hairer vol.1 p.191
            return y0 + dt * th * (
                k1 * (1.0
                     - (183.0/64.0)*th
                     + (37.0/12.0)*th2
                     - (145.0/128.0)*th3)
              + k3 * (0.0
                     + (1500.0/371.0)*th
                     - (1000.0/159.0)*th2
                     + (1000.0/371.0)*th3)
              + k4 * (0.0
                     - (125.0/32.0)*th
                     + (125.0/12.0)*th2
                     - (375.0/64.0)*th3)
              + k5 * (0.0
                     + (9477.0/3392.0)*th
                     - (729.0/106.0)*th2
                     + (25515.0/6784.0)*th3)
              + k6 * (0.0
                     - (11.0/7.0)*th
                     + (11.0/3.0)*th2
                     - (55.0/28.0)*th3)
            );
        }
    };

    using Result = AdaptiveStepResult<State, Dense>;

    Result step(const Problem& prob,
                double t, const State& y, double dt) const
    {
        // Tableau de Dormand-Prince — Hairer, Nørsett, Wanner, Table 5.2
        auto k1 = prob.f(t, y);

        auto k2 = prob.f(t + (1.0/5.0)*dt,
                    y + dt*(1.0/5.0*k1));

        auto k3 = prob.f(t + (3.0/10.0)*dt,
                    y + dt*(3.0/40.0*k1 + 9.0/40.0*k2));

        auto k4 = prob.f(t + (4.0/5.0)*dt,
                    y + dt*(44.0/45.0*k1 - 56.0/15.0*k2 + 32.0/9.0*k3));

        auto k5 = prob.f(t + (8.0/9.0)*dt,
                    y + dt*(19372.0/6561.0*k1  - 25360.0/2187.0*k2
                          + 64448.0/6561.0*k3  -   212.0/729.0*k4));

        auto k6 = prob.f(t + dt,
                    y + dt*(9017.0/3168.0*k1  -  355.0/33.0*k2
                          + 46732.0/5247.0*k3 +   49.0/176.0*k4
                          - 5103.0/18656.0*k5));

        // Solution ordre 5 (coefficients b_i)
        auto y5 = y + dt*(  35.0/384.0*k1
                          + 500.0/1113.0*k3
                          + 125.0/192.0*k4
                          - 2187.0/6784.0*k5
                          +  11.0/84.0*k6);

        // FSAL
        auto k7 = prob.f(t + dt, y5);

        // Solution ordre 4 (coefficients b*_i, embedded)
        auto y4 = y + dt*(5179.0/57600.0*k1
                        + 7571.0/16695.0*k3
                        + 393.0/640.0*k4
                        - 92097.0/339200.0*k5
                        + 187.0/2100.0*k6
                        +    1.0/40.0*k7);

        return { y5, y5 - y4, Dense{y, k1, k3, k4, k5, k6, t, dt} };
    }
};

// ════════════════════════════════════════════════════════════════════════════
// Méthode SYMPLECTIQUE
// ════════════════════════════════════════════════════════════════════════════

// ─── Velocity Verlet (Störmer-Verlet, ordre 2 symplectique) ─────────────────
// Pour SeparableProblem : x'' = a(x)
// 2 évaluations d'accélération par pas.

template<typename Problem>
struct VelocityVerletStepper {
    using State = typename Problem::state_type;

    struct Result {
        State x;
        State v;
    };

    Result step(const Problem& prob,
                double /*t*/, const State& x, const State& v, double dt) const
    {
        State a0    = prob.accel(x);
        State v_mid = v    + (dt/2.0) * a0;
        State x_new = x    + dt       * v_mid;
        State a1    = prob.accel(x_new);
        State v_new = v_mid + (dt/2.0) * a1;
        return {x_new, v_new};
    }
};

} // namespace ode
