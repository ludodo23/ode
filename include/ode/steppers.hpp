#pragma once

#include "problem.hpp"
#include "step_result.hpp"
#include <cmath>
#include <array>
#include <utility>

namespace ode {

// ════════════════════════════════════════════════════════════════════════════
// Méthodes FIXES
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Steppers génériques Runge-Kutta explicites
// - Évite la duplication
// - Sépare la "méthode" (tableau de Butcher) de l’algorithme
// ════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// Base générique : Runge-Kutta explicite via tableau de Butcher
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Generic explicit Runge-Kutta stepper.
 * @tparam Problem Problem type.
 * @tparam Stages Number of RK stages.
 * @tparam Tableau Butcher tableau provider.
 */
template<typename Problem, std::size_t Stages, typename Tableau>
struct ExplicitRKStepper {
    using State = typename Problem::state_type;

    StepResult<State> step(const Problem& prob,
                           double t,
                           const State& y,
                           double dt) const
    {
        std::array<State, Stages> k;

        // Calcul des étapes intermédiaires
        for (std::size_t i = 0; i < Stages; ++i) {
            State yi = y;

            for (std::size_t j = 0; j < i; ++j) {
                yi = yi + (dt * Tableau::a[i][j]) * k[j];
            }

            k[i] = prob.f(t + Tableau::c[i] * dt, yi);
        }

        // Combinaison finale
        State result = y;
        for (std::size_t i = 0; i < Stages; ++i) {
            result = result + (dt * Tableau::b[i]) * k[i];
        }

        return { result };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tableaux de Butcher
// ─────────────────────────────────────────────────────────────────────────────

// Euler explicite (ordre 1)
struct EulerTableau {
    static constexpr std::array<double, 1> c = {0.0};
    static constexpr std::array<std::array<double, 1>, 1> a = {{
        {{0.0}}
    }};
    static constexpr std::array<double, 1> b = {1.0};
};

// RK2 Midpoint (ordre 2)
struct RK2Tableau {
    static constexpr std::array<double, 2> c = {0.0, 0.5};
    static constexpr std::array<std::array<double, 2>, 2> a = {{
        {{0.0, 0.0}},
        {{0.5, 0.0}}
    }};
    static constexpr std::array<double, 2> b = {0.0, 1.0};
};

// RK4 classique (ordre 4)
struct RK4Tableau {
    static constexpr std::array<double, 4> c = {0.0, 0.5, 0.5, 1.0};
    static constexpr std::array<std::array<double, 4>, 4> a = {{
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.5, 0.0, 0.0, 0.0}},
        {{0.0, 0.5, 0.0, 0.0}},
        {{0.0, 0.0, 1.0, 0.0}}
    }};
    static constexpr std::array<double, 4> b = {
        1.0 / 6.0,
        1.0 / 3.0,
        1.0 / 3.0,
        1.0 / 6.0
    };
};

// ─────────────────────────────────────────────────────────────────────────────
// Aliases lisibles
// ─────────────────────────────────────────────────────────────────────────────

template<typename Problem>
using EulerStepper = ExplicitRKStepper<Problem, 1, EulerTableau>;

template<typename Problem>
using RK2Stepper = ExplicitRKStepper<Problem, 2, RK2Tableau>;

template<typename Problem>
using RK4Stepper = ExplicitRKStepper<Problem, 4, RK4Tableau>;

// ════════════════════════════════════════════════════════════════════════════
// Méthodes ADAPTATIVES (embedded RK)
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// Steppers adaptatifs génériques (Embedded Runge-Kutta)
// - Sépare tableau de Butcher + solution haute/basse précision
// - Mutualise RK23 / RK45 / futurs DOP853, Cash-Karp, Fehlberg...
// - Dense output spécialisé par méthode
// ════════════════════════════════════════════════════════════════════════════

#include <array>

// ─────────────────────────────────────────────────────────────────────────────
// Base générique Embedded RK
// ─────────────────────────────────────────────────────────────────────────────

template<typename Problem,
         std::size_t Stages,
         typename Tableau,
         DenseType<typename Problem::state_type> DenseOutput>
struct EmbeddedRKStepper {
    using State  = typename Problem::state_type;
    using Dense  = DenseOutput;
    using Result = AdaptiveStepResult<State, Dense>;

    Result step(const Problem& prob,
                double t,
                const State& y,
                double dt) const
    {
        std::array<State, Stages> k;

        // Calcul des stages
        for (std::size_t i = 0; i < Stages; ++i) {
            State yi = y;

            for (std::size_t j = 0; j < i; ++j) {
                yi = yi + (dt * Tableau::a[i][j]) * k[j];
            }

            k[i] = prob.f(t + Tableau::c[i] * dt, yi);
        }

        // Solution ordre haut
        State y_high = y;
        for (std::size_t i = 0; i < Stages; ++i) {
            y_high = y_high + (dt * Tableau::b_high[i]) * k[i];
        }

        // FSAL éventuel
        State k_fsal{};
        if constexpr (Tableau::fsal) {
            k_fsal = prob.f(t + dt, y_high);
        }

        // Solution embedded ordre bas
        State y_low = y;
        for (std::size_t i = 0; i < Stages; ++i) {
            y_low = y_low + (dt * Tableau::b_low[i]) * k[i];
        }

        if constexpr (Tableau::fsal_embedded) {
            y_low = y_low + (dt * Tableau::b_low_fsal) * k_fsal;
        }

        return {
            y_high,
            y_high - y_low,
            DenseOutput::make(y, y_high, k, k_fsal, t, dt)
        };
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RK23 Bogacki–Shampine
// ─────────────────────────────────────────────────────────────────────────────

template<typename State>
struct RK23DenseOutput {
    State y0, y1, k1, k4;
    double t0, dt;

    State operator()(double t) const {
                double th = (t - t0) / dt;
                return (1.0 - th) * y0 + th * y1
                     + th * (th - 1.0) * (
                           (1.0 - 2.0 * th) * (y1 - y0)
                         + (th - 1.0) * dt * k1
                         + th * dt * k4
                       );
            }
    template<std::size_t N>
    static auto make(const State& y0,
                     const State& y1,
                     const std::array<State, N>& k,
                     const State& k4,
                     double t0,
                     double dt) {
        return RK23DenseOutput{y0, y1, k[0], k4, t0, dt};
    }
};

struct RK23Tableau {
    static constexpr bool fsal = true;
    static constexpr bool fsal_embedded = true;
    static constexpr double b_low_fsal = 1.0 / 8.0;

    static constexpr std::array<double, 3> c = {
        0.0, 0.5, 0.75
    };

    static constexpr std::array<std::array<double, 3>, 3> a = {{
        {{0.0, 0.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.0, 0.75, 0.0}}
    }};

    static constexpr std::array<double, 3> b_high = {
        2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0
    };

    static constexpr std::array<double, 3> b_low = {
        7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0
    };
};

template<typename Problem>
using RK23Stepper = EmbeddedRKStepper<Problem, 3, RK23Tableau, RK23DenseOutput<typename Problem::state_type>>;


// ─────────────────────────────────────────────────────────────────────────────
// RK45 Dormand–Prince
// ─────────────────────────────────────────────────────────────────────────────

template<typename State>
struct RK45DenseOutput {
    State y0, k1, k3, k4, k5, k6;
    double t0, dt;

    State operator()(double t) const {
        double th  = (t - t0) / dt;
        double th2 = th * th;
        double th3 = th2 * th;

        return y0 + dt * th * (
            k1 * (1.0
                    - (183.0/64.0)*th
                    + (37.0/12.0)*th2
                    - (145.0/128.0)*th3)
            + k3 * ((1500.0/371.0)*th
                    - (1000.0/159.0)*th2
                    + (1000.0/371.0)*th3)
            + k4 * (-(125.0/32.0)*th
                    + (125.0/12.0)*th2
                    - (375.0/64.0)*th3)
            + k5 * ((9477.0/3392.0)*th
                    - (729.0/106.0)*th2
                    + (25515.0/6784.0)*th3)
            + k6 * (-(11.0/7.0)*th
                    + (11.0/3.0)*th2
                    - (55.0/28.0)*th3)
        );
    }
    
    template<std::size_t N>
    static auto make(const State& y0,
                     const State&,
                     const std::array<State, N>& k,
                     const State&,
                     double t0,
                     double dt) {
        return RK45DenseOutput{y0, k[0], k[2], k[3], k[4], k[5], t0, dt};
    }
};

struct RK45Tableau {
    static constexpr bool fsal = true;
    static constexpr bool fsal_embedded = true;
    static constexpr double b_low_fsal = 1.0 / 40.0;

    static constexpr std::array<double, 6> c = {
        0.0,
        1.0/5.0,
        3.0/10.0,
        4.0/5.0,
        8.0/9.0,
        1.0
    };

    static constexpr std::array<std::array<double, 6>, 6> a = {{
        {{0,0,0,0,0,0}},
        {{1.0/5.0,0,0,0,0,0}},
        {{3.0/40.0,9.0/40.0,0,0,0,0}},
        {{44.0/45.0,-56.0/15.0,32.0/9.0,0,0,0}},
        {{19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0,0,0}},
        {{9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0,0}}
    }};

    static constexpr std::array<double, 6> b_high = {
        35.0/384.0,
        0.0,
        500.0/1113.0,
        125.0/192.0,
        -2187.0/6784.0,
        11.0/84.0
    };

    static constexpr std::array<double, 6> b_low = {
        5179.0/57600.0,
        0.0,
        7571.0/16695.0,
        393.0/640.0,
        -92097.0/339200.0,
        187.0/2100.0
    };
};

template<typename Problem>
using RK45Stepper = EmbeddedRKStepper<Problem, 6, RK45Tableau, RK45DenseOutput<typename Problem::state_type>>;

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
