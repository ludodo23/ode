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

// ─────────────────────────────────────────────────────────────────────────────
// Base générique Embedded RK
// ─────────────────────────────────────────────────────────────────────────────
/**
 * @brief Generic embedded Runge-Kutta stepper for adaptive time-stepping.
 * @tparam Problem Problem type.
 * @tparam Stages Number of RK stages.
 * @tparam Tableau Butcher tableau provider (with b_high, b_low, fsal, etc.).
 * @tparam DenseOutput Type for dense output interpolation. Must satisfy DenseType<State>.
 * 
 * The step() method computes both the high-order and low-order solutions, estimates the error, and constructs
 * the dense output polynomial.
 * 
 */
template<typename Problem,
         std::size_t Stages,
         typename Tableau,
         DenseType<typename Problem::state_type> DenseOutput>
struct EmbeddedRKStepper {
    using State  = typename Problem::state_type;
    using Dense  = DenseOutput;
    using Result = AdaptiveStepResult<State, Dense>;

    /**
     * @brief Compute a single adaptive step.
     * 
     * This method performs the following steps:
     * 1. Compute the intermediate stages k[i] using the tableau coefficients a[i][j] following the standard equation: k[i] = f(t + c[i]*dt, y + dt * sum(a[i][j] * k[j], j=0..i-1)) where the sum is over the previous stages j < i.
     * 2. Compute the high-order solution y_high using b_high w.r.t. equation: y_high = y + dt * sum(b_high[i] * k[i], i=0..Stages-1).
     * 3. If FSAL is enabled, compute the FSAL stage k_fsal for the next step using the high-order solution: k_fsal = f(t + dt, y_high).
     * 4. Compute the low-order solution y_low using b_low and optionally the FSAL stage if fsal_embedded is enabled w.r.t. equation: y_low = y + dt * (sum(b_low[i] * k[i], i=0..Stages-1) + b_low_fsal * k_fsal).
     * 5. Construct the dense output polynomial using the provided DenseOutput type.
     * 
     * The returned AdaptiveStepResult contains:
     * - y: the high-order solution at the end of the step.
     * - error: the estimated error (y_high - y_low).
     * - dense: the dense output polynomial for interpolation within the step.
     * 
     * @param prob Problem instance.
     * @param t Current time.
     * @param y Current state.
     * @param dt Time step.
     * @return Adaptive step result.
     */
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

/**
 * @brief Dense output for RK23 Bogacki–Shampine method.
 * 
 * This struct provides a dense output polynomial for the RK23 method, which allows
 * for interpolation of the solution at any point within the time step. The dense output
 * is constructed using the initial state y0, the final state y1, the intermediate stages k1 and k4,
 * and the time information t0 and dt. The operator() method evaluates 
 * the dense output polynomial at a given time t within the interval [t0, t0 + dt].
 * 
 * @tparam State State type.
 * 
 */
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

/**
 * @brief Butcher tableau for the RK23 Bogacki–Shampine method.
 * 
 * This tableau defines the coefficients for the RK23 method, which is a 3-stage embedded Runge-Kutta method of order 2(3). It has the following properties:
 * - FSAL (First Same As Last): The last stage of the current step is the first stage of the next step, which can save one function evaluation per step.
 * - Embedded: It provides both a second-order solution (y_low) and a third-order solution (y_high) for error estimation and adaptive time-stepping.
 * - The low-order solution includes a contribution from the FSAL stage if fsal_embedded is true, which can improve the accuracy of the error estimate
 * 
 * The tableau is defined as follows:
 * - c: The time points for the stages.
 * - a: The coefficients for the intermediate stages.
 * - b_high: The coefficients for the high-order solution (order 3).
 * - b_low: The coefficients for the low-order solution (order 2).
 * 
 * The RK23 method is particularly efficient for problems where a moderate accuracy is sufficient and where the cost of function evaluations is significant, as it allows for adaptive time-stepping with a good balance between accuracy and computational effort.
 * We remind the equations for the RK23 method:
 * - k1 = f(t, y)
 * - k2 = f(t + 0.5*dt, y + 0.5*dt*k1)
 * - k3 = f(t + 0.75*dt, y + 0.75*dt*k2)
 * - k4 = f(t + dt, y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3))
 * - y_high = y + dt*(2/9*k1 + 1/3*k2 + 4/9*k3)
 * - y_low = y + dt*(7/24*k1 + 1/4*k2 + 1/3*k3 + 1/8*k4) (if fsal_embedded is true, add 1/8*k4 to the low-order solution)
 *  
 */
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

/** @brief RK23 stepper for adaptive time-stepping.
 * 
 * This struct implements the RK23 method, which is a 3-stage embedded Runge-Kutta method of order 2(3). It uses the RK23Tableau for the coefficients and the RK23DenseOutput for interpolation. The step() method computes both the high-order and low-order solutions, estimates the error, and constructs the dense output polynomial for interpolation within the step.
 */
template<typename Problem>
using RK23Stepper = EmbeddedRKStepper<Problem, 3, RK23Tableau, RK23DenseOutput<typename Problem::state_type>>;


// ─────────────────────────────────────────────────────────────────────────────
// RK45 Dormand–Prince
// ─────────────────────────────────────────────────────────────────────────────

/** @brief Dense output for RK45 Dormand–Prince method.
 * 
 * This struct provides a dense output polynomial for the RK45 method, which allows
 * for interpolation of the solution at any point within the time step. The dense
 * output is constructed using the initial state y0, the intermediate stages k1, k3,
 * k4, k5, k6, and the time information t0 and dt. The operator() method evaluates 
 * the dense output polynomial at a given time t within the interval [t0, t0 + dt].
 * 
 *  @tparam State State type.
 */
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

/** @brief Tableau for RK45 Dormand–Prince method.
 * 
 * This tableau defines the coefficients for the RK45 method, which is a 6-stage embedded Runge-Kutta method of order 4(5). It has the following properties:
 * - FSAL (First Same As Last): The last stage of the current step is the first stage of the next step, which can save one function evaluation per step.
 * - Embedded: It provides both a fourth-order solution (y_high) and a fifth-order solution (y_low) for error estimation and adaptive time-stepping.
 * - The low-order solution includes a contribution from the FSAL stage if fsal_embedded is true, which can improve the accuracy of the error estimate
 * 
 * The tableau is defined as follows:
 * - c: The time points for the stages.
 * - a: The coefficients for the intermediate stages.
 * - b_high: The coefficients for the high-order solution (order 5).
 * - b_low: The coefficients for the low-order solution (order 4).
 * 
 * The RK45 method is particularly efficient for problems where a high accuracy is required and where the cost of function evaluations is significant, as it allows for adaptive time-stepping with a good balance between accuracy and computational effort.
 * We remind the equations for the RK45 method:
 * - k1 = f(t, y)
 * - k2 = f(t + 1/5*dt, y + dt*(1/5*k1))
 * - k3 = f(t + 3/10*dt, y + dt*(3/40*k1 + 9/40*k2))
 * - k4 = f(t + 4/5*dt, y + dt*(44/45*k1 - 56/15*k2 + 32/9*k3))
 * - k5 = f(t + 8/9*dt, y + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
 * - k6 = f(t + dt, y + dt*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5))
 * - y_high = y + dt*(35/384*k1 + 0*k2 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
 * - y_low = y + dt*(5179/57600*k1 + 0*k2 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6) (if fsal_embedded is true, add
 *  
 * 
 *  @tparam State State type.
 */
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

/**
 * @brief RK45 Dormand–Prince stepper for adaptive time-stepping.
 * 
 * This struct implements the RK45 method, which is a 6-stage embedded Runge-Kutta method of order 4(5). It uses the RK45Tableau for the coefficients and the RK45DenseOutput for interpolation. The step() method computes both the high-order and low-order solutions, estimates the error, and constructs the dense output polynomial for interpolation within the step.
 */
template<typename Problem>
using RK45Stepper = EmbeddedRKStepper<Problem, 6, RK45Tableau, RK45DenseOutput<typename Problem::state_type>>;

// ════════════════════════════════════════════════════════════════════════════
// Méthode SYMPLECTIQUE
// ════════════════════════════════════════════════════════════════════════════

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
