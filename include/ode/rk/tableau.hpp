#pragma once

#include <array>
#include <cmath>

#include "ode/concept_api.hpp"

namespace ode {

// ─────────────────────────────────────────────────────────────────────────────
// Tableaux de Butcher
// ─────────────────────────────────────────────────────────────────────────────

// Euler explicite (ordre 1)
struct EulerTableau {
    static constexpr std::size_t stages = 1;
    static constexpr std::array<double, 1> c = {0.0};
    static constexpr std::array<std::array<double, 1>, 1> a = {{
        {{0.0}}
    }};
    static constexpr std::array<double, 1> b = {1.0};
};

// RK2 Midpoint (ordre 2)
struct RK2Tableau {
    static constexpr std::size_t stages = 2;
    static constexpr std::array<double, 2> c = {0.0, 0.5};
    static constexpr std::array<std::array<double, 2>, 2> a = {{
        {{0.0, 0.0}},
        {{0.5, 0.0}}
    }};
    static constexpr std::array<double, 2> b = {0.0, 1.0};
};

// RK4 classique (ordre 4)
struct RK4Tableau {
    static constexpr std::size_t stages = 4;
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


// -----------------------------------------------------------------------------
// Default embedded RK error = single state difference (RK23/RK45)
// -----------------------------------------------------------------------------
template<StateType State>
struct DefaultErrorNorm {
    State err;
    State y;
    State y_next;
    DefaultErrorNorm() = delete;
    DefaultErrorNorm(const State & err_, const State& y_, const State& y_next_) : err(err_), y(y_), y_next(y_next_) {};

    double operator()(double atol,
                      double rtol) const
    {
        double sum = 0.0;
        std::size_t n = state_size(err);

        for (std::size_t i = 0; i < n; ++i) {
            double sc = atol + rtol * std::max(std::abs(state_at(y, i)), std::abs(state_at(y_next, i)));
            double e  = state_at(err, i) / sc;
            sum += e * e;
        }

        return std::sqrt(sum / static_cast<double>(n));
    }
};

// -----------------------------------------------------------------------------
// DOP853 dual estimator
// -----------------------------------------------------------------------------
template<StateType State>
struct DOP853Error {
    State err5;
    State err3;
};

template<StateType State>
struct DOP853ErrorNorm {
    DOP853Error<State> err;
    State y;
    State y_next;

    DOP853ErrorNorm() = delete;
    DOP853ErrorNorm(const DOP853Error<State>& err_, const State& y_, const State& y_next_) : err(err_), y(y_), y_next(y_next_) {};
    double operator()(double atol,
                      double rtol) const
    {
        double err5 = 0.0;
        double err3 = 0.0;
        std::size_t n = state_size(y);

        for (std::size_t i = 0; i < n; ++i) {
            double sc = atol + rtol * std::max(std::abs(state_at(y, i)), std::abs(state_at(y_next, i)));

            double e5 = state_at(err.err5, i) / sc;
            double e3 = state_at(err.err3, i) / sc;

            err5 += e5 * e5;
            err3 += e3 * e3;
        }

        if (err5 == 0.0 && err3 == 0.0) {
            return 0.0;
        }

        double deno = err5 + 0.01 * err3;
        return std::sqrt(err5 / (static_cast<double>(n) * deno));
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
template<StateType State>
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
template <StateType State>
struct RK23Tableau {
    using error_type = DefaultErrorNorm<State>;
    using dense_type = RK23DenseOutput<State>;
    static constexpr bool fsal = true;
    static constexpr bool fsal_embedded = true;
    static constexpr double b_low_fsal = 1.0 / 8.0;
    static constexpr std::size_t stages = 3;

    static constexpr std::array<double, stages> c = {
        0.0, 0.5, 0.75
    };

    static constexpr std::array<std::array<double, stages>, stages> a = {{
        {{0.0, 0.0, 0.0}},
        {{0.5, 0.0, 0.0}},
        {{0.0, 0.75, 0.0}}
    }};

    static constexpr std::array<double, stages> b_high = {
        2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0
    };

    static constexpr std::array<double, stages> b_low = {
        7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0
    };

    template<std::size_t N>
    static error_type compute_error(const State& y,
                                const State& y_high,
                                const std::array<State, N>& k,
                                const State& k_fsal,
                                double dt)
    {
        State y_low = y;

        for (std::size_t i = 0; i < stages; ++i)
            y_low = y_low + dt * b_low[i] * k[i];

        if constexpr (fsal_embedded)
            y_low = y_low + dt * b_low_fsal * k_fsal;

        return DefaultErrorNorm<State>(y_high - y_low, y, y_high);
    }

template<typename Problem, std::size_t N>
    static dense_type make_dense(
        const Problem& prob,
        double t,
        const State& y,
        const State& y_next,
        double dt,
        const std::array<State, N>& k,
        const State& k_fsal)
    {
return {y, y_next, k[0], k_fsal, t, dt};
}
};

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
template<StateType State>
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
template <StateType State>
struct RK45Tableau {
    using error_type = DefaultErrorNorm<State>;
    using dense_type = RK45DenseOutput<State>;
    static constexpr bool fsal = true;
    static constexpr bool fsal_embedded = true;
    static constexpr double b_low_fsal = 1.0 / 40.0;
    static constexpr std::size_t stages = 6;

    static constexpr std::array<double, stages> c = {
        0.0,
        1.0/5.0,
        3.0/10.0,
        4.0/5.0,
        8.0/9.0,
        1.0
    };

    static constexpr std::array<std::array<double, stages>, stages> a = {{
        {{0,0,0,0,0,0}},
        {{1.0/5.0,0,0,0,0,0}},
        {{3.0/40.0,9.0/40.0,0,0,0,0}},
        {{44.0/45.0,-56.0/15.0,32.0/9.0,0,0,0}},
        {{19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0,0,0}},
        {{9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0,0}}
    }};

    static constexpr std::array<double, stages> b_high = {
        35.0/384.0,
        0.0,
        500.0/1113.0,
        125.0/192.0,
        -2187.0/6784.0,
        11.0/84.0
    };

    static constexpr std::array<double, stages> b_low = {
        5179.0/57600.0,
        0.0,
        7571.0/16695.0,
        393.0/640.0,
        -92097.0/339200.0,
        187.0/2100.0
    };

    template<std::size_t N>
    static error_type compute_error(const State& y,
                                const State& y_high,
                                const std::array<State, N>& k,
                                const State& k_fsal,
                                double dt)
    {
        State y_low = y;

        for (std::size_t i = 0; i < stages; ++i)
            y_low = y_low + dt * b_low[i] * k[i];

        if constexpr (fsal_embedded)
            y_low = y_low + dt * b_low_fsal * k_fsal;

        return DefaultErrorNorm<State>(y_high - y_low, y, y_high);
    }

    template<typename Problem, std::size_t N>
    static dense_type make_dense(
        const Problem& prob,
        double t,
        const State& y,
        const State& y_next,
        double dt,
        const std::array<State, N>& k,
        const State&)
    {
return RK45DenseOutput{y, k[0], k[2], k[3], k[4], k[5], t, dt};
}

};

template<StateType State>
struct DOP853DenseOutput
{
    State y0;
    State F[8];

    double t0;
    double dt;

    State operator()(double t) const
    {
        double x = (t - t0) / dt;

        State y{};

        for (std::size_t k = 0; k < 8; ++k)
        {
            const State& f = F[7 - k];

            y = y + f;

            if ((k & 1) == 0)
                y = x * y;
            else
                y = (1.0 - x) * y;
        }

        return y0 + y;
    }
};

// values from https://docs.rs/ode_solvers/latest/src/ode_solvers/butcher_tableau.rs.html and https://github.com/scipy/scipy/blob/v1.13.1/scipy/integrate/_ivp/dop853_coefficients.py
template<StateType State>
struct DOP853Tableau
{
    using error_type = DOP853ErrorNorm<State>;
    using dense_type = DOP853DenseOutput<State>;

    static constexpr std::size_t stages = 12;
    static constexpr std::size_t stages_extended = 16;
    static constexpr std::size_t interpolator_power = 7;
    static constexpr bool fsal = false;
    static constexpr bool fsal_embedded = false;

    // ------------------------------------------------------------
    // RK core (Hairer coefficients)
    // ------------------------------------------------------------

    static constexpr std::array<double, stages_extended> c = { 
        /* 0 */  0.0,
        /* 1 */  0.526001519587677318785587544488E-01,
        /* 2 */  0.789002279381515978178381316732E-01,
        /* 3 */  0.118350341907227396726757197510E+00,
        /* 4 */  0.281649658092772603273242802490E+00,
        /* 5 */  0.333333333333333333333333333333E+00,
        /* 6 */  0.25E+00,
        /* 7 */  0.307692307692307692307692307692E+00,
        /* 8 */  0.651282051282051282051282051282E+00,
        /* 9 */  0.6E+00,
        /* 10 */ 0.857142857142857142857142857142E+00,
        /* 11 */ 0.0,
        /* 12 */ 0.0,
        /* 13 */ 0.1E+00,
        /* 14 */ 0.2E+00,
        /* 15 */ 0.777777777777777777777777777778E+00,
    };

    static constexpr std::array<std::array<double, stages_extended>, stages_extended> a = {{
        {
            /* (1, 0) */ 5.26001519587677318785587544488e-2,
            /* (1, 1) */ 0.0,
            /* (1, 2) */ 0.0,
            /* (1, 3) */ 0.0,
            /* (1, 4) */ 0.0,
            /* (1, 5) */ 0.0,
            /* (1, 6) */ 0.0,
            /* (1, 7) */ 0.0,
            /* (1, 8) */ 0.0,
            /* (1, 9) */ 0.0,
            /* (1, 10) */ 0.0,
            /* (1, 12) */ 0.0,
            /* (1, 13) */ 0.0,
            /* (1, 14) */ 0.0,
            /* (1, 15) */ 0.0
        },
        {
            /* (2, 0) */ 1.97250569845378994544595329183e-2,
            /* (2, 1) */ 5.91751709536136983633785987549e-2,
            /* (2, 2) */ 0.0,
            /* (2, 3) */ 0.0,
            /* (2, 4) */ 0.0,
            /* (2, 5) */ 0.0,
            /* (2, 6) */ 0.0,
            /* (2, 7) */ 0.0,
            /* (2, 8) */ 0.0,
            /* (2, 9) */ 0.0,
            /* (2, 10) */ 0.0,
            /* (2, 12) */ 0.0,
            /* (2, 13) */ 0.0,
            /* (2, 14) */ 0.0,
            /* (2, 15) */ 0.0
        },
        {     
            /* (3, 0) */ 2.95875854768068491816892993775e-2,
            /* (3, 1) */ 0.0,
            /* (3, 2) */ 8.87627564304205475450678981324e-2,
            /* (3, 3) */ 0.0,
            /* (3, 4) */ 0.0,
            /* (3, 5) */ 0.0,
            /* (3, 6) */ 0.0,
            /* (3, 7) */ 0.0,
            /* (3, 8) */ 0.0,
            /* (3, 9) */ 0.0,
            /* (3, 10) */ 0.0,
            /* (3, 12) */ 0.0,
            /* (3, 13) */ 0.0,
            /* (3, 14) */ 0.0,
            /* (3, 15) */ 0.0
        },
        {
            /* (4, 0) */ 2.41365134159266685502369798665e-1,
            /* (4, 1) */ 0.0,
            /* (4, 2) */ -8.84549479328286085344864962717e-1,
            /* (4, 3) */ 9.24834003261792003115737966543e-1,
            /* (4, 4) */ 0.0,
            /* (4, 5) */ 0.0,
            /* (4, 6) */ 0.0,
            /* (4, 7) */ 0.0,
            /* (4, 8) */ 0.0,
            /* (4, 9) */ 0.0,
            /* (4, 10) */ 0.0,
            /* (4, 12) */ 0.0,
            /* (4, 13) */ 0.0,
            /* (4, 14) */ 0.0,
            /* (4, 15) */ 0.0
        },
        {   
            /* (5, 0) */ 3.7037037037037037037037037037e-2,
            /* (5, 1) */ 0.0,
            /* (5, 2) */ 0.0,
            /* (5, 3) */ 1.70828608729473871279604482173e-1,
            /* (5, 4) */ 1.25467687566822425016691814123e-1,
            /* (5, 5) */ 0.0,
            /* (5, 6) */ 0.0,
            /* (5, 7) */ 0.0,
            /* (5, 8) */ 0.0,
            /* (5, 9) */ 0.0,
            /* (5, 10) */ 0.0,
            /* (5, 12) */ 0.0,
            /* (5, 13) */ 0.0,
            /* (5, 14) */ 0.0,
            /* (5, 15) */ 0.0
        },
        {      
            /* (6, 0) */ 3.7109375e-2,
            /* (6, 1) */ 0.0,
            /* (6, 2) */ 0.0,
            /* (6, 3) */ 1.70252211019544039314978060272e-1,
            /* (6, 4) */ 6.02165389804559606850219397283e-2,
            /* (6, 5) */ -1.7578125e-2,
            /* (6, 6) */ 0.0,
            /* (6, 7) */ 0.0,
            /* (6, 8) */ 0.0,
            /* (6, 9) */ 0.0,
            /* (6, 10) */ 0.0,
            /* (6, 12) */ 0.0,
            /* (6, 13) */ 0.0,
            /* (6, 14) */ 0.0,
            /* (6, 15) */ 0.0
        },
        {      
            /* (7, 0) */ 3.70920001185047927108779319836e-2,
            /* (7, 1) */ 0.0,
            /* (7, 2) */ 0.0,
            /* (7, 3) */ 1.70383925712239993810214054705e-1,
            /* (7, 4) */ 1.07262030446373284651809199168e-1,
            /* (7, 5) */ -1.53194377486244017527936158236e-2,
            /* (7, 6) */ 8.27378916381402288758473766002e-3,
            /* (7, 7) */ 0.0,
            /* (7, 8) */ 0.0,
            /* (7, 9) */ 0.0,
            /* (7, 10) */ 0.0,
            /* (7, 12) */ 0.0,
            /* (7, 13) */ 0.0,
            /* (7, 14) */ 0.0,
            /* (7, 15) */ 0.0
        },
        {      
            /* (8, 0) */ 6.24110958716075717114429577812e-1,
            /* (8, 1) */ 0.0,
            /* (8, 2) */ 0.0,
            /* (8, 3) */ -3.36089262944694129406857109825,
            /* (8, 4) */ -8.68219346841726006818189891453e-1,
            /* (8, 5) */ 2.75920996994467083049415600797e1,
            /* (8, 6) */ 2.01540675504778934086186788979e1,
            /* (8, 7) */ -4.34898841810699588477366255144e1,
            /* (8, 8) */ 0.0,
            /* (8, 9) */ 0.0,
            /* (8, 10) */ 0.0,
            /* (8, 11) */ 0.0,
            /* (8, 12) */ 0.0,
            /* (8, 13) */ 0.0,
            /* (8, 14) */ 0.0,
            /* (8, 15) */ 0.0
        },
        {     
            /* (9, 0) */ 4.77662536438264365890433908527e-1,
            /* (9, 1) */ 0.0,
            /* (9, 2) */ 0.0,
            /* (9, 3) */ -2.48811461997166764192642586468,
            /* (9, 4) */ -5.90290826836842996371446475743e-1,
            /* (9, 5) */ 2.12300514481811942347288949897e1,
            /* (9, 6) */ 1.52792336328824235832596922938e1,
            /* (9, 7) */ -3.32882109689848629194453265587e1,
            /* (9, 8) */ -2.03312017085086261358222928593e-2,
            /* (9, 9) */ 0.0,
            /* (9, 10) */ 0.0,
            /* (9, 11) */ 0.0,
            /* (9, 12) */ 0.0,
            /* (9, 13) */ 0.0,
            /* (9, 14) */ 0.0,
            /* (9, 15) */ 0.0
        },
        {    
            /* (10, 0) */ -9.3714243008598732571704021658e-1,
            /* (10, 1) */ 0.0,
            /* (10, 2) */ 0.0,
            /* (10, 3) */ 5.18637242884406370830023853209,
            /* (10, 4) */ 1.09143734899672957818500254654,
            /* (10, 5) */ -8.14978701074692612513997267357,
            /* (10, 6) */ -1.85200656599969598641566180701e1,
            /* (10, 7) */ 2.27394870993505042818970056734e1,
            /* (10, 8) */ 2.49360555267965238987089396762,
            /* (10, 9) */ -3.0467644718982195003823669022,
            /* (10, 10) */ 0.0,
            /* (10, 11) */ 0.0,
            /* (10, 12) */ 0.0,
            /* (10, 13) */ 0.0,
            /* (10, 14) */ 0.0,
            /* (10, 15) */ 0.0
        },
        {    
            /* (11, 0) */ 2.27331014751653820792359768449,
            /* (11, 1) */ 0.0,
            /* (11, 2) */ 0.0,
            /* (11, 3) */ -1.05344954667372501984066689879e1,
            /* (11, 4) */ -2.00087205822486249909675718444,
            /* (11, 5) */ -1.79589318631187989172765950534e1,
            /* (11, 6) */ 2.79488845294199600508499808837e1,
            /* (11, 7) */ -2.85899827713502369474065508674,
            /* (11, 8) */ -8.87285693353062954433549289258,
            /* (11, 9) */ 1.23605671757943030647266201528e1,
            /* (11, 10) */ 6.43392746015763530355970484046e-1,
            /* (11, 11) */ 0.0,
            /* (11, 12) */ 0.0,
            /* (11, 13) */ 0.0,
            /* (11, 14) */ 0.0,
            /* (11, 15) */ 0.0
        },
        {   
            /* (12, 0) */ 5.42937341165687622380535766363e-2,
            /* (12, 1) */ 0.0,
            /* (12, 2) */ 0.0,
            /* (12, 3) */ 0.0,
            /* (12, 4) */ 0.0,
            /* (12, 5) */ 4.45031289275240888144113950566,
            /* (12, 6) */ 1.89151789931450038304281599044,
            /* (12, 7) */ -5.8012039600105847814672114227,
            /* (12, 8) */ 3.1116436695781989440891606237e-1,
            /* (12, 9) */ -1.52160949662516078556178806805e-1,
            /* (12, 10) */ 2.01365400804030348374776537501e-1,
            /* (12, 11) */ 4.47106157277725905176885569043e-2,
            /* (12, 12) */ 0.0,
            /* (12, 13) */ 0.0,
            /* (12, 14) */ 0.0,
            /* (12, 15) */ 0.0
        },
        {   
            /* (13, 0) */ 5.61675022830479523392909219681e-2,
            /* (13, 1) */ 0.0,
            /* (13, 2) */ 0.0,
            /* (13, 3) */ 0.0,
            /* (13, 4) */ 0.0,
            /* (13, 5) */ 0.0,
            /* (13, 6) */ 2.53500210216624811088794765333e-1,
            /* (13, 7) */ -2.46239037470802489917441475441e-1,
            /* (13, 8) */ -1.24191423263816360469010140626e-1,
            /* (13, 9) */ 1.5329179827876569731206322685e-1,
            /* (13, 10) */ 8.20105229563468988491666602057e-3,
            /* (13, 11) */ 7.56789766054569976138603589584e-3,
            /* (13, 12) */ -8.298e-3,
            /* (13, 13) */ 0.0,
            /* (13, 14) */ 0.0,
            /* (13, 15) */ 0.0
        },
        {   
            /* (14, 0) */ 3.18346481635021405060768473261e-2,
            /* (14, 1) */ 0.0,
            /* (14, 2) */ 0.0,
            /* (14, 3) */ 0.0,
            /* (14, 4) */ 0.0,
            /* (14, 5) */ 2.83009096723667755288322961402e-2,
            /* (14, 6) */ 5.35419883074385676223797384372e-2,
            /* (14, 7) */ -5.49237485713909884646569340306e-2,
            /* (14, 8) */ 0.0,
            /* (14, 9) */ 0.0,
            /* (14, 10) */ -1.08347328697249322858509316994e-4,
            /* (14, 11) */ 3.82571090835658412954920192323e-4,
            /* (14, 12) */ -3.40465008687404560802977114492e-4,
            /* (14, 13) */ 1.41312443674632500278074618366e-1,
            /* (14, 14) */ 0.0,
            /* (14, 15) */ 0.0
        },
        {    
            /* (15, 0) */ -4.28896301583791923408573538692e-1,
            /* (15, 1) */ 0.0,
            /* (15, 2) */ 0.0,
            /* (15, 3) */ 0.0,
            /* (15, 4) */ 0.0,
            /* (15, 5) */ -4.69762141536116384314449447206,
            /* (15, 6) */ 7.68342119606259904184240953878,
            /* (15, 7) */ 4.06898981839711007970213554331,
            /* (15, 8) */ 3.56727187455281109270669543021e-1,
            /* (15, 9) */ 0.0,
            /* (15, 10) */ 0.0,
            /* (15, 11) */ 0.0,
            /* (15, 12) */ -1.39902416515901462129418009734e-3,
            /* (15, 13) */ 2.9475147891527723389556272149,
            /* (15, 14) */ -9.15095847217987001081870187138,
            /* (15, 15) */ 0.0
        }
    }};

    static constexpr std::array<double, stages> b_high = {
        /* 0 */  5.42937341165687622380535766363E-2,
        /* 1 */  0.0,
        /* 2 */  0.0,
        /* 3 */  0.0,
        /* 4 */  0.0,
        /* 5 */  4.45031289275240888144113950566E0,
        /* 6 */  1.89151789931450038304281599044E0,
        /* 7 */  -5.8012039600105847814672114227E0,
        /* 8 */  3.1116436695781989440891606237E-1,
        /* 9 */  -1.52160949662516078556178806805E-1,
        /* 10 */ 2.01365400804030348374776537501E-1,
        /* 11 */ 4.47106157277725905176885569043E-2,
    };

    // ------------------------------------------------------------
    // Error estimation (embedded 5th order)
    // ------------------------------------------------------------

    static constexpr std::array<double, stages_extended> e5 = {
        /* 0 */ 0.1312004499419488073250102996e-1,
        /* 1 */ 0.0,
        /* 2 */ 0.0,
        /* 3 */ 0.0,
        /* 4 */ 0.0,
        /* 5 */ -0.1225156446376204440720569753e+1,
        /* 6 */ -0.4957589496572501915214079952,
        /* 7 */ 0.1664377182454986536961530415e+1,
        /* 8 */ -0.3503288487499736816886487290,
        /* 9 */ 0.3341791187130174790297318841,
        /* 10 */ 0.8192320648511571246570742613e-1,
        /* 11 */ -0.2235530786388629525884427845e-1,
        /* 12 */ 0.0,
        /* 13 */ 0.0,
        /* 14 */ 0.0,
        /* 15 */ 0.0,
    };

    static constexpr std::array<double, stages_extended> e3 = {
        /* 0 */  5.42937341165687622380535766363E-2 - 0.244094488188976377952755905512,
        /* 1 */  0.0,
        /* 2 */  0.0,
        /* 3 */  0.0,
        /* 4 */  0.0,
        /* 5 */  4.45031289275240888144113950566E0,
        /* 6 */  1.89151789931450038304281599044E0,
        /* 7 */  -5.8012039600105847814672114227E0,
        /* 8 */  3.1116436695781989440891606237E-1 - 0.733846688281611857341361741547,
        /* 9 */  -1.52160949662516078556178806805E-1,
        /* 10 */ 2.01365400804030348374776537501E-1,
        /* 11 */ 4.47106157277725905176885569043E-2 - 0.220588235294117647058823529412e-1,
        /* 12 */ 0.0,
        /* 13 */ 0.0,
        /* 14 */ 0.0,
        /* 15 */ 0.0,
    };

    // ------------------------------------------------------------
    // Dense extension (IMPORTANT PART)
    // ------------------------------------------------------------

    static constexpr std::array<std::array<double, stages_extended>, interpolator_power - 3> d = {{
        {
            /* (0, 0) */ -0.84289382761090128651353491142e+1,
            /* (0, 1) */ 0.0,
            /* (0, 2) */ 0.0,
            /* (0, 3) */ 0.0,
            /* (0, 4) */ 0.0,
            /* (0, 5) */ 0.56671495351937776962531783590,
            /* (0, 6) */ -0.30689499459498916912797304727e+1,
            /* (0, 7) */ 0.23846676565120698287728149680e+1,
            /* (0, 8) */ 0.21170345824450282767155149946e+1,
            /* (0, 9) */ -0.87139158377797299206789907490,
            /* (0, 10) */ 0.22404374302607882758541771650e+1,
            /* (0, 11) */ 0.63157877876946881815570249290,
            /* (0, 12) */ -0.88990336451333310820698117400e-1,
            /* (0, 13) */ 0.18148505520854727256656404962e+2,
            /* (0, 14) */ -0.91946323924783554000451984436e+1,
            /* (0, 15) */ -0.44360363875948939664310572000e+1,
        },
        {
            /* (1, 0) */ 0.10427508642579134603413151009e+2,
            /* (1, 1) */ 0.0,
            /* (1, 2) */ 0.0,
            /* (1, 3) */ 0.0,
            /* (1, 4) */ 0.0,
            /* (1, 5) */ 0.24228349177525818288430175319e+3,
            /* (1, 6) */ 0.16520045171727028198505394887e+3,
            /* (1, 7) */ -0.37454675472269020279518312152e+3,
            /* (1, 8) */ -0.22113666853125306036270938578e+2,
            /* (1, 9) */ 0.77334326684722638389603898808e+1,
            /* (1, 10) */ -0.30674084731089398182061213626e+2,
            /* (1, 11) */ -0.93321305264302278729567221706e+1,
            /* (1, 12) */ 0.15697238121770843886131091075e+2,
            /* (1, 13) */ -0.31139403219565177677282850411e+2,
            /* (1, 14) */ -0.93529243588444783865713862664e+1,
            /* (1, 15) */ 0.35816841486394083752465898540e+2,
        },
        {
            /* (2, 0) */  0.19985053242002433820987653617e+2,
            /* (2, 1) */  0.0,
            /* (2, 2) */  0.0,
            /* (2, 3) */  0.0,
            /* (2, 4) */  0.0,
            /* (2, 5) */  -0.38703730874935176555105901742e+3,
            /* (2, 6) */  -0.18917813819516756882830838328e+3,
            /* (2, 7) */  0.52780815920542364900561016686e+3,
            /* (2, 8) */  -0.11573902539959630126141871134e+2,
            /* (2, 9) */  0.68812326946963000169666922661e+1,
            /* (2, 10) */  -0.10006050966910838403183860980e+1,
            /* (2, 11) */  0.77771377980534432092869265740,
            /* (2, 12) */  -0.27782057523535084065932004339e+1,
            /* (2, 13) */  -0.60196695231264120758267380846e+2,
            /* (2, 14) */  0.84320405506677161018159903784e+2,
            /* (2, 15) */  0.11992291136182789328035130030e+2,
        },
        {
            /* (3, 0) */  -0.25693933462703749003312586129e+2,
            /* (3, 1) */  0.0,
            /* (3, 2) */  0.0,
            /* (3, 3) */  0.0,
            /* (3, 4) */  0.0,   
            /* (3, 5) */  -0.15418974869023643374053993627e+3,
            /* (3, 6) */  -0.23152937917604549567536039109e+3,
            /* (3, 7) */  0.35763911791061412378285349910e+3,
            /* (3, 8) */  0.93405324183624310003907691704e+2,
            /* (3, 9) */  -0.37458323136451633156875139351e+2,
            /* (3, 10) */  0.10409964950896230045147246184e+3,
            /* (3, 11) */  0.29840293426660503123344363579e+2,
            /* (3, 12) */  -0.43533456590011143754432175058e+2,
            /* (3, 13) */  0.96324553959188282948394950600e+2,
            /* (3, 14) */  -0.39177261675615439165231486172e+2,
            /* (3, 15) */  -0.14972683625798562581422125276e+3
        }
    }};

    // ------------------------------------------------------------
    // ERROR (clean separation)
    // ------------------------------------------------------------

    template<std::size_t N>
    static error_type compute_error(
        const State& y,
        const State& y_high,
        const std::array<State, N>& k,
        const State&,
        double dt)
    {
        State err5{};
        State err3{};

        for (std::size_t i = 0; i < stages; ++i)
        {
            err5 = err5 + dt * e5[i] * k[i];
            err3 = err3 + dt * e3[i] * k[i];
        }

        return DOP853ErrorNorm<State>({err5, err3}, y, y_high);
    }

    // ------------------------------------------------------------
    // DENSE (Hairer structure, explicit reconstruction)
    // ------------------------------------------------------------

    template<typename Problem, std::size_t N>
    static dense_type make_dense(
        const Problem& prob,
        double t,
        const State& y,
        const State& y_next,
        double dt,
        const std::array<State, N>& k,
        const State&)
    {

        std::array<State, stages_extended> ks{};

        for (std::size_t i = 0; i < stages; ++i) {
            ks[i] = k[i];
        }

        for (std::size_t i = stages; i < stages_extended; ++i) {
            State yi = y;

            for (std::size_t j = 0; j < i; ++j)
                yi = yi + dt * a[i][j] * ks[j];

            ks[i] = prob.f(t + c[i] * dt, yi);
        }

        std::array<State, interpolator_power + 1> F{};

        F[0] = y_next - y;
        F[1] = dt * ks[0] - F[0];
        F[2] = 2.0 * F[0] - dt * (ks[11] + ks[0]);

        for (std::size_t m = 3; m < interpolator_power + 1; ++m)
        {
            F[m] = State{};

            for (std::size_t j = 0; j < stages_extended; ++j)
            {
                F[m] = F[m] + dt * d[m - 3][j] * ks[j];
            }
        }

        return {
            y,
            {F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7]},
            t,
            dt
        };
    }
};

} // namespace ode