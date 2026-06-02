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
template <typename State>
struct RK23Tableau {
    using error_type = DefaultErrorNorm<State>;
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

    template<std::size_t N>
    static DefaultErrorNorm<State> compute_error(const State& y,
                                const State& y_high,
                                const std::array<State, N>& k,
                                const State& k_fsal,
                                double dt)
    {
        State y_low = y;

        for (std::size_t i = 0; i < 3; ++i)
            y_low = y_low + dt * b_low[i] * k[i];

        if constexpr (fsal_embedded)
            y_low = y_low + dt * b_low_fsal * k_fsal;

        return DefaultErrorNorm<State>(y_high - y_low, y, y_high);
    }
};



template<typename State>
struct DOP853Tableau
{
    using error_type = DOP853Error<State>;
    using dense_type = DOP853DenseOutput<State>;

    static constexpr std::size_t stages = 12;
    static constexpr std::size_t extra = 3;

    // ------------------------------------------------------------
    // RK core (Hairer coefficients)
    // ------------------------------------------------------------

    static constexpr std::array<double, stages> c = { /* Hairer c1..c12 */ };

    static constexpr std::array<std::array<double, stages>, stages> a = { /* Hairer A */ };

    static constexpr std::array<double, stages> b_high = { /* 8th order solution */ };

    // ------------------------------------------------------------
    // Error estimation (embedded 5th order)
    // ------------------------------------------------------------

    static constexpr std::array<double, stages> e5 = { /* difference coeff */ };

    static constexpr std::array<double, stages> e3 = { /* optional second estimator */ };

    // ------------------------------------------------------------
    // Dense extension (IMPORTANT PART)
    // ------------------------------------------------------------

    static constexpr std::array<double, extra> c_extra = { /* Hairer */ };

    static constexpr std::array<std::array<double, stages + extra>, extra> a_extra = { /* Hairer */ };

    static constexpr std::array<std::array<double, stages + extra>, 8> d = { /* Hairer dense coefficients */ };

    // ------------------------------------------------------------
    // ERROR (clean separation)
    // ------------------------------------------------------------

    template<std::size_t N>
    static error_type compute_error(
        const State&,
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

        return {err5, err3};
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
        constexpr std::size_t total = stages + extra;

        std::array<State, total> ks;

        for (std::size_t i = 0; i < stages; ++i)
            ks[i] = k[i];

        for (std::size_t i = 0; i < extra; ++i)
        {
            State yi = y;

            for (std::size_t j = 0; j < stages + i; ++j)
                yi = yi + dt * a_extra[i][j] * ks[j];

            ks[stages + i] =
                prob.f(t + c_extra[i] * dt, yi);
        }

        std::array<State, 8> r{};

        // r0, r1 structure Hairer
        r[0] = y;
        r[1] = y_next - y;

        for (std::size_t m = 2; m < 8; ++m)
        {
            r[m] = State{};

            for (std::size_t j = 0; j < total; ++j)
                r[m] = r[m] + dt * d[m][j] * ks[j];
        }

        return {
            r[0], r[1], r[2], r[3],
            r[4], r[5], r[6], r[7],
            t, dt
        };
    }
};