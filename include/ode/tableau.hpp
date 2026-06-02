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