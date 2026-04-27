#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <array>
#include <cmath>

#include "test_problems.hpp"
#include "ode.hpp"

using namespace ode;
using namespace ode::test;
using Catch::Matchers::WithinAbs;

// ════════════════════════════════════════════════════════════════════════════
// Vérification des concepts à la compilation
// ════════════════════════════════════════════════════════════════════════════

// double satisfait StateType
static_assert(StateType<double>);

// State2D satisfait StateType
static_assert(StateType<State2D>);

// State4D satisfait StateType
static_assert(StateType<State4D>);

// Vec2 satisfait StateType
static_assert(StateType<Vec2>);

// ════════════════════════════════════════════════════════════════════════════
// std::array<double, N> comme State
// ════════════════════════════════════════════════════════════════════════════

// On enrichit std::array avec les opérateurs nécessaires
template<std::size_t N>
struct ArrayState {
    std::array<double, N> data{};

    ArrayState operator+(const ArrayState& o) const {
        ArrayState r;
        for (std::size_t i = 0; i < N; ++i) r.data[i] = data[i] + o.data[i];
        return r;
    }
    ArrayState operator-(const ArrayState& o) const {
        ArrayState r;
        for (std::size_t i = 0; i < N; ++i) r.data[i] = data[i] - o.data[i];
        return r;
    }
    friend ArrayState operator*(double c, const ArrayState& s) {
        ArrayState r;
        for (std::size_t i = 0; i < N; ++i) r.data[i] = c * s.data[i];
        return r;
    }
    friend ArrayState operator*(const ArrayState& s, double c) { return c * s; }

    double& operator[](std::size_t i) { return data[i]; }
    double  operator[](std::size_t i) const { return data[i]; }
};

namespace ode {
    template<std::size_t N>
    double norm(const ArrayState<N>& s) {
        double sum = 0;
        for (auto x : s.data) sum += x*x;
        return std::sqrt(sum);
    }
}

static_assert(StateType<ArrayState<2>>);
static_assert(StateType<ArrayState<6>>);

TEST_CASE("ArrayState<2> - RK4 harmonic oscillator", "[state][array]")
{
    using S = ArrayState<2>;

    auto f = [](double /*t*/, const S& y) -> S {
        return { y[1], -y[0] };  // y' = [v, -x]
    };

    auto prob = make_problem(f, 0.0, S{{1.0, 0.0}});
    Options opts;
    opts.t_end = 2.0 * M_PI;
    opts.dt    = 1e-3;

    auto sol = solve_ivp(prob, RK4{}, opts);

    CHECK_THAT(sol.y.back()[0], WithinAbs(1.0, 1e-6));
    CHECK_THAT(sol.y.back()[1], WithinAbs(0.0, 1e-6));
}

TEST_CASE("ArrayState<6> - 6D state (astrodynamical style)", "[state][array]")
{
    // Décroissance exponentielle sur 6 composantes indépendantes
    using S = ArrayState<6>;

    auto f = [](double /*t*/, const S& y) -> S {
        S dy;
        for (int i = 0; i < 6; ++i) dy[i] = -y[i];
        return dy;
    };

    S y0; for (int i = 0; i < 6; ++i) y0[i] = 1.0;
    auto prob = make_problem(f, 0.0, y0);

    Options opts;
    opts.t_end = 1.0;
    opts.dt    = 1e-3;

    auto sol = solve_ivp(prob, RK4{}, opts);

    double exact = std::exp(-1.0);
    for (int i = 0; i < 6; ++i) {
        CHECK_THAT(sol.y.back()[i], WithinAbs(exact, 1e-6));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests des opérations arithmétiques sur State2D
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("State2D - arithmetic operations", "[state][arithmetic]")
{
    State2D a{1.0, 2.0};
    State2D b{3.0, 4.0};

    auto sum = a + b;
    CHECK_THAT(sum.x, WithinAbs(4.0, 1e-15));
    CHECK_THAT(sum.v, WithinAbs(6.0, 1e-15));

    auto scaled = 2.0 * a;
    CHECK_THAT(scaled.x, WithinAbs(2.0, 1e-15));
    CHECK_THAT(scaled.v, WithinAbs(4.0, 1e-15));

    auto diff = b - a;
    CHECK_THAT(diff.x, WithinAbs(2.0, 1e-15));
    CHECK_THAT(diff.v, WithinAbs(2.0, 1e-15));
}

// ════════════════════════════════════════════════════════════════════════════
// Tests de make_problem / make_separable
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("make_problem - type deduction", "[problem][deduction]")
{
    auto f    = ExponentialDecay{};
    auto prob = make_problem(f, 0.0, 1.0);

    static_assert(std::same_as<typename decltype(prob)::state_type, double>);
    CHECK_THAT(prob.t0, WithinAbs(0.0, 1e-15));
    CHECK_THAT(prob.y0, WithinAbs(1.0, 1e-15));
}

TEST_CASE("make_separable - type deduction", "[problem][separable]")
{
    auto prob = make_separable(
        [](const double& x) { return -x; },
        1.0, 0.0, 0.0
    );

    static_assert(std::same_as<typename decltype(prob)::state_type, double>);
    CHECK_THAT(prob.t0, WithinAbs(0.0, 1e-15));
    CHECK_THAT(prob.x0, WithinAbs(1.0, 1e-15));
    CHECK_THAT(prob.v0, WithinAbs(0.0, 1e-15));
}
