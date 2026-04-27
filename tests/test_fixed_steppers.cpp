#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "test_problems.hpp"
#include "ode.hpp"

using namespace ode;
using namespace ode::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

// Intègre un problème scalaire avec un stepper fixe et retourne l'erreur finale
template<typename Stepper>
double fixed_step_error(double dt, double t_end = 1.0)
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Stepper stepper;
    FixedController ctrl{dt};
    BasicSampler<double> sampler;

    auto sol = integrate(prob, stepper, ctrl, sampler, t_end);
    return std::abs(sol.y.back() - std::exp(-t_end));
}

// Estime l'ordre de convergence par régression sur deux pas
double convergence_order(double err_coarse, double err_fine, double ratio = 2.0)
{
    return std::log(err_coarse / err_fine) / std::log(ratio);
}

// ════════════════════════════════════════════════════════════════════════════
// Tests Euler
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("Euler - exponential decay", "[euler][fixed]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1.0;
    opts.dt    = 1e-3;

    auto sol = solve_ivp(prob, Euler{}, opts);

    REQUIRE(sol.success);
    REQUIRE(sol.t.size() > 0);
    REQUIRE(sol.y.size() == sol.t.size());

    double exact = std::exp(-1.0);
    CHECK_THAT(sol.y.back(), WithinAbs(exact, 1e-3));  // Euler O(h)
}

TEST_CASE("Euler - convergence order  1", "[euler][convergence]")
{
    double err_h    = fixed_step_error<EulerStepper<ODEProblem<double, ExponentialDecay>>>(1e-2);
    double err_h2   = fixed_step_error<EulerStepper<ODEProblem<double, ExponentialDecay>>>(5e-3);
    double order    = convergence_order(err_h, err_h2);

    // Ordre 1 attendu (ratio 2 → erreur divisée par ~2)
    CHECK(order > 0.9);
    CHECK(order < 1.2);
}

TEST_CASE("Euler - initial condition", "[euler][initial]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1e-10;  // quasi-nul
    opts.dt    = 1e-12;

    auto sol = solve_ivp(prob, Euler{}, opts);
    CHECK_THAT(sol.y.front(), WithinAbs(1.0, 1e-15));
}

// ════════════════════════════════════════════════════════════════════════════
// Tests RK2
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("RK2 - exponential decay", "[rk2][fixed]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1.0;
    opts.dt    = 1e-2;

    auto sol = solve_ivp(prob, RK2{}, opts);

    double exact = std::exp(-1.0);
    CHECK_THAT(sol.y.back(), WithinAbs(exact, 1e-4));
}

TEST_CASE("RK2 - convergence order  2", "[rk2][convergence]")
{
    using P = ODEProblem<double, ExponentialDecay>;
    double err_h  = fixed_step_error<RK2Stepper<P>>(1e-2);
    double err_h2 = fixed_step_error<RK2Stepper<P>>(5e-3);
    double order  = convergence_order(err_h, err_h2);

    CHECK(order > 1.8);
    CHECK(order < 2.2);
}

// ════════════════════════════════════════════════════════════════════════════
// Tests RK4
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("RK4 - exponential decay", "[rk4][fixed]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1.0;
    opts.dt    = 1e-2;

    auto sol = solve_ivp(prob, RK4{}, opts);

    double exact = std::exp(-1.0);
    CHECK_THAT(sol.y.back(), WithinAbs(exact, 1e-8));  // RK4 O(h^4)
}

TEST_CASE("RK4 - convergence order  4", "[rk4][convergence]")
{
    using P = ODEProblem<double, ExponentialDecay>;
    double err_h  = fixed_step_error<RK4Stepper<P>>(1e-2);
    double err_h2 = fixed_step_error<RK4Stepper<P>>(5e-3);
    double order  = convergence_order(err_h, err_h2);

    CHECK(order > 3.8);
    CHECK(order < 4.2);
}

TEST_CASE("RK4 - harmonic oscillator", "[rk4][oscillator]")
{
    HarmonicOscillatorRHS rhs{1.0};
    auto prob = make_problem(rhs, 0.0, State2D{1.0, 0.0});

    Options opts;
    opts.t_end = 2.0 * M_PI;  // une période complète
    opts.dt    = 1e-3;

    auto sol = solve_ivp(prob, RK4{}, opts);

    // Après une période : retour à y(0)=1, y'(0)=0
    CHECK_THAT(sol.y.back().x, WithinAbs(1.0, 1e-6));
    CHECK_THAT(sol.y.back().v, WithinAbs(0.0, 1e-6));
}

TEST_CASE("RK4 - energy conservation (harmonic oscillator)", "[rk4][energy]")
{
    HarmonicOscillatorRHS rhs{1.0};
    HarmonicOscillator    osc{1.0};
    auto prob = make_problem(rhs, 0.0, State2D{1.0, 0.0});

    Options opts;
    opts.t_end = 10.0;
    opts.dt    = 1e-3;

    auto sol = solve_ivp(prob, RK4{}, opts);

    double E0 = osc.energy(1.0, 0.0);  // énergie initiale = 0.5
    for (size_t i = 0; i < sol.y.size(); ++i) {
        double E = osc.energy(sol.y[i].x, sol.y[i].v);
        CHECK_THAT(E, WithinAbs(E0, 1e-5));
    }
}

TEST_CASE("RK4 - correct number of steps", "[rk4][steps]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1.0;
    opts.dt    = 0.1;

    auto sol = solve_ivp(prob, RK4{}, opts);

    // t0 + 10 pas de 0.1 = t_end, donc 11 points (t0 inclus)
    REQUIRE(sol.t.size() == 11);
    CHECK_THAT(sol.t.back(), WithinAbs(1.0, 1e-14));
}

TEST_CASE("RK4 - different t0 values", "[rk4][t0]")
{
    auto prob = make_problem(ExponentialDecay{}, 2.0, std::exp(-2.0));
    Options opts;
    opts.t_end = 3.0;
    opts.dt    = 1e-3;

    auto sol = solve_ivp(prob, RK4{}, opts);

    double exact = std::exp(-3.0);
    CHECK_THAT(sol.y.back(), WithinAbs(exact, 1e-8));
}
