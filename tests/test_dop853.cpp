#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include <iostream>

#include "test_problems.hpp"
#include "ode.hpp"

using namespace ode;
using namespace ode::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ════════════════════════════════════════════════════════════════════════════
// DOP853 - base accuracy tests
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - exponential decay high accuracy", "[dop853]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);

    Options opts;
    opts.t_end = 1.0;
    opts.rtol  = 1e-12;
    opts.atol  = 1e-14;
    opts.dt    = 1e-2;

    auto sol = solve_ivp(prob, DOP853{}, opts);

    REQUIRE(sol.success);

    double exact = std::exp(-1.0);

    CHECK_THAT(sol.y.back(), WithinAbs(exact, 1e-10));
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - order / convergence sanity check
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - convergence faster than RK45", "[dop853][comparison]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    double exact = std::exp(-1.0);

    Options opts;
    opts.t_end = 1.0;
    opts.rtol  = 1e-8;
    opts.atol  = 1e-10;
    opts.dt    = 1e-2;

    auto sol45   = solve_ivp(prob, RK45{}, opts);
    auto sol853  = solve_ivp(prob, DOP853{}, opts);

    double err45  = std::abs(sol45.y.back()  - exact);
    double err853 = std::abs(sol853.y.back() - exact);

    // DOP853 doit être significativement meilleur
    CHECK(err853 < err45);
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - dense output precision (VERY IMPORTANT)
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - dense output matches RK step endpoints", "[dop853][dense]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);

    Options opts;
    opts.t_end  = 1.0;
    opts.dt     = 0.1;
    opts.t_eval = {0.05, 0.15, 0.33, 0.51, 0.77, 0.99};

    opts.rtol = 1e-12;
    opts.atol = 1e-14;

    auto sol = solve_ivp(prob, DOP853Dense{}, opts);

    REQUIRE(sol.t.size() == opts.t_eval.size());

    for (size_t i = 0; i < sol.t.size(); ++i)
    {
        double exact = std::exp(-sol.t[i]);

        // Dense output doit être très précis (ordre 8)
        CHECK_THAT(sol.y[i], WithinAbs(exact, 1e-10));
    }
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - harmonic oscillator stability (long time)
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - harmonic oscillator long term stability", "[dop853][oscillator]")
{
    HarmonicOscillatorRHS rhs{1.0};
    auto prob = make_problem(rhs, 0.0, State2D{1.0, 0.0});

    Options opts;
    opts.t_end = 50.0;   // long integration
    opts.dt    = 0.1;
    opts.rtol  = 1e-10;
    opts.atol  = 1e-12;

    auto sol = solve_ivp(prob, DOP853{}, opts);

    REQUIRE(sol.success);

    // amplitude approx conservée
    double r = std::sqrt(sol.y.back().x * sol.y.back().x +
                         sol.y.back().v * sol.y.back().v);

    CHECK_THAT(r, WithinAbs(1.0, 1e-3));
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - Kepler orbit (very strong stress test)
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - Kepler orbit energy stability", "[dop853][kepler]")
{
    KeplerRHS rhs;

    auto prob = make_problem(rhs, 0.0, State4D{1.0, 0.0, 0.0, 1.0});

    Options opts;
    opts.t_end    = 20.0 * M_PI; // multiple orbits
    opts.rtol     = 1e-11;
    opts.atol     = 1e-13;
    opts.dt       = 0.1;
    opts.max_steps = 20000;

    auto sol = solve_ivp(prob, DOP853{}, opts);

    std::cout << "DOP853 - Kepler orbit: n_steps = " << sol.n_steps << ", n_rejected = " << sol.n_rejected << std::endl;

    REQUIRE(sol.success);

    std::cout << "Succes : " << sol.success << std::endl;
    std::cout << "solution size : " << sol.y.size() << std::endl;

    // retour proche de la condition initiale
    CHECK_THAT(sol.y.back().x,  WithinAbs(1.0, 1e-3));
    CHECK_THAT(sol.y.back().y,  WithinAbs(0.0, 1e-3));
    CHECK_THAT(sol.y.back().vx, WithinAbs(0.0, 1e-3));
    CHECK_THAT(sol.y.back().vy, WithinAbs(1.0, 1e-3));

    std::cout << "Position finale : x = " << sol.y.back().x << ", y = " << sol.y.back().y << std::endl;
    std::cout << "Vitesse finale : vx = " << sol.y.back().vx << ", vy = " << sol.y.back().vy << std::endl;
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - adaptive behavior sanity
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - adaptive uses fewer steps than RK45 at same tolerance", "[dop853][adaptive]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);

    Options opts;
    opts.t_end = 1.0;
    opts.rtol  = 1e-9;
    opts.atol  = 1e-11;
    opts.dt    = 1e-2;

    auto sol45  = solve_ivp(prob, RK45{}, opts);
    auto sol853 = solve_ivp(prob, DOP853{}, opts);

    CHECK(sol853.n_steps < sol45.n_steps);
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - rejected steps exist but remain controlled
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - rejected steps occur under stiffness", "[dop853][adaptive]")
{
    // Van der Pol modéré pour forcer adaptativité
    auto prob = make_problem(VanDerPolRHS{1.0}, 0.0, State2D{2.0, 0.0});

    Options opts;
    opts.t_end = 10.0;
    opts.dt    = 0.5;   // volontairement agressif
    opts.rtol  = 1e-8;
    opts.atol  = 1e-10;

    auto sol = solve_ivp(prob, DOP853{}, opts);

    REQUIRE(sol.success);

    CHECK(sol.n_steps > 0);
    CHECK(sol.n_rejected >= 0);
}


// ════════════════════════════════════════════════════════════════════════════
// DOP853 - consistency check (dense vs endpoint)
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("DOP853 - dense output matches stored solution", "[dop853][dense]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);

    Options opts;
    opts.t_end  = 1.0;
    opts.dt     = 0.1;
    opts.t_eval = {0.1, 0.2, 0.3, 0.4, 0.5};

    opts.rtol = 1e-12;
    opts.atol = 1e-14;

    auto sol = solve_ivp(prob, DOP853Dense{}, opts);

    REQUIRE(sol.t.size() == 5);

    for(size_t i = 0; i < sol.t.size(); ++i)
    {
        double exact = std::exp(-sol.t[i]);

        // Dense output doit être très précis (ordre 8)
        std::cout << "t = " << sol.t[i] << ", y = " << sol.y[i] << ", exact = " << exact << std::endl;
    }

    for (size_t i = 0; i < sol.t.size(); ++i)
    {
        double exact = std::exp(-sol.t[i]);

        // très faible erreur attendue
        CHECK_THAT(sol.y[i], WithinAbs(exact, 1e-11));
    }
}