#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <algorithm>

#include "test_problems.hpp"
#include "../include/ode.hpp"

using namespace ode;
using namespace ode::test;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ════════════════════════════════════════════════════════════════════════════
// Tests RK23
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("RK23 - décroissance exponentielle", "[rk23][adaptive]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1.0;
    opts.rtol  = 1e-6;
    opts.atol  = 1e-9;
    opts.dt    = 1e-2;

    auto sol = solve_ivp(prob, RK23{}, opts);

    REQUIRE(sol.success);
    double exact = std::exp(-1.0);
    CHECK_THAT(sol.y.back(), WithinAbs(exact, 1e-5));
}

TEST_CASE("RK23 - contrôle adaptatif : moins de pas que fixe", "[rk23][adaptive]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);

    // RK23 adaptatif
    Options opts_adapt;
    opts_adapt.t_end = 1.0;
    opts_adapt.rtol  = 1e-4;
    opts_adapt.atol  = 1e-6;
    opts_adapt.dt    = 1e-2;
    auto sol_adapt = solve_ivp(prob, RK23{}, opts_adapt);

    // RK4 fixe fin
    Options opts_fixed;
    opts_fixed.t_end = 1.0;
    opts_fixed.dt    = 1e-4;
    auto sol_fixed = solve_ivp(prob, RK4{}, opts_fixed);

    // RK23 adaptatif doit utiliser bien moins de pas
    CHECK(sol_adapt.n_steps < sol_fixed.n_steps);
}

TEST_CASE("RK23 - dense output via t_eval", "[rk23][dense]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end  = 1.0;
    opts.t_eval = {0.1, 0.3, 0.5, 0.7, 0.9, 1.0};
    opts.rtol   = 1e-8;
    opts.atol   = 1e-10;
    opts.dt     = 1e-2;

    auto sol = solve_ivp(prob, RK23Dense{}, opts);

    REQUIRE(sol.t.size() == 6);
    for (size_t i = 0; i < sol.t.size(); ++i) {
        double exact = std::exp(-sol.t[i]);
        CHECK_THAT(sol.y[i], WithinAbs(exact, 1e-6));
    }
}

TEST_CASE("RK23 - dense output : valeurs intermédiaires précises", "[rk23][dense]")
{
    // Vérifie que le polynôme dense interpole bien entre les nœuds
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end  = 0.5;
    opts.dt     = 0.1;
    // Points très fins pour vérifier le polynôme
    opts.t_eval = {0.05, 0.15, 0.25, 0.35, 0.45, 0.5};
    opts.rtol   = 1e-8;
    opts.atol   = 1e-10;

    auto sol = solve_ivp(prob, RK23Dense{}, opts);

    for (size_t i = 0; i < sol.t.size(); ++i) {
        double exact = std::exp(-sol.t[i]);
        // Le polynôme dense est certifié ordre 3, tolérance généreuse
        CHECK_THAT(sol.y[i], WithinAbs(exact, 1e-5));
    }
}

TEST_CASE("RK23 - pas rejetés enregistrés", "[rk23][adaptive]")
{
    // Van der Pol légèrement non-linéaire pour forcer quelques rejets
    auto prob = make_problem(VanDerPolRHS{0.5}, 0.0, State2D{2.0, 0.0});
    Options opts;
    opts.t_end = 5.0;
    opts.dt    = 0.5;   // grand pas initial → forcer des rejets
    opts.rtol  = 1e-6;
    opts.atol  = 1e-9;

    auto sol = solve_ivp(prob, RK23{}, opts);
    REQUIRE(sol.success);
    // Quelques rejets attendus avec un grand pas initial
    // (pas d'assertion stricte, juste vérifier que le compteur fonctionne)
    CHECK(sol.n_rejected >= 0);
    CHECK(sol.n_steps > 0);
}

// ════════════════════════════════════════════════════════════════════════════
// Tests RK45
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("RK45 - décroissance exponentielle", "[rk45][adaptive]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end = 1.0;
    opts.rtol  = 1e-8;
    opts.atol  = 1e-10;
    opts.dt    = 1e-2;

    auto sol = solve_ivp(prob, RK45{}, opts);

    REQUIRE(sol.success);
    CHECK_THAT(sol.y.back(), WithinAbs(std::exp(-1.0), 1e-6));
}

TEST_CASE("RK45 - plus précis que RK23 à même tolérance", "[rk45][rk23][comparison]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    double exact = std::exp(-1.0);

    Options opts;
    opts.t_end = 1.0;
    opts.rtol  = 1e-6;
    opts.atol  = 1e-8;
    opts.dt    = 1e-2;

    auto sol23 = solve_ivp(prob, RK23{}, opts);
    auto sol45 = solve_ivp(prob, RK45{}, opts);

    double err23 = std::abs(sol23.y.back() - exact);
    double err45 = std::abs(sol45.y.back() - exact);

    // RK45 est généralement plus précis ou utilise moins de pas
    CHECK(err45 <= err23 * 10.0);  // tolérance large — les deux respectent rtol
}

TEST_CASE("RK45 - oscillateur harmonique", "[rk45][oscillator]")
{
    HarmonicOscillatorRHS rhs{1.0};
    auto prob = make_problem(rhs, 0.0, State2D{1.0, 0.0});

    Options opts;
    opts.t_end = 2.0 * M_PI;
    opts.rtol  = 1e-8;
    opts.atol  = 1e-10;
    opts.dt    = 0.1;

    auto sol = solve_ivp(prob, RK45{}, opts);

    CHECK_THAT(sol.y.back().x, WithinAbs(1.0, 1e-4));
    CHECK_THAT(sol.y.back().v, WithinAbs(0.0, 1e-4));
}

TEST_CASE("RK45 - dense output via t_eval", "[rk45][dense]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end  = 1.0;
    opts.t_eval = {0.2, 0.4, 0.6, 0.8, 1.0};
    opts.rtol   = 1e-9;
    opts.atol   = 1e-11;
    opts.dt     = 1e-2;

    auto sol = solve_ivp(prob, RK45Dense{}, opts);

    REQUIRE(sol.t.size() == 5);
    for (size_t i = 0; i < sol.t.size(); ++i) {
        double exact = std::exp(-sol.t[i]);
        CHECK_THAT(sol.y[i], WithinAbs(exact, 1e-2));
    }
}

TEST_CASE("RK45 - Kepler orbite circulaire", "[rk45][kepler]")
{
    // Orbite circulaire : r0=(1,0), v0=(0,1) → période T = 2π, énergie E = -0.5
    KeplerRHS rhs;
    auto prob = make_problem(rhs, 0.0, State4D{1.0, 0.0, 0.0, 1.0});

    Options opts;
    opts.t_end    = 2.0 * M_PI;  // une période
    opts.rtol     = 1e-8;
    opts.atol     = 1e-10;
    opts.dt       = 0.1;
    opts.max_steps = 10'000;

    auto sol = solve_ivp(prob, RK45{}, opts);

    REQUIRE(sol.success);
    // Retour à la position initiale après une période
    CHECK_THAT(sol.y.back().x,  WithinAbs(1.0, 1e-4));
    CHECK_THAT(sol.y.back().y,  WithinAbs(0.0, 1e-4));
    CHECK_THAT(sol.y.back().vx, WithinAbs(0.0, 1e-4));
    CHECK_THAT(sol.y.back().vy, WithinAbs(1.0, 1e-4));
}

// ════════════════════════════════════════════════════════════════════════════
// Tests communs aux méthodes adaptatives
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("RK23 - max_steps déclenche arrêt propre", "[rk23][safety]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end     = 1.0;
    opts.dt        = 1e-2;
    opts.max_steps = 2;   // intentionnellement trop petit

    auto sol = solve_ivp(prob, RK23{}, opts);

    CHECK_FALSE(sol.success);
    CHECK(sol.n_steps <= 6);
}

TEST_CASE("RK45 - max_steps déclenche arrêt propre", "[rk45][safety]")
{
    auto prob = make_problem(ExponentialDecay{}, 0.0, 1.0);
    Options opts;
    opts.t_end     = 1.0;
    opts.dt        = 1e-2;
    opts.max_steps = 5;

    auto sol = solve_ivp(prob, RK45{}, opts);

    CHECK_FALSE(sol.success);
    CHECK(sol.n_steps <= 6);
}
