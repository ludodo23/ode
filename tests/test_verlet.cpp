#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "test_problems.hpp"
#include "ode.hpp"

using namespace ode;
using namespace ode::test;
using Catch::Matchers::WithinAbs;

// ════════════════════════════════════════════════════════════════════════════
// Tests Velocity Verlet
// ════════════════════════════════════════════════════════════════════════════

TEST_CASE("Verlet - harmonic oscillator : return to initial conditions after one period", "[verlet][oscillator]")
{
    // x'' = -ω²x, x(0)=1, x'(0)=0  →  x(t) = cos(t), T = 2π
    auto prob = make_separable(
        [](const double& x) { return -x; },  // a(x) = -x, ω=1
        1.0, 0.0, 0.0
    );

    Options opts;
    opts.t_end = 2.0 * M_PI;
    opts.dt    = 1e-3;

    auto sol = solve_separable(prob, opts);

    CHECK_THAT(sol.y.back(), WithinAbs(1.0, 1e-5));
}

TEST_CASE("Verlet - energy conservation (harmonic oscillator)", "[verlet][energy]")
{
    // E = 0.5*v² + 0.5*x² = 0.5 (constante pour ω=1, x0=1, v0=0)
    // Verlet est symplectique → E oscille mais ne dérive pas séculairement

    auto accel = [](const double& x) { return -x; };
    auto prob  = make_separable(accel, 1.0, 0.0, 0.0);

    // On intègre via la boucle manuelle pour avoir x ET v
    VelocityVerletStepper<decltype(prob)> stepper;
    double t = 0.0, x = 1.0, v = 0.0;
    double dt = 1e-2;
    double E0 = 0.5*(v*v + x*x);

    double E_max_drift = 0.0;
    for (int i = 0; i < 10000; ++i) {  // 100 périodes
        auto res = stepper.step(prob, t, x, v, dt);
        t += dt;
        x = res.x;
        v = res.v;
        double E = 0.5*(v*v + x*x);
        E_max_drift = std::max(E_max_drift, std::abs(E - E0));
    }

    // Énergie bornée (pas de dérive séculaire pour un intégrateur symplectique)
    CHECK(E_max_drift < 1e-4);
}

TEST_CASE("Verlet - Kepler circular orbit : energy conservation", "[verlet][kepler]")
{
    // r0 = (1,0), v0 = (0,1) → orbite circulaire, E = -0.5
    auto prob = make_separable(
        KeplerAccel{},
        Vec2{1.0, 0.0},
        Vec2{0.0, 1.0},
        0.0
    );

    VelocityVerletStepper<decltype(prob)> stepper;
    Vec2   r{1.0, 0.0}, v{0.0, 1.0};
    double t  = 0.0;
    double dt = 1e-3;
    double E0 = kepler_energy(r, v);  // = -0.5

    double E_max_drift = 0.0;
    int    n_periods   = 10;
    int    n_steps     = static_cast<int>(n_periods * 2.0*M_PI / dt);

    for (int i = 0; i < n_steps; ++i) {
        auto res = stepper.step(prob, t, r, v, dt);
        t += dt;
        r = res.x;
        v = res.v;
        double E = kepler_energy(r, v);
        E_max_drift = std::max(E_max_drift, std::abs(E - E0));
    }

    // Énergie bornée sur 10 orbites
    CHECK(E_max_drift < 1e-3);
}

TEST_CASE("Verlet - Kepler : return to initial position after one orbit", "[verlet][kepler]")
{
    auto prob = make_separable(
        KeplerAccel{},
        Vec2{1.0, 0.0},
        Vec2{0.0, 1.0},
        0.0
    );

    Options opts;
    opts.t_end = 2.0 * M_PI;
    opts.dt    = 1e-4;

    // Intégration manuelle
    VelocityVerletStepper<decltype(prob)> stepper;
    Vec2   r{1.0, 0.0}, v{0.0, 1.0};
    double t = 0.0;

    while (t < opts.t_end - opts.dt/2.0) {
        double step = std::min(opts.dt, opts.t_end - t);
        auto res = stepper.step(prob, t, r, v, step);
        t += step;
        r = res.x;
        v = res.v;
    }

    CHECK_THAT(r.x, WithinAbs(1.0, 1e-4));
    CHECK_THAT(r.y, WithinAbs(0.0, 1e-4));
}

TEST_CASE("Verlet - order 2 : convergence on harmonic oscillator", "[verlet][convergence]")
{
    auto accel = [](const double& x) { return -x; };
    auto prob  = make_separable(accel, 1.0, 0.0, 0.0);

    double t_end = 1.0;   // non-commensurable pour forcer un vrai dernier pas tronqué
    double exact = std::cos(t_end);

    auto integrate_verlet = [&](double dt) {
        VelocityVerletStepper<decltype(prob)> stepper;
        double t = 0.0, x = 1.0, v = 0.0;
        while (t < t_end) {
            double step = std::min(dt, t_end - t);
            if (step < 1e-14) break;
            auto res = stepper.step(prob, t, x, v, step);
            t += step; x = res.x; v = res.v;
        }
        return x;
    };

    double err_h  = std::abs(integrate_verlet(1e-2) - exact);
    double err_h2 = std::abs(integrate_verlet(5e-3) - exact);
    double order  = std::log(err_h / err_h2) / std::log(2.0);

    CHECK(order > 1.8);
    CHECK(order < 2.2);
}
