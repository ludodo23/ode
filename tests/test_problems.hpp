#pragma once

// Problèmes de référence avec solutions analytiques
// Utilisés dans tous les tests pour valider les intégrateurs

#include <cmath>

namespace ode::test {

// ─── Décroissance exponentielle : y' = -y,  y(0) = 1 ────────────────────────
// Solution : y(t) = exp(-t)
// Non-stiff, scalaire, référence universelle

struct ExponentialDecay {
    double lambda = 1.0;

    double operator()(double /*t*/, double y) const {
        return -lambda * y;
    }

    double exact(double t) const {
        return std::exp(-lambda * t);
    }
};

// ─── Oscillateur harmonique : y'' + ω²y = 0 ─────────────────────────────────
// Reformulé en système : [y, y']' = [y', -ω²y]
// Solution : y(t) = cos(ωt), y'(t) = -ω sin(ωt)
// Bon pour tester la conservation de l'énergie

struct HarmonicOscillator {
    double omega = 1.0;

    // Pour solve_ivp : state = double (y seulement, reformulé en 2D via State2D)
    // Utiliser avec State2D ci-dessous

    double exact_y(double t) const { return std::cos(omega * t); }
    double exact_dy(double t) const { return -omega * std::sin(omega * t); }
    double energy(double y, double dy) const {
        return 0.5*(dy*dy + omega*omega*y*y);
    }
};

// ─── État 2D minimal (sans Eigen) ────────────────────────────────────────────
// Supporte +, * double — compatible StateType concept

struct State2D {
    double x = 0.0, v = 0.0;

    State2D operator+(const State2D& o) const { return {x+o.x, v+o.v}; }
    State2D operator-(const State2D& o) const { return {x-o.x, v-o.v}; }

    friend State2D operator*(double c, const State2D& s) { return {c*s.x, c*s.v}; }
    friend State2D operator*(const State2D& s, double c) { return {c*s.x, c*s.v}; }
};

// f(t, [y, y']) = [y', -ω²y]
struct HarmonicOscillatorRHS {
    double omega = 1.0;

    State2D operator()(double /*t*/, const State2D& s) const {
        return { s.v, -omega*omega * s.x };
    }
};

// ─── Équation de Van der Pol (μ petit → non-stiff) ───────────────────────────
// y'' - μ(1-y²)y' + y = 0
// Pour μ=0 : oscillateur harmonique. Pour μ petit : légèrement non-linéaire.

struct VanDerPolRHS {
    double mu = 0.1;

    State2D operator()(double /*t*/, const State2D& s) const {
        return { s.v, mu*(1.0 - s.x*s.x)*s.v - s.x };
    }
};

// ─── Problème de Kepler (2D) ─────────────────────────────────────────────────
// r'' = -r / |r|³  (en unités GM=1)
// Utilisé pour tester les méthodes symplectiques
// Orbite circulaire : r0 = (1, 0), v0 = (0, 1) → T = 2π

struct State4D {
    double x = 0, y = 0, vx = 0, vy = 0;

    State4D operator+(const State4D& o) const { return {x+o.x, y+o.y, vx+o.vx, vy+o.vy}; }
    State4D operator-(const State4D& o) const { return {x-o.x, y-o.y, vx-o.vx, vy-o.vy}; }

    friend State4D operator*(double c, const State4D& s) {
        return {c*s.x, c*s.y, c*s.vx, c*s.vy};
    }
    friend State4D operator*(const State4D& s, double c) { return c * s; }
};

struct KeplerRHS {
    State4D operator()(double /*t*/, const State4D& s) const {
        double r3 = std::pow(s.x*s.x + s.y*s.y, 1.5);
        return { s.vx, s.vy, -s.x/r3, -s.y/r3 };
    }
};

// Position 2D pour le problème séparable de Kepler
struct Vec2 {
    double x = 0, y = 0;

    Vec2 operator+(const Vec2& o) const { return {x+o.x, y+o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x-o.x, y-o.y}; }

    friend Vec2 operator*(double c, const Vec2& v) { return {c*v.x, c*v.y}; }
    friend Vec2 operator*(const Vec2& v, double c) { return {c*v.x, c*v.y}; }
};

struct KeplerAccel {
    Vec2 operator()(const Vec2& r) const {
        double r3 = std::pow(r.x*r.x + r.y*r.y, 1.5);
        return { -r.x/r3, -r.y/r3 };
    }
};

// Énergie de Kepler : E = 0.5*|v|² - 1/|r|
inline double kepler_energy(const Vec2& r, const Vec2& v) {
    double r_norm = std::sqrt(r.x*r.x + r.y*r.y);
    double v2     = v.x*v.x + v.y*v.y;
    return 0.5*v2 - 1.0/r_norm;
}

} // namespace ode::test


namespace ode {
    inline double norm(const test::State2D& s) {
        return std::sqrt(s.x*s.x + s.v*s.v);
    }
    inline double norm(const test::State4D& s) {
        return std::sqrt(s.x*s.x + s.y*s.y + s.vx*s.vx + s.vy*s.vy);
    }
    inline double norm(const test::Vec2& v) {
        return std::sqrt(v.x*v.x + v.y*v.y);
    }
}
