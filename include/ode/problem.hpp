#pragma once

#include "state.hpp"
#include <concepts>

namespace ode {

// ─── Concept : fonction f(t, y) -> State ────────────────────────────────────
/**
 * @brief Concept for ODE functions.
 * 
 * A type F is an ODE function for state type S if it can be called with a double and a const S& and returns something convertible to S.
 */
template<typename F, typename S>
concept ODEFunction =
    StateType<S> &&
    requires(F f, double t, const S& y) {
        { f(t, y) } -> std::convertible_to<S>;
    };

// ─── Problème de Cauchy : dy/dt = f(t, y),  y(t0) = y0 ─────────────────────

/**
 * @brief Structure for representing an ODE problem.
 * 
 * Encapsulates the ODE function, initial time, and initial state.
 * 
 * @tparam S The state type.
 * @tparam F The ODE function type.
 */
template<StateType S, ODEFunction<S> F>
struct ODEProblem {
    using state_type    = S;
    using function_type = F;

    F      f;
    double t0;
    S      y0;

    constexpr ODEProblem(F func, double t0_, S y0_)
        : f(func), t0(t0_), y0(std::move(y0_)) {}
};

// Helper de déduction de types
template<typename F, typename S>
auto make_problem(F f, double t0, S y0) {
    return ODEProblem<S, F>(std::move(f), t0, std::move(y0));
}

// ─── Concept : accélération a(x) -> State (problème séparable) ──────────────

/**
 * @brief Concept for acceleration functions in separable problems.
 * 
 * A type A is an acceleration function for state type S if it can be called with a const S& and returns something convertible to S.
 */
template<typename A, typename S>
concept AccelFunction =
    StateType<S> &&
    requires(A a, const S& x) {
        { a(x) } -> std::convertible_to<S>;
    };

// ─── Problème séparable : x'' = a(x),  x(t0)=x0, x'(t0)=v0 ────────────────
// Utilisé par les intégrateurs symplectiques (Verlet, Yoshida…)

/**
 * @brief Structure for representing a separable ODE problem.
 * 
 * Encapsulates the acceleration function, initial position, initial velocity, and initial time.
 * 
 * @tparam S The state type.
 * @tparam A The acceleration function type.
 */
template<StateType S, AccelFunction<S> A>
struct SeparableProblem {
    using state_type = S;
    using accel_type = A;

    A      accel;
    S      x0;
    S      v0;
    double t0;

    SeparableProblem(A a, S x0_, S v0_, double t0_)
        : accel(std::move(a)), x0(std::move(x0_)),
          v0(std::move(v0_)), t0(t0_) {}
};

template<typename A, typename S>
auto make_separable(A a, S x0, S v0, double t0) {
    return SeparableProblem<S, A>(std::move(a), std::move(x0),
                                  std::move(v0), t0);
}

} // namespace ode
