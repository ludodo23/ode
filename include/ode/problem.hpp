#pragma once

#include "ode/concept_api.hpp"

namespace ode {

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

template<StateType S, SecondOrderODEFunction<S> F>
struct SecondOrderODEProblem {
    using state_type    = S;
    using function_type = F;

    F      f;
    double t0;
    S      y0;
    S     yp0;

    constexpr SecondOrderODEProblem(F func, double t0_, S y0_, S yp0_)
        : f(func), t0(t0_), y0(std::move(y0_)), yp0(std::move(yp0_)) {}
};

// Helper de déduction de types
template<typename F, typename S>
auto make_second_order_problem(F f, double t0, S y0, S yp0) {
    return SecondOrderODEProblem<S, F>(std::move(f), t0, std::move(y0), std::move(yp0));
}


// ─── Problème séparable : y'' = a(y),  y(t0)=y0, y'(t0)=v0 ────────────────
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
struct SeparableODEProblem {
    using state_type = S;
    using accel_type = A;

    A      accel;
    S      y0;
    S      v0;
    double t0;

    SeparableODEProblem(A a, S y0_, S v0_, double t0_)
        : accel(std::move(a)), y0(std::move(y0_)),
          v0(std::move(v0_)), t0(t0_) {}
};

template<typename A, typename S>
auto make_separable(A a, S y0, S v0, double t0) {
    return SeparableODEProblem<S, A>(std::move(a), std::move(y0),
                                  std::move(v0), t0);
}

// ─── Traits de détection du type de problème ─────────────────────────────────

template<typename T>
struct is_second_order : std::false_type {};

template<typename S, typename F>
struct is_second_order<SecondOrderODEProblem<S, F>> : std::true_type {};

template<typename T>
inline constexpr bool is_second_order_v = is_second_order<T>::value;


template<typename T>
struct is_separable : std::false_type {};

template<typename S, typename F>
struct is_separable<SeparableODEProblem<S, F>> : std::true_type {};

template<typename T>
inline constexpr bool is_separable_v = is_separable<T>::value;

} // namespace ode
