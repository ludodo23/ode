#pragma once

#include <concepts>

namespace ode {

/**
 * @brief Concept for state types.
 * 
 * A type is a state type if it supports addition and scalar multiplication.
 */
template<typename S>
concept ScalarState =
    std::convertible_to<S, double>;

template<typename S>
concept VectorState =
    requires(S s, std::size_t i)
    {
        { s.size() } -> std::convertible_to<std::size_t>;
        { s[i] } -> std::convertible_to<double>;
    };

template<typename S>
concept StateType =
    (ScalarState<S> || VectorState<S>)
    &&
    requires(S a, S b, double c)
    {
        { a + b } -> std::convertible_to<S>;
        { c * a } -> std::convertible_to<S>;
        { a * c } -> std::convertible_to<S>;
    };

template<ScalarState S>
constexpr std::size_t state_size(const S&)
{
    return 1;
}

template<VectorState S>
constexpr std::size_t state_size(const S& s)
{
    return s.size();
}

template<ScalarState S>
double state_at(const S& s, std::size_t)
{
    return s;
}

template<VectorState S>
double state_at(const S& s, std::size_t i)
{
    return s[i];
}

/**
 * @brief Concept for dense output types.
 * 
 * A type is a dense output type if it can be called with a double argument and returns a state.
 * This is used to represent the local interpolation polynomial for dense output in adaptive methods.
 */
template<typename DenseOutput, typename State>
concept DenseType =
    requires(DenseOutput d, double t) {
        { d(t) } -> std::convertible_to<State>;
    };


template<typename ErrorNorm, typename State>
concept ErrorNormType =
requires(
    const ErrorNorm& norm,
    double atol,
    double rtol)
{
    { norm(atol, rtol) } -> std::convertible_to<double>;
};

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

/**
 * @brief Concept for second-order ODE functions.
 * 
 * A type F is a second-order ODE function for state type S if it can be called with a double, a const S&, and another const S& and returns something convertible to S.
 */
template<typename F, typename S>
concept SecondOrderODEFunction =
    StateType<S> &&
    requires(F f, double t, const S& y, const S& yp) {
        { f(t, y, yp) } -> std::convertible_to<S>;
    };


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

} // namespace ode

