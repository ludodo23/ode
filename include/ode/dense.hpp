#pragma once

#include <concepts>

namespace ode {

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

}