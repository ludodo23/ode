#pragma once

#include <concepts>
#include <cmath>

namespace ode {

template<typename S>
concept StateType =
    requires(S a, S b, double c) {
        { a + b }  -> std::convertible_to<S>;
        { c * a }  -> std::convertible_to<S>;
        { a * c }  -> std::convertible_to<S>;
    };

// norm() — point de customisation
// Règle : définir une surcharge libre dans le même namespace que votre type.
// Elle sera trouvée par ADL depuis integrate.hpp.
// Surcharges fournies pour les scalaires :
inline double norm(double s) { return std::abs(s); }
inline double norm(float  s) { return std::abs(s); }

} // namespace ode
