#pragma once
#include "concept_api.hpp"

namespace ode {

// ─── État augmenté [y, yp] pour la réduction d'ordre ────────────────────────
//
// Satisfait StateType : addition composante par composante, multiplication scalaire.

template<StateType S>
struct AugmentedState {
    S y;
    S yp;

    AugmentedState(S y_, S yp_) : y(std::move(y_)), yp(std::move(yp_)) {}

    AugmentedState operator+(const AugmentedState& o) const {
        return {y + o.y, yp + o.yp};
    }

    AugmentedState operator*(double c) const {
        return {y * c, yp * c};
    }

    friend AugmentedState operator*(double c, const AugmentedState& a) {
        return {c * a.y, c * a.yp};
    }

    // Pour satisfaire VectorState : size() et operator[]
    std::size_t size() const {
        return state_size(y) + state_size(yp);
    }

    double operator[](std::size_t i) const {
        std::size_t n = state_size(y);
        return (i < n) ? state_at(y, i) : state_at(yp, i - n);
    }
};

// Trait de détection
template<typename T>
struct is_augmented : std::false_type {};

template<StateType S>
struct is_augmented<AugmentedState<S>> : std::true_type {};

template<typename T>
inline constexpr bool is_augmented_v = is_augmented<T>::value;

// Helper pour extraire S depuis AugmentedState<S> ou S
template<typename T>
struct inner_state { using type = T; };

template<StateType S>
struct inner_state<AugmentedState<S>> { using type = S; };

template<typename T>
using inner_state_t = typename inner_state<T>::type;

} // namespace ode