#pragma once

#include "solution.hpp"
#include "states.hpp"
#include <vector>
#include <stdexcept>

namespace ode {

// ─── BasicSampler : collecte tous les pas ────────────────────────────────────
// Compatible avec les méthodes fixes et adaptatives (sans dense output requis).
// Si S = AugmentedState<U>, décompose automatiquement en (y, yp) via if constexpr.

/**
 * @brief Basic sampler for collecting all steps.
 *
 * This sampler collects all steps during the integration process.
 * If the state type is AugmentedState<U> (second-order problem), the sampler
 * automatically decomposes the augmented state into position (y) and velocity (yp).
 *
 * @tparam S The state type (or AugmentedState<U> for second-order problems).
 */
template<StateType S>
class BasicSampler {
    using Inner = inner_state_t<S>;

public:
    /**
     * @brief Initialize the sampler.
     *
     * @param t The initial time.
     * @param y The initial state.
     */
    void init(double t, const S& z) { push(t, z); }

    // Surcharge sans dense (méthodes fixes)
    /**
     * @brief Observe a step during integration.
     *
     * @param t The time.
     * @param y The state.
     */
    void observe(double t, const S& z) { push(t, z); }

    // Surcharge avec dense (ignoré ici)
    /**
     * @brief Observe a step during integration with dense output.
     *
     * @param t The time.
     * @param y The state.
     * @param dense The dense output (ignored in this sampler).
     * @tparam Dense The type of the dense output.
     */
    template<typename Dense>
    void observe(double t, const S& z, const Dense&) { push(t, z); }

    /**
     * @brief Get the result of the sampling.
     *
     * @return The solution containing the sampled steps.
     *         If S = AugmentedState<U>, returns Solution<U> with both y and yp filled.
     *         Otherwise returns Solution<S> with yp empty.
     */
    Solution<Inner> result() {
        return Solution<Inner>{
            std::move(t_),
            std::move(y_),
            std::move(yp_),
            {},       // pas d'interpolant global
            true, 0, 0, "OK"
        };
    }

private:
    /**
     * @brief Push a state into the internal storage.
     *        Decomposes AugmentedState into (y, yp) if needed.
     */
    void push(double t, const S& z) {
        t_.push_back(t);
        if constexpr (is_augmented_v<S>) {
            y_.push_back(z.y);
            yp_.push_back(z.yp);
        } else {
            y_.push_back(z);
        }
    }

    /** @brief The times at which the solution is sampled. */
    std::vector<double> t_;
    /** @brief The states at the sampled times. */
    std::vector<Inner>  y_;
    /** @brief The derivative states (only filled for second-order problems). */
    std::vector<Inner>  yp_;
};

// ─── TEvalSampler : évalue aux instants t_eval via dense output ──────────────
// Requiert que le stepper fournisse un dense output.
// Les t_eval doivent être triés par ordre croissant.
// Si S = AugmentedState<U>, décompose automatiquement en (y, yp) via if constexpr.

/**
 * @brief Sampler for evaluating the solution at specific time points.
 *
 * This sampler evaluates the solution at the specified time points using dense output.
 * If the state type is AugmentedState<U> (second-order problem), the sampler
 * automatically decomposes the augmented state into position (y) and velocity (yp).
 *
 * @tparam S The state type (or AugmentedState<U> for second-order problems).
 */
template<StateType S>
class TEvalSampler {
    using Inner = inner_state_t<S>;

public:
    /**
     * @brief Construct a TEvalSampler with the specified evaluation time points.
     *
     * @param t_eval The time points at which to evaluate the solution.
     */
    explicit TEvalSampler(std::vector<double> t_eval)
        : t_eval_(std::move(t_eval))
    {}

    /**
     * @brief Initialize the sampler.
     *
     * @param t The initial time.
     * @param y The initial state.
     */
    void init(double t, const S& z) {
        t_prev_ = t;
        store_prev(z);
        // Absorber les t_eval == t0
        while (idx_ < t_eval_.size() && t_eval_[idx_] <= t) {
            t_out_.push_back(t_eval_[idx_]);
            push_state(z);
            ++idx_;
        }
    }

    // Sans dense : interpolation linéaire de secours
    void observe(double t, const S& z) {
        while (idx_ < t_eval_.size() && t_eval_[idx_] <= t) {
            double alpha = (t_eval_[idx_] - t_prev_) / (t - t_prev_);
            t_out_.push_back(t_eval_[idx_]);
            push_lerp(z, alpha);
            ++idx_;
        }
        t_prev_ = t;
        store_prev(z);
    }

    template<typename Dense>
    void observe(double t, const S& z, const Dense& dense) {
        while (idx_ < t_eval_.size() && t_eval_[idx_] <= t) {
            t_out_.push_back(t_eval_[idx_]);
            push_dense(dense, t_eval_[idx_]);
            ++idx_;
        }
        t_prev_ = t;
        store_prev(z);
    }

    Solution<Inner> result() {
        return Solution<Inner>{
            std::move(t_out_),
            std::move(y_out_),
            std::move(yp_out_),
            {},
            true, 0, 0, "OK"
        };
    }

private:
    /**
     * @brief Store the current state as the previous state for interpolation.
     */
    void store_prev(const S& z) {
        if constexpr (is_augmented_v<S>) {
            y_prev_  = z.y;
            yp_prev_ = z.yp;
        } else {
            y_prev_ = z;
        }
    }

    /**
     * @brief Push a state directly into the output storage.
     */
    void push_state(const S& z) {
        if constexpr (is_augmented_v<S>) {
            y_out_.push_back(z.y);
            yp_out_.push_back(z.yp);
        } else {
            y_out_.push_back(z);
        }
    }

    /**
     * @brief Push a linearly interpolated state into the output storage.
     *
     * @param z  The state at the end of the current step.
     * @param alpha Interpolation factor in [0, 1].
     */
    void push_lerp(const S& z, double alpha) {
        if constexpr (is_augmented_v<S>) {
            y_out_.push_back((1.0 - alpha)*y_prev_  + alpha*z.y);
            yp_out_.push_back((1.0 - alpha)*yp_prev_ + alpha*z.yp);
        } else {
            y_out_.push_back((1.0 - alpha)*y_prev_ + alpha*z);
        }
    }

    /**
     * @brief Push a dense-output-interpolated state into the output storage.
     *        If S = AugmentedState<U>, extracts y and yp from the dense value.
     *
     * @tparam Dense The type of the dense output callable.
     * @param dense The dense output callable.
     * @param t     The time at which to evaluate the dense output.
     */
    template<typename Dense>
    void push_dense(const Dense& dense, double t) {
        if constexpr (is_augmented_v<S>) {
            auto val = dense(t);
            y_out_.push_back(val.y);
            yp_out_.push_back(val.yp);
        } else {
            y_out_.push_back(dense(t));
        }
    }

    std::vector<double> t_eval_;
    std::vector<double> t_out_;
    std::vector<Inner>  y_out_;
    std::vector<Inner>  yp_out_;
    size_t              idx_     = 0;
    double              t_prev_  = 0.0;
    Inner               y_prev_  = {};
    Inner               yp_prev_ = {};  // inutilisé si premier ordre, coût nul
};

} // namespace ode