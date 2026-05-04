#pragma once

#include "solution.hpp"
#include <vector>
#include <stdexcept>

namespace ode {

// ─── BasicSampler : collecte tous les pas ────────────────────────────────────
// Compatible avec les méthodes fixes et adaptatives (sans dense output requis)

/**
 * @brief Basic sampler for collecting all steps.
 * 
 * This sampler collects all steps during the integration process.
 * 
 * @tparam S The state type.
 */
template<StateType S>
class BasicSampler {
public:
    /**
     * @brief Initialize the sampler.
     * 
     * @param t The initial time.
     * @param y The initial state.
     */
    void init(double t, const S& y) {
        t_.push_back(t);
        y_.push_back(y);
    }

    // Surcharge sans dense (méthodes fixes)
    /**
     * @brief Observe a step during integration.
     * 
     * @param t The time.
     * @param y The state.
     */
    void observe(double t, const S& y) {
        t_.push_back(t);
        y_.push_back(y);
    }

    // Surcharge avec dense (ignoré ici)
    /**
     * @brief Observe a step during integration with dense output.
     * 
     * @param t The time.
     * @param y The state.
     * @param dense The dense output.
     * @tparam Dense The type of the dense output.
     */
    template<typename Dense>
    void observe(double t, const S& y, const Dense&) {
        observe(t, y);
    }

    /**
     * @brief Get the result of the sampling.
     * 
     * @return The solution containing the sampled steps.
     */
    Solution<S> result() {
        return Solution<S>{
            std::move(t_),
            std::move(y_),
            {},       // pas d'interpolant global
            true, 0, 0, "OK"
        };
    }

private:
    /** @brief The times at which the solution is sampled. */
    std::vector<double> t_;
    /** @brief The states at the sampled times. */
    std::vector<S>      y_;
};

// ─── TEvalSampler : évalue aux instants t_eval via dense output ──────────────
// Requiert que le stepper fournisse un dense output.
// Les t_eval doivent être triés par ordre croissant.

/**
 * @brief Sampler for evaluating the solution at specific time points.
 * 
 * This sampler evaluates the solution at the specified time points using dense output.
 * 
 * @tparam S The state type.
 */
template<StateType S>
class TEvalSampler {
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
    void init(double t, const S& y) {
        t_prev_ = t;
        y_prev_ = y;
        // Absorber les t_eval == t0
        while (idx_ < t_eval_.size() && t_eval_[idx_] <= t) {
            t_out_.push_back(t_eval_[idx_]);
            y_out_.push_back(y);
            ++idx_;
        }
    }

    // Sans dense : on stocke quand même (fallback)
    void observe(double t, const S& y) {
        while (idx_ < t_eval_.size() && t_eval_[idx_] <= t) {
            // Interpolation linéaire de secours
            double alpha = (t_eval_[idx_] - t_prev_) / (t - t_prev_);
            y_out_.push_back((1.0 - alpha)*y_prev_ + alpha*y);
            t_out_.push_back(t_eval_[idx_]);
            ++idx_;
        }
        t_prev_ = t;
        y_prev_ = y;
    }

    template<typename Dense>
    void observe(double t, const S& y, const Dense& dense) {
        while (idx_ < t_eval_.size() && t_eval_[idx_] <= t) {
            t_out_.push_back(t_eval_[idx_]);
            y_out_.push_back(dense(t_eval_[idx_]));
            ++idx_;
        }
        t_prev_ = t;
        y_prev_ = y;
    }

    Solution<S> result() {
        return Solution<S>{
            std::move(t_out_),
            std::move(y_out_),
            {},
            true, 0, 0, "OK"
        };
    }

private:
    std::vector<double> t_eval_;
    std::vector<double> t_out_;
    std::vector<S>      y_out_;
    size_t              idx_    = 0;
    double              t_prev_ = 0.0;
    S                   y_prev_ = {};
};

} // namespace ode
