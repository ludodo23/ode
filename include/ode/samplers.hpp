#pragma once

#include "solution.hpp"
#include <vector>
#include <stdexcept>

namespace ode {

// ─── BasicSampler : collecte tous les pas ────────────────────────────────────
// Compatible avec les méthodes fixes et adaptatives (sans dense output requis)

template<StateType S>
class BasicSampler {
public:
    void init(double t, const S& y) {
        t_.push_back(t);
        y_.push_back(y);
    }

    // Surcharge sans dense (méthodes fixes)
    void observe(double t, const S& y) {
        t_.push_back(t);
        y_.push_back(y);
    }

    // Surcharge avec dense (ignoré ici)
    template<typename Dense>
    void observe(double t, const S& y, const Dense&) {
        observe(t, y);
    }

    Solution<S> result() {
        return Solution<S>{
            std::move(t_),
            std::move(y_),
            {},       // pas d'interpolant global
            true, 0, 0, "OK"
        };
    }

private:
    std::vector<double> t_;
    std::vector<S>      y_;
};

// ─── TEvalSampler : évalue aux instants t_eval via dense output ──────────────
// Requiert que le stepper fournisse un dense output.
// Les t_eval doivent être triés par ordre croissant.

template<StateType S>
class TEvalSampler {
public:
    explicit TEvalSampler(std::vector<double> t_eval)
        : t_eval_(std::move(t_eval))
    {}

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
