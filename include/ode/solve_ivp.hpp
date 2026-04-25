#pragma once

#include "integrate.hpp"
#include "steppers.hpp"
#include "controllers.hpp"
#include "samplers.hpp"

namespace ode {

// ════════════════════════════════════════════════════════════════════════════
// Tags de méthodes
// ════════════════════════════════════════════════════════════════════════════

struct Euler {
    template<typename P> static auto make_stepper(const P&) { return EulerStepper<P>{}; }
    static auto make_controller(const Options& o) { return FixedController{o.dt}; }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

struct RK2 {
    template<typename P> static auto make_stepper(const P&) { return RK2Stepper<P>{}; }
    static auto make_controller(const Options& o) { return FixedController{o.dt}; }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

struct RK4 {
    template<typename P> static auto make_stepper(const P&) { return RK4Stepper<P>{}; }
    static auto make_controller(const Options& o) { return FixedController{o.dt}; }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

// Pour les méthodes adaptatives, on ne peut pas retourner des types différents
// selon t_eval depuis une même fonction (types incompatibles).
// On utilise BasicSampler par défaut ; TEvalSampler nécessite un appel explicite.
// TODO : utiliser std::variant ou une abstraction polymorphe pour unifier.

struct RK23 {
    template<typename P> static auto make_stepper(const P&) { return RK23Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 3};
    }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

struct RK45 {
    template<typename P> static auto make_stepper(const P&) { return RK45Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 5};
    }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

// ─── Variantes avec t_eval (dense output) ────────────────────────────────────

struct RK23Dense {
    template<typename P> static auto make_stepper(const P&) { return RK23Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 3};
    }
    template<typename S> static auto make_sampler(const Options& o) {
        return TEvalSampler<S>{o.t_eval};
    }
};

struct RK45Dense {
    template<typename P> static auto make_stepper(const P&) { return RK45Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 5};
    }
    template<typename S> static auto make_sampler(const Options& o) {
        return TEvalSampler<S>{o.t_eval};
    }
};

// ════════════════════════════════════════════════════════════════════════════
// API principale
// ════════════════════════════════════════════════════════════════════════════

template<typename Problem, typename Method>
auto solve_ivp(const Problem& prob, Method, Options opts = {})
{
    using State = typename Problem::state_type;

    auto stepper    = Method::template make_stepper<Problem>(prob);
    auto controller = Method::make_controller(opts);
    auto sampler    = Method::template make_sampler<State>(opts);

    return integrate(prob, stepper, controller, sampler,
                     opts.t_end, opts.max_steps);
}

// Surcharge pour les problèmes séparables (Verlet)
template<typename Problem>
auto solve_separable(const Problem& prob, Options opts = {})
{
    auto stepper = VelocityVerletStepper<Problem>{};
    BasicSampler<typename Problem::state_type> sampler;

    return integrate_separable(prob, stepper, sampler,
                                opts.t_end, opts.dt, opts.max_steps);
}

} // namespace ode
