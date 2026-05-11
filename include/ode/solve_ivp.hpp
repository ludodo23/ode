#pragma once

#include "integrate.hpp"
#include "steppers.hpp"
#include "controllers.hpp"
#include "samplers.hpp"

namespace ode {

// ════════════════════════════════════════════════════════════════════════════
// Tags de méthodes
// ════════════════════════════════════════════════════════════════════════════

/**
 * @brief Tag struct for the Euler method.
 * 
 * This struct serves as a tag for the Euler method, which is a first-order explicit Runge-Kutta method. It provides static member functions to create the appropriate stepper, controller, and sampler for the Euler method. The make_stepper function returns an instance of EulerStepper, which implements the Euler integration scheme. The make_controller function returns a FixedController with a time step specified in the options. The make_sampler function returns a BasicSampler that will be used to store the results of the integration at each time step.
 * 
 * We remind the reader that the Euler method is a simple and straightforward integration scheme, but it is not very accurate and can be unstable for stiff problems or long integration times. It is often used for educational purposes or as a baseline for comparison with more advanced methods.
 * It follows the update rule: y_{n+1} = y_n + dt * f(t_n, y_n), where f is the derivative function defined by the problem.
 */
struct Euler {
    template<typename P> static auto make_stepper(const P&) { return EulerStepper<P>{}; }
    static auto make_controller(const Options& o) { return FixedController{o.dt}; }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

/** @brief Tag struct for the RK2 method.
 * 
 * This struct serves as a tag for the RK2 method, which is a second-order explicit Runge-Kutta method. It provides static member functions to create the appropriate stepper, controller, and sampler for the RK2 method. The make_stepper function returns an instance of RK2Stepper, which implements the RK2 integration scheme. The make_controller function returns a FixedController with a time step specified in the options. The make_sampler function returns a BasicSampler that will be used to store the results of the integration at each time step.
 * 
 * We remind the reader that the RK2 method, also known as the midpoint method, is a simple improvement over the Euler method that provides better accuracy by using an intermediate slope. It is still not suitable for stiff problems, but it can be a good choice for non-stiff problems where a moderate level of accuracy is sufficient.
 * The RK2 method follows the update rule:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + 0.5*dt, y_n + 0.5*dt*k1)
 * - y_{n+1} = y_n + dt * k2
 */
struct RK2 {
    template<typename P> static auto make_stepper(const P&) { return RK2Stepper<P>{}; }
    static auto make_controller(const Options& o) { return FixedController{o.dt}; }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

/** @brief Tag struct for the RK4 method.
 * 
 * This struct serves as a tag for the RK4 method, which is a fourth-order explicit Runge-Kutta method. It provides static member functions to create the appropriate stepper, controller, and sampler for the RK4 method. The make_stepper function returns an instance of RK4Stepper, which implements the RK4 integration scheme. The make_controller function returns a FixedController with a time step specified in the options. The make_sampler function returns a BasicSampler that will be used to store the results of the integration at each time step.
 * 
 * We remind the reader that the RK4 method is a widely used integration scheme that provides good accuracy for non-stiff problems. It is more computationally expensive than lower-order methods but offers a good balance between accuracy and computational effort.
 * The RK4 method follows the update rule:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + 0.5*dt, y_n + 0.5*dt*k1)
 * - k3 = f(t_n + 0.5*dt, y_n + 0.5*dt*k2)
 * - k4 = f(t_n + dt, y_n + dt*k3)
 * - y_{n+1} = y_n + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
 */
struct RK4 {
    template<typename P> static auto make_stepper(const P&) { return RK4Stepper<P>{}; }
    static auto make_controller(const Options& o) { return FixedController{o.dt}; }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

// Pour les méthodes adaptatives, on ne peut pas retourner des types différents
// selon t_eval depuis une même fonction (types incompatibles).
// On utilise BasicSampler par défaut ; TEvalSampler nécessite un appel explicite.
// TODO : utiliser std::variant ou une abstraction polymorphe pour unifier.

/** @brief Tag struct for the RK23 method.
 * 
 * This struct serves as a tag for the RK23 method, which is a 3-stage embedded Runge-Kutta method of order 2(3). It provides static member functions to create the appropriate stepper, controller, and sampler for the RK23 method. The make_stepper function returns an instance of RK23Stepper, which implements the RK23 integration scheme. The make_controller function returns an AdaptiveController with the specified tolerances and time step bounds. The make_sampler function returns a BasicSampler that will be used to store the results of the integration at each time step.
 * 
 * We remind the reader that the RK23 method is a widely used integration scheme that provides good accuracy for non-stiff problems. It is more computationally expensive than lower-order methods but offers a good balance between accuracy and computational effort.
 * The RK23 method follows the update rule:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + 0.5*dt, y_n + 0.5*dt*k1)
 * - k3 = f(t_n + 0.75*dt, y_n + 0.75*dt*k2)
 * - k4 = f(t_n + dt, y_n + dt*(2/9*k1 + 1/3*k2 + 4/9*k3))
 * - y_{n+1} = y_n + dt*(7/24*k1 + 1/4*k2 + 1/3*k3 + 1/8*k4)
 */
struct RK23 {
    template<typename P> static auto make_stepper(const P&) { return RK23Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 3};
    }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

/** @brief Tag struct for the RK45 method.
 * 
 * This struct serves as a tag for the RK45 method, which is a 6-stage embedded Runge-Kutta method of order 4(5). It provides static member functions to create the appropriate stepper, controller, and sampler for the RK45 method. The make_stepper function returns an instance of RK45Stepper, which implements the RK45 integration scheme. The make_controller function returns an AdaptiveController with the specified tolerances and time step bounds. The make_sampler function returns a BasicSampler that will be used to store the results of the integration at each time step.
 * 
 * We remind the reader that the RK45 method is a widely used integration scheme that provides good accuracy for non-stiff problems. It is more computationally expensive than lower-order methods but offers a good balance between accuracy and computational effort.
 * The RK45 method follows the update rule:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + 1/5*dt, y_n + dt*(1/5*k1))
 * - k3 = f(t_n + 3/10*dt, y_n + dt*(3/40*k1 + 9/40*k2))
 * - k4 = f(t_n + 4/5*dt, y_n + dt*(44/45*k1 - 56/15*k2 + 32/9*k3))
 * - k5 = f(t_n + 8/9*dt, y_n + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
 * - k6 = f(t_n + dt, y_n + dt*(9017/3168*k1 - 355/33*k2 + 480/11*k3 - 3577/1320*k4 + 900/1321*k5))
 * - y_{n+1} = y_n + dt*(900/1321*k5)
 */
struct RK45 {
    template<typename P> static auto make_stepper(const P&) { return RK45Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 5};
    }
    template<typename S> static auto make_sampler(const Options&) { return BasicSampler<S>{}; }
};

// ─── Variantes avec t_eval (dense output) ────────────────────────────────────

/** @brief Tag struct for the RK23 method with dense output.
 * 
 * This struct serves as a tag for the RK23 method with dense output, which is a 3-stage embedded Runge-Kutta method of order 2(3). It provides static member functions to create the appropriate stepper, controller, and sampler for the RK23 method with dense output.
 * 
 * We remind the reader that the RK23 method is a widely used integration scheme that provides good accuracy for non-stiff problems. It is more computationally expensive than lower-order methods but offers a good balance between accuracy and computational effort.
 * The RK23 method follows the update rule:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + 0.5*dt, y_n + 0.5*dt*k1)
 * - k3 = f(t_n + 0.75*dt, y_n + 0.75*dt*k2)
 * - k4 = f(t_n + dt, y_n + dt*(2/9*k1 + 1/3*k2 + 4/9*k3))
 * - y_{n+1} = y_n + dt*(7/24*k1 + 1/4*k2 + 1/3*k3 + 1/8*k4)
 * 
 * The dense output allows for interpolation of the solution at any point within the time step, which can be useful for outputting results at specific times or for post-processing. The make_sampler function returns a TEvalSampler that will store the results at the specified evaluation times in t_eval.
 * The equations for the dense output polynomial are derived from the intermediate stages and the time step, and they allow for accurate interpolation of the solution at any point within the interval [t_n, t_n + dt].
 * It implements the following dense output formula:
 * y(t) = y_n + (t - t_n) * (k1 + (t - t_n) * (k2 + (t - t_n) * k3)))
 * where k1, k2, and k3 are the intermediate stages computed during the RK23 step.  
 */
struct RK23Dense {
    template<typename P> static auto make_stepper(const P&) { return RK23Stepper<P>{}; }
    static auto make_controller(const Options& o) {
        return AdaptiveController{o.dt, o.rtol, o.atol, o.dt_min, o.dt_max, 3};
    }
    template<typename S> static auto make_sampler(const Options& o) {
        return TEvalSampler<S>{o.t_eval};
    }
};

/** @brief Tag struct for the RK45 method with dense output.
 * 
 * This struct serves as a tag for the RK45 method with dense output, which is a 5-stage embedded Runge-Kutta method of order 4(5). It provides static member functions to create the appropriate stepper, controller, and sampler for the RK45 method with dense output.
 * 
 * We remind the reader that the RK45 method is a widely used integration scheme that provides good accuracy for non-stiff problems. It is more computationally expensive than lower-order methods but offers a good balance between accuracy and computational effort.
 * The RK45 method follows the update rule:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + 1/5*dt, y_n + dt*(1/5*k1))
 * - k3 = f(t_n + 3/10*dt, y_n + dt*(3/40*k1 + 9/40*k2))
 * - k4 = f(t_n + 4/5*dt, y_n + dt*(44/45*k1 - 56/15*k2 + 32/9*k3))
 * - k5 = f(t_n + 8/9*dt, y_n + dt*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4))
 * - k6 = f(t_n + dt, y_n + dt*(9017/3168*k1 - 355/33*k2 + 480/11*k3 - 3577/1320*k4 + 900/1321*k5))
 * - y_{n+1} = y_n + dt*(900/1321*k5)
 * 
 * The dense output allows for interpolation of the solution at any point within the time step, which can be useful for outputting results at specific times or for post-processing. The make_sampler function returns a TEvalSampler that will store the results at the specified evaluation times in t_eval.
 * The equations for the dense output polynomial are derived from the intermediate stages and the time step, and they allow for accurate interpolation of the solution at any point within the interval [t_n, t_n + dt].
 * The dense output for RK45 is more complex than for RK23 due to the higher order and the additional stages, but it provides a powerful tool for obtaining accurate results at specific times without the need for additional function evaluations. It is particularly useful for problems where the solution needs to be evaluated at specific time points that do not coincide with the integration steps, or for post-processing and visualization of the solution. The dense output polynomial is constructed using the intermediate stages and the time step,
 * and it allows for accurate interpolation of the solution at any point within the interval [t_n, t_n + dt]. The specific form of the dense output polynomial for RK45 is derived from the Butcher tableau and the coefficients of the method, and it can be evaluated efficiently using the intermediate stages computed during the RK45 step.
 * It implements the following dense output formula:
 * y(t) = y_n + (t - t_n) * (k1 + (t - t_n) * (k2 + (t - t_n) * (k3 + (t - t_n) * (k4 + (t - t_n) * k5)))))
 * where k1, k2, k3, k4, and k5 are the intermediate stages computed during the RK45 step.
 */
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
