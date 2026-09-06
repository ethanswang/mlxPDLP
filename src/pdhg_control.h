// Scalar safeguards shared by the device solver and FP64 continuation.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace mlxpdlp::detail {

inline double bounded_weight(double weight, double fallback = 1.0) {
    if (!std::isfinite(weight) || weight <= 0.0)
        weight = std::isfinite(fallback) && fallback > 0.0 ? fallback : 1.0;
    return std::clamp(weight, 1e-12, 1e12);
}

inline double weight_from_log(double log_weight, double fallback) {
    if (!std::isfinite(log_weight))
        return bounded_weight(fallback);
    constexpr double log_limit = 27.631021115928547; // log(1e12)
    return bounded_weight(std::exp(std::clamp(log_weight, -log_limit, log_limit)));
}

inline double pid_weight(double weight, double error, double kp, double ki, double kd,
                         double smoothing, double &integral, double &last_error, double fallback) {
    integral = smoothing * integral + error;
    const double log_weight =
        std::log(bounded_weight(weight)) + kp * error + ki * integral + kd * (error - last_error);
    if (!std::isfinite(log_weight) || !std::isfinite(integral)) {
        integral = last_error = 0.0;
        return bounded_weight(fallback);
    }
    last_error = error;
    return weight_from_log(log_weight, fallback);
}

// HPR's sigma is the primal step eta/w: multiplying sigma by kappa
// therefore divides w by kappa. Blend weights in log coordinates.
inline double hpr_weight(double distance_ratio, double fp_error, double best_gap,
                         double best_weight, double primal_residual, double dual_residual,
                         double gap) {
    best_weight = bounded_weight(best_weight);
    if (!(distance_ratio > 0.0) || !std::isfinite(distance_ratio) || !std::isfinite(fp_error) ||
        fp_error < 0.0 || !std::isfinite(primal_residual) || primal_residual < 0.0 ||
        !std::isfinite(dual_residual) || dual_residual < 0.0 || !std::isfinite(gap))
        return best_weight;
    const double denominator = best_gap > 0.0 ? best_gap : fp_error;
    const double fact = denominator > 0.0 ? std::exp(-0.05 * fp_error / denominator) : 1.0;
    double log_weight =
        fact * std::log(0.998 * distance_ratio) + (1.0 - fact) * std::log(best_weight);
    const double floor =
        std::max(std::min(dual_residual, primal_residual), std::min(std::fabs(gap), fp_error));
    if (floor <= 9e-10 && primal_residual > 0.0) {
        const double ratio = dual_residual / primal_residual;
        const double kappa = std::clamp(floor > 5e-10 ? std::sqrt(ratio) : ratio, 1e-2, 100.0);
        log_weight -= std::log(kappa);
    }
    return weight_from_log(log_weight, best_weight);
}

inline bool hpr_necessary_restart(double current, double at_restart, double previous) {
    return std::isfinite(current) && std::isfinite(at_restart) && current <= 0.6 * at_restart &&
           current > previous;
}

inline bool invalid_fixed_point_metric(double movement, double interaction, double epsilon) {
    if (!std::isfinite(movement) || !std::isfinite(interaction) ||
        !std::isfinite(movement + interaction))
        return true;
    // Only cancellation at the arithmetic's rounding scale may be clamped.
    return movement + interaction < -64.0 * epsilon * std::max(movement, std::fabs(interaction));
}

inline double certificate_merit(double primal, double dual, double gap, double feasible_tolerance,
                                double optimal_tolerance) {
    if (!std::isfinite(primal) || !std::isfinite(dual) || !std::isfinite(gap))
        return std::numeric_limits<double>::infinity();
    const double feasible_scale = std::max(feasible_tolerance, std::numeric_limits<double>::min());
    const double optimal_scale = std::max(optimal_tolerance, std::numeric_limits<double>::min());
    return std::max({primal / feasible_scale, dual / feasible_scale, gap / optimal_scale});
}

} // namespace mlxpdlp::detail
