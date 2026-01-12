#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Evaluates the absolute value of the derivative of a spline at a given point.
 *
 * @param t - The point at which to evaluate the derivative.
 * @param params - The spline data.
 * @returns - The absolute value of the derivative of the spline at the given point.
 */
double eval_abs_deriv(double t, void *params) {
  spline_data *sdata = (spline_data *)params;
  return fabs(gsl_spline_eval_deriv(sdata->spline, t, sdata->acc));
} // END FUNCTION eval_abs_deriv
