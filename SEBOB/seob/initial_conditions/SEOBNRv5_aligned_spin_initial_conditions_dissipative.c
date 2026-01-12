#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Evaluates the dissipative initial conditions for the SEOBNRv5 ODE integration.
 *
 * @params commondata - The Common data structure containing the model parameters.
 * @returns - GSL_SUCCESS (0) upon success.
 */
int SEOBNRv5_aligned_spin_initial_conditions_dissipative(commondata_struct *restrict commondata) {
  REAL x_lo = -3e-2;
  REAL x_hi = 0.0;
  gsl_function F;
  F.function = &SEOBNRv5_aligned_spin_radial_momentum_condition;
  F.params = commondata;
  commondata->prstar = root_finding_1d(x_lo, x_hi, &F);
  return GSL_SUCCESS;
} // END FUNCTION SEOBNRv5_aligned_spin_initial_conditions_dissipative
