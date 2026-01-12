#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_vector.h>

/**
 * Evaluates the conservative initial conditions for the SEOBNRv5 ODE integration.
 * The conservative initial conditions are given by a circular orbit (p_{r_*} = 0) such that the
 * the time derivative of the tortoise momentum is zero and the time derivative of the orbital phase equals the input orbital frequency.
 * @params commondata - The Common data structure containing the model parameters.
 * @returns - GSL_SUCCESS (0) upon success.
 */
void SEOBNRv5_aligned_spin_initial_conditions_conservative(commondata_struct *restrict commondata) {
  const size_t n = 2;
  gsl_multiroot_function f = {&SEOBNRv5_aligned_spin_Hamiltonian_circular_orbit, n, commondata};
  REAL omega = commondata->initial_omega;
  REAL pphi = pow(omega, -1. / 3.);
  REAL r = pphi * pphi;
  const REAL x_guess[2] = {r, pphi};
  REAL x_result[2] = {0., 0.};
  root_finding_multidimensional(2, x_guess, &f, x_result);
  commondata->r = x_result[0];
  commondata->pphi = x_result[1];
} // END FUNCTION SEOBNRv5_aligned_spin_initial_conditions_conservative
