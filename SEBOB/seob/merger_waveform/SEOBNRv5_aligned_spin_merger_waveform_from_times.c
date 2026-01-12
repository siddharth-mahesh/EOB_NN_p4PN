#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Calculates the (2,2) mode of the native SEOBNRv5 merger-ringdown model for a given array of times.
 *
 * @params times - Array of times at which to evaluate the waveform.
 * @params amps - Array to store the calculated amplitudes.
 * @params phases - Array to store the calculated phases.
 * @params t_0 - Attachment time.
 * @params h_0 - Amplitude at attachment time.
 * @params hdot_0 - Amplitude derivative at attachment time.
 * @params phi_0 - Phase at attachment time.
 * @params phidot_0 - Angular frequency at attachment time.
 * @params nsteps_MR - length of the times array.
 * @params commondata - Common data structure containing the model parameters.
 */
void SEOBNRv5_aligned_spin_merger_waveform_from_times(REAL *restrict times, REAL *restrict amps, REAL *restrict phases, const REAL t_0,
                                                      const REAL h_0, const REAL hdot_0, const REAL phi_0, const REAL phidot_0,
                                                      const size_t nsteps_MR, commondata_struct *restrict commondata) {
  size_t i;
  REAL waveform[2];
  for (i = 0; i < nsteps_MR; i++) {
    // compute
    SEOBNRv5_aligned_spin_merger_waveform(times[i], t_0, h_0, hdot_0, phi_0, phidot_0, commondata, waveform);
    // store
    amps[i] = waveform[0];
    phases[i] = waveform[1];
  }
} // END FUNCTION SEOBNRv5_aligned_spin_merger_waveform_from_times
