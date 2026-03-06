#include "BHaH_defines.h"

/**
 * Set commondata_struct to default values specified within NRPy.
 */
void commondata_struct_set_to_default(commondata_struct *restrict commondata) {
  // Set commondata_struct variables to default
  commondata->Delta_t = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::Delta_t
  commondata->M_f = 0.0;                   // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::M_f
  commondata->NUMGRIDS = 1;                // nrpy.grid::NUMGRIDS
  commondata->a6 = 0.0;                    // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::a6
  commondata->a_1_NQC = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::a_1_NQC
  commondata->a_2_NQC = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::a_2_NQC
  commondata->a_3_NQC = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::a_3_NQC
  commondata->a_f = 0.0;                   // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::a_f
  commondata->b_1_NQC = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::b_1_NQC
  commondata->b_2_NQC = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::b_2_NQC
  commondata->chi1 = 0.4;                  // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::chi1
  commondata->chi2 = -0.3;                 // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::chi2
  commondata->dSO = 0.0;                   // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::dSO
  commondata->dT = 0.0;                    // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::dT
  commondata->dt = 2.4627455127717882e-05; // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::dt
  commondata->initial_omega = 0.01118;     // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::initial_omega
  commondata->m1 = 0.5;                    // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::m1
  commondata->m2 = 0.5;                    // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::m2
  commondata->mass_ratio = 1;              // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::mass_ratio
  commondata->nr_amp_1 = 0.0;              // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::nr_amp_1
  commondata->nr_amp_2 = 0.0;              // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::nr_amp_2
  commondata->nr_amp_3 = 0.0;              // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::nr_amp_3
  commondata->nr_omega_1 = 0.0;            // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::nr_omega_1
  commondata->nr_omega_2 = 0.0;            // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::nr_omega_2
  commondata->omega_qnm = 0.0;             // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm
  commondata->omega_qnm_l2m1 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l2m1
  commondata->omega_qnm_l2m2 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l2m2
  commondata->omega_qnm_l3m2 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l3m2
  commondata->omega_qnm_l3m3 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l3m3
  commondata->omega_qnm_l4m3 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l4m3
  commondata->omega_qnm_l4m4 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l4m4
  commondata->omega_qnm_l5m5 = 0.0;        // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::omega_qnm_l5m5
  commondata->phi = 0;                     // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::phi
  commondata->pphi = 3.3;                  // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::pphi
  commondata->prstar = 0.0;                // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::prstar
  commondata->r = 20;                      // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::r
  commondata->r_ISCO = 0.0;                // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::r_ISCO
  commondata->r_stop = 0.0;                // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::r_stop
  commondata->t_ISCO = 0.0;                // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::t_ISCO
  commondata->t_attach = 0.0;              // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::t_attach
  commondata->t_stepback = 250.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::t_stepback
  commondata->tau_qnm = 0.0;               // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm
  commondata->tau_qnm_l2m1 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l2m1
  commondata->tau_qnm_l2m2 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l2m2
  commondata->tau_qnm_l3m2 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l3m2
  commondata->tau_qnm_l3m3 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l3m3
  commondata->tau_qnm_l4m3 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l4m3
  commondata->tau_qnm_l4m4 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l4m4
  commondata->tau_qnm_l5m5 = 0.0;          // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::tau_qnm_l5m5
  commondata->total_mass = 50;             // nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients::total_mass
} // END FUNCTION commondata_struct_set_to_default
