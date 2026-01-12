#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Calculate the SEOBNRv5 NR-informed right-hand sides for the Non Quasi-Circular (NQC) corrections.
 *
 * @params commondata - Common data structure containing the model parameters.
 * @params amps - Array to store the amplitude and its higher derivatives.
 * @params omegas - Array to store the angular frequency and its derivative.
 */
void SEOBNRv5_aligned_spin_NQC_rhs(commondata_struct *restrict commondata, REAL *restrict amps, REAL *restrict omegas) {
  const REAL m1 = commondata->m1;
  const REAL m2 = commondata->m2;
  const REAL chi1 = commondata->chi1;
  const REAL chi2 = commondata->chi2;
  // compute
  const REAL tmp0 = m1 + m2;
  const REAL tmp7 = ((m1) * (m1) * (m1));
  const REAL tmp8 = ((m2) * (m2) * (m2));
  const REAL tmp11 = ((m1) * (m1) * (m1) * (m1));
  const REAL tmp12 = ((m2) * (m2) * (m2) * (m2));
  const REAL tmp16 = m1 / m2;
  const REAL tmp1 = (1.0 / ((tmp0) * (tmp0)));
  const REAL tmp5 = (1.0 / ((tmp0) * (tmp0) * (tmp0) * (tmp0)));
  const REAL tmp9 = pow(tmp0, -6);
  const REAL tmp13 = pow(tmp0, -8);
  const REAL tmp2 = m1 * m2 * tmp1;
  const REAL tmp6 = ((m1) * (m1)) * ((m2) * (m2)) * tmp5;
  const REAL tmp10 = tmp7 * tmp8 * tmp9;
  const REAL tmp17 = ((1.0 / 2.0) * chi1 - 1.0 / 2.0 * chi2) * (tmp16 - 1) / ((1 - 2 * tmp2) * (tmp16 + 1));
  const REAL tmp18 = (1.0 / 2.0) * chi1 + (1.0 / 2.0) * chi2 + tmp17;
  const REAL tmp19 = ((tmp18) * (tmp18));
  const REAL tmp20 = ((tmp18) * (tmp18) * (tmp18));
  const REAL tmp21 = tmp18 * tmp2;
  const REAL tmp22 = tmp18 * tmp6;
  const REAL h_t_attach = tmp2 * fabs((1869399114678491.0 / 20000000000000000.0) * chi1 + (1869399114678491.0 / 20000000000000000.0) * chi2 -
                                      6678807011156761.0 / 500000000000000.0 * tmp10 * tmp18 - 4687585958426211.0 / 100000000000000.0 * tmp10 +
                                      (3598984888018441.0 / 50000000000000.0) * tmp11 * tmp12 * tmp13 +
                                      (1869399114678491.0 / 10000000000000000.0) * tmp17 + (3099447225891283.0 / 5000000000000000.0) * tmp19 * tmp6 -
                                      10413107147836477.0 / 500000000000000000.0 * tmp19 + (860293461381563.0 / 2000000000000000.0) * tmp2 * tmp20 -
                                      217072339408107.0 / 250000000000000.0 * tmp2 - 8493901280736431.0 / 100000000000000000.0 * tmp20 -
                                      217891933479267.0 / 125000000000000.0 * tmp21 + (7194264161892297.0 / 1000000000000000.0) * tmp22 +
                                      (12440404909323101.0 / 1000000000000000.0) * tmp6 + 14670966347991181.0 / 10000000000000000.0);
  const REAL hdot_t_attach = 0;
  const REAL hddot_t_attach = tmp2 * ((13256427724964417.0 / 20000000000000000000.0) * chi1 + (13256427724964417.0 / 20000000000000000000.0) * chi2 +
                                      (13256427724964417.0 / 10000000000000000000.0) * tmp17 + (965383211569707.0 / 2500000000000000000.0) * tmp19 -
                                      5615209579018517.0 / 1000000000000000000.0 * tmp2 + (358851415965951.0 / 100000000000000000.0) * tmp21 -
                                      3353002258827749.0 / 1000000000000000000.0 * tmp6 - 1228489545575993.0 / 500000000000000000.0);
  const REAL w_t_attach =
      -4566965972049467.0 / 100000000000000000.0 * chi1 - 4566965972049467.0 / 100000000000000000.0 * chi2 +
      (7134801214011141.0 / 25000000000000000.0) * ((m1) * (m1)) * ((m2) * (m2)) * tmp5 +
      (392132032264821.0 / 1562500000000000.0) * m1 * m2 * tmp1 * tmp18 + (647517837725651.0 / 1250000000000000.0) * m1 * m2 * tmp1 * tmp19 +
      (24194837236629313.0 / 100000000000000000.0) * m1 * m2 * tmp1 * tmp20 - 6698610724189451.0 / 2000000000000000.0 * tmp10 +
      (5893523296177077.0 / 1000000000000000.0) * tmp11 * tmp12 * tmp13 - 4566965972049467.0 / 50000000000000000.0 * tmp17 -
      381474417039561.0 / 25000000000000000.0 * ((tmp18) * (tmp18) * (tmp18) * (tmp18)) +
      (7502911609839309.0 / 2000000000000000.0) * tmp18 * tmp7 * tmp8 * tmp9 - 9714093262519423.0 / 10000000000000000.0 * tmp19 * tmp6 -
      871517604568457.0 / 10000000000000000.0 * tmp19 - 31709602351033533.0 / 100000000000000000.0 * tmp2 -
      1673164620878479.0 / 25000000000000000.0 * tmp20 - 16973430239436997.0 / 10000000000000000.0 * tmp22 - 1342707196092513.0 / 5000000000000000.0;
  const REAL wdot_t_attach = (2131427096514931.0 / 2500000000000000000.0) * chi1 + (2131427096514931.0 / 2500000000000000000.0) * chi2 -
                             948504518530783.0 / 4000000000000000.0 * tmp10 + (2131427096514931.0 / 1250000000000000000.0) * tmp17 -
                             839285104015017.0 / 100000000000000000.0 * tmp19 * tmp2 + (19481328417233967.0 / 10000000000000000000.0) * tmp19 -
                             31039709730298883.0 / 1000000000000000000.0 * tmp2 + (1534856684343527.0 / 2500000000000000000.0) * tmp20 -
                             14385878246751733.0 / 500000000000000000.0 * tmp21 + (3899508160993091.0 / 50000000000000000.0) * tmp22 +
                             (2305369786365707.0 / 25000000000000000.0) * tmp6 - 1370988203959337.0 / 250000000000000000.0;

  amps[0] = h_t_attach;
  amps[1] = hdot_t_attach;
  amps[2] = hddot_t_attach;
  omegas[0] = fabs(w_t_attach);
  omegas[1] = fabs(wdot_t_attach);
  commondata->nr_amp_1 = amps[0];
  commondata->nr_amp_2 = amps[1];
  commondata->nr_amp_3 = amps[2];
  commondata->nr_omega_1 = omegas[0];
  commondata->nr_omega_2 = omegas[1];
} // END FUNCTION SEOBNRv5_aligned_spin_NQC_rhs
