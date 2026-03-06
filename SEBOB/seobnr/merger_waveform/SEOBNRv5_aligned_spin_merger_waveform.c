#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Calculates the (2,2) mode of the native SEOBNRv5 merger-ringdown model at a single timestep.
 *
 * @params t - Time at which to evaluate the waveform.
 * @params t_0 - Attachment time.
 * @params h_0 - Amplitude at attachment time.
 * @params hdot_0 - Amplitude derivative at attachment time.
 * @params phi_0 - Phase at attachment time.
 * @params phidot_0 - Angular frequency at attachment time.
 * @params commondata - Common data structure containing the model parameters.
 * @params waveform - Array to store the amplitude and phase of the waveform.
 */
void SEOBNRv5_aligned_spin_merger_waveform(const REAL t, const REAL t_0, const REAL h_0, const REAL hdot_0, const REAL phi_0, const REAL phidot_0,
                                           commondata_struct *restrict commondata, REAL *restrict waveform) {
  const REAL m1 = commondata->m1;
  const REAL m2 = commondata->m2;
  const REAL chi1 = commondata->chi1;
  const REAL chi2 = commondata->chi2;
  const REAL omega_qnm = commondata->omega_qnm;
  const REAL tau_qnm = commondata->tau_qnm;
  // compute
  const REAL tmp0 = (1.0 / (tau_qnm));
  const REAL tmp1 = t - t_0;
  const REAL tmp2 = m1 + m2;
  const REAL tmp5 = ((m1) * (m1));
  const REAL tmp6 = ((m2) * (m2));
  const REAL tmp9 = ((m1) * (m1) * (m1));
  const REAL tmp10 = ((m2) * (m2) * (m2));
  const REAL tmp16 = m1 / m2;
  const REAL tmp3 = (1.0 / ((tmp2) * (tmp2)));
  const REAL tmp7 = (1.0 / ((tmp2) * (tmp2) * (tmp2) * (tmp2)));
  const REAL tmp11 = pow(tmp2, -6);
  const REAL tmp13 = ((m1) * (m1) * (m1) * (m1)) * ((m2) * (m2) * (m2) * (m2)) / pow(tmp2, 8);
  const REAL tmp4 = m1 * m2 * tmp3;
  const REAL tmp8 = tmp5 * tmp6 * tmp7;
  const REAL tmp12 = tmp10 * tmp11 * tmp9;
  const REAL tmp17 = ((1.0 / 2.0) * chi1 - 1.0 / 2.0 * chi2) * (tmp16 - 1) / ((1 - 2 * tmp4) * (tmp16 + 1));
  const REAL tmp18 = (1.0 / 2.0) * chi1 + (1.0 / 2.0) * chi2 + tmp17;
  const REAL tmp19 = ((tmp18) * (tmp18));
  const REAL tmp20 = ((tmp18) * (tmp18) * (tmp18));
  const REAL tmp24 = tmp18 * tmp8;
  const REAL tmp29 = ((tmp18) * (tmp18) * (tmp18) * (tmp18));
  const REAL tmp22 = tmp19 * tmp4;
  const REAL tmp30 =
      -5805750633866389.0 / 1000000000000000000.0 * chi1 - 5805750633866389.0 / 1000000000000000000.0 * chi2 +
      (338852775814739.0 / 10000000000000000.0) * m1 * m2 * tmp18 * tmp3 + (24693956198650533.0 / 500000000000000000.0) * m1 * m2 * tmp19 * tmp3 +
      (6284152952186423.0 / 100000000000000000.0) * m1 * m2 * tmp20 * tmp3 + (5358902726316703.0 / 100000000000000000.0) * m1 * m2 * tmp3 +
      (233497207038307.0 / 125000000000000.0) * tmp10 * tmp11 * tmp18 * tmp9 + (204371156181773.0 / 100000000000000.0) * tmp10 * tmp11 * tmp9 -
      4238246400992723.0 / 1000000000000000.0 * tmp13 - 5805750633866389.0 / 500000000000000000.0 * tmp17 +
      (1316051994812337.0 / 100000000000000000.0) * tmp19 * tmp5 * tmp6 * tmp7 - 1931426101231131.0 / 100000000000000000.0 * tmp19 -
      18908151134871907.0 / 1000000000000000000.0 * tmp20 - 280995251995187.0 / 400000000000000.0 * tmp24 -
      17767400411303719.0 / 10000000000000000000.0 * tmp29 - 1017480616331343.0 / 2500000000000000.0 * tmp8 +
      1725087176121217.0 / 20000000000000000.0;
  const REAL tmp27 = (1353801859378071.0 / 10000000000000000.0) * chi1 + (1353801859378071.0 / 10000000000000000.0) * chi2 -
                     2503684066293349.0 / 1250000000000000.0 * tmp12 * tmp18 - 2216569449718197.0 / 50000000000000.0 * tmp12 +
                     (6328645899089733.0 / 100000000000000.0) * tmp13 + (1353801859378071.0 / 5000000000000000.0) * tmp17 -
                     1148577802431279.0 / 625000000000000.0 * tmp18 * tmp4 + (1778086469611943.0 / 500000000000000.0) * tmp19 * tmp8 +
                     (26429701927331617.0 / 100000000000000000.0) * tmp19 - 2043750369085779.0 / 2000000000000000.0 * tmp20 * tmp4 +
                     (2034803251432791.0 / 10000000000000000.0) * tmp20 - 394016376243669.0 / 200000000000000.0 * tmp22 +
                     (5585850576543833.0 / 1000000000000000.0) * tmp24 - 577847612626177.0 / 500000000000000.0 * tmp4 +
                     (2382393318207833.0 / 250000000000000.0) * tmp8 + 330476747176907.0 / 625000000000000.0;
  const REAL tmp31 = (h_0 * tmp0 + hdot_0) / tmp30;
  const REAL tmp32 = -908341098771731.0 / 10000000000000000.0 * m1 * m2 * tmp18 * tmp3 -
                     9461048976443967.0 / 200000000000000000.0 * m1 * m2 * tmp20 * tmp3 - 3900462972696403.0 / 5000000000000000.0 * m1 * m2 * tmp3 -
                     4113462898396891.0 / 1000000000000000.0 * tmp10 * tmp11 * tmp18 * tmp9 -
                     4143974724044049.0 / 200000000000000.0 * tmp10 * tmp11 * tmp9 + (28423701011399217.0 / 1000000000000000.0) * tmp13 -
                     1291690187883191.0 / 1250000000000000.0 * tmp19 * tmp5 * tmp6 * tmp7 + (1961705613365691.0 / 250000000000000000.0) * tmp19 +
                     (24203355614948303.0 / 1000000000000000000.0) * tmp20 + (5087017219867469.0 / 20000000000000000.0) * tmp22 +
                     (1033077689742347.0 / 625000000000000.0) * tmp24 + (13320569794516407.0 / 1000000000000000000.0) * tmp29 +
                     (12151357370227279.0 / 2000000000000000.0) * tmp8 - 108606240822233.0 / 800000000000000.0;
  const REAL tmp33 = (140672585931819.0 / 500000000000000.0) * chi1 + (140672585931819.0 / 500000000000000.0) * chi2 +
                     (4528653178655107.0 / 500000000000000.0) * tmp12 * tmp18 + (1379217496010499.0 / 5000000000000.0) * tmp12 -
                     7044987659379691.0 / 20000000000000.0 * tmp13 + (140672585931819.0 / 250000000000000.0) * tmp17 +
                     (1161383067307001.0 / 2500000000000000.0) * tmp18 * tmp4 + (5993783063497041.0 / 250000000000000.0) * tmp19 * tmp8 +
                     (12114999080794129.0 / 10000000000000000.0) * tmp19 - 8497144162281911.0 / 2500000000000000.0 * tmp20 * tmp4 +
                     (7288163703575561.0 / 10000000000000000.0) * tmp20 - 1006495407151063.0 / 100000000000000.0 * tmp22 -
                     2634148094540663.0 / 500000000000000.0 * tmp24 - 2038911386031449.0 / 12500000000000000.0 * tmp29 +
                     (2236915171778621.0 / 200000000000000.0) * tmp4 - 2037082849089339.0 / 25000000000000.0 * tmp8 +
                     3570970180918431.0 / 100000000000000000.0;
  const REAL tmp28 = cosh(tmp27);
  const REAL tmp34 = exp(tmp33);
  const REAL h = (h_0 + ((tmp28) * (tmp28)) * tmp31 * tanh(tmp1 * tmp30 - tmp27) + tmp28 * tmp31 * sinh(tmp27)) * exp(-tmp0 * tmp1);
  const REAL phi =
      -omega_qnm * tmp1 + phi_0 + (omega_qnm + phidot_0) * (tmp34 + 1) * exp(-tmp33) * log((tmp34 * exp(tmp1 * tmp32) + 1) / (tmp34 + 1)) / tmp32;

  waveform[0] = h;
  waveform[1] = phi;
} // END FUNCTION SEOBNRv5_aligned_spin_merger_waveform
