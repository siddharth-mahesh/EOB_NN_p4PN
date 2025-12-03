"""
NR Surrogate Loader
Calls the NR surrogate model for waveforms that are then processed into training data for the EOB_NN_p4PN model.
"""
import gwsurrogate
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

surrogate_name = 'NRHybSur3dq8_CCE'
sur = gwsurrogate.LoadSurrogate(surrogate_name)
nu = 0.10142355 
omega = 0.01156083
q = (- (2 - 1/nu) + np.sqrt((2 - 1/nu)**2 - 4)) / (2)
if q < 1:
    q = (- (2 - 1/nu) - np.sqrt((2 - 1/nu)**2 - 4)) / (2)                           # mass ratio, mA/mB >= 1.
chiA = [0,0,0]         # Dimensionless spin of heavier BH
chiB = [0,0,0]        # Dimensionless of lighter BH
dt = 0.1                        # timestep size, Units of total mass M
f_low = omega/np.pi                       # initial frequency, f_low=0 returns the full surrogate

def get_amp_phase(time,strain):
    amp = np.abs(strain)
    phase = np.unwrap(np.angle(strain))
    return amp, phase

def get_amp_frequency(time,strain):
    amp = np.abs(strain)
    phase = np.unwrap(np.angle(strain))
    frequency = CubicSpline(time, phase)(time,nu=1)
    return amp, frequency

# h is dictionary of spin-weighted spherical harmonic modes
# t is the corresponding time array in units of M
t, h, _ = sur(q, chiA, chiB, dt=dt, f_low=f_low)
t = t[:np.argmax(np.abs(h[(2,2)]))]
h22 = h[(2,2)][:len(t)]

plt.plot(t,np.real(h22))
plt.plot(t,np.imag(h22))
plt.show()

plt.plot(np.real(h22),np.imag(h22))
plt.show()