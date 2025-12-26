"""
NR Surrogate Loader
Calls the NR surrogate model for waveforms that are then processed into training data for the EOB_NN_p4PN model.
"""
import gwsurrogate
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from scipy.interpolate import CubicSpline

class NRSurrogateLoader:
    def __init__(self, surrogate_name: str, srate: int):
        self.surrogate_name = surrogate_name
        self.surrogate = gwsurrogate.LoadSurrogate(surrogate_name)
        self.q_max = 8.
        self.srate = srate
    
    def _single_surrogate_load(self,x):
        nu = x[0]
        omega = x[1]
        q = (- (2 - 1/nu) + np.sqrt((2 - 1/nu)**2 - 4)) / (2)
        if q < 1:
            q = (- (2 - 1/nu) - np.sqrt((2 - 1/nu)**2 - 4)) / (2)
        chiA = [0,0,0]         # Dimensionless spin of heavier BH
        chiB = [0,0,0]        # Dimensionless of lighter BH
        dt = 0.1                        # timestep size, Units of total mass M
        f_low = omega/jnp.pi                       # initial frequency, f_low=0 returns the full surrogate
        t, h, _ = self.surrogate(q, chiA, chiB, dt=dt, f_low=f_low)
        t_sampled = jnp.linspace(t[0],t[-1],self.srate)
        h22 = jnp.interp(t_sampled,t,h[(2,2)])
        return jnp.stack([t_sampled,h22],axis=1,dtype=jnp.complex128)
        
    def __call__(self,key: jax.random.PRNGKey, num_waveforms: int, omega_min: float, omega_max: float):
        nu_min = self.q_max/(self.q_max + 1)**2
        nu_max = 1./4. 
        nu = np.random.uniform(nu_min,nu_max,num_waveforms)
        omega = np.random.uniform(omega_min,omega_max,num_waveforms)
        x = np.stack([nu,omega],axis=1)
        y = []
        for i in range(num_waveforms):
            y.append(self._single_surrogate_load(x[i]))
        return jnp.array(x), jnp.array(y)
        
        
if __name__ == "__main__":
    nr_surrogate_loader = NRSurrogateLoader("NRHybSur3dq8_CCE", 8192)
    x, y = nr_surrogate_loader(jax.random.PRNGKey(42), 10, 0.01, 0.03)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    h22 = y[0]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(jnp.abs(h22[:, 0]), jnp.abs(h22[:, 1]))
    ax[0].set_title(rf"Extrapolated waveform, $\nu = {x[0][0]:.2e}$, $\Omega_0 = {x[0][1]:.2e}$")
    ax[0].set_xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    ax[0].set_yscale('log')
    ax[0].grid(True)
    ax[1].plot(jnp.abs(h22[:, 0]), jnp.unwrap(jnp.angle(h22[:, 1])))
    ax[1].set_xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()    