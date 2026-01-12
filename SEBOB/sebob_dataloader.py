"""
EOB Surrogate Loader
Calls the SEOBNRv5 model for waveforms that are then processed into training data for the EOB_NN_p4PN model.
"""
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import os , subprocess
jax.config.update("jax_enable_x64", True)
from scipy.interpolate import CubicSpline
from SEBOB.commondata_parser import read_commondata_from_binary

class SEOBLoader:
    def __init__(self, srate: int):
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.exec_path = os.path.join(repo_path, "SEBOB", "seob")
        self.srate = srate
    
    def _single_waveform_load(self,x):
        nu = x[0]
        omega = x[1]
        q = (- (2 - 1/nu) + np.sqrt((2 - 1/nu)**2 - 4)) / (2)
        if q < 1:
            q = (- (2 - 1/nu) - np.sqrt((2 - 1/nu)**2 - 4)) / (2)
        with open(os.path.join(self.exec_path,"parfile.par"),"w") as parfile:
            out_str = f"""
#### seobnrv5_aligned_spin_inspiral BH@H parameter file. NOTE: only commondata CodeParameters appear here ###
###########################
###########################
### Module: nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients
Delta_t_NS = 0.0             # (REAL)
a6 = 0.0                     # (REAL)
chi1 = 0.                   # (REAL)
chi2 = 0.                  # (REAL)
dt = 2.4627455127717882e-05  # (REAL)
initial_omega = {omega}      # (REAL)
mass_ratio = {q}               # (REAL)
total_mass = 50              # (REAL)
"""
            parfile.write(outstr)
        parfile_path = os.path.join(self.exec_path,"parfile.par")
        exec_path = os.path.join(self.exec_path,"seobnrv5_aligned_spin_inspiral")
        subprocess.run(
            [
                exec_path,
                parfile_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        commondata = read_commondata_from_binary(os.path.join(self.exec_path,"commondata.bin"))
        
        
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