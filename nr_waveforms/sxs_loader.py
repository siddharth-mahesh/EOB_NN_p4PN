import matplotlib.pyplot as plt
import numpy as np  
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import json
import sxs
import os

class SXSLoader:
    def __init__(
        self, 
        sxs_id_loc: str,
        dict_key: str
    ):
        """
        Initialize SXS loader with JSON file containing SXS IDs.

        Args:
            sxs_id_loc: Path to JSON file containing SXS IDs
            dict_key: Key in SXS dictionary to select the desired set of binaries.
            Options currently include:
             - 'chimax_1em3': quasicircular configurations with maximum individual spin magnitude 1e-3
             - 'chimax_1em4': quasicircular configurations with maximum individual spin magnitude 1e-4
             - 'chimax_1em5': quasicircular configurations with maximum individual spin magnitude 1e-5
        """
        if not sxs_id_loc.endswith('.json'):
            raise ValueError("sxs_id_loc must be a JSON file path")
        with open(os.path.join(os.getcwd(),"nr_waveforms",sxs_id_loc), 'r') as f:
            data = json.load(f)
            self.sxsids = data[dict_key]
        self.srate = 8192
    
    def visualize_parameter_space(self):
        """
        Visualize the parameter space of the dataset.
        """
        import seaborn as sns
        import pandas as pd 
        # create a pandas dataframe from the SXS metadata
        nus = []
        omegas = []
        for sxsid in self.sxsids:
            sxs_bbh = sxs.load(sxsid)
            q = sxs_bbh.metadata.reference_mass_ratio
            nus.append(q/(q+1)**2)
            omegas.append(np.linalg.norm(np.array(sxs_bbh.metadata.reference_orbital_frequency)))
        df = pd.DataFrame({'nu': nus, 'omega': omegas})
        pp = sns.pairplot(df,corner=True)
        pp.savefig("sxs_parameter_space.png")
        

    def __call__(self, num_waveforms: int):
        """
        Load specified number of waveforms from SXS catalog.
        
        Args:
            num_waveforms: Number of waveforms to load

        Returns:
            List of SXS waveform objects
        """
        # Ensure that there are sufficient available waveforms
        # Randomize the set of SXS IDs
        if num_waveforms > len(self.sxsids):
            raise ValueError(f"Requested {num_waveforms} waveforms, but only {len(self.sxsids)} available")
        elif num_waveforms < len(self.sxsids):
            # Randomly sample num_waveforms from self.sxsids
            import random
            selected_ids = random.sample(self.sxsids, num_waveforms)
        else:
            # Use all available waveforms
            selected_ids = self.sxsids
        x = []
        y = []
        for sxsid in selected_ids:
            sxs_bbh = sxs.load(sxsid)
            w = sxs_bbh.h
            q = sxs_bbh.metadata.reference_mass_ratio
            nu = q/(q+1)**2
            omega = jnp.linalg.norm(jnp.array(sxs_bbh.metadata.reference_orbital_frequency))
            x.append(
                [
                    nu,
                    omega
                ]
            )
            reference_time = sxs_bbh.metadata.reference_time
            # add a small offset to the relaxation time
            reference_index = w.index_closest_to(reference_time + 10)
            w_sliced = w[reference_index:]
            new_times = jnp.linspace(w_sliced.t[0], w_sliced.t[-1], self.srate)
            w_sliced = w_sliced.interpolate(new_times)
            ell, m = 2, 2
            h22 = w_sliced[:, w_sliced.index(ell, m)]
            y.append(
                jnp.hstack([
                    jnp.array(h22.t,dtype=jnp.complex128).reshape(-1, 1),
                    jnp.array(h22.data,dtype=jnp.complex128).reshape(-1, 1)
                ])
            )
        return jnp.array(x), jnp.array(y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataloader = SXSLoader("sxs_non_spinning.json", "chimax_1em3")
    dataloader.visualize_parameter_space()
    x, y = dataloader(10)
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