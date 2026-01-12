import matplotlib.pyplot as plt
import numpy as np  
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
#import jax
#import jax.numpy as np
#jax.config.update('jax_enable_x64', True)
import json
import sxs
import os

class SXSLoader:
    def __init__(
        self, 
        srate: int,
        sxs_id_loc: str,
        dict_key: str
    ):
        """
        Initialize SXS loader with JSON file containing SXS IDs.

        Args:
            srate: Sampling rate for the strain.
            sxs_id_loc: Path to JSON file containing SXS IDs
            dict_key: Key in SXS dictionary to select the desired set of binaries.
            Options currently include:
             - 'chimax_1em3': quasicircular configurations with maximum individual spin magnitude 1e-3
             - 'chimax_1em4': quasicircular configurations with maximum individual spin magnitude 1e-4
             - 'chimax_1em5': quasicircular configurations with maximum individual spin magnitude 1e-5
        """
        self.dict_key = dict_key
        if not sxs_id_loc.endswith('.json'):
            raise ValueError("sxs_id_loc must be a JSON file path")
        with open(os.path.join(os.getcwd(),"nr_waveforms",sxs_id_loc), 'r') as f:
            data = json.load(f)
            self.sxsids = data[dict_key]
        self.srate = srate
    
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
        pp.savefig(f"sxs_{self.dict_key}_parameter_space.png")
    
    def find_interpolated_maximum(self,strain,time):
        strain_abs = np.abs(strain)
        spl = CubicSpline(time,strain_abs)
        root_func = lambda x,args: args(x,nu=1)
        sol = root_scalar(
            root_func,
            (spl),
            method='bisect',
            bracket=[time[np.argmax(strain_abs)] - 10, time[np.argmax(strain_abs)] + 10],
            xtol=1e-12,
            rtol=1e-12
        )
        return sol.root
 

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
            omega = np.linalg.norm(np.array(sxs_bbh.metadata.reference_orbital_frequency))
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
            h22 = w_sliced[:, w_sliced.index(2, 2)]
            tmax = self.find_interpolated_maximum(h22.data,h22.t)
            new_times = np.linspace(h22.t[0], tmax, self.srate)
            h22 = h22.interpolate(new_times)
            y.append(
                np.hstack([
                    np.array(h22.t - h22.t[0],dtype=np.complex128).reshape(-1, 1),
                    np.array(h22.data,dtype=np.complex128).reshape(-1, 1)
                ])
            )
        return np.array(x), np.array(y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataloader = SXSLoader(4096,"sxs_non_spinning.json", "chimax_1em4")
    dataloader.visualize_parameter_space()
    x, y = dataloader(len(dataloader.sxsids))
    np.save("x_sxs_1em4.npy", x)
    np.save("y_sxs_1em4.npy", y)
    print("x val shape:", x.shape)
    print("y val shape:", y.shape)
    dataloader = SXSLoader(4096,"sxs_non_spinning.json", "chimax_1em5")
    dataloader.visualize_parameter_space()
    x, y = dataloader(len(dataloader.sxsids))
    np.save("x_sxs_1em5.npy", x)
    np.save("y_sxs_1em5.npy", y)
    print("x train shape:", x.shape)
    print("y train shape:", y.shape)
    dataloader = SXSLoader(4096,"sxs_non_spinning.json", "chimax_1em3")
    dataloader.visualize_parameter_space()
    x, y = dataloader(len(dataloader.sxsids))
    np.save("x_sxs_1em3.npy", x)
    np.save("y_sxs_1em3.npy", y)
    print("x train shape:", x.shape)
    print("y train shape:", y.shape)
    h22 = y[0]
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(np.abs(h22[:, 0]), np.abs(h22[:, 1]))
    ax[0].set_title(rf"Extrapolated waveform, $\nu = {x[0][0]:.2e}$, $\Omega_0 = {x[0][1]:.2e}$")
    ax[0].set_xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    ax[0].set_yscale('log')
    ax[0].grid(True)
    ax[1].plot(np.abs(h22[:, 0]), np.unwrap(np.angle(h22[:, 1])))
    ax[1].set_xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig('sxs_example.png',dpi=300)
    plt.show()