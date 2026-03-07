"""
EOB Surrogate Loader
Calls the SEOBNRv5 model for waveforms that are then processed into training data for the EOB_NN_p4PN model.
"""
import matplotlib.pyplot as plt
import numpy as np
import os , subprocess
from scipy.interpolate import CubicSpline
from SEBOB.commondata_parser import read_commondata_from_binary

class SEOBLoader:
    def __init__(self, srate: int):
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.exec_path = os.path.join(repo_path, "SEBOB", "seobnr")
        self.srate = srate
        self.q_max = 10

    def _single_commondata_load(self,x):
        nu = x[0]
        omega = x[1]
        q = (- (2 - 1/nu) + np.sqrt((2 - 1/nu)**2 - 4)) / (2)
        if q < 1:
            q = (- (2 - 1/nu) - np.sqrt((2 - 1/nu)**2 - 4)) / (2)
        commondata_path = os.path.join(self.exec_path, "commondata.bin")
        # Stale-file guard: remove previous output so failed runs cannot reuse it.
        try:
            os.remove(commondata_path)
        except FileNotFoundError:
            pass
        with open(os.path.join(self.exec_path,"parfile.par"),"w") as parfile:
            out_str = f"""
#### seobnrv5_aligned_spin_inspiral BH@H parameter file. NOTE: only commondata CodeParameters appear here ###
###########################
###########################
### Module: nrpy.infrastructures.BHaH.seobnr.SEOBNRv5_aligned_spin_coefficients
chi1 = 0.                   # (REAL)
chi2 = 0.                  # (REAL)
dt = 2.4627455127717882e-05  # (REAL)
initial_omega = {omega}      # (REAL)
mass_ratio = {q}               # (REAL)
total_mass = 50              # (REAL)
"""
            parfile.write(out_str)
        parfile_path = os.path.join(self.exec_path,"parfile.par")
        exec_path = os.path.join(self.exec_path,"seobnrv5_nrpy")
        result = subprocess.run(
            [
                exec_path,
                parfile_path
            ],
            cwd=self.exec_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            stdout_tail = (result.stdout or "").strip()[-500:]
            stderr_tail = (result.stderr or "").strip()[-500:]
            raise RuntimeError(
                "SEOBNR execution failed with return code "
                f"{result.returncode}. stdout_tail='{stdout_tail}' stderr_tail='{stderr_tail}'"
            )
        if not os.path.exists(commondata_path):
            raise RuntimeError(
                "SEOBNR execution completed but commondata.bin was not produced. "
                "Stale-file guard prevented reusing old output."
            )
        commondata = read_commondata_from_binary(commondata_path)
        return commondata

    def _single_waveform_load(self,x):
        commondata = self._single_commondata_load(x)
        h22 = commondata.waveform_IMR
        t = np.real(h22[:-2:2])
        h22 = h22[1:-1:2]
        habs = np.abs(h22)
        hphase = np.unwrap(np.angle(h22))
        new_times = np.linspace(t[0],t[-1],self.srate) + 0.0j
        habs_interp = CubicSpline(t,habs)(new_times)
        hphase_interp = CubicSpline(t,hphase)(new_times)
        h22_interp = habs_interp * np.exp(1j * hphase_interp)
        return np.stack([
            new_times,
            h22_interp
            ], axis=1)

    def _single_trajectory_load(self,x):
        NUMVARS = 8
        TIME = 0
        R = 1
        PHI = 2
        PRSTAR = 3
        PPHI = 4
        idx = lambda point, var: NUMVARS*point + var 
        commondata = self._single_commondata_load(x)
        nu = x[0]
        dynamics_raw = commondata.dynamics_raw
        radii = np.array([dynamics_raw[idx(i,R)] for i in range(commondata.nsteps_raw)])
        times = np.array([dynamics_raw[idx(i,TIME)] for i in range(commondata.nsteps_raw)])
        r = np.linspace(radii[0],radii[-1],self.srate)
        t_spl = CubicSpline(-radii,times)
        new_times = t_spl(-r)
        r_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,R)] for i in range(commondata.nsteps_raw)]))
        phi_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,PHI)] for i in range(commondata.nsteps_raw)]))
        prstar_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,PRSTAR)] for i in range(commondata.nsteps_raw)]))
        pphi_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,PPHI)] for i in range(commondata.nsteps_raw)]))
        phi = phi_spl(new_times)
        prstar = prstar_spl(new_times)
        pphi = pphi_spl(new_times)
        rdot = r_spl(new_times,1)
        phidot = phi_spl(new_times,1)
        prstardot = prstar_spl(new_times,1)
        pphidot = pphi_spl(new_times,1)
        
        # new_x = [nu,r,phi,prstar,pphi]
        # new_y = [rdot,phidot,prstardot,pphidot]

        new_x = np.stack([
            nu * np.ones_like(r),
            r,
            phi,
            prstar,
            pphi
            ], axis=1)
        new_y = np.stack([
            rdot,
            phidot,
            prstardot,
            pphidot
            ], axis=1)
        return new_x, new_y
    def _single_trajectory_load_validation(self,x):
        NUMVARS = 8
        TIME = 0
        R = 1
        PHI = 2
        PRSTAR = 3
        PPHI = 4
        idx = lambda point, var: NUMVARS*point + var 
        commondata = self._single_commondata_load(x)
        nu = x[0]
        dynamics_raw = commondata.dynamics_raw
        radii = np.array([dynamics_raw[idx(i,R)] for i in range(commondata.nsteps_raw)])
        times = np.array([dynamics_raw[idx(i,TIME)] for i in range(commondata.nsteps_raw)])
        r = np.linspace(radii[0],radii[-1],self.srate)
        t_spl = CubicSpline(-radii,times)
        new_times = t_spl(-r)
        r_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,R)] for i in range(commondata.nsteps_raw)]))
        phi_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,PHI)] for i in range(commondata.nsteps_raw)]))
        prstar_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,PRSTAR)] for i in range(commondata.nsteps_raw)]))
        pphi_spl = CubicSpline(times,np.array([dynamics_raw[idx(i,PPHI)] for i in range(commondata.nsteps_raw)]))
        
        x_pert = x * (1 - 1e-13)
        commondata_pert = self._single_commondata_load(x_pert)
        dynamics_raw_pert = commondata_pert.dynamics_raw
        radii_pert = np.array([dynamics_raw_pert[idx(i,R)] for i in range(commondata_pert.nsteps_raw)])
        times_pert = np.array([dynamics_raw_pert[idx(i,TIME)] for i in range(commondata_pert.nsteps_raw)])
        t_spl_pert = CubicSpline(-radii_pert,times_pert)
        r_pert = np.linspace(radii_pert[0],radii_pert[-1],self.srate)
        new_times_pert = t_spl_pert(-r_pert)
        r_pert_spl = CubicSpline(times_pert,np.array([dynamics_raw_pert[idx(i,R)] for i in range(commondata_pert.nsteps_raw)]))
        phi_pert_spl = CubicSpline(times_pert,np.array([dynamics_raw_pert[idx(i,PHI)] for i in range(commondata_pert.nsteps_raw)]))
        prstar_pert_spl = CubicSpline(times_pert,np.array([dynamics_raw_pert[idx(i,PRSTAR)] for i in range(commondata_pert.nsteps_raw)]))
        pphi_pert_spl = CubicSpline(times_pert,np.array([dynamics_raw_pert[idx(i,PPHI)] for i in range(commondata_pert.nsteps_raw)]))

        
        actual_times = new_times if new_times[-1] < new_times_pert[-1] else new_times_pert

        r = r_spl(actual_times)
        phi = phi_spl(actual_times)
        prstar = prstar_spl(actual_times)
        pphi = pphi_spl(actual_times)
        rdot = r_spl(actual_times,1)
        phidot = phi_spl(actual_times,1)
        prstardot = prstar_spl(actual_times,1)
        pphidot = pphi_spl(actual_times,1)

        r_pert = r_pert_spl(actual_times)
        phi_pert = phi_pert_spl(actual_times)
        prstar_pert = prstar_pert_spl(actual_times)
        pphi_pert = pphi_pert_spl(actual_times)
        rdot_pert = r_pert_spl(actual_times,1)
        phidot_pert = phi_pert_spl(actual_times,1)
        prstardot_pert = prstar_pert_spl(actual_times,1)
        pphidot_pert = pphi_pert_spl(actual_times,1)

        e_rel_rdot = np.abs((rdot - rdot_pert)/rdot)
        e_rel_phidot = np.abs((phidot - phidot_pert)/phidot)
        e_rel_prstardot = np.abs((prstardot - prstardot_pert)/prstardot)
        e_rel_pphidot = np.abs((pphidot - pphidot_pert)/pphidot)

        # new_x = [nu,r,phi,prstar,pphi]
        # new_y = [rdot,phidot,prstardot,pphidot]

        new_x = np.stack([
            nu * np.ones_like(r),
            r,
            phi,
            prstar,
            pphi
            ], axis=1)
        new_y = np.stack([
            rdot,
            phidot,
            prstardot,
            pphidot
            ], axis=1)
        new_e_rel = np.stack([
            e_rel_rdot,
            e_rel_phidot,
            e_rel_prstardot,
            e_rel_pphidot
            ], axis=1)
        return new_x, new_y , new_e_rel

    def __call__(
                 self,
                 seed: int,
                 num_waveforms: int, 
                 omega_min: float, 
                 omega_max: float,
                 trajectory: bool = False,
                 validation: bool = False):
        nu_min = self.q_max/(self.q_max + 1)**2
        nu_max = 1./4. 
        rng = np.random.default_rng(seed)
        nu = rng.uniform(nu_min,nu_max,num_waveforms)
        omega = rng.uniform(omega_min,omega_max,num_waveforms)
        x = np.stack([nu,omega],axis=1)
        if trajectory and not validation:
            x_final = []
            y_final = []
            for i in range(num_waveforms):
                x_i, y_i = self._single_trajectory_load(x[i])
                x_final.append(x_i)
                y_final.append(y_i)
            # final shape must be (num_waveforms*srate, 5) and (num_waveforms*srate, 4)
            return np.array(x_final).reshape(-1,5), np.array(y_final).reshape(-1,4)
        if trajectory and validation:
            x_final = []
            y_final = []
            e_rel_final = []
            for i in range(num_waveforms):
                x_i, y_i, e_rel_i = self._single_trajectory_load_validation(x[i])
                x_final.append(x_i)
                y_final.append(y_i)
                e_rel_final.append(e_rel_i)
            # final shape must be (num_waveforms*srate, 5) and (num_waveorms*srate, 4)
            return np.array(x_final).reshape(-1,5), np.array(y_final).reshape(-1,4), np.array(e_rel_final).reshape(-1,4)

        y = []
        for i in range(num_waveforms):
            y.append(self._single_waveform_load(x[i]))
        return np.array(x), np.array(y)
        
        
if __name__ == "__main__":
    seob_loader = SEOBLoader(512)
    
    trajectory = True
    if not trajectory:
        # test waveform
        x, y = seob_loader(42, 10, 0.01, 0.03)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        h22 = y[0]
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(np.real(h22[:, 0]), np.abs(h22[:, 1]))
        ax[0].set_title(rf"Extrapolated waveform, $\nu = {x[0][0]:.2e}$, $\Omega_0 = {x[0][1]:.2e}$")
        ax[0].set_xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
        ax[0].set_yscale('log')
        ax[0].grid(True)
        ax[1].plot(np.real(h22[:, 0]), np.unwrap(np.angle(h22[:, 1])))
        ax[1].set_xlabel(r"$(t_{\mathrm{corr}} - r_\ast)/M$")
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()    
    else:    
        # training trajectories
        x, y = seob_loader(42, 80, 0.01, 0.03, trajectory=True)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        fig, ax = plt.subplots(2, 1)
        x_first = x[:seob_loader.srate].reshape(seob_loader.srate,5)
        ax[0].plot(x_first[:, 1], x_first[:, 3])
        ax[0].set_title(rf"EOB Phase Space, $\nu = {x_first[0,0]:.2e}$")
        ax[0].set_xlabel(r"$r/M$")
        ax[0].set_ylabel(r'$p_r^*/\mu$')
        ax[0].grid(True)
        ax[1].plot(x_first[:, 2], x_first[:, 4])
        ax[1].set_xlabel(r"$\phi$")
        ax[1].set_ylabel(r'$p_\phi/M\mu$')
        ax[1].grid(True)
        plt.tight_layout()
        plt.show()    
        # save the training data
        np.save("seob_x_train_prelim.npy", x)
        np.save("seob_y_train_prelim.npy", y)
        
        # validation trajectories
        x, y, erel = seob_loader(21, 20, 0.01, 0.03, trajectory=True, validation=True)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("erel shape:", erel.shape)
        print(erel)
        # save the validation data
        np.save("seob_x_val_prelim.npy", x)
        np.save("seob_y_val_prelim.npy", y)
        np.save("seob_erel_val_prelim.npy", erel)
        
        # test trajectories
        x, y, erel = seob_loader(10, 10, 0.01, 0.03, trajectory=True, validation=True)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("erel shape:", erel.shape)
        # save the training data
        np.save("seob_x_test_prelim.npy", x)
        np.save("seob_y_test_prelim.npy", y)
        np.save("seob_erel_test_prelim.npy", erel)
        
