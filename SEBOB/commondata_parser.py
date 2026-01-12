import struct
import numpy as np

class CommondataStruct:
    """
    Structure to hold the C commondata struct from SEBOB.
    This mirrors the C struct definition from the EOB code used in https://arxiv.org/abs/2508.20418.
    """
    
    def __init__(self):
        """
        Initialize the CommondataStruct with default values.
        This corresponds to the C struct definition in the EOB code.
        """
        self.dynamics_fine = None
        self.dynamics_low = None
        self.dynamics_raw = None
        self.Delta_t = 0.0
        self.Hreal = 0.0
        self.M_f = 0.0
        self.Omega_circ = 0.0
        self.a6 = 0.0
        self.a_1_NQC = 0.0
        self.a_2_NQC = 0.0
        self.a_3_NQC = 0.0
        self.nr_amp_1 = 0.0
        self.nr_amp_2 = 0.0
        self.nr_amp_3 = 0.0
        self.a_f = 0.0
        self.b_1_NQC = 0.0
        self.b_2_NQC = 0.0
        self.nr_omega_1 = 0.0
        self.nr_omega_2 = 0.0
        self.chi1 = 0.0
        self.chi2 = 0.0
        self.dHreal_dpphi = 0.0
        self.dHreal_dpphi_circ = 0.0
        self.dHreal_dprstar = 0.0
        self.dHreal_dr = 0.0
        self.dHreal_dr_circ = 0.0
        self.dHreal_dr_dpphi = 0.0
        self.dHreal_dr_dr = 0.0
        self.dSO = 0.0
        self.dT = 0.0
        self.dt = 0.0
        self.flux = 0.0
        self.initial_omega = 0.0
        self.m1 = 0.0
        self.m2 = 0.0
        self.mass_ratio = 0.0
        self.omega_qnm = 0.0
        self.phi = 0.0
        self.pphi = 0.0
        self.prstar = 0.0
        self.r = 0.0
        self.r_ISCO = 0.0
        self.r_stop = 0.0
        self.t_ISCO = 0.0
        self.t_attach = 0.0
        self.t_stepback = 0.0
        self.tau_qnm = 0.0
        self.total_mass = 0.0
        self.xi = 0.0
        self.waveform_IMR = None
        self.waveform_fine = None
        self.waveform_inspiral = None
        self.waveform_low = None
        self.NUMGRIDS = 0
        self.nsteps_IMR = 0
        self.nsteps_fine = 0
        self.nsteps_inspiral = 0
        self.nsteps_low = 0
        self.nsteps_raw = 0

def read_commondata_from_binary(filename):
    """
    Read binary commondata file and populate CommondataStruct.
    
    Args:
        filename (str): Path to the binary commondata file
        
    Returns:
        CommondataStruct: Populated structure with data from file
    """
    data = CommondataStruct()
    with open(filename, "rb") as f:
        # Read non-pointer members
        # All REAL are double (8 bytes)
        # int is 4 bytes
        # size_t is 8 bytes (assuming 64-bit system)
        format_string = "<" + "d" * 46 + "i" + "Q" * 5 # 46 doubles, 1 int, 5 size_t (unsigned long long)
        (data.Delta_t, data.Hreal, data.M_f, data.Omega_circ, data.a6, data.a_1_NQC, data.a_2_NQC, data.a_3_NQC, 
         data.nr_amp_1, data.nr_amp_2, data.nr_amp_3, data.a_f, \
         data.b_1_NQC, data.b_2_NQC, data.nr_omega_1, data.nr_omega_2, data.chi1, data.chi2, data.dHreal_dpphi, data.dHreal_dpphi_circ, data.dHreal_dprstar, \
         data.dHreal_dr, data.dHreal_dr_circ, data.dHreal_dr_dpphi, data.dHreal_dr_dr, data.dSO, data.dT, data.dt, \
         data.flux, data.initial_omega, data.m1, data.m2, data.mass_ratio, data.omega_qnm, data.phi, data.pphi, \
         data.prstar, data.r, data.r_ISCO, data.r_stop, data.t_ISCO, data.t_attach, data.t_stepback, data.tau_qnm, \
         data.total_mass, data.xi, data.NUMGRIDS, data.nsteps_IMR, data.nsteps_fine, data.nsteps_inspiral, data.nsteps_low, data.nsteps_raw) = \
            struct.unpack(format_string, f.read(struct.calcsize(format_string)))
        # Read pointer members (arrays)
        if data.nsteps_fine > 0:
            data.dynamics_fine = np.fromfile(f, dtype=np.float64, count=data.nsteps_fine*8)
        if data.nsteps_low > 0:
            data.dynamics_low = np.fromfile(f, dtype=np.float64, count=data.nsteps_low*8)
        if data.nsteps_raw > 0:
            data.dynamics_raw = np.fromfile(f, dtype=np.float64, count=data.nsteps_raw*8)
        if data.nsteps_IMR > 0:
            # double complex is two doubles (real and imaginary part)
            data.waveform_IMR = np.fromfile(f, dtype=np.complex128, count=data.nsteps_IMR*2)
        if data.nsteps_fine > 0:
            data.waveform_fine = np.fromfile(f, dtype=np.complex128, count=data.nsteps_fine*2)
        if data.nsteps_inspiral > 0:
            data.waveform_inspiral = np.fromfile(f, dtype=np.complex128, count=data.nsteps_inspiral*2)
        if data.nsteps_low > 0:
            data.waveform_low = np.fromfile(f, dtype=np.complex128, count=data.nsteps_low*2)

    return data

if __name__ == "__main__":
    import sys,os
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(repo_path, "SEBOB", "seob", "commondata.bin")
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found")
        sys.exit(1)
    data = read_commondata_from_binary(data_file)
    print("Commondata loaded successfully!")
    print(f"Delta_t: {data.Delta_t}")
    print(f"Hreal: {data.Hreal}")
    print(f"M_f: {data.M_f}")
    print(f"Number of fine steps: {data.nsteps_fine}")
    print(f"Number of low steps: {data.nsteps_low}")
    import matplotlib.pyplot as plt
    print(data.waveform_low.shape)
    plt.plot(np.real(data.waveform_IMR[0:-2:2]),np.real(data.waveform_IMR[1:-1:2]))
    plt.plot(np.real(data.waveform_IMR[0:-2:2]),np.imag(data.waveform_IMR[1:-1:2]))
    plt.show()
