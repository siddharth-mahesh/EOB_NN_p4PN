import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


class BOB:
    def __init__(self):
        pass

    def _qnm_interps(self,fin_state):
        M_f , a_f = fin_state.T
        afinallist = jnp.array([ -0.9996, -0.9995, -0.9994, -0.9992, -0.999, -0.9989, -0.9988,
            -0.9987, -0.9986, -0.9985, -0.998, -0.9975, -0.997, -0.996, -0.995, -0.994, -0.992, -0.99, -0.988,
            -0.986, -0.984, -0.982, -0.98, -0.975, -0.97, -0.96, -0.95, -0.94, -0.92, -0.9, -0.88, -0.86, -0.84,
            -0.82, -0.8, -0.78, -0.76, -0.74, -0.72, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3,
            -0.25, -0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
            0.65, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97,
            0.975, 0.98, 0.982, 0.984, 0.986, 0.988, 0.99, 0.992, 0.994, 0.995, 0.996, 0.997, 0.9975, 0.998,
            0.9985, 0.9986, 0.9987, 0.9988, 0.9989, 0.999, 0.9992, 0.9994, 0.9995, 0.9996
        ])
        
        reomegaqnm22 = jnp.array([ 0.2915755,0.291581,0.2915866,0.2915976,0.2916086,0.2916142,0.2916197,0.2916252,
            0.2916307,0.2916362,0.2916638,0.2916915,0.2917191,0.2917744,0.2918297,0.291885,0.2919958,0.2921067,0.2922178,0.2923289,
            0.2924403,0.2925517,0.2926633,0.292943,0.2932235,0.2937871,0.2943542,0.2949249,0.2960772,0.2972442,0.2984264,0.299624,
            0.3008375,0.3020672,0.3033134,0.3045767,0.3058573,0.3071558,0.3084726,0.3098081,0.3132321,0.316784,0.3204726,0.3243073,
            0.3282986,0.3324579,0.336798,0.3413329,0.3460786,0.3510526,0.3562748,0.3617677,0.3675569,0.3736717,0.3801456,0.3870175,
            0.394333,0.4021453,0.4105179,0.4195267,0.4292637,0.4398419,0.4514022,0.464123,0.4782352,0.4940448,0.5119692,0.5326002,
            0.5417937,0.5516303,0.5622007,0.5736164,0.586017,0.5995803,0.6145391,0.631206,0.6500179,0.6716143,0.6969947,0.7278753,
            0.74632,0.7676741,0.7932082,0.8082349,0.8254294,0.8331,0.8413426,0.8502722,0.8600456,0.8708927,0.8830905,0.8969183,0.9045305
            ,0.912655,0.9213264,0.9258781,0.9305797,0.9354355,0.9364255,0.937422,0.9384248,0.9394341,0.9404498,0.9425009,0.9445784,
            0.9456271,0.9466825
        ])

        imomegaqnm22 = jnp.array([ 0.0880269,0.0880272,0.0880274,0.088028,0.0880285,0.0880288,0.088029,0.0880293,
            0.0880296,0.0880298,0.0880311,0.0880325,0.0880338,0.0880364,0.0880391,0.0880417,0.088047,0.0880523,0.0880575,0.0880628,0.088068,
            0.0880733,0.0880785,0.0880915,0.0881045,0.0881304,0.088156,0.0881813,0.0882315,0.0882807,0.0883289,0.0883763,0.0884226,0.0884679,
            0.0885122,0.0885555,0.0885976,0.0886386,0.0886785,0.0887172,0.0888085,0.0888917,0.0889663,0.0890315,0.0890868,0.0891313,0.0891643,
            0.0891846,0.0891911,0.0891825,0.0891574,0.0891138,0.0890496,0.0889623,0.0888489,0.0887057,0.0885283,0.0883112,0.0880477,0.0877293,
            0.0873453,0.086882,0.0863212,0.0856388,0.0848021,0.0837652,0.0824618,0.0807929,0.0799908,0.0790927,0.0780817,0.0769364,0.0756296,
            0.0741258,0.072378,0.0703215,0.0678642,0.0648692,0.0611186,0.0562313,0.053149,0.0494336,0.0447904,0.0419586,0.0386302,0.0371155,
            0.0354677,0.033659,0.0316517,0.0293904,0.0268082,0.0238377,0.0221857,0.0204114,0.0185063,0.0175021,0.016462,0.015385,0.0151651,
            0.0149437,0.0147207,0.0144962,0.0142701,0.0138132,0.0133501,0.0131161,0.0128806
        ])
        omega_qnm = jnp.interp(a_f, afinallist, reomegaqnm22) / M_f
        tau_qnm = 1./(jnp.interp(a_f, afinallist, imomegaqnm22) / M_f)
        
        return jnp.column_stack([omega_qnm, tau_qnm])

    def _single_pass(self,bob_inits):
        # bob_inits has shape (omega_qnm,tau_qnm,A_p,Omega_p)
        omega_qnm , tau_qnm , A_p , omega_p, t_start, t_end = bob_inits
        times = jnp.linspace(t_start,t_end,2048)
        Omega_inf = omega_qnm / 2
        Omega_p = - omega_p / 2
        A_BOB = A_p / jnp.cosh(times/tau_qnm)
        b = Omega_p**2
        a = Omega_inf**2 - Omega_p**2
        Omega_minf = jnp.sqrt(b - a)
        Omega = jnp.sqrt(b + a * jnp.tanh(times/tau_qnm))
        phi = -4 * tau_qnm * a * (Omega_inf * jnp.arctanh(Omega/Omega_inf) - Omega_minf * jnp.arctanh(Omega_minf/Omega)) / (Omega_inf**2 - Omega_minf**2)
        phi_p = -4 * tau_qnm * a * (Omega_inf * jnp.arctanh(Omega_p/Omega_inf) - Omega_minf * jnp.arctanh(Omega_minf/Omega_p)) / (Omega_inf**2 - Omega_minf**2)
        phi_BOB = phi - phi_p
        return jnp.column_stack([times,A_BOB,phi_BOB,A_p*jnp.exp(-times/tau_qnm),-times * omega_qnm])

    def __call__(self,nr_in):
        # nr_in has format (batch_size,m_f,a_f,A_p,omega_p,t_start,t_end)
        fin_state = nr_in[:,:2]
        qnms = self._qnm_interps(fin_state)
        bob_inits = jnp.column_stack([qnms,nr_in[:,2:]])
        return jax.vmap(self._single_pass, in_axes=(0))(bob_inits)

if __name__ == "__main__":
    bob = BOB()
    nr_in = jnp.load("nrin_merger_sxs_1em4.npy")
    nr_news = jnp.load("nrnews_merger_sxs_1em4.npy")
    bob_out = bob(nr_in)
    idx = 11
    times_nr = nr_news[idx, :, 0]
    phi_nr = nr_news[idx, :, 1]
    A22_nr = nr_news[idx, :, 2]
    A22_bob = bob_out[idx, :, 1]
    phi_bob = bob_out[idx, :, 2]
    A22_qnm = bob_out[idx, :, 3]
    phi_qnm = bob_out[idx, :, 4]
    qnm_factorized_A = A22_nr/A22_qnm
    qnm_factorized_phi = phi_nr - phi_qnm
    bob_factorized_A = A22_nr/A22_bob
    bob_factorized_phi = phi_nr - phi_bob
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,10))
    ax[0].plot(times_nr,A22_nr,label="NR")
    ax[0].plot(times_nr,A22_bob,linestyle='dashed',label="BOB")
    #ax[0].plot(times_nr,A22_qnm,linestyle='dotted',label="QNM")
    ax[0].set_ylabel("News Amplitude")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(times_nr,jnp.ones(times_nr.shape),color='black',label=r"$\bar{h} = 1$")
    #ax[1].plot(times_nr,qnm_factorized_A,label=r"QNM factorized")
    ax[1].plot(times_nr,bob_factorized_A,label=r"BOB factorized")
    ax[1].set_ylabel("News Amplitude Factorization")
    ax[1].set_yscale('log')
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("bob_factorized_amplitude.png",dpi=300)
    plt.show()

    fig,ax = plt.subplots(2,1,sharex=True,figsize=(10,10))
    ax[0].plot(times_nr,phi_nr,label="NR")
    ax[0].plot(times_nr,phi_bob,linestyle='dashed',label="BOB")
    #ax[0].plot(times_nr,phi_qnm,linestyle='dotted',label="QNM")
    ax[0].set_ylabel("News Phase")
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(times_nr,jnp.zeros(times_nr.shape),color='black',label=r"$\bar{\phi} = 0$")
    #ax[1].plot(times_nr,qnm_factorized_phi,label=r"QNM factorized")
    ax[1].plot(times_nr,bob_factorized_phi,label=r"BOB factorized")
    ax[1].set_ylabel("News Phase Factorization")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("bob_factorized_phase.png",dpi=300)
    plt.show()





    