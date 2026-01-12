CC ?= gcc  # assigns the value CC to gcc only if environment variable CC is not already set

CFLAGS = -std=gnu99 -O2 -march=native -g -Wall -I. $(shell gsl-config --cflags)
CXXFLAGS = -I. -O2 -g -Wall -Wno-unknown-pragmas -march=native
VALGRIND_CFLAGS = -I. -std=gnu99 -O2 -g -Wall -Wno-unknown-pragmas $(shell gsl-config --cflags)
INCLUDEDIRS =
LDFLAGS = $(shell gsl-config --libs) -lm
# Check for OpenMP support
OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)
endif

OBJ_FILES = cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o dynamics/eval_abs_deriv.o dynamics/find_local_minimum_index.o dynamics/SEOBNRv5_aligned_spin_augments.o dynamics/SEOBNRv5_aligned_spin_flux.o dynamics/SEOBNRv5_aligned_spin_interpolate_dynamics.o dynamics/SEOBNRv5_aligned_spin_iterative_refinement.o dynamics/SEOBNRv5_aligned_spin_ode_integration.o dynamics/SEOBNRv5_aligned_spin_right_hand_sides.o initial_conditions/SEOBNRv5_aligned_spin_Hamiltonian_circular_orbit.o initial_conditions/SEOBNRv5_aligned_spin_initial_conditions_conservative.o initial_conditions/SEOBNRv5_aligned_spin_initial_conditions_dissipative.o initial_conditions/SEOBNRv5_aligned_spin_multidimensional_root_wrapper.o initial_conditions/SEOBNRv5_aligned_spin_radial_momentum_condition.o inspiral_waveform/SEOBNRv5_aligned_spin_gamma_wrapper.o inspiral_waveform/SEOBNRv5_aligned_spin_interpolate_modes.o inspiral_waveform/SEOBNRv5_aligned_spin_waveform.o inspiral_waveform/SEOBNRv5_aligned_spin_waveform_from_dynamics.o main.o merger_waveform/SEOBNRv5_aligned_spin_merger_waveform.o merger_waveform/SEOBNRv5_aligned_spin_merger_waveform_from_times.o nqc_corrections/SEOBNRv5_aligned_spin_NQC_corrections.o nqc_corrections/SEOBNRv5_aligned_spin_NQC_rhs.o params_struct_set_to_default.o SEOBNRv5_aligned_spin_coefficients.o SEOBNRv5_aligned_spin_IMR_waveform.o utils/commondata_io.o utils/handle_gsl_return_status.o utils/root_finding_1d.o utils/root_finding_multidimensional.o utils/SEOBNRv5_aligned_spin_unwrap.o

all: seobnrv5_aligned_spin_inspiral

%.o: %.c $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

seobnrv5_aligned_spin_inspiral: $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

valgrind: clean
	$(MAKE) CFLAGS="$(VALGRIND_CFLAGS)" all
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -s ./seobnrv5_aligned_spin_inspiral

# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o */*/*.o *~ */*~ */lib*.a ./#* *.txt *.gp *.dat *.avi *.png seobnrv5_aligned_spin_inspiral
