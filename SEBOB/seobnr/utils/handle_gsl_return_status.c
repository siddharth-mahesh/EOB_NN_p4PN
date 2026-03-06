#include "BHaH_defines.h"

/**
 * Error handler for calls to GSL-dependent routines.
 *
 * @param status - The return value of the GSL-dependent call.
 * @param status_desired - The desired return statuses of the GSL call.
 * @param num_desired - The number of desired return statuses.
 * @param function_name - The name of the GSL-dependent function that was called.
 */
void handle_gsl_return_status(int status, int status_desired[], int num_desired, const char *restrict function_name) {
  int count = 0;
  for (int i = 0; i < num_desired; i++) {
    if (status == status_desired[i]) {
      count++;
    }
  }
  if (count == 0) {
    printf("In function %s, gsl returned error: %s\nAborted", function_name, gsl_strerror(status));
    exit(EXIT_FAILURE);
  }
} // END FUNCTION handle_gsl_return_status
