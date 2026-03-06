#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Finds the local minimum index in an array.
 * Implements the logic of scipy.argrelmin with order = 3 and clipped indexing.
 * However, it returns the index of the first minimum from the left,
 * as required by SEOBNRv5_aligned_spin_iterative_refinement.
 *
 * @param arr - The array to search for the minimum in.
 * @param size - The size of the array.
 * @param order - The maximum array index shift to consider for the minimum.
 * @returns - The index of the local minimum in the array or -1 if no minimum is found.
 */
size_t find_local_minimum_index(REAL *restrict arr, size_t size, int order) {
  if (size < 2 * order + 1) {
    return -1; // Not enough points to apply the order
  }

  for (size_t i = 0; i < size; ++i) {
    bool is_min = true;
    for (int shift = 1; shift <= order; ++shift) {
      // clipped indexing: return 0 or size-1 if out of bounds
      size_t left_idx = MAX(i, shift) - shift;     // returns 0 if i < shift else i - shift
      size_t right_idx = MIN(i + shift, size - 1); // returns size-1 if i > size-1 else i + shift
      is_min = is_min && (arr[i] < arr[left_idx]) && (arr[i] < arr[right_idx]);
      if (is_min)
        return i;
    }
  }
  return -1; // No local minimum found
} // END FUNCTION find_local_minimum_index
