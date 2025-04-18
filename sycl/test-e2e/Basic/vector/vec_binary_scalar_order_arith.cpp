// REQUIRES: preview-breaking-changes-supported
// RUN: %{build} -fpreview-breaking-changes -o %t.out
// RUN: %{run} %t.out

// Checks scalar/vec arithmetic operator ordering.

#include "vec_binary_scalar_order.hpp"

#define CHECK_SIZES_AND_COMMON_OPS(Q, C, T)                                    \
  CHECK_SIZES(Q, Failures, T, false, +)                                        \
  CHECK_SIZES(Q, Failures, T, false, -)                                        \
  CHECK_SIZES(Q, Failures, T, false, /)                                        \
  CHECK_SIZES(Q, Failures, T, false, *)
#define CHECK_SIZES_AND_INT_ONLY_OPS(Q, C, T)                                  \
  CHECK_SIZES(Q, Failures, T, false, %)

int main() {
  sycl::queue Q;
  int Failures = 0;

  // Check operators on types with requirements if they are supported.
  if (Q.get_device().has(sycl::aspect::fp16)) {
    CHECK_SIZES_AND_COMMON_OPS(Q, Failures, sycl::half);
  }
  if (Q.get_device().has(sycl::aspect::fp64)) {
    CHECK_SIZES_AND_COMMON_OPS(Q, Failures, double);
  }

  // Check operators without requirements.
  CHECK_SIZES_AND_COMMON_OPS(Q, Failures, float);
  CHECK_SIZES_AND_COMMON_OPS(Q, Failures, int8_t);
  CHECK_SIZES_AND_COMMON_OPS(Q, Failures, uint16_t);

  // Check integer only operators.
  CHECK_SIZES_AND_INT_ONLY_OPS(Q, Failures, int16_t);
  CHECK_SIZES_AND_INT_ONLY_OPS(Q, Failures, uint8_t);
  return Failures;
}
