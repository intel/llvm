// REQUIRES: preview-breaking-changes-supported
// RUN: %{build} -fpreview-breaking-changes -o %t.out
// RUN: %{run} %t.out

// Checks scalar/vec bitwise operator ordering.

#include "vec_binary_scalar_order.hpp"

#define CHECK_SIZES_AND_OPS(Q, C, T)                                           \
  CHECK_SIZES(Q, Failures, T, false, >>)                                       \
  CHECK_SIZES(Q, Failures, T, false, <<)                                       \
  CHECK_SIZES(Q, Failures, T, false, &)                                        \
  CHECK_SIZES(Q, Failures, T, false, |)                                        \
  CHECK_SIZES(Q, Failures, T, false, ^)

int main() {
  sycl::queue Q;
  int Failures = 0;

  // Check operators.
  CHECK_SIZES_AND_OPS(Q, Failures, int8_t);
  CHECK_SIZES_AND_OPS(Q, Failures, int16_t);
  CHECK_SIZES_AND_OPS(Q, Failures, int32_t);
  CHECK_SIZES_AND_OPS(Q, Failures, int64_t);
  CHECK_SIZES_AND_OPS(Q, Failures, uint8_t);
  CHECK_SIZES_AND_OPS(Q, Failures, uint16_t);
  CHECK_SIZES_AND_OPS(Q, Failures, uint32_t);
  CHECK_SIZES_AND_OPS(Q, Failures, uint64_t);
  return Failures;
}
