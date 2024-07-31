// REQUIRES: preview-breaking-changes-supported
// RUN: %{build} -fpreview-breaking-changes -o %t.out
// RUN: %{run} %t.out

// Checks scalar/vec relational operator ordering.

#include "vec_binary_scalar_order.hpp"

// NOTE: For the sake of compile-time we pick only a few operators per category.
#define CHECK_SIZES_AND_OPS(Q, C, T)                                           \
  CHECK_SIZES(Q, Failures, T, true, ==)                                        \
  CHECK_SIZES(Q, Failures, T, true, !=)                                        \
  CHECK_SIZES(Q, Failures, T, true, <)                                         \
  CHECK_SIZES(Q, Failures, T, true, >)                                         \
  CHECK_SIZES(Q, Failures, T, true, <=)                                        \
  CHECK_SIZES(Q, Failures, T, true, >=)

int main() {
  sycl::queue Q;
  int Failures = 0;

  // Check operators on types with requirements if they are supported.
  if (Q.get_device().has(sycl::aspect::fp16)) {
    CHECK_SIZES_AND_OPS(Q, Failures, sycl::half);
  }
  if (Q.get_device().has(sycl::aspect::fp64)) {
    CHECK_SIZES_AND_OPS(Q, Failures, double);
  }

  // Check all operators without requirements.
  CHECK_SIZES_AND_OPS(Q, Failures, float);
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
