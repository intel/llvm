// RUN: %{build} %{embed-ir} -fsycl-preserve-device-nonsemantic-metadata -fsycl-range-rounding=disable -o %t.out
// RUN: %{run} %t.out

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test no local internalization is performed for USM pointers if one kernel
// does not use an 'annotated_ptr' with the necessary annotations.

#include "usm-internalization.hpp"

int main() {
  test_usm_internalization<false>(sycl_ext::access_scope_work_group);
  return 0;
}
