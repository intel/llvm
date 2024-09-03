// RUN: %{build} %{embed-ir} -fsycl-preserve-device-nonsemantic-metadata -fsycl-range-rounding=disable -o %t.out
// RUN: %{run} %t.out

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test private internalization for USM.

#include "usm-internalization.hpp"

int main() {
  test_usm_internalization<true>(sycl_ext::access_scope_work_item);
  return 0;
}
