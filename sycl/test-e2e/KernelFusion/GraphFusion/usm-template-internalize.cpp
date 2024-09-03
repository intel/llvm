// RUN: %{build} %{embed-ir} -fsycl-preserve-device-nonsemantic-metadata -fsycl-range-rounding=disable -o %t.out
// RUN: %{run} %t.out

// REQUIRES: linux, fusion, (level_zero || cuda)

// Test internalization for USM, but using template instantiations of the
// property rather than pre-defined values.

#include "usm-internalization.hpp"

int main() {
  test_usm_internalization<true, class Kernel11, class Kernel12>(
      sycl_ext::property::access_scope<sycl::memory_scope::work_item>{});
  test_usm_internalization<true, class Kernel21, class Kernel22>(
      sycl_ext::property::access_scope<sycl::memory_scope::work_group>{});
  return 0;
}
