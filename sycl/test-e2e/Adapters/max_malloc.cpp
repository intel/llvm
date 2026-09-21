// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PROGRAM_COMPILE_OPTIONS=-ze-intel-greater-than-4GB-buffer-required %{run} %t.out

// debug_R_U_N: %if linux %{ env ZE_DEBUG=-1 ZE_ENABLE_VALIDATION_LAYER=1
// SYCL_UR_TRACE=-1 %} env UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
// SYCL_PROGRAM_COMPILE_OPTIONS=-ze-intel-greater-than-4GB-buffer-required
// %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

const double Gb = 1024 * 1024 * 1024;
const size_t Kb = 1024;

// Maximum GPU page size assumed for this sweep: 1GB.
const size_t MaxPageSize = 1024 * 1024 * Kb;

int main() {
  auto D = device(gpu_selector_v);

  std::cout << "name = " << D.get_info<info::device::name>() << std::endl;

  auto global_mem_size = D.get_info<info::device::global_mem_size>() / Gb;
  std::cout << "global_mem_size = " << global_mem_size << std::endl;
  std::cout << "max_mem_alloc_size = "
            << D.get_info<info::device::max_mem_alloc_size>() / Gb << std::endl;

  auto Q = queue(D);
  // Sweep allocation sizes as powers of two, from 1KB up to the maximum GPU
  // page size (1GB), to exercise every allocation-size class (tiny, small,
  // medium, and page-sized) rather than only whole-Gb-sized allocations.
  for (size_t Size = Kb; Size <= MaxPageSize; Size *= 2) {
    void *p;
    p = malloc_device(Size, Q);
    std::cout << "malloc_device(" << Size / Kb << "Kb) = " << p << std::endl;
    if (p == nullptr) {
      std::cout << "FAILED" << std::endl;
      return -1;
    }
    sycl::free(p, Q);

    p = malloc_shared(Size, Q);
    std::cout << "malloc_shared(" << Size / Kb << "Kb) = " << p << std::endl;
    if (p == nullptr) {
      std::cout << "FAILED" << std::endl;
      return -1;
    }
    sycl::free(p, Q);
  }

  return 0;
}
