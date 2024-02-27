// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 SYCL_PROGRAM_COMPILE_OPTIONS=-ze-intel-greater-than-4GB-buffer-required %{run} %t.out

// TODO: Temporarily disabled on Linux due to failures.
// UNSUPPORTED: linux

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

const double Gb = 1024 * 1024 * 1024;

int main() {
  auto D = device(gpu_selector_v);

  std::cout << "name = " << D.get_info<info::device::name>() << std::endl;

  auto global_mem_size = D.get_info<info::device::global_mem_size>() / Gb;
  std::cout << "global_mem_size = " << global_mem_size << std::endl;
  std::cout << "max_mem_alloc_size = "
            << D.get_info<info::device::max_mem_alloc_size>() / Gb << std::endl;

  auto Q = queue(D);
  for (int I = 1; I < global_mem_size; I++) {
    void *p;
    p = malloc_device(I * Gb, Q);
    std::cout << "malloc_device(" << I << "Gb) = " << p << std::endl;
    if (p == nullptr) {
      std::cout << "FAILED" << std::endl;
      return -1;
    }
    sycl::free(p, Q);

    p = malloc_shared(I * Gb, Q);
    std::cout << "malloc_shared(" << I << "Gb) = " << p << std::endl;
    if (p == nullptr) {
      std::cout << "FAILED" << std::endl;
      return -1;
    }
    sycl::free(p, Q);
  }

  return 0;
}
