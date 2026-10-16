// REQUIRES: gpu
// RUN: %{build} -o %t.exe
// RUN: %{run} %t.exe 2> %t.out

// RUN: %{run-aux} FileCheck %s --check-prefix=CHECK-EXPECTED-ERROR --input-file %t.out
// CHECK-EXPECTED-ERROR: error: backend compiler failed build

#include <array>
#include <iostream>
#include <sycl/detail/core.hpp>

void test() {
  using namespace sycl;
#if CPU
  auto a = cpu_selector_v;
#else
  auto a = gpu_selector_v;
#endif
  queue q(a);
  constexpr size_t MaxKernelSize = 1 + 1024 * 32;
  size_t Limit =
      q.get_device().get_info<sycl::info::device::max_parameter_size>();

  if (Limit >= MaxKernelSize) {
    std::cerr << "max_parameter_size (" << Limit << ") >= " << MaxKernelSize
              << ": increase MaxKernelSize in this test\n";
    std::exit(1);
  }

  std::array<char, MaxKernelSize> Big;
  Big[0] = 'a';
  Big[MaxKernelSize - 1] = 'a';

  try {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsycl-strict"
    q.single_task([=] {
       volatile int local = Big[0] + Big[MaxKernelSize - 1];
       (void)local;
     }).wait();
#pragma clang diagnostic pop
  } catch (exception &e) {
    std::cerr << e.what() << '\n';
  }
}

int main() { test(); }
