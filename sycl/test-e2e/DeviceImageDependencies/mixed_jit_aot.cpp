// Test a kernel launch (implicit build path) whose JIT (SPIR-V) device image
// depends on an image that is only available as native AOT.

// REQUIRES: ocloc, gpu, level_zero

// UNSUPPORTED: windows && gpu-intel-gen12
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/21556

// RUN: %clangxx -fsycl %S/Inputs/a.cpp -I %S/Inputs -c -o %t_a.o
// RUN: %clangxx -fsycl %S/Inputs/b.cpp -I %S/Inputs -c -o %t_b.o
// RUN: %clangxx -fsycl %S/Inputs/c.cpp -I %S/Inputs -c -o %t_c.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen %S/Inputs/d.cpp -I %S/Inputs -c -o %t_d.o
// RUN: %clangxx -fsycl %s -I %S/Inputs -c -o %t_main.o
// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts -fsycl-device-code-split=per_kernel -fsycl-allow-device-image-dependencies -ftarget-export-symbols %t_a.o %t_b.o %t_c.o %t_d.o %t_main.o -o %t.out
// RUN: %{run} %t.out

#include "a.hpp"

#include <sycl/detail/core.hpp>

#include <cassert>
#include <iostream>

int main() {
  sycl::queue q;
  int val = 0;
  {
    sycl::buffer<int, 1> buf(&val, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc{buf, cgh, sycl::read_write};
      cgh.single_task([=]() { acc[0] = levelA(acc[0]); });
    });
  }
  std::cout << "val=" << std::hex << val << "\n";
  assert(val == 0xDCBA);
  return 0;
}
