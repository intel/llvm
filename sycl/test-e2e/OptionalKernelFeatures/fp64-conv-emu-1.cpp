// REQUIRES: ocloc, gpu, linux, arch-intel_gpu_dg2_g10
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-REASON: FP64 emulation is an Intel specific feature.

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10 -fsycl-fp64-conv-emu -O0 %s -o %t_opt.out
// RUN: %{run} %t_opt.out

// Tests that aspect::fp64 is not emitted correctly when -fsycl-fp64-conv-emu
// flag is used.

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    h.single_task([=]() {
      double a[10];
      double b[10];
      int i = 4;
      b[i] = (double)((float)(a[i]));
    });
  });
  return 0;
}
