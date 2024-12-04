// We require a certain HW here, because we specifically want to exercise AOT
// compilation and not JIT fallback. However, to make this test run in more
// environments (and therefore cover more scenarios), the list of HW is bigger
// than just a single target.
//
// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12 || arch-intel_gpu_pvc || arch-intel_gpu_mtl_h || arch-intel_gpu_mtl_u
//
// UNSUPPORTED: cuda, hip
// UNSUPPORTED-REASON: FP64 emulation is an Intel specific feature.

// RUN: %clangxx -fsycl -fsycl-targets=intel_gpu_dg2_g10,intel_gpu_dg2_g11,intel_gpu_dg2_g12,intel_gpu_pvc,intel_gpu_mtl_h,intel_gpu_mtl_u -fsycl-fp64-conv-emu -O0 %s -o %t.out
// RUN: %{run} %t.out

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
