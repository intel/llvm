// REQUIRES: ocloc, gpu, linux
// UNSUPPORTED: cuda, hip

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen  -Xsycl-target-backend "-device pvc" -fsycl-fp64-conv-emu -O0 %s -o %t_opt.out
// TODO: Enable when GPU driver is updated.
// RUNx: %{run} %t_opt.out

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
