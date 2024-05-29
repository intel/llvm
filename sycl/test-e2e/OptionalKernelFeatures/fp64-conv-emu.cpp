// REQUIRES: ocloc, gpu
// UNSUPPORTED: cuda, hip

// RUN: %clangxx -fsycl -fsycl-targets=spir64_gen  -Xsycl-target-backend "-device *" -fsycl-fp64-conv-emu -O0 %s -o %t_opt.out
// To be enabled soon.
// RUNx: %{run} %t_opt.out

// Tests that aspect::fp64 is not emitted correctly when -fsycl-fp64-conv-emu
// flag is used.

#include <sycl/detail/core.hpp>

template <typename t, typename Func>
__attribute__((sycl_kernel)) void kernel_convert(const Func &func) {
  double a[3] = {1, 2, 3};
  double b[3] = {1, 2, 3};
  int i = 1;
  func(a, b, i);
}

extern "C" {
// symbols so that linker find them and doesn't fail.
void __sycl_register_lib(void *) {}
void __sycl_unregister_lib(void *) {}
}

int main() {
  kernel_convert<class fake_kernel_convert>(
      [](double *a, double *b, int i) { b[i] = (double)((float)(a[i])); });
  return 0;
}
