// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// REQUIRES: level_zero

// Regression test to check that Level Zero backend doesn't fail when using a
// long kernel name.

#include <sycl/sycl.hpp>

int main() {
  cl::sycl::queue Q;
  {
    cl::sycl::program Program(Q.get_context());
    using SingleTask = class
        nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn;

    Program.build_with_kernel_type<SingleTask>();
    cl::sycl::kernel Kernel = Program.get_kernel<SingleTask>();

    Q.submit([&](cl::sycl::handler &CGH) {
      CGH.single_task<SingleTask>(Kernel, [=]() {});
    });
  }

  return 0;
}
