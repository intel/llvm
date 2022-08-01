// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// REQUIRES: cuda

#include <sycl/sycl.hpp>

int main() {
  sycl::queue Queue{sycl::default_selector{}};
  Queue.submit(
      [](sycl::handler &CGH) { CGH.single_task<class TestKernel>([] {}); });

  sycl::program Prog{Queue.get_context()};
  Prog.build_with_kernel_type<class TestKernel>();

  auto NativeProgram = sycl::get_native<sycl::backend::ext_oneapi_cuda>(Prog);

  assert(NativeProgram != 0);

  // TODO check program interop constructor, once it is available.

  return 0;
}
