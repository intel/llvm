// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// REQUIRES: cuda

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

int main() {
  sycl::queue Queue{sycl::default_selector{}};
  Queue.submit(
      [](sycl::handler &CGH) { CGH.single_task<class TestKernel>([] {}); });

  sycl::program Prog{Queue.get_context()};
  Prog.build_with_kernel_type<class TestKernel>();

  auto NativeProgram = Prog.get_native<sycl::backend::cuda>();

  assert(NativeProgram != 0);

  // TODO check program interop constructor, once it is available.

  return 0;
}
