// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=GPU %t.out
// REQUIRES: cuda

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

int main() {
  sycl::queue Queue{sycl::default_selector{}};
  Queue.submit(
      [](sycl::handler &CGH) { CGH.single_task<class TestKernel>([] {}); });

  sycl::program Prog{Queue.get_context()};

  auto NativeProgram = Prog.get_native<sycl::backend::cuda>();

  assert(NativeProgram == 0 && "CUmodule is zero");

  // TODO check program interop constructor, once it is available.

  return 0;
}
