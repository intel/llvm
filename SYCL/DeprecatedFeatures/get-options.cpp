// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// This test passes OpenCL specific compiler and linker swiches to the backend,
// so it is unsupported on any other backend.
// UNSUPPORTED: cuda || hip || level_zero

#include <CL/sycl.hpp>

#include <cassert>

// Check program::get_compile/link/build_options functions

class KernelName;
void submitKernel() {
  cl::sycl::queue q;
  q.submit(
      [&](cl::sycl::handler &cgh) { cgh.single_task<KernelName>([]() {}); });
}

int main() {
  const std::string CompileOpts{"-cl-opt-disable"};
  const std::string LinkOpts{"-cl-fast-relaxed-math"};
  const std::string BuildOpts{"-cl-opt-disable -cl-fast-relaxed-math"};

  cl::sycl::context Ctx;
  cl::sycl::program PrgA{Ctx};
  assert(PrgA.get_compile_options().empty());
  assert(PrgA.get_link_options().empty());
  assert(PrgA.get_build_options().empty());

  PrgA.build_with_kernel_type<KernelName>(BuildOpts);
  assert(PrgA.get_compile_options().empty());
  assert(PrgA.get_link_options().empty());
  assert(PrgA.get_build_options() == (PrgA.is_host() ? "" : BuildOpts));

  cl::sycl::program PrgB{Ctx};
  PrgB.compile_with_kernel_type<KernelName>(CompileOpts);
  assert(PrgB.get_compile_options() == (PrgB.is_host() ? "" : CompileOpts));
  assert(PrgB.get_link_options().empty());
  assert(PrgB.get_build_options() == (PrgB.is_host() ? "" : CompileOpts));

  PrgB.link(LinkOpts);
  assert(PrgB.get_compile_options() == (PrgB.is_host() ? "" : CompileOpts));
  assert(PrgB.get_link_options() == (PrgB.is_host() ? "" : LinkOpts));
  assert(PrgB.get_build_options() == (PrgB.is_host() ? "" : LinkOpts));
}
