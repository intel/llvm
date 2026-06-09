// RUN: %{build} -DMAIN -c -o %t.main.o
// RUN: %{build} -c -o %t.kernel.o
// RUN: %clangxx -fsycl %{sycl_target_opts} %t.main.o %t.kernel.o -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

class KernelA;
void submitKernel(sycl::queue &Q);
#ifndef MAIN
void submitKernel(sycl::queue &Q) {
  Q.submit([&](sycl::handler &CGH) { CGH.single_task<KernelA>([=]() {}); });
}
#else
int main() {
  sycl::queue Queue;
  sycl::device Dev = Queue.get_device();
  sycl::context Ctx = Queue.get_context();
  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Ctx, {Dev});

  auto FoundKernel = KernelBundle.get_kernel<KernelA>();
  auto ExpectedKernel = KernelBundle.get_kernel(sycl::get_kernel_id<KernelA>());

  // operator== isn't guaranteed to work in this scenario, so compare traits
  // about the kernels instead.
  auto FoundKernelName =
      FoundKernel.get_info<sycl::info::kernel::function_name>();
  auto ExpectedKernelName =
      ExpectedKernel.get_info<sycl::info::kernel::function_name>();

  auto FoundKernelContext = FoundKernel.get_info<sycl::info::kernel::context>();
  auto ExpectedKernelContext =
      ExpectedKernel.get_info<sycl::info::kernel::context>();

  assert(FoundKernelName == ExpectedKernelName);
  assert(FoundKernelContext == ExpectedKernelContext);

  return 0;
}
#endif
