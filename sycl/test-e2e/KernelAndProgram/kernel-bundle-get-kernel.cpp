// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test checks whether the Kernel object returned from
// kernel_bundle::get_kernel<typename KernelName>() is the same as a Kernel
// object retrieved via other methods.

#include <sycl/detail/core.hpp>

class KernelA;

int main() {
  sycl::queue Queue;

  sycl::device Dev = Queue.get_device();

  sycl::context Ctx = Queue.get_context();

  Queue.submit([&](sycl::handler &CGH) { CGH.single_task<KernelA>([=]() {}); });

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
