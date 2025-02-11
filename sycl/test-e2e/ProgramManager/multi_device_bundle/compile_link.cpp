// REQUIRES: gpu && linux && (opencl || level_zero)

// RUN: %{build} -o %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 %{run} %t.out

// Test to check that we can compile and link a kernel bundle for multiple
// devices and run the kernel on each device.
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>

class Kernel;

int main() {
  sycl::platform platform;
  auto devices = platform.get_devices();
  if (!(devices.size() >= 3))
    return 0;
  auto dev1 = devices[0], dev2 = devices[1], dev3 = devices[2];

  auto ctx = sycl::context({dev1, dev2, dev3});
  sycl::queue queues[3] = {sycl::queue(ctx, dev1), sycl::queue(ctx, dev2),
                           sycl::queue(ctx, dev3)};
  sycl::kernel_id kid = sycl::get_kernel_id<Kernel>();
  sycl::kernel_bundle kernelBundleInput =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(ctx, {kid});
  auto KernelBundleCompiled = compile(kernelBundleInput, {dev1, dev2, dev3});
  auto KernelBundleLinked = link(KernelBundleCompiled, {dev1, dev2, dev3});
  for (int i = 0; i < 3; i++) {
    queues[i].submit([=](sycl::handler &cgh) {
      cgh.use_kernel_bundle(KernelBundleLinked);
      cgh.single_task<Kernel>([=]() {});
    });
    queues[i].wait();
  }
  return 0;
}
