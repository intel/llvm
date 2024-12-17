// REQUIRES: gpu && linux && (opencl || level_zero)

// Test to check that we can create input kernel bundle and call build twice for
// overlapping set of devices and execute the kernel on each device.

// RUN: %{build} -o %t.out
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

// XFAIL: arch-intel_gpu_pvc
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16401

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
  // CHECK: urProgramCreateWithIL(
  // CHECK: urProgramBuildExp(
  auto KernelBundleExe1 = build(kernelBundleInput, {dev1, dev2});
  // CHECK: urProgramCreateWithIL(
  // CHECK: urProgramBuildExp(
  auto KernelBundleExe2 = build(kernelBundleInput, {dev2, dev3});
  // No other program creation calls are expected.
  // CHECK-NOT: urProgramCreateWithIL(
  auto KernelObj1 = KernelBundleExe1.get_kernel(kid);
  auto KernelObj2 = KernelBundleExe2.get_kernel(kid);
  queues[0].submit([=](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe1);
    cgh.single_task<Kernel>([=]() {});
  });
  queues[1].submit([=](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe1);
    cgh.single_task(KernelObj1);
  });
  queues[1].submit([=](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe2);
    cgh.single_task(KernelObj2);
  });
  queues[2].submit([=](sycl::handler &cgh) {
    cgh.use_kernel_bundle(KernelBundleExe2);
    cgh.single_task(KernelObj2);
  });
  return 0;
}
