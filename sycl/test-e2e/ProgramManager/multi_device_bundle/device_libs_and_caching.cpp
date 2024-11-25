// REQUIRES: ocloc && gpu && linux && (opencl || level_zero)

// Test to check several use cases for multi-device kernel bundles.
// Test covers AOT and JIT cases. Kernel is using some math functions to enforce
// using device libraries to excersise additional logic in the program manager.
// Checks are used to test that program and device libraries caching works as
// expected.

// Test JIT first.
// Intentionally use jit linking of device libraries to check that program
// manager can handle this as well. With this option program manager will
// compile the main program, load and compile device libraries and then link
// everything together.
// RUN: %{build} -fsycl-device-lib-jit-link -o %t.out

// Check the default case when in-memory caching is enabled.
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=4 SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-SPIRV-JIT-LINK-TRACE

// Check the case when in-memory caching of the programs is disabled.
// RUN: env SYCL_CACHE_IN_MEM=0 NEOReadDebugKeys=1 CreateMultipleRootDevices=4 %{run} %t.out

// Test AOT next.
// RUN: %{build} -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device *" -o %t.out

// Check the default case when in-memory caching is enabled.
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=4 SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s --check-prefixes=CHECK-AOT-TRACE

// Check the case when in-memory caching of the programs is disabled.
// RUN: env SYCL_CACHE_IN_MEM=0 NEOReadDebugKeys=1 CreateMultipleRootDevices=4 %{run} %t.out

#include <cmath>
#include <complex>
#include <sycl/detail/core.hpp>
#include <sycl/ext/intel/math.hpp>
#include <sycl/usm.hpp>

class Kernel;
class Kernel2;
class Kernel3;

int main() {
  sycl::platform platform;
  auto devices = platform.get_devices();
  if (!(devices.size() >= 4))
    return 0;
  auto dev1 = devices[0], dev2 = devices[1], dev3 = devices[2],
       dev4 = devices[3];
  auto ctx = sycl::context({dev1, dev2, dev3, dev4});
  sycl::queue queues[4] = {sycl::queue(ctx, dev1), sycl::queue(ctx, dev2),
                           sycl::queue(ctx, dev3), sycl::queue(ctx, dev4)};

  auto res = sycl::malloc_host<int>(3, ctx);
  auto KernelLambda = [=]() {
    res[0] = sycl::ext::intel::math::float2int_rd(4.0) + (int)sqrtf(4.0f) +
             std::exp(std::complex<float>(0.f, 0.f)).real();
  };
  // Test case 1
  // Get bundle in executable state for multiple devices in a context, enqueue a
  // kernel to each device.
  {
    sycl::kernel_id kid = sycl::get_kernel_id<Kernel>();
    // Create the main program containing the kernel.
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCreateWithIL(

    // Create and compile the program for required device libraries (2 of them
    // in this case).
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCreateWithIL(
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCompileExp(
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCreateWithIL(
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCompileExp(

    // Compile the main program
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCompileExp(

    // Link main program and device libraries.
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramLinkExp(

    // CHECK-AOT-TRACE: urProgramCreateWithBinary(
    // CHECK-AOT-TRACE: urProgramBuildExp(
    sycl::kernel_bundle kernelBundleExecutable =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev1, dev2, dev3}, {kid});

    for (int i = 0; i < 3; i++) {
      queues[i].submit([=](sycl::handler &cgh) {
        cgh.use_kernel_bundle(kernelBundleExecutable);
        cgh.single_task<Kernel>(KernelLambda);
      });
      queues[i].wait();
    }
    std::cout << "Test #1 passed." << std::endl;
  }

  // Test case 2
  // Get two bundles in executable state: for the first two devices in the
  // context and for the new set of devices which includes the dev4. This checks
  // caching of the programs and device libraries.
  {
    sycl::kernel_id kid = sycl::get_kernel_id<Kernel>();
    // Program associated with {dev1, dev2, dev3} is supposed to be cached from
    // the first test case, we don't expect any additional program creation and
    // compilation calls for the following bundles because they are all created
    // for subsets of {dev1, dev2, dev3} which means that the program handle
    // from cache will be used.
    sycl::kernel_bundle kernelBundleExecutableSubset1 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev1, dev2}, {kid});
    sycl::kernel_bundle kernelBundleExecutableSubset2 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev2, dev3}, {kid});
    sycl::kernel_bundle kernelBundleExecutableSubset3 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev1, dev3}, {kid});
    sycl::kernel_bundle kernelBundleExecutableSubset4 =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx, {dev3},
                                                                {kid});

    // Here we create a bundle with a different set of devices which includes
    // dev4, so we expect new UR program creation.
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCreateWithIL(

    // Device libraries will be additionally compiled for dev4, but no program
    // creation is expected for device libraries as program handle already
    // exists in the per-context cache.
    // CHECK-SPIRV-JIT-LINK-TRACE-NOT: urProgramCreateWithIL(
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCompileExp(

    // Main program will be compiled for new set of devices.
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramCompileExp(

    // Main program will be linked with device libraries.
    // CHECK-SPIRV-JIT-LINK-TRACE: urProgramLinkExp(

    // CHECK-AOT-TRACE: urProgramCreateWithBinary(
    // CHECK-AOT-TRACE: urProgramBuildExp(
    sycl::kernel_bundle kernelBundleExecutableNewSet =
        sycl::get_kernel_bundle<sycl::bundle_state::executable>(
            ctx, {dev2, dev3, dev4}, {kid});

    for (int i = 0; i < 3; i++) {
      queues[0].submit([=](sycl::handler &cgh) {
        cgh.use_kernel_bundle(kernelBundleExecutableSubset1);
        cgh.single_task<Kernel>(KernelLambda);
      });
      queues[0].wait();

      queues[2].submit([=](sycl::handler &cgh) {
        cgh.use_kernel_bundle(kernelBundleExecutableNewSet);
        cgh.single_task<Kernel>(KernelLambda);
      });
      queues[2].wait();
    }
    std::cout << "Test #2 passed." << std::endl;
  }
  return 0;
}
