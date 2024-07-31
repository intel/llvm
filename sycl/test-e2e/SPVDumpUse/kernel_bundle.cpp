// REQUIRES: opencl || level_zero
//
// SYCL_USE_KERNEL_SPV assumes no dead arguments elimination, need to produce
// SPV under the same conditions.
// RUN: %{build} -fno-sycl-dead-args-optimization -fno-sycl-instrument-device-code -DVALUE=1 -o %t.out
// RUN: %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefix ONE
// RUN: env SYCL_DUMP_IMAGES_PREFIX=%t.sycl_ SYCL_DUMP_IMAGES=1 %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefix ONE
// RUN: env SYCL_USE_KERNEL_SPV=%t.sycl_spir64.spv %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefix ONE
//
// This can perform dead arguments elimination, SYCL RT will ignore the
// ArgElimMask produced by the device compiler.
// RUN: %{build} -fno-sycl-instrument-device-code -DVALUE=2 -o %t.out
// RUN: %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefix TWO
// FIXME: SYCL_USE_KERNEL_SPV is ignored for kernel_bundles.
// RUN: env SYCL_USE_KERNEL_SPV=%t.sycl_spir64.spv %{run-unfiltered-devices} %t.out | FileCheck %s --check-prefix TWO
#include <sycl/detail/core.hpp>

using namespace sycl;

int main() {
  constexpr int N = 16;
  buffer<int> b(N);
  queue q;
  auto bundle = get_kernel_bundle<bundle_state::executable>(
      q.get_context(), {q.get_device()}, {get_kernel_id<class KernelName>()});
  q.submit([&](handler &cgh) {
    cgh.use_kernel_bundle(bundle);
    accessor acc{b, cgh};
    cgh.parallel_for<class KernelName>(nd_range<1>{N, N}, [=](nd_item<1> id) {
      acc[id.get_global_id(0)] = VALUE;
    });
  });

  // ONE: Result: 1
  // TWO: Result: 2
  std::cout << "Result: " << host_accessor{b}[0] << std::endl;

  return 0;
}
