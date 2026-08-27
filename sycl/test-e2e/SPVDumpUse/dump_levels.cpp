// REQUIRES: target-spir
//
// Check the two SYCL_DUMP_IMAGES levels. Device code split is disabled so that
// both kernels below live in a single device image.
// RUN: %{build} -fsycl-device-code-split=off -o %t.out
//
// Level 2 dumps an image when it is first used and reports the file it was
// dumped to. The second kernel uses the same image, so it is reported as
// already dumped instead of being dumped again.
// RUN: env SYCL_DUMP_IMAGES_PREFIX=%t.used_ SYCL_DUMP_IMAGES=2 %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefix USED
//
// USED: SYCL_DUMP_IMAGES: dumped device image to "{{.*}}used_spir64_1.spv"
// USED: SYCL_DUMP_IMAGES: device image already dumped to "{{.*}}_1.spv"
//
// Level 1 dumps all images when they are loaded, without any reporting.
// RUN: env SYCL_DUMP_IMAGES_PREFIX=%t.all_ SYCL_DUMP_IMAGES=1 %{run-unfiltered-devices} %t.out 2>&1 | FileCheck %s --check-prefix ALL --allow-empty
//
// ALL-NOT: SYCL_DUMP_IMAGES:

#include <cassert>
#include <sycl/detail/core.hpp>

using namespace sycl;

class KernelA;
class KernelB;

int main() {
  constexpr int N = 16;
  buffer<int> B(N);
  queue Q;

  Q.submit([&](handler &CGH) {
    accessor Acc{B, CGH};
    CGH.parallel_for<KernelA>(N, [=](id<1> I) { Acc[I] = 1; });
  });

  // A second kernel from the same device image must not dump it again.
  Q.submit([&](handler &CGH) {
    accessor Acc{B, CGH};
    CGH.parallel_for<KernelB>(N, [=](id<1> I) { Acc[I] += 1; });
  });

  host_accessor HostAcc{B};
  assert(HostAcc[0] == 2 && "Kernels did not execute");
  return 0;
}
