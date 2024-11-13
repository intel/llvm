// REQUIRES: (level_zero || opencl) && linux && gpu

// RUN: %{build} -o %t.out
// RUN: rm -rf %t/cache_dir
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=1 SYCL_CACHE_DIR=%t/cache_dir env -u XDG_CACHE_HOME env -u HOME %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-BUILD
// RUN: env NEOReadDebugKeys=1 CreateMultipleRootDevices=3 SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_TRACE=1 SYCL_CACHE_DIR=%t/cache_dir env -u XDG_CACHE_HOME env -u HOME %{run} %t.out 2>&1 | FileCheck %s --check-prefixes=CHECK-CACHE

// Test checks that persistent cache works correctly with multiple devices.

#include <sycl/detail/core.hpp>

using namespace sycl;

class SimpleKernel;

int main(void) {
  platform plt;
  auto devs = plt.get_devices();
  context ctx(devs);
  assert(devs.size() >= 3);

  constexpr size_t sz = 1024;
  sycl::buffer<int, 1> bufA(sz);
  auto bundle = sycl::get_kernel_bundle<bundle_state::input>(ctx);
  // CHECK-BUILD: [Persistent Cache]: device binary has been cached
  // CHECK-CACHE: [Persistent Cache]: using cached device binary
  auto bundle_exe = sycl::build(bundle, {devs[0], devs[2]});
  auto kernel = bundle_exe.get_kernel(sycl::get_kernel_id<SimpleKernel>());
  sycl::queue q(devs[2]);
  q.submit([&](sycl::handler &cgh) {
    sycl::accessor accA(bufA, cgh, sycl::write_only);
    cgh.parallel_for<SimpleKernel>(sycl::range<1>(sz), [=](sycl::item<1> item) {
      accA[item] = item.get_linear_id();
    });
  });
  q.wait();
  return 0;
}
