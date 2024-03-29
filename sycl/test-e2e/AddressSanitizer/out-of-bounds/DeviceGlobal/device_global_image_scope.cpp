// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -O2 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/device_global/device_global.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

sycl::ext::oneapi::experimental::device_global<
    int[4], decltype(properties(device_image_scope, host_access_read_write))>
    dev_global;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Test>([=]() {
      dev_global[4] = 42;
      // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device Global
      // CHECK: {{WRITE of size 4 at kernel <.*Test> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
      // CHECK: {{#0 .* .*device_global_image_scope.cpp:}}[[@LINE-3]]
    });
  }).wait();

  int val;
  Q.copy(dev_global, &val).wait();

  return 0;
}
