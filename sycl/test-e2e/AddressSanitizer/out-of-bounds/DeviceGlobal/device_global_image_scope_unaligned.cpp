// REQUIRES: linux
// RUN: %{build} %device_asan_flags -O0 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t.out
// RUN: env SYCL_PREFER_UR=1 %{run} not %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/device_global/device_global.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

sycl::ext::oneapi::experimental::device_global<
    char[5], decltype(properties(device_image_scope, host_access_read_write))>
    dev_global;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
     h.single_task<class Test>([=]() { dev_global[8] = 42; });
     // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device Global
     // CHECK: {{WRITE of size 1 at kernel <.*Test> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
     // CHECK: {{#0 .* .*device_global_image_scope_unaligned.cpp:}}[[@LINE-3]]
   }).wait();

  char val;
  Q.copy(dev_global, &val).wait();

  return 0;
}
