// REQUIRES: linux, cpu
// RUN: %{build} %device_sanitizer_flags -O2 -g -DUSER_CODE_1 -c -o %t1.o
// RUN: %{build} %device_sanitizer_flags -O2 -g -DUSER_CODE_2 -c -o %t2.o
// RUN: %clangxx -fsycl %device_sanitizer_flags -O2 -g %t1.o %t2.o -o %t.out
// RUN: env SYCL_PREFER_UR=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu %{run-unfiltered-devices} not %t.out 2>&1 | FileCheck %s

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/device_global/device_global.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

#ifdef USER_CODE_1

sycl::ext::oneapi::experimental::device_global<
    int[4], decltype(properties(device_image_scope, host_access_read_write))>
    dev_global2;

void foo() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Test2>([=]() {
      dev_global2[4] = 42;
      // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device Global
      // CHECK: {{WRITE of size 4 at kernel <.*Test2> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
      // CHECK: {{#0 .* .*multi_device_images.cpp:}}[[@LINE-3]]
    });
  }).wait();
}

#else

sycl::ext::oneapi::experimental::device_global<
    int, decltype(properties(device_image_scope, host_access_read_write))>
    dev_global;

extern void foo();

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Test1>([=]() {
      dev_global = 42;
    });
  }).wait();

  foo();

  return 0;
}

#endif
