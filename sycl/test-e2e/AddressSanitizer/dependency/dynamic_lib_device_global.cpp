// REQUIRES: linux, cpu || (gpu && level_zero)
// RUN: rm -rf %t.dir; mkdir -p %t.dir
// RUN: %{build} %device_asan_flags -O0 -g -fPIC -shared -fsycl-allow-device-image-dependencies -DBUILD_LIB -o %t.dir/libdevice_oob.so
// RUN: %{build} %device_asan_flags -O0 -g -fsycl-allow-device-image-dependencies -o %t.out -L%t.dir -ldevice_oob -Wl,-rpath=%t.dir
// RUN: %{run} not --crash %t.out 2>&1 | FileCheck %s
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/device_global/device_global.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

#ifdef BUILD_LIB

sycl::ext::oneapi::experimental::device_global<
    int[4], decltype(properties(device_image_scope, host_access_read_write))>
    LibDeviceGlobal;

SYCL_EXTERNAL void touchLibDeviceGlobal() {
  LibDeviceGlobal[4] = 42;
  // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device Global
  // CHECK: {{WRITE of size 4 at kernel <.*MainKernel> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
  // CHECK: {{#0 .* .*dynamic_lib_device_global.cpp:}}[[@LINE-3]]
}

#else

extern SYCL_EXTERNAL void touchLibDeviceGlobal();

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
     h.single_task<class MainKernel>([=]() { touchLibDeviceGlobal(); });
   }).wait();

  return 0;
}

#endif
