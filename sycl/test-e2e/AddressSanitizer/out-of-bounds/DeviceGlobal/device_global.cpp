// REQUIRES: linux, cpu || (gpu && level_zero)

// The following is an ugly hack to make CI pass. The issue here is that when
// sycl-toolchain is built with assertions enabled, then we hit one at
// `DeviceGlobalUSMMem::~DeviceGlobalUSMMem()` and exit with abort. If not, then
// sanitizer does `exit(1)`.

// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %{run} not %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t2.out
// RUN: %{run} not %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t3.out
// RUN: %{run} not %t3.out 2>&1 | FileCheck %s

// Flakily timesout on PVC
// UNSUPPORTED: arch-intel_gpu_pvc
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16401

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/device_global/device_global.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi;
using namespace sycl::ext::oneapi::experimental;

sycl::ext::oneapi::experimental::device_global<char[5]> dev_global;

int main() {
  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
     h.single_task<class Test>([=]() { dev_global[8] = 42; });
     // CHECK: ERROR: DeviceSanitizer: out-of-bounds-access on Device USM
     // CHECK: {{WRITE of size 1 at kernel <.*Test> LID\(0, 0, 0\) GID\(0, 0, 0\)}}
     // CHECK: {{#0 .* .*device_global.cpp:}}[[@LINE-3]]
   }).wait();

  return 0;
}
