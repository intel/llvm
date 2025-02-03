// REQUIRES: linux, cpu || (gpu && level_zero)

// The following is an ugly hack to make CI pass. The issue here is that when
// sycl-toolchain is built with assertions enabled, then we hit one at
// `DeviceGlobalUSMMem::~DeviceGlobalUSMMem()` and exit with abort. If not, then
// sanitizer does `exit(1)`.
//
// That doesn't matter as long as LIT does *not* use "external shell", but when
// it does, the difference between `not` and `not --crash` becomes crucial and
// we cannot have a single command that would match both behaviors. Ideally,
// we'd need to make a choice based on how the toolchain was built.
// Unfortunately, we don't have that information at the LIT level and
// propagating it all the way down to it would be ugly. Instead, rely on the
// fact that "no-assertion" mode doesn't use "run-only" lit invocations and make
// a choice based on that. This is rather fragile but workarounds the issue for
// the time being.

// DEFINE: %{not} = not %if test-mode-run-only %{ --crash %}

// RUN: %{build} %device_asan_flags -O0 -g -o %t1.out
// RUN: %{run} %{not} %t1.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O1 -g -o %t2.out
// RUN: %{run} %{not} %t2.out 2>&1 | FileCheck %s
// RUN: %{build} %device_asan_flags -O2 -g -o %t3.out
// RUN: %{run} %{not} %t3.out 2>&1 | FileCheck %s

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
