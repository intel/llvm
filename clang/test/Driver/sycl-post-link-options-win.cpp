// REQUIRES: system-windows
/// Verify same set of sycl-post-link options generated for old and new offloading model
// Test for JIT compilation
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --offload-new-driver \
// RUN:          -Xdevice-post-link -O0 -v %s 2>&1 \
// RUN:   | FileCheck -check-prefix OPTIONS_POSTLINK_JIT %s
// RUN: %clangxx --target=x86_64-pc-windows-msvc -fsycl --no-offload-new-driver \
// RUN:          -Xdevice-post-link -O0 -v %s 2>&1 \
// RUN:   | FileCheck -check-prefix OPTIONS_POSTLINK_JIT %s
// OPTIONS_POSTLINK_JIT: sycl-post-link{{.*}} -O0 -O2 -device-globals -spec-const=native -split=auto -emit-only-kernels-as-entry-points -emit-param-info -symbols -emit-exported-symbols -split-esimd -lower-esimd

#include <sycl/sycl.hpp>
using namespace sycl;

int main(void) {
  sycl::queue queue;
  sycl::event event = queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class set_range>(sycl::range<1>{16},
                                      [=](sycl::id<1> idx) {});
  });
  return 0;
}
