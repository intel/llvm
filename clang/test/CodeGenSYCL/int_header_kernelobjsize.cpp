// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks that the getSize() member function is generated
// into the integration header and that it returns the size of the
// kernel object.

#include "sycl.hpp"

using namespace cl::sycl;

void testA() {
  queue q;
  constexpr int N = 256;
  int A[N] = {10};
  kernel_single_task<class KernelName>([=]() {
    for (int k = 0; k < N; ++k) {
      (void)A[k];
    }
  });
}
// CHECK: template <> struct KernelInfo<KernelName> {
// CHECK:   static constexpr unsigned long getSize() { return 1024; }
