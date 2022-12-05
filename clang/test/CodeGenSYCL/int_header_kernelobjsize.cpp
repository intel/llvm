// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsycl-int-header=%t.h %s
// RUN: FileCheck -input-file=%t.h %s

// This test checks that the getKernelSize() member function is
// generated into the integration header and that it returns the
// size of the kernel object in bytes.

#include "sycl.hpp"

using namespace sycl;

void testA() {
  queue q;
  constexpr int N = 256;
  int A[N] = {10};
  q.submit([&](handler &h) {
    h.single_task<class KernelName>([=]() {
      for (int k = 0; k < N; ++k) {
        (void)A[k];
      }
    });
  });
}
// CHECK: template <> struct KernelInfo<KernelName> {
// CHECK: // Returns the size of the kernel object in bytes.
// CHECK: static constexpr long{{.*}} getKernelSize() { return 1024; }
