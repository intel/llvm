// RUN: %clangxx --offload-new-driver -fsyclbin -o %t.syclbin %s
// RUN: syclbin-dump %t.syclbin | FileCheck %s

// Checks the generated SYCLBIN contents of a simple SYCL free function kernel.

#include <sycl/sycl.hpp>

extern "C" {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
}
}

// CHECK:      Version: {{[1-9]+}}
// CHECK-NEXT: Global metadata:
// CHECK-NEXT:   SYCLBIN/global metadata:
// CHECK-NEXT:     state: 0
// CHECK-NEXT: Number of Abstract Modules: 1
// CHECK-NEXT: Abstract Module 0:
// CHECK-NEXT:   Metadata:
// CHECK:      Number of IR Modules: 1
// CHECK-NEXT:   IR module 0:
// CHECK-NEXT:       Metadata:
// CHECK-NEXT:         SYCLBIN/ir module metadata:
// CHECK-NEXT:           type: 0
// CHECK-NEXT:     Raw IR bytes: <Binary blob of {{.*}} bytes>
// CHECK-NEXT:   Number of Native Device Code Images: 0
