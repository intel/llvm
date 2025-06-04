// RUN: %clangxx --offload-new-driver -fsyclbin=input -o %t.input.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=object -o %t.object.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=executable -o %t.executable.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin -o %t.default.syclbin %s
// RUN: syclbin-dump %t.input.syclbin | FileCheck %s --check-prefix CHECK-INPUT
// RUN: syclbin-dump %t.object.syclbin | FileCheck %s --check-prefix CHECK-OBJECT
// RUN: syclbin-dump %t.executable.syclbin | FileCheck %s --check-prefix CHECK-EXECUTABLE
// RUN: syclbin-dump %t.default.syclbin | FileCheck %s --check-prefix CHECK-EXECUTABLE

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

// CHECK-INPUT:      Version: {{[1-9]+}}
// CHECK-INPUT-NEXT: Global metadata:
// CHECK-INPUT-NEXT:   SYCLBIN/global metadata:
// CHECK-INPUT-NEXT:     state: 0
// CHECK-INPUT-NEXT: Number of Abstract Modules: 1
// CHECK-INPUT-NEXT: Abstract Module 0:
// CHECK-INPUT-NEXT:   Metadata:
// CHECK-INPUT:      Number of IR Modules: 1
// CHECK-INPUT-NEXT:   IR module 0:
// CHECK-INPUT-NEXT:       Metadata:
// CHECK-INPUT-NEXT:         SYCLBIN/ir module metadata:
// CHECK-INPUT-NEXT:           type: 0
// CHECK-INPUT-NEXT:     Raw IR bytes: <Binary blob of {{.*}} bytes>
// CHECK-INPUT-NEXT:   Number of Native Device Code Images: 0

// CHECK-OBJECT:      Version: {{[1-9]+}}
// CHECK-OBJECT-NEXT: Global metadata:
// CHECK-OBJECT-NEXT:   SYCLBIN/global metadata:
// CHECK-OBJECT-NEXT:     state: 1
// CHECK-OBJECT-NEXT: Number of Abstract Modules: 1
// CHECK-OBJECT-NEXT: Abstract Module 0:
// CHECK-OBJECT-NEXT:   Metadata:
// CHECK-OBJECT:      Number of IR Modules: 1
// CHECK-OBJECT-NEXT:   IR module 0:
// CHECK-OBJECT-NEXT:       Metadata:
// CHECK-OBJECT-NEXT:         SYCLBIN/ir module metadata:
// CHECK-OBJECT-NEXT:           type: 0
// CHECK-OBJECT-NEXT:     Raw IR bytes: <Binary blob of {{.*}} bytes>
// CHECK-OBJECT-NEXT:   Number of Native Device Code Images: 0

// CHECK-EXECUTABLE:      Version: {{[1-9]+}}
// CHECK-EXECUTABLE-NEXT: Global metadata:
// CHECK-EXECUTABLE-NEXT:   SYCLBIN/global metadata:
// CHECK-EXECUTABLE-NEXT:     state: 2
// CHECK-EXECUTABLE-NEXT: Number of Abstract Modules: 1
// CHECK-EXECUTABLE-NEXT: Abstract Module 0:
// CHECK-EXECUTABLE-NEXT:   Metadata:
// CHECK-EXECUTABLE:      Number of IR Modules: 1
// CHECK-EXECUTABLE-NEXT:   IR module 0:
// CHECK-EXECUTABLE-NEXT:       Metadata:
// CHECK-EXECUTABLE-NEXT:         SYCLBIN/ir module metadata:
// CHECK-EXECUTABLE-NEXT:           type: 0
// CHECK-EXECUTABLE-NEXT:     Raw IR bytes: <Binary blob of {{.*}} bytes>
// CHECK-EXECUTABLE-NEXT:   Number of Native Device Code Images: 0
