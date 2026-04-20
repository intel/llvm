// RUN: %clangxx --offload-new-driver -fsyclbin=input -o %t.input.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=object -o %t.object.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin=executable -o %t.executable.syclbin %s
// RUN: %clangxx --offload-new-driver -fsyclbin -o %t.default.syclbin %s
// RUN: syclbin-dump %t.input.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-INPUT
// RUN: syclbin-dump %t.object.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-OBJECT
// RUN: syclbin-dump %t.executable.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-EXECUTABLE
// RUN: syclbin-dump %t.default.syclbin | FileCheck %s --check-prefixes=CHECK-GENERAL,CHECK-EXECUTABLE

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

// CHECK-GENERAL:      Global metadata:
// CHECK-GENERAL-NEXT:   SYCLBIN/global metadata:
// CHECK-INPUT:            state: 0
// CHECK-OBJECT:           state: 1
// CHECK-EXECUTABLE:       state: 2
// CHECK-GENERAL:          abstract_modules_num: 1
// CHECK-GENERAL-NEXT: Number of Abstract Modules: 1
// CHECK-GENERAL-NEXT: Abstract Module 0:
// CHECK-GENERAL-NEXT: Metadata:
// CHECK-GENERAL-NEXT:   SYCL/device requirements:
// CHECK-GENERAL-NEXT:     aspects:
// CHECK-GENERAL-NEXT:   SYCL/kernel names:
// CHECK-GENERAL-NEXT:     __sycl_kernel_TestKernel: 1
// CHECK-GENERAL-NEXT:   SYCL/misc properties:
// CHECK-GENERAL-NEXT:     optLevel: 2
// CHECK-GENERAL-NEXT:  Number of IR Modules: 1
// CHECK-GENERAL-NEXT:  IR module 0:
// CHECK-GENERAL-NEXT:    Image Kind: spv
// CHECK-GENERAL-NEXT:    Triple: spir64-unknown-unknown
// CHECK-GENERAL-NEXT:    Arch:
// CHECK-GENERAL-NEXT:    Raw bytes: <Binary blob of {{.*}} bytes>
// CHECK-GENERAL-NEXT:  Number of Native Device Code Images: 0
