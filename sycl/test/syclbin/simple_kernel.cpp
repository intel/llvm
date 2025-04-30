// RUN: %clangxx -fsycl-device-only -Xclang -emit-llvm-bc -o %t.bc %s
// RUN: clang-offload-packager -o %t.out "--image=file=%t.bc,triple=spir64-unknown-unknown,arch=,kind=sycl,compile-opts="
// RUN: clang-linker-wrapper --syclbin --host-triple=x86_64-unknown-linux-gnu -sycl-device-libraries="libsycl-crt.new.o" -sycl-device-library-location=%sycl_libs_dir --sycl-post-link-options="-device-globals" --llvm-spirv-options=-spirv-max-version=1.4 -o %t.syclbin %t.out
// RUN: syclbin-dump %t.syclbin | FileCheck %s

// Checks the generated SYCLBIN contents of a simple SYCL free function kernel.

// TODO: Replace clang tooling invocation with -fsyclbin clang driver command
//       when available. Once this is in place, Windows should also be
//       supported.
// UNSUPPORTED: windows
// UNSUPPORTED-TRACKER: CMPLRLLVM-65259

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
