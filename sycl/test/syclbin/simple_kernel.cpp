// RUN: %clangxx -fsycl-device-only -Xclang -emit-llvm-bc -o %t.bc %s
// RUN: clang-offload-packager -o %t.out "--image=file=%t.bc,triple=spir64-unknown-unknown,arch=,kind=sycl,compile-opts="
// RUN: clang-linker-wrapper --syclbin --host-triple=x86_64-unknown-linux-gnu -sycl-device-libraries="libsycl-crt.new.o,libsycl-complex.new.o,libsycl-complex-fp64.new.o,libsycl-cmath.new.o,libsycl-cmath-fp64.new.o,libsycl-imf.new.o,libsycl-imf-fp64.new.o,libsycl-imf-bf16.new.o,libsycl-fallback-cassert.new.o,libsycl-fallback-cstring.new.o,libsycl-fallback-complex.new.o,libsycl-fallback-complex-fp64.new.o,libsycl-fallback-cmath.new.o,libsycl-fallback-cmath-fp64.new.o,libsycl-fallback-imf.new.o,libsycl-fallback-imf-fp64.new.o,libsycl-fallback-imf-bf16.new.o,libsycl-itt-user-wrappers.new.o,libsycl-itt-compiler-wrappers.new.o,libsycl-itt-stubs.new.o" -sycl-device-library-location=%sycl_libs_dir --sycl-post-link-options="-device-globals" --llvm-spirv-options=-spirv-max-version=1.4 -o %t.syclbin %t.out
// RUN: syclbin-dump %t.syclbin | FileCheck %s

// Checks the generated SYCLBIN contents of a simple SYCL free function kernel.

// TODO: Replace clang tooling invocation with -fsyclbin clang driver command
//       when available.

#include <sycl/sycl.hpp>

extern "C" {
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (sycl::ext::oneapi::experimental::single_task_kernel))
void TestKernel(int *Ptr, int Size) {
  for (size_t I = 0; I < Size; ++I)
    Ptr[I] = I;
}
}

// CHECK:      Version:                    {{[1-9]+}}
// CHECK-NEXT: State:                      input
// CHECK-NEXT: Number of Abstract Modules: 1
// CHECK-NEXT: Abstract Module 0:
// CHECK-NEXT:   Metadata:
// CHECK-NEXT:     Kernel names:
// CHECK-NEXT:       __sycl_kernel_TestKernel
// CHECK-NEXT:     Imported symbols:
// CHECK-NEXT:     Exported symbols:
// CHECK-NEXT:     Properties: <Binary blob of {{.*}} bytes>
// CHECK-NEXT:   Number of IR Modules: 1
// CHECK-NEXT:   IR module 0:
// CHECK-NEXT:     IR type: SPIRV
// CHECK-NEXT:     Raw IR bytes: <Binary blob of {{.*}} bytes>
// CHECK-NEXT:   Number of Native Device Code Images: 0
