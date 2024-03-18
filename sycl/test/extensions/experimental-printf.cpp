// This test is intended to check that internal
// __SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ works as expected, i.e. we can
// see printf ExtInst regardless of the macro presence and that argument
// promotion is disabled if the macro is present.
//
// RUN: %clangxx -fsycl -fsycl-device-only -fno-sycl-use-bitcode %s -o %t.spv
// RUN: llvm-spirv -to-text %t.spv -o %t.spt
// RUN: FileCheck %s --check-prefixes CHECK,CHECK-FLOAT < %t.spt
//
// RUN: %clangxx -fsycl -fsycl-device-only -fno-sycl-use-bitcode -D__SYCL_USE_VARIADIC_SPIRV_OCL_PRINTF__ %s -o %t.spv
// RUN: llvm-spirv -to-text %t.spv -o %t.spt
// RUN: FileCheck %s --check-prefixes CHECK,CHECK-DOUBLE < %t.spt

// CHECK-FLOAT: TypeFloat [[#TYPE:]] 32
// CHECK-DOUBLE: TypeFloat [[#TYPE:]] 64
// CHECK: Constant [[#TYPE]] [[#CONST:]]
// CHECK: ExtInst [[#]] [[#]] [[#]] printf [[#]] [[#CONST]]

#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif

const __SYCL_CONSTANT_AS char fmt[] = "Hello, World! %f\n";

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() {
      float f = 3.14;
      sycl::ext::oneapi::experimental::printf(fmt, f);
    });
  });

  return 0;
}
