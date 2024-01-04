// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s --check-prefix=MLIR
#include <sycl/sycl.hpp>

void test_global_string(sycl::device d) {
  auto q = sycl::queue(d); 
  {
    q.submit([&] (sycl::handler& cgh) {
       cgh.single_task<class printkernel>([=] {
          // Test that the string is generated in the gpu module
          // MLIR: gpu.module @device_functions {
          // MLIR-NEXT: llvm.mlir.global internal constant @str0("Hello\00") {addr_space = 1 : i32, alignment = 1 : i64}
          char str1[10] = "Hello";
       });
    });
  }
}

