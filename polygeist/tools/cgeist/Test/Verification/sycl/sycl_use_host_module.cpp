// RUN: clang++ -fsycl -fsycl-device-only -Xcgeist -sycl-use-host-module=%S/host_module.mlir -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK:        gpu.module @device_functions
// CHECK:        memref.global constant @c : memref<i32, 1> {alignment = 4 : i64}
// CHECK-NEXT:   func.func @do_nothing() -> memref<i32, 1> {
// CHECK-NEXT:     %0 = memref.get_global @c : memref<i32, 1>
// CHECK-NEXT:     return %0 : memref<i32, 1>
// CHECK-NEXT:   }
// Check that the function is at the end of the top-level module
// CHECK-NEXT: }
// CHECK-NOT:  {{.+}}
SYCL_EXTERNAL void do_nothing() {}
