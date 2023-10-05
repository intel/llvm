// RUN: clang++ -fsycl -fsycl-device-only -Xcgeist \
// RUN:     -sycl-use-host-module=%S/host_module.mlir -O0 -w -emit-mlir -o - %s \
// RUN:     -Xcgeist -sycl-device-only=false | FileCheck %s
// RUN: clang++ -fsycl -fsycl-device-only -Xcgeist \
// RUN:     -sycl-use-host-module=%S/host_module.mlir -O0 -w -emit-mlir -o - %s \
// RUN:     | FileCheck %s --check-prefix=CHECK-DROP
// RUN: clang++ -fsycl -fsycl-device-only -Xcgeist \
// RUN:     -sycl-use-host-module=%S/host_module.mlir -O0 -w -emit-llvm -o - %s \
// RUN:     -Xcgeist -sycl-device-only=false | FileCheck %s --check-prefix=CHECK-LLVM
// RUN: clang++ -fsycl -fsycl-device-only -Xcgeist \
// RUN:     -sycl-use-host-module=%S/host_module.mlir -O0 -w -emit-llvm -o - %s \
// RUN:     | FileCheck %s --check-prefix=CHECK-LLVM

#include <sycl/sycl.hpp>

#include "nop_kernel.hpp"

// CHECK:        gpu.module @device_functions
// CHECK:        memref.global constant @c : memref<i32, 1> {alignment = 4 : i64}
// CHECK-NEXT:   func.func @host_foo() -> memref<i32, 1> {
// CHECK-NEXT:     %0 = memref.get_global @c : memref<i32, 1>
// CHECK-NEXT:     return %0 : memref<i32, 1>
// CHECK-NEXT:   }
// Check that the function is at the end of the top-level module
// CHECK-NEXT: }
// CHECK-NOT:  {{.+}}

// CHECK-DROP:     gpu.module @device_functions
// CHECK-DROP-NOT: memref.global constant @c : memref<i32, 1> {alignment = 4 : i64}
// CHECK-DROP-NOT: func.func @host_foo() -> memref<i32, 1> {

// CHECK-LLVM-NOT: host_foo
