// RUN: clang++ -fsycl -fsycl-targets=spir64-unknown-unknown-syclmlir -emit-mlir -O0 %s -o - -S -emit-mlir -fsycl-device-only | FileCheck %s

#include <sycl/sycl.hpp>

namespace NS {
// CHECK-LABEL:     llvm.mlir.global internal @_ZN2NSL1CE() {addr_space = 0 : i32, sym_visibility = "private"} : f64 {
// CHECK:             %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:             llvm.return %[[VAL_0]] : f64
// CHECK:           }
const double C = 0.0;
}

// CHECK-LABEL:     func.func @_Z3foov() -> (f64 {llvm.noundef})
// CHECK:             %[[VAL_152:.*]] = llvm.mlir.addressof @_ZN2NSL1CE : !llvm.ptr
// CHECK:             %[[VAL_153:.*]] = llvm.load %[[VAL_152]] : !llvm.ptr -> f64
// CHECK:             return %[[VAL_153]] : f64
// CHECK:           }
SYCL_EXTERNAL double foo() { return NS::C; }
