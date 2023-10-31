// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

struct w { int el; };

SYCL_EXTERNAL void foo(w *x);

// CHECK-LABEL:     func.func @_Z4testv()
// CHECK:             %[[VAL_151:.*]] = arith.constant 1 : i64
// CHECK:             %[[VAL_152:.*]] = llvm.alloca %[[VAL_151]] x !llvm.array<24 x struct<(i32)>> : (i64) -> !llvm.ptr
// CHECK:             %[[VAL_153:.*]] = llvm.getelementptr inbounds %[[VAL_152]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x struct<(i32)>>
// CHECK:             %[[VAL_154:.*]] = llvm.addrspacecast %[[VAL_153]] : !llvm.ptr to !llvm.ptr<4>
// CHECK:             call @_Z3fooP1w(%[[VAL_154]]) : (!llvm.ptr<4>) -> ()
// CHECK:             return
// CHECK:           }
SYCL_EXTERNAL void test() {
  w ws[24];
  foo(ws);
}
