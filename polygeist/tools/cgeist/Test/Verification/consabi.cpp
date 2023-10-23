// RUN: cgeist  -O0 -w %s  --function=* -S | FileCheck %s

class D {
  double a;
  double b;
};

class QStream {
  D device_;
  int id;
};

QStream ilaunch_kernel(QStream x) {
  return x;
}

// CHECK-LABEL:   func.func @_Z14ilaunch_kernel7QStream(
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           %[[VAL_1:.*]] = arith.constant 24 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(f64, f64)>, i32)> : (i64) -> !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_0]], %[[VAL_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_4:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:           return %[[VAL_4]] : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK:         }
