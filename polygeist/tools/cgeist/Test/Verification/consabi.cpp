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
// CHECK-SAME:                                          %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.struct<(struct<(f64, f64)>, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(struct<(f64, f64)>, i32)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN7QStreamC1EOS_(%[[VAL_2]], %[[VAL_0]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:      return %[[VAL_3]] : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN7QStreamC1EOS_(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                 %[[VAL_1:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:      call @_ZN1DC1EOS_(%[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> i32
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:      llvm.store %[[VAL_5]], %[[VAL_6]] : i32, !llvm.ptr
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN1DC1EOS_(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64, f64)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr -> f64
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64, f64)>
// CHECK-NEXT:      llvm.store %[[VAL_3]], %[[VAL_4]] : f64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64, f64)>
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_5]] : !llvm.ptr -> f64
// CHECK-NEXT:      %[[VAL_7:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64, f64)>
// CHECK-NEXT:      llvm.store %[[VAL_6]], %[[VAL_7]] : f64, !llvm.ptr
// CHECK-NEXT:      return
// CHECK-NEXT:    }
