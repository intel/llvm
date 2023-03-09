// RUN: cgeist -O0 -w %s  --function=* -S | FileCheck %s

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

// CHECK:   func @_Z14ilaunch_kernel7QStream(%arg0: !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.struct<(struct<(f64, f64)>, i32)> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(f64, f64)>, i32)> : (i64) -> !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     call @_ZN7QStreamC1EOS_(%0, %arg0) : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>, !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> ()
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     return %1 : !llvm.struct<(struct<(f64, f64)>, i32)>
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZN7QStreamC1EOS_(%arg0: !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>, %arg1: !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<struct<(f64, f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<struct<(f64, f64)>>
// CHECK-NEXT:     call @_ZN1DC1EOS_(%0, %1) : (!llvm.ptr<struct<(f64, f64)>>, !llvm.ptr<struct<(f64, f64)>>) -> ()
// CHECK-NEXT:     %2 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i32>
// CHECK-NEXT:     %4 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %3, %4 : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZN1DC1EOS_(%arg0: !llvm.ptr<struct<(f64, f64)>>, %arg1: !llvm.ptr<struct<(f64, f64)>>)
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<f64>
// CHECK-NEXT:     %2 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %1, %2 : !llvm.ptr<f64>
// CHECK-NEXT:     %3 = llvm.getelementptr inbounds %arg1[0, 1] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<f64>
// CHECK-NEXT:     %5 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr<struct<(f64, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %4, %5 : !llvm.ptr<f64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
