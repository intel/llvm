// RUN: cgeist %s -O2 --function=* -S | FileCheck %s

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
// CHECK-NEXT:     %0 = llvm.bitcast %arg1 : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>> to !llvm.ptr<f64>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<f64>
// CHECK-NEXT:     %2 = llvm.bitcast %arg0 : !llvm.ptr<struct<(struct<(f64, f64)>, i32)>> to !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %1, %2 : !llvm.ptr<f64>
// CHECK-NEXT:     %[[a4:.+]] = llvm.getelementptr %0[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %[[a5:.+]] = llvm.load %[[a4]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[a7:.+]] = llvm.getelementptr %2[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %[[a5]], %[[a7]] : !llvm.ptr<f64>
// CHECK-NEXT:     %[[a8:.+]] = llvm.getelementptr %arg1[0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     %[[a9:.+]] = llvm.load %[[a8]] : !llvm.ptr<i32>
// CHECK-NEXT:     %[[a10:.+]] = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %[[a9]], %[[a10]] : !llvm.ptr<i32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   func @_ZN1DC1EOS_(%arg0: memref<?x2xf64>, %arg1: memref<?x2xf64>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = affine.load %arg1[0, 0] : memref<?x2xf64>
// CHECK-NEXT:     affine.store %0, %arg0[0, 0] : memref<?x2xf64>
// CHECK-NEXT:     %1 = affine.load %arg1[0, 1] : memref<?x2xf64>
// CHECK-NEXT:     affine.store %1, %arg0[0, 1] : memref<?x2xf64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
