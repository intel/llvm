// RUN: cgeist %s --function=* -S | FileCheck %s

extern void print(char*);

class Root {
public:
    int x;
    Root(int y) : x(y) {
        print("Calling root");
    }
};

class FRoot {
public:
    float f;
    FRoot() : f(2.18) {
        print("Calling froot");
    }
};

class Sub : public Root, public FRoot {
public:
    double d;
    Sub(int i, double y) : Root(i), d(y) {
        print("Calling Sub");
    }
};

void make() {
    Sub s(3, 3.14);
}

// CHECK:   func @_Z4makev() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %cst = arith.constant 3.140000e+00 : f64
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(i32)>, struct<(f32)>, f64)> : (i64) -> !llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>
// CHECK-NEXT:     call @_ZN3SubC1Eid(%0, %c3_i32, %cst) : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>, i32, f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN3SubC1Eid(%arg0: !llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>, %arg1: i32, %arg2: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = "polygeist.pointer2memref"(%arg0) : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> memref<?x1xi32>
// CHECK-NEXT:     call @_ZN4RootC1Ei(%0, %arg1) : (memref<?x1xi32>, i32) -> ()
// CHECK-NEXT:     %1 = llvm.getelementptr %arg0[0, 1] : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> !llvm.ptr<struct<(f32)>>
// CHECK-NEXT:     %2 = "polygeist.pointer2memref"(%1) : (!llvm.ptr<struct<(f32)>>) -> memref<?x1xf32>
// CHECK-NEXT:     call @_ZN5FRootC1Ev(%2) : (memref<?x1xf32>) -> ()
// CHECK-NEXT:     %3 = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr<struct<(struct<(i32)>, struct<(f32)>, f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %arg2, %3 : !llvm.ptr<f64>
// CHECK-NEXT:     %4 = llvm.mlir.addressof @str0 : !llvm.ptr<array<12 x i8>>
// CHECK-NEXT:     %5 = llvm.getelementptr %4[0, 0] : (!llvm.ptr<array<12 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @_Z5printPc(%5) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN4RootC1Ei(%arg0: memref<?x1xi32>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     affine.store %arg1, %arg0[0, 0] : memref<?x1xi32>
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str1 : !llvm.ptr<array<13 x i8>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr<array<13 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @_Z5printPc(%1) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN5FRootC1Ev(%arg0: memref<?x1xf32>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %cst = arith.constant 2.180000e+00 : f32
// CHECK-NEXT:     affine.store %cst, %arg0[0, 0] : memref<?x1xf32>
// CHECK-NEXT:     %[[i1:.+]] = llvm.mlir.addressof @str2 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %[[i2:.+]] = llvm.getelementptr %[[i1]][0, 0] : (!llvm.ptr<array<14 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:     call @_Z5printPc(%[[i2]]) : (!llvm.ptr<i8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
