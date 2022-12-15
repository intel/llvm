// RUN: cgeist %s --function=* -S | FileCheck %s

extern void print(char*);

class Root {
public:
    Root(int y) {
        print("Calling root");
    }
};

class FRoot {
public:
    FRoot() {
        print("Calling froot");
    }
};

class Sub : public Root, public FRoot {
public:
    Sub(int i, double y) : Root(i) {
        print("Calling Sub");
    }
};

void make() {
    Sub s(3, 3.14);
}

// clang-format off
// CHECK:   func @_Z4makev() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %cst = arith.constant 3.140000e+00 : f64
// CHECK-DAG:     %c3_i32 = arith.constant 3 : i32
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i8)> : (i64) -> !llvm.ptr<struct<(i8)>>
// CHECK-NEXT:     call @_ZN3SubC1Eid(%0, %c3_i32, %cst) : (!llvm.ptr<struct<(i8)>>, i32, f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN3SubC1Eid(%arg0: !llvm.ptr<struct<(i8)>>, %arg1: i32, %arg2: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>}
// CHECK-NEXT:     call @_ZN4RootC1Ei(%arg0, %arg1) : (!llvm.ptr<struct<(i8)>>, i32) -> ()
// CHECK-NEXT:     call @_ZN5FRootC1Ev(%arg0) : (!llvm.ptr<struct<(i8)>>) -> ()
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str0 : !llvm.ptr<array<12 x i8>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<array<12 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     call @_Z5printPc(%1) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN4RootC1Ei(%arg0: !llvm.ptr<struct<(i8)>>, %arg1: i32) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str1 : !llvm.ptr<array<13 x i8>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<array<13 x i8>>) -> memref<?xi8> 
// CHECK-NEXT:     call @_Z5printPc(%1) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN5FRootC1Ev(%arg0: !llvm.ptr<struct<(i8)>>) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str2 : !llvm.ptr<array<14 x i8>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<array<14 x i8>>) -> memref<?xi8> 
// CHECK-NEXT:     call @_Z5printPc(%1) : (memref<?xi8>) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
