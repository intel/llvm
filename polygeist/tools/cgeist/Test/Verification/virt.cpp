// RUN: cgeist  -O0 -w %s --function=* -S | FileCheck %s

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

// CHECK-LABEL:   func.func @_Z4makev() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3.140000e+00 : f64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(i32)>, struct<(f32)>, f64)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN3SubC1Eid(%[[VAL_3]], %[[VAL_1]], %[[VAL_0]]) : (!llvm.ptr, i32, f64) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN3SubC1Eid(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                            %[[VAL_1:.*]]: i32,
// CHECK-SAME:                            %[[VAL_2:.*]]: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>
// CHECK-NEXT:      call @_ZN4RootC1Ei(%[[VAL_3]], %[[VAL_1]]) : (!llvm.ptr, i32) -> ()
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>
// CHECK-NEXT:      call @_ZN5FRootC1Ev(%[[VAL_4]]) : (!llvm.ptr) -> ()
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32)>, struct<(f32)>, f64)>
// CHECK-NEXT:      llvm.store %[[VAL_2]], %[[VAL_5]] : f64, !llvm.ptr
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = "polygeist.pointer2memref"(%[[VAL_6]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:      call @_Z5printPc(%[[VAL_7]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN4RootC1Ei(
// CHECK-SAME:                            %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                            %[[VAL_1:.*]]: i32) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_2]] : i32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:      call @_Z5printPc(%[[VAL_4]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN5FRootC1Ev(
// CHECK-SAME:                             %[[VAL_0:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 2.180000e+00 : f32
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_2]] : f32, !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.addressof @str2 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:      call @_Z5printPc(%[[VAL_4]]) : (memref<?xi8>) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @_Z5printPc(memref<?xi8>) attributes {llvm.linkage = #llvm.linkage<external>}
