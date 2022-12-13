// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

// CHECK-LABEL: func.func @_Z7vecinitv() -> !llvm.struct<(vector<4xi8>)>
// CHECK-DAG:     %[[VAL_0:.*]] = arith.constant 4 : i64
// CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 1 : i8
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 0 : i8
// CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : i64
// CHECK-NEXT:    %[[VAL_4:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(vector<4xi8>)> : (i64) -> !llvm.ptr<struct<(vector<4xi8>)>>
// CHECK-NEXT:    %[[VAL_5:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(vector<4xi8>)> : (i64) -> !llvm.ptr<struct<(vector<4xi8>)>>
// CHECK-NEXT:    %[[VAL_6:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(ptr<i8, 4>, i64)> : (i64) -> !llvm.ptr<struct<(ptr<i8, 4>, i64)>>
// CHECK-NEXT:    %[[VAL_7:.*]] = llvm.alloca %[[VAL_3]] x !llvm.array<4 x i8> : (i64) -> !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:    %[[VAL_8:.*]] = llvm.alloca %[[VAL_3]] x !llvm.array<4 x i8> : (i64) -> !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:    %[[VAL_9:.*]] = llvm.alloca %[[VAL_3]] x !llvm.struct<(vector<4xi8>)> : (i64) -> !llvm.ptr<struct<(vector<4xi8>)>>
// CHECK-NEXT:    %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_8]][0, 0] : (!llvm.ptr<array<4 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %[[VAL_2]], %[[VAL_10]] : !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_11:.*]] = llvm.getelementptr %[[VAL_8]][0, 1] : (!llvm.ptr<array<4 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %[[VAL_1]], %[[VAL_11]] : !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_12:.*]] = llvm.getelementptr %[[VAL_8]][0, 2] : (!llvm.ptr<array<4 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %[[VAL_2]], %[[VAL_12]] : !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_8]][0, 3] : (!llvm.ptr<array<4 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    llvm.store %[[VAL_2]], %[[VAL_13]] : !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_14:.*]] = llvm.load %[[VAL_8]] : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:    llvm.store %[[VAL_14]], %[[VAL_7]] : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:    %[[VAL_15:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i8, 4>, i64)>
// CHECK-NEXT:    %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_7]][0, 0] : (!llvm.ptr<array<4 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:    %[[VAL_17:.*]] = llvm.addrspacecast %[[VAL_16]] : !llvm.ptr<i8> to !llvm.ptr<i8, 4>
// CHECK-NEXT:    %[[VAL_18:.*]] = llvm.insertvalue %[[VAL_17]], %[[VAL_15]][0] : !llvm.struct<(ptr<i8, 4>, i64)>
// CHECK-NEXT:    %[[VAL_19:.*]] = llvm.insertvalue %[[VAL_0]], %[[VAL_18]][1] : !llvm.struct<(ptr<i8, 4>, i64)>
// CHECK-NEXT:    %[[VAL_20:.*]] = llvm.addrspacecast %[[VAL_9]] : !llvm.ptr<struct<(vector<4xi8>)>> to !llvm.ptr<struct<(vector<4xi8>)>, 4>
// CHECK-NEXT:    llvm.store %[[VAL_19]], %[[VAL_6]] : !llvm.ptr<struct<(ptr<i8, 4>, i64)>>
// CHECK-NEXT:    sycl.call(%[[VAL_20]], %[[VAL_6]]) {FunctionName = @Boolean, MangledFunctionName = @_ZN4sycl3_V16detail7BooleanILi4EEC1ESt16initializer_listIaE, TypeName = @Boolean} : (!llvm.ptr<struct<(vector<4xi8>)>, 4>, !llvm.ptr<struct<(ptr<i8, 4>, i64)>>) -> ()
// CHECK-NEXT:    %[[VAL_21:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr<struct<(vector<4xi8>)>>
// CHECK-NEXT:    llvm.store %[[VAL_21]], %[[VAL_5]] : !llvm.ptr<struct<(vector<4xi8>)>>
// CHECK-NEXT:    %[[VAL_22:.*]] = llvm.addrspacecast %[[VAL_4]] : !llvm.ptr<struct<(vector<4xi8>)>> to !llvm.ptr<struct<(vector<4xi8>)>, 4>
// CHECK-NEXT:    %[[VAL_23:.*]] = llvm.addrspacecast %[[VAL_5]] : !llvm.ptr<struct<(vector<4xi8>)>> to !llvm.ptr<struct<(vector<4xi8>)>, 4>
// CHECK-NEXT:    sycl.call(%[[VAL_22]], %[[VAL_23]]) {FunctionName = @Boolean, MangledFunctionName = @_ZN4sycl3_V16detail7BooleanILi4EEC1ERKS3_, TypeName = @Boolean} : (!llvm.ptr<struct<(vector<4xi8>)>, 4>, !llvm.ptr<struct<(vector<4xi8>)>, 4>) -> ()
// CHECK-NEXT:    %[[VAL_24:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr<struct<(vector<4xi8>)>>
// CHECK-NEXT:    return %[[VAL_24]] : !llvm.struct<(vector<4xi8>)>
// CHECK-NEXT:  }

SYCL_EXTERNAL sycl::detail::Boolean<4> vecinit() {
  sycl::detail::Boolean<4> t{false, true, false, false};
  return t;
}
