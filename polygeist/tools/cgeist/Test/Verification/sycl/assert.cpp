// RUN: clang++ -Xcgeist --use-opaque-pointers=1 -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

// CHECK-LABEL:   gpu.module @device_functions {

// CHECK-LABEL:     llvm.func @__assert_fail(!llvm.ptr<4>, !llvm.ptr<4>, i32, !llvm.ptr<4>)

// CHECK-LABEL:     func.func @_Z12check_assertN4sycl3_V12idILi1EEES2_(
// CHECK-SAME:          %[[VAL_151:.*]]: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef},
// CHECK-SAME:          %[[VAL_152:.*]]: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef})
// CHECK-NEXT:        %[[C26:.*]] = arith.constant 26 : i32
// CHECK:               %[[VAL_157:.*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK-NEXT:          %[[VAL_158:.*]] = llvm.getelementptr inbounds %[[VAL_157]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<11 x i8>
// CHECK-NEXT:          %[[VAL_159:.*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK-NEXT:          %[[VAL_160:.*]] = llvm.getelementptr inbounds %[[VAL_159]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<[[#PATHLENGTH:]] x i8>
// CHECK-NEXT:          %[[VAL_161:.*]] = llvm.mlir.addressof @str2 : !llvm.ptr
// CHECK-NEXT:          %[[VAL_162:.*]] = llvm.getelementptr inbounds %[[VAL_161]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<32 x i8>
// CHECK-NEXT:          %[[VAL_163:.*]] = llvm.addrspacecast %[[VAL_158]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:          %[[VAL_164:.*]] = llvm.addrspacecast %[[VAL_160]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:          %[[VAL_165:.*]] = llvm.addrspacecast %[[VAL_162]] : !llvm.ptr to !llvm.ptr<4>
// CHECK-NEXT:          llvm.call @__assert_fail(%[[VAL_163]], %[[VAL_164]], %[[C26]], %[[VAL_165]]) : (!llvm.ptr<4>, !llvm.ptr<4>, i32, !llvm.ptr<4>) -> ()

SYCL_EXTERNAL void check_assert(id<1> id1, id<1> id2) {
  assert(id1 == id2);
}
