// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

// CHECK-LABEL:  gpu.module @device_functions {
// CHECK-LABEL:    llvm.func @__assert_fail(!llvm.ptr<i8, 4>, !llvm.ptr<i8, 4>, i32, !llvm.ptr<i8, 4>)
// CHECK-LABEL:    func.func @_Z12check_assertN4sycl3_V12idILi1EEES2_(%arg0: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef}, %arg1: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef})
// CHECK-NEXT:        [[C22:%.*]] = arith.constant 22 : i32
// CHECK:             %7 = llvm.mlir.addressof @str0 : !llvm.ptr<array<11 x i8>>
// CHECK-NEXT:        %8 = llvm.getelementptr %7[0, 0] : (!llvm.ptr<array<11 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:        %9 = llvm.mlir.addressof @str1 : !llvm.ptr<array<[[PATH_LENGTH:.*]] x i8>>
// CHECK-NEXT:        %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr<array<[[PATH_LENGTH]] x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:        %11 = llvm.mlir.addressof @str2 : !llvm.ptr<array<32 x i8>>
// CHECK-NEXT:        %12 = llvm.getelementptr %11[0, 0] : (!llvm.ptr<array<32 x i8>>) -> !llvm.ptr<i8>
// CHECK-NEXT:        %13 = llvm.addrspacecast %8 : !llvm.ptr<i8> to !llvm.ptr<i8, 4>
// CHECK-NEXT:        %14 = llvm.addrspacecast %10 : !llvm.ptr<i8> to !llvm.ptr<i8, 4>
// CHECK-NEXT:        %15 = llvm.addrspacecast %12 : !llvm.ptr<i8> to !llvm.ptr<i8, 4>
// CHECK-NEXT:        llvm.call @__assert_fail(%13, %14, [[C22]], %15) : (!llvm.ptr<i8, 4>, !llvm.ptr<i8, 4>, i32, !llvm.ptr<i8, 4>) -> ()

SYCL_EXTERNAL void check_assert(id<1> id1, id<1> id2) {
  assert(id1 == id2);
}
