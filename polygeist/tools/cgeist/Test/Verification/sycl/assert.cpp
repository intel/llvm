// RUN: clang++  -fsycl -fsycl-device-only -O0 -w -emit-mlir %s -o - | FileCheck %s

#include <sycl/sycl.hpp>
using namespace sycl;

// CHECK-LABEL:   gpu.module @device_functions {

// CHECK-LABEL:     func.func @_Z12check_assertN4sycl3_V12idILi1EEES2_(
// CHECK-SAME:          %[[VAL_151:.*]]: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef},
// CHECK-SAME:          %[[VAL_152:.*]]: memref<?x!sycl_id_1_> {llvm.align = 8 : i64, llvm.byval = !sycl_id_1_, llvm.noundef})
// CHECK:               %[[VAL_153:.*]] = arith.constant 26 : i32
// CHECK:               %[[VAL_157:.*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK:               %[[VAL_158:.*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK:               %[[VAL_159:.*]] = llvm.mlir.addressof @str2 : !llvm.ptr
// CHECK:               %[[VAL_160:.*]] = "polygeist.pointer2memref"(%[[VAL_157]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:               %[[VAL_161:.*]] = memref.memory_space_cast %[[VAL_160]] : memref<?xi8> to memref<?xi8, 4>
// CHECK:               %[[VAL_162:.*]] = "polygeist.pointer2memref"(%[[VAL_158]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:               %[[VAL_163:.*]] = memref.memory_space_cast %[[VAL_162]] : memref<?xi8> to memref<?xi8, 4>
// CHECK:               %[[VAL_164:.*]] = "polygeist.pointer2memref"(%[[VAL_159]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK:               %[[VAL_165:.*]] = memref.memory_space_cast %[[VAL_164]] : memref<?xi8> to memref<?xi8, 4>
// CHECK:               func.call @__assert_fail(%[[VAL_161]], %[[VAL_163]], %[[VAL_153]], %[[VAL_165]]) : (memref<?xi8, 4>, memref<?xi8, 4>, i32, memref<?xi8, 4>) -> ()

// CHECK-LABEL:     func.func private @__assert_fail(memref<?xi8, 4> {llvm.noundef}, memref<?xi8, 4> {llvm.noundef}, i32 {llvm.noundef}, memref<?xi8, 4> {llvm.noundef})

SYCL_EXTERNAL void check_assert(id<1> id1, id<1> id2) {
  assert(id1 == id2);
}
