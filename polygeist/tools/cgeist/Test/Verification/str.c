// RUN: cgeist -O0 -w %s --function=meta -S | FileCheck %s

int foo(const char*);

int meta() {
    return foo("bar") + foo(__PRETTY_FUNCTION__);
}

// CHECK-LABEL:   llvm.mlir.global internal constant @str1("int meta()\00") {addr_space = 0 : i32}
// CHECK-NEXT:    llvm.mlir.global internal constant @str0("bar\00") {addr_space = 0 : i32}

// CHECK-LABEL:   func.func @meta() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_1:.*]] = "polygeist.pointer2memref"(%[[VAL_0]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:      %[[VAL_2:.*]] = call @foo(%[[VAL_1]]) : (memref<?xi8>) -> i32
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.mlir.addressof @str1 : !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = "polygeist.pointer2memref"(%[[VAL_3]]) : (!llvm.ptr) -> memref<?xi8>
// CHECK-NEXT:      %[[VAL_5:.*]] = call @foo(%[[VAL_4]]) : (memref<?xi8>) -> i32
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.addi %[[VAL_2]], %[[VAL_5]] : i32
// CHECK-NEXT:      return %[[VAL_6]] : i32
// CHECK-NEXT:    }
// CHECK-LABEL:   func.func private @foo(memref<?xi8>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
