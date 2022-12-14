// RUN: cgeist -O0 -w %s --function=meta -S | FileCheck %s

int foo(const char*);

int meta() {
    return foo("bar") + foo(__PRETTY_FUNCTION__);
}

// CHECK:   llvm.mlir.global internal constant @str1("int meta()\00")
// CHECK:   llvm.mlir.global internal constant @str0("bar\00")
// CHECK:   func @meta() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.mlir.addressof @str0 : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:     %1 = "polygeist.pointer2memref"(%0) : (!llvm.ptr<array<4 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     %2 = call @foo(%1) : (memref<?xi8>) -> i32
// CHECK-NEXT:     %3 = llvm.mlir.addressof @str1 : !llvm.ptr<array<11 x i8>>
// CHECK-NEXT:     %4 = "polygeist.pointer2memref"(%3) : (!llvm.ptr<array<11 x i8>>) -> memref<?xi8>
// CHECK-NEXT:     %5 = call @foo(%4) : (memref<?xi8>) -> i32
// CHECK-NEXT:     %6 = arith.addi %2, %5 : i32
// CHECK-NEXT:     return %6 : i32
// CHECK-NEXT:   }

// CHECK:   func.func private @foo(memref<?xi8>) -> i32
