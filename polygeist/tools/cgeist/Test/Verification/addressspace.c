// RUN: cgeist %s --function=* -S | FileCheck %s

// CHECK:  func.func @test() -> memref<?xi32, 4> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:    [[CALL:%.*]] = call @foo() : () -> memref<?xi32, 1>
// CHECK:    [[M2P:%.*]] = "polygeist.memref2pointer"([[CALL]]) : (memref<?xi32, 1>) -> !llvm.ptr<i32, 1>
// CHECK:    [[ACAST:%.*]] = llvm.addrspacecast [[M2P]] : !llvm.ptr<i32, 1> to !llvm.ptr<i32, 4>
// CHECK:    [[P2M:%.*]] = "polygeist.pointer2memref"([[ACAST]]) : (!llvm.ptr<i32, 4>) -> memref<?xi32, 4>
// CHECK:    return [[P2M]] : memref<?xi32, 4>
// CHECK:  }
// CHECK:  func.func private @foo() -> memref<?xi32, 1>

int __attribute__((address_space(1))) *foo();

int __attribute__((address_space(4))) *test() {
  return (int __attribute__((address_space(4))) *)foo();
}
