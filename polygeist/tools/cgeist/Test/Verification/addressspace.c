// RUN: cgeist %s --function=* -S | FileCheck %s

// CHECK:  func.func @test() -> memref<?xi32, 4> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:    [[CALL:%.*]] = call @foo() : () -> memref<?xi32, 1>
// CHECK:    [[MSC:%.*]] = memref.memory_space_cast [[CALL]] : memref<?xi32, 1> to memref<?xi32, 4>
// CHECK:    return [[MSC]] : memref<?xi32, 4>
// CHECK:  }
// CHECK:  func.func private @foo() -> memref<?xi32, 1>

int __attribute__((address_space(1))) *foo();

int __attribute__((address_space(4))) *test() {
  return (int __attribute__((address_space(4))) *)foo();
}
