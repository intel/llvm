// RUN: cgeist %s --function=* -S | FileCheck %s

struct x {};

int foo(x arg);

// CHECK-LABEL:   func.func @_Z3barv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:           %[[VAL_0:.*]] = call @_Z3foo1x() : () -> i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }
int bar() {
  return foo(x{});
}

// CHECK:         func.func private @_Z3foo1x() -> i32 attributes {llvm.linkage = #llvm.linkage<external>}
