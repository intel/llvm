// RUN: cgeist -O0 %s -w -o - -S --function=* | FileCheck %s

// The function being called here should not be generated as it won't be called.
// CHECK-NOT: func.func{{.*}}@{{.*}}g

bool g();

// Assume with a boolean variable as an argument

// CHECK-LABEL:   func.func @_Z2f0b(
// CHECK-SAME:                      %[[VAL_0:.*]]: i1)
// CHECK-NEXT:      "llvm.intr.assume"(%[[VAL_0]]) : (i1) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }
void f0(bool b) {
  __builtin_assume(b);
}

// Assume with a boolean expression as an argument

// CHECK-LABEL:   func.func @_Z2f1i(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32)
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.cmpi sgt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK-NEXT:      "llvm.intr.assume"(%[[VAL_2]]) : (i1) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

void f1(int x) {
  __builtin_assume(x > 0);
}

// Assume with an expression which might have side effects as an argument.
// Should be omitted.

// CHECK-LABEL:   func.func @_Z2f2v()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

void f2() {
  __builtin_assume(g());
}
