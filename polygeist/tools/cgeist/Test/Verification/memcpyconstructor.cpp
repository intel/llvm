// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

#include <utility>

// A and C present trivial copy and move constructors.

class C {
public:
  C() = default;
  C(const C &) = default;
  C(C &&) = default;
private:
  long long a;
  double b;
  void *c;
};

class A {
private:
  int a;
  float b;
  C c;
  int d[10];
};

// CHECK-LABEL:   func.func @_Z3barv()
// CHECK:           %[[VAL_0:.*]] = arith.constant 72 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)> : (i64) -> !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_4]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           "llvm.intr.memcpy"(%[[VAL_2]], %[[VAL_4]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           return
// CHECK:         }
void bar() {
  A a;
  A cpy(a);
  A mv(std::move(a));
}

union U {
  U(const A &a) : a(a) {}
  U() : i(0) {}

  int i;
  A a;
};

// CHECK-LABEL:   func.func @_Z5bar_uv()
// CHECK:           %[[VAL_0:.*]] = arith.constant 72 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)> : (i64) -> !llvm.ptr
// CHECK:           call @_ZN1UC1Ev(%[[VAL_4]]) : (!llvm.ptr) -> ()
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_4]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           "llvm.intr.memcpy"(%[[VAL_2]], %[[VAL_4]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           return
// CHECK:         }
void bar_u() {
  U u;
  U cpu(u);
  U mv(std::move(u));
}

class Empty {};

// COM: Lowers to nop as no action shall be performed in the constructor.

// CHECK-LABEL:   func.func @_Z9bar_emptyv()
// CHECK:           return
// CHECK:         }

void bar_empty() {
  Empty e;
  Empty cpy(e);
  Empty mv(std::move(e));
}
