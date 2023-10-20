// RUN: cgeist -O0 %s --function=* -S | FileCheck %s

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

void foo(A);

// CHECK-LABEL:   func.func @_Z3barv()
// CHECK:           %[[VAL_0:.*]] = arith.constant 72 : i64
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_1]] x !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)> : (i64) -> !llvm.ptr
// CHECK:           "llvm.intr.memcpy"(%[[VAL_3]], %[[VAL_4]], %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_5:.*]] = llvm.load %[[VAL_3]] : !llvm.ptr -> !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_2]] : !llvm.struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>, !llvm.ptr
// CHECK:           call @_Z3foo1A(%[[VAL_2]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }

void bar() {
  A a;
  foo(a);
}

union U {
  U(const A &a) : a(a) {}
  U(int i) : i(i) {}

  int i;
  A a;
};

void foo_u(U);

// CHECK-LABEL:   func.func @_Z5bar_ui(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32)
// CHECK:           %[[VAL_1:.*]] = arith.constant 72 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)> : (i64) -> !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)> : (i64) -> !llvm.ptr
// CHECK:           call @_ZN1UC1Ei(%[[VAL_5]], %[[VAL_0]]) : (!llvm.ptr, i32) -> ()
// CHECK:           "llvm.intr.memcpy"(%[[VAL_4]], %[[VAL_5]], %[[VAL_1]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK:           %[[VAL_6:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)>
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_3]] : !llvm.struct<(struct<(i32, f32, struct<(i64, f64, ptr)>, array<10 x i32>)>)>, !llvm.ptr
// CHECK:           call @_Z5foo_u1U(%[[VAL_3]]) : (!llvm.ptr) -> ()
// CHECK:           return
// CHECK:         }

void bar_u(int i) {
  U u(i);
  foo_u(u);
}
