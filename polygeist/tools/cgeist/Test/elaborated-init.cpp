// RUN: cgeist  %s --function='*' -S -std=c++14 | FileCheck %s
struct A {
  using TheType = int[4];
};

void testArrayInitExpr()
{
  A::TheType a{1,2,3,4};
  auto l = [a]{
  };
}

// CHECK-LABEL:   func.func @_Z17testArrayInitExprv()
// CHECK:           %[[VAL_0:.*]] = arith.constant 16 : i64
// CHECK:           "llvm.intr.memcpy"(%{{.*}}, %{{.*}}, %[[VAL_0]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
