// RUN: cgeist %s --function='*' -S | FileCheck %s
struct A {
  using TheType = int[4];
};

void testArrayInitExpr()
{
  A::TheType a{1,2,3,4};
  auto l = [a]{
  };
}

// CHECK: func.func private @_ZZ17testArrayInitExprvEN3$_0C1EOS_(%arg0: !llvm.ptr<struct<(array<4 x i32>)>>, %arg1: !llvm.ptr<struct<(array<4 x i32>)>>) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[%c0_i32, 0] : (!llvm.ptr<struct<(array<4 x i32>)>>, i32) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:     %1 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<array<4 x i32>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.getelementptr %arg1[%c0_i32, 0] : (!llvm.ptr<struct<(array<4 x i32>)>>, i32) -> !llvm.ptr<array<4 x i32>>
// CHECK-NEXT:     %3 = llvm.getelementptr %2[%c0_i32, %c0_i32] : (!llvm.ptr<array<4 x i32>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     affine.for %arg2 = 0 to 4 {
// CHECK-NEXT:       %4 = arith.index_cast %arg2 : index to i64
// CHECK-NEXT:       %5 = llvm.getelementptr %1[%4] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:       %6 = llvm.getelementptr %3[%4] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:       %7 = llvm.load %6 : !llvm.ptr<i32>
// CHECK-NEXT:       llvm.store %7, %5 : !llvm.ptr<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
