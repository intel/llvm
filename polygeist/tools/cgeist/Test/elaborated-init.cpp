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

// CHECK-LABEL:   func.func private @_ZZ17testArrayInitExprvEN3$_0C1EOS_(
// CHECK-SAME:                                                           %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                                           %[[VAL_1:.*]]: !llvm.ptr) attributes {llvm.linkage = #llvm.linkage<internal>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x i32>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(array<4 x i32>)>
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.getelementptr inbounds %[[VAL_4]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<4 x i32>
// CHECK-NEXT:      affine.for %[[VAL_6:.*]] = 0 to 4 {
// CHECK-NEXT:        %[[VAL_7:.*]] = arith.index_cast %[[VAL_6]] : index to i64
// CHECK-NEXT:        %[[VAL_8:.*]] = llvm.getelementptr %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:        %[[VAL_9:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_7]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK-NEXT:        %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> i32
// CHECK-NEXT:        llvm.store %[[VAL_10]], %[[VAL_8]] : i32, !llvm.ptr
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
