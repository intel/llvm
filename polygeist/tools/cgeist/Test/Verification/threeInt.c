// RUN: cgeist -O0 -w %s --function=struct_pass_all_same -S | FileCheck %s

typedef struct {
  int a, b, c;
} threeInt;

int struct_pass_all_same(threeInt* a) {
  return a->b;
}

// CHECK-LABEL:   func.func @struct_pass_all_same(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !llvm.ptr) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, i32)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> i32
// CHECK-NEXT:      return %[[VAL_2]] : i32
// CHECK-NEXT:    }
