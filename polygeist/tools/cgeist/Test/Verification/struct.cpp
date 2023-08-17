// RUN: cgeist  %s %stdinclude --function=func -S | FileCheck %s

float hload(const void* data);

struct OperandInfo {
  char dtype = 'a';

  void* data;

  bool end;
};

extern "C" {
float func(struct OperandInfo* op) {
    return hload(op->data);
}
}

// CHECK-LABEL:   func.func @func(
// CHECK-SAME:                    %[[VAL_0:.*]]: !llvm.ptr) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_1:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, ptr, i8)>
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.load %[[VAL_1]] : !llvm.ptr -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_3:.*]] = call @_Z5hloadPKv(%[[VAL_2]]) : (!llvm.ptr) -> f32
// CHECK-NEXT:      return %[[VAL_3]] : f32
// CHECK-NEXT:    }
