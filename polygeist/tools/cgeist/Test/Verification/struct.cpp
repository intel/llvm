// RUN: cgeist %s %stdinclude --function=func -S | FileCheck %s

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

// CHECK:   func @func(%arg0: !llvm.ptr<struct<(i8, ptr<i8>, i8)>>) -> f32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %[[i2:.+]] = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr<struct<(i8, ptr<i8>, i8)>>) -> !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %[[i3:.+]] = llvm.load %[[i2]] : !llvm.ptr<ptr<i8>>
// CHECK-NEXT:     %[[i4:.+]] = call @_Z5hloadPKv(%[[i3]]) : (!llvm.ptr<i8>) -> f32
// CHECK-NEXT:     return %[[i4]] : f32
// CHECK-NEXT:   }
