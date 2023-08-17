// RUN: cgeist  -O0 -w %s --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

struct MOperandInfo* begin();

struct MOperandInfo& inner() {
  return begin()[0];
}

// CHECK-LABEL:   func.func @_Z5innerv() -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = call @_Z5beginv() : () -> !llvm.ptr
// CHECK-NEXT:      return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @_Z5beginv() -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>}
