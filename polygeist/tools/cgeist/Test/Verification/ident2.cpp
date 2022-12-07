// RUN: cgeist -O0 -w %s --function=* -S | FileCheck %s

struct MOperandInfo {
  char device;
  char dtype;
};

struct MOperandInfo* begin();

struct MOperandInfo& inner() {
  return begin()[0];
}

// CHECK:   func.func @_Z5innerv() -> !llvm.ptr<struct<(i8, i8)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = call @_Z5beginv() : () -> !llvm.ptr<struct<(i8, i8)>>
// CHECK-NEXT:     return %0 : !llvm.ptr<struct<(i8, i8)>>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func private @_Z5beginv() -> !llvm.ptr<struct<(i8, i8)>> attributes {llvm.linkage = #llvm.linkage<external>}
