// RUN: cgeist %s --function=* -S | FileCheck %s

void* malloc(unsigned long);

struct Meta {
    float* f;
    char x;
};

struct Meta* create() {
    return (struct Meta*)malloc(sizeof(struct Meta));
}

// CHECK-LABEL:   func.func @create() -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:      %[[VAL_0:.*]] = "polygeist.typeSize"() <{source = !llvm.struct<(memref<?xf32>, i8)>}> : () -> index
// CHECK-NEXT:      %[[VAL_1:.*]] = arith.index_cast %[[VAL_0]] : index to i64
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.call @malloc(%[[VAL_1]]) : (i64) -> !llvm.ptr
// CHECK-NEXT:      return %[[VAL_2]] : !llvm.ptr
// CHECK-NEXT:    }
