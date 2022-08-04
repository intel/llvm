// RUN: cgeist %s --function=* -S | FileCheck %s

void* malloc(unsigned long);

struct Meta {
    float* f;
    char x;
};

struct Meta* create() {
    return (struct Meta*)malloc(sizeof(struct Meta));
}

// CHECK:   func @create() -> !llvm.ptr<!llvm.struct<(memref<?xf32>, i8)>> attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "polygeist.typeSize"() {source = !llvm.struct<(memref<?xf32>, i8)>} : () -> index
// CHECK-NEXT:     %1 = arith.index_cast %0 : index to i64
// CHECK-NEXT:     %2 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %3 = llvm.bitcast %2 : !llvm.ptr<i8> to !llvm.ptr<!llvm.struct<(memref<?xf32>, i8)>>
// CHECK-NEXT:     return %3 : !llvm.ptr<!llvm.struct<(memref<?xf32>, i8)>>
// CHECK-NEXT:   }
