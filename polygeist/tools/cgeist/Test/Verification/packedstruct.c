// RUN: cgeist %s --function=* -S | FileCheck %s

struct meta {
    long long a;
    char dtype;
};

struct fin {
    struct meta f;
    char dtype;
} __attribute__((packed)) ;

long long run(struct meta m, char c);

void compute(struct fin f) {
    run(f.f, f.dtype);
}

// CHECK:   func @compute(%arg0: !llvm.ptr<struct<(struct<(i64, i8)>, i8)>>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(struct<(i64, i8)>, i8)>>) -> !llvm.ptr<struct<(i64, i8)>>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<struct<(i64, i8)>>
// CHECK-NEXT:     %2 = llvm.getelementptr inbounds %arg0[0, 1] : (!llvm.ptr<struct<(struct<(i64, i8)>, i8)>>) -> !llvm.ptr<i8>
// CHECK-NEXT:     %3 = llvm.load %2 : !llvm.ptr<i8>
// CHECK-NEXT:     %4 = call @run(%1, %3) : (!llvm.struct<(i64, i8)>, i8) -> i64
// CHECK-NEXT:     return
// CHECK-NEXT:   }

