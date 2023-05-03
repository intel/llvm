// RUN: cgeist --use-opaque-pointers %s --function=* -S | FileCheck %s

struct X {
   int a;
   long long b;
};
struct Y {
  int c;
  struct X x;
};
unsigned long long size() {
  return sizeof(struct Y);
}

// CHECK:   func @size() -> i64 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:     %0 = "polygeist.typeSize"() {source = !polygeist.struct<(i32, !polygeist.struct<(i32, i64)>)>} : () -> index
// CHECK-NEXT:     %1 = arith.index_cast %0 : index to i64
// CHECK-NEXT:     return %1 : i64
// CHECK-NEXT:   }
