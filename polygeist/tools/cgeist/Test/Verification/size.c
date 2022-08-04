// RUN: cgeist %s --function=* -S | FileCheck %s

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
// CHECK-NEXT:     %c24_i64 = arith.constant 24 : i64
// CHECK-NEXT:     return %c24_i64 : i64
// CHECK-NEXT:   }
