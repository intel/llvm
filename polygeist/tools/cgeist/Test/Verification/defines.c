// RUN: cgeist -DPASS -DTEST0 -DTEST1 -DTEST2 %s -S -o -

// CHECK-LABEL: func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    return %c0_i32 : i32
// CHECK-NEXT:  }

int main() {
#ifdef PASS
  return 0;
#else
  return 1;
#endif
}
