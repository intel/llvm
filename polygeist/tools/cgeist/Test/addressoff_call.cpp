// RUN: cgeist %s --function=* -S | FileCheck %s
unsigned long foo(unsigned);
inline unsigned long inlineFunc(unsigned i) noexcept {
  return foo(i);
}

struct S {
  static unsigned long bar() noexcept {
    return (&inlineFunc)(0);
  }
};

void f(){
  auto res = S::bar();
}
// CHECK: func.func @_Z10inlineFuncj(%arg0: i32) -> i64 attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:   %0 = call @_Z3fooj(%arg0) : (i32) -> i64
// CHECK-NEXT:   return %0 : i64
// CHECK-NEXT: }
