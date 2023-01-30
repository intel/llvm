// RUN:  %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// This test checks that static variables defined in host code and used in
// device code do not force emission of their parent host functions.

#include "sycl.hpp"

// CHECK-NOT: {{class.sycl::.*queue}}

// CHECK: @_ZZ4mainE3Loc = internal addrspace(1) constant i32 42, align 4
// CHECK: @_ZZ4mainE6Struct = internal addrspace(1) constant %struct.S { i32 38 }, align 4
// CHECK: @_ZL4Glob = internal addrspace(1) constant i64 100500, align 8
// CHECK: @_ZZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEvE6InKern = internal addrspace(1) constant i32 2022, align 4
// CHECK: @_ZN1SIiE6MemberE = available_externally addrspace(1) constant i32 1, align 4
// CHECK: @_ZZ3fooiE5InFoo = internal addrspace(1) constant i32 300, align 4

// CHECK: define{{.*}}@_Z3fooi(i32 noundef %In)
// CHECK-NOT: define{{.*}}@main()

template <class T> struct S {
  static const T Member = 1;
  int Parrots = 38;
};

static constexpr unsigned long Glob = 100500;
int foo(const int In) {
  static constexpr int InFoo = 300;
  return InFoo + In;
}

int main() {
  sycl::queue q;
  static constexpr int Loc = 42;
  static const S<int> Struct;
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class TheKernel>([=]() {
      (void)Loc;
      (void)Struct;

      // Make sure other use cases with statics are not broken by the change.
      (void)Glob;
      static const int InKern = 2022;
      foo(Loc);
      (void)S<int>::Member;
    });
  });

  return 0;
}
