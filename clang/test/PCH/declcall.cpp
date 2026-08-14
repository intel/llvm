// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -include-pch %t -emit-llvm %s -o - | FileCheck %s

// Check that the devirtualized flag of a declcall expression survives
// serialization. The declcall is written into the PCH and re-evaluated in the
// second run.

#ifndef HEADER
#define HEADER
struct B { virtual int g(int); };
constexpr auto getptr() { return declcall(((B *)0)->B::g(0)); }
#else
auto p = getptr();
// A direct pointer to the member (devirtualized), not a vtable index
// ({ i64 1, i64 0 }).
// CHECK: @p = {{.*}}global { i64, i64 } { i64 ptrtoint (ptr @_ZN1B1gEi to i64), i64 0 }
#endif
