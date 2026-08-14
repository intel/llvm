// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-module-interface %t/M.cppm -o %t/M.pcm
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -fmodule-file=M=%t/M.pcm %t/user.cpp -emit-llvm -o - | FileCheck %s

// The devirtualized flag of a declcall must survive C++20 module
// serialization, just as it does for a PCH.

//--- M.cppm
export module M;
struct B { virtual int g(int); };
export constexpr auto getptr() { return declcall(((B *)0)->B::g(0)); }

//--- user.cpp
import M;
auto p = getptr();
// A direct pointer to the member (devirtualized), not a vtable index.
// CHECK: @p = {{.*}}global { i64, i64 } { i64 ptrtoint (ptr @_ZNW1M1B1gEi to i64), i64 0 }
