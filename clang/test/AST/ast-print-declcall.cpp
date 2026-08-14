// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -triple x86_64-unknown-linux-gnu -ast-print %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -triple x86_64-unknown-linux-gnu -ast-print %s -o - | %clang_cc1 -std=c++2c -fsycl-is-host -triple x86_64-unknown-linux-gnu -fsyntax-only -x c++ -

// declcall prints the original call expression it was written with (not the
// resolved callee), so the pretty-printed output re-parses.

int f(int);
int f(char);
struct B { virtual int g(int); };

// CHECK: auto p = declcall(f(0));
auto p = declcall(f(0));

// CHECK: auto q = declcall(((B *)0)->B::g(0));
auto q = declcall(((B *)0)->B::g(0));
