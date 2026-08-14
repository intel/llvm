// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++2c -triple x86_64-pc-windows-msvc -emit-llvm %s -o /dev/null

int f(int);
int g(int);

template <class T> void sink(T);
template <class T> void h(decltype(declcall(f(T{}))) p) { sink(p); }
template <class T> void k(decltype(declcall(g(T{}))) p) { sink(p); }

// A dependent declcall reaches the Itanium mangler. It is mangled as a vendor
// extended expression (u <source-name> <template-arg>* E), so h and k -- which
// differ only in the declcall operand -- mangle distinctly, instead of both
// collapsing to a malformed, non-demanglable empty decltype ("DTE").
// CHECK-DAG: define {{.*}} @_Z1hIiEvDTu8declcallXcl1ftlT_EEEEE
// CHECK-DAG: define {{.*}} @_Z1kIiEvDTu8declcallXcl1gtlT_EEEEE
template void h<int>(int (*)(int));
template void k<int>(int (*)(int));
