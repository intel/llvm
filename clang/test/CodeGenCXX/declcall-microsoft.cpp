// RUN: %clang_cc1 -std=c++2c -fsycl-is-host -triple x86_64-pc-windows-msvc -emit-llvm %s -o - | FileCheck %s

struct B { virtual int g(int); int h(int); };

// A qualified call to a virtual member devirtualizes to a direct function
// pointer, not a vftable (vcall) thunk.
// CHECK: @"?pv@@{{[^"]*}}" = {{.*}}global ptr @"?g@B@@UEAAHH@Z"
auto pv = declcall(((B *)0)->B::g(0));

// An unqualified virtual call keeps virtual dispatch: a vcall thunk.
// CHECK: @"?pu@@{{[^"]*}}" = {{.*}}global ptr @"??_9B@@$BA@AA"
auto pu = declcall(((B *)0)->g(0));

// A non-virtual member is a direct pointer.
// CHECK: @"?ph@@{{[^"]*}}" = {{.*}}global ptr @"?h@B@@QEAAHH@Z"
auto ph = declcall(((B *)0)->h(0));
