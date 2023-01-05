// RUN: %clang_cc1 -flto=thin -flto-unit -fsplit-lto-unit -triple x86_64-unknown-linux -fvisibility=hidden -emit-llvm-bc -o %t %s
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-modextract -opaque-pointers -o - -n 1 %t | llvm-dis -opaque-pointers | FileCheck %s
// Added -opaque-pointers.
// FIXME: Align with the community code when project is ready to enable opaque
// pointers by default
// RUN: llvm-modextract -opaque-pointers -b -o - -n 1 %t | llvm-bcanalyzer -dump | FileCheck %s --check-prefix=LTOUNIT
// LTOUNIT: <FLAGS op0=8/>

// CHECK: @_ZTV1A = linkonce_odr
class A {
  virtual void f() {}
};

A *f() {
  return new A;
}
