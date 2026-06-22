// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -ast-dump %s 2> /dev/null | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -verify -verify-ignore-unexpected=note -fsyntax-only %s

// This function verifies we don't trigger the error
//  SYCL kernel cannot call an undefined function without SYCL_EXTERNAL attribute
// due to the failure to fully instantiate Ker

#include "sycl.hpp"

sycl::queue deviceQueue;

// Checking we generate the AST of interest

// CHECK:      CXXOperatorCallExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}} 'void () const' lvalue CXXMethod [[MethodAddr:[0-9a-fx]*]] 'operator()' 'void () const'
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: SYCLKernelAttr

// CHECK:      ClassTemplateSpecializationDecl {{.*}} struct Ker definition implicit_instantiation
// CHECK:      CXXRecordDecl {{.*}} implicit struct Ker
// CHECK-NEXT: CXXMethodDecl [[MethodAddr]] {{.*}} used invalid operator() 'void () const'

struct NoCopy {
  NoCopy() {}
  NoCopy(const NoCopy&) = delete;
};

template <typename Cb>
struct Ker {
  void operator()() const {
    NoCopy Obj;
    Cb{}(Obj); //  expected-error {{call to deleted constructor of 'NoCopy'}}
  }
};

struct TakeByValue{
    void operator()(NoCopy) {}
};

int main() {
  deviceQueue.submit([&](sycl::handler &h) {
  Ker<TakeByValue> K;
    h.single_task<decltype(K)>(K);
  });
}
