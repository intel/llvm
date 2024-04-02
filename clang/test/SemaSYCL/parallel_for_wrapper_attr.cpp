// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -sycl-std=2017 -fsycl-is-device -triple spir64 | FileCheck %s

#include "Inputs/sycl.hpp"

template <typename T> class Fobj {
public:
  Fobj() {}
  void operator()() const {
    auto L0 = []() [[intel::reqd_sub_group_size(4)]]{};
    L0();
  }
};

void invoke() {
  sycl::queue q;
  q.submit([&](sycl::handler &h) {
    Fobj<int> fobj1;
    h.parallel_for<class __pf_kernel_wrapper>(fobj1);
  });
  q.submit([&](sycl::handler &h) {
    Fobj<short> fobj2;
    h.parallel_for<class PPP>(fobj2);
  });
}

// CHECK-LABEL: ClassTemplateSpecializationDecl {{.*}} class Fobj definition
// CHECK:       TemplateArgument type 'int'
// CHECK:       CXXMethodDecl {{.*}} used operator() 'void () const' implicit_instantiation implicit-inline
// CHECK:       CXXMethodDecl {{.*}} used constexpr operator() 'void () const' implicit_instantiation inline
// CHECK-NEXT:  CompoundStmt
// CHECK-NEXT:  IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr {{.*}} 'int'
// CHECK-NEXT:  value: Int 4
// CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
// CHECK:       CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK:       IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr {{.*}} 'int'
// CHECK-NEXT:  value: Int 4
// CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
// CHECK:       CXXConstructorDecl
// CHECK:       CXXConstructorDecl

// CHECK-LABEL: ClassTemplateSpecializationDecl {{.*}} class Fobj definition
// CHECK:       TemplateArgument type 'short'
// CHECK:       CXXMethodDecl {{.*}} used operator() 'void () const' implicit_instantiation implicit-inline
// CHECK:       CXXMethodDecl {{.*}} used constexpr operator() 'void () const' implicit_instantiation inline
// CHECK-NEXT:  CompoundStmt
// CHECK-NEXT:  IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT:  ConstantExpr {{.*}} 'int'
// CHECK-NEXT:  value: Int 4
// CHECK-NEXT:  IntegerLiteral{{.*}}4{{$}}
// CHECK:       CXXOperatorCallExpr {{.*}} 'void' '()'
// CHECK-NOT:   IntelReqdSubGroupSizeAttr {{.*}}
// CHECK:       CXXConstructorDecl
