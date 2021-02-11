// RUN: %clang_cc1 %s -fsyntax-only -ast-dump -fsycl -fsycl-is-device -triple spir64 | FileCheck %s

template <typename Name, typename Type>
[[clang::sycl_kernel]] void __my_kernel__(Type bar) {
  bar();
}

template <typename Name, typename Type> void parallel_for(Type lambda) {
  __my_kernel__<Name>(lambda);
}

template <typename T> class Fobj {
public:
  Fobj() {}
  void operator()() const {
    auto L0 = []() [[intel::reqd_sub_group_size(4)]]{};
    L0();
  }
};

void invoke() {
  Fobj<int> fobj1;
  parallel_for<class __pf_kernel_wrapper>(fobj1);
  Fobj<short> fobj2;
  parallel_for<class PPP>(fobj2);
}

// CHECK-LABEL: ClassTemplateSpecializationDecl {{.*}} class Fobj definition
// CHECK:       TemplateArgument type 'int'
// CHECK:       CXXMethodDecl {{.*}} used operator() 'void () const'
// CHECK:       CXXMethodDecl {{.*}} used operator() 'void () const' inline
// CHECK-NEXT:  CompoundStmt
// CHECK-NEXT:  IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 4
// CHECK:       CXXOperatorCallExpr {{.*}} 'void':'void' '()'
// CHECK:       IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 4
// CHECK:       CXXDestructorDecl

// CHECK-LABEL: ClassTemplateSpecializationDecl {{.*}} class Fobj definition
// CHECK:       TemplateArgument type 'short'
// CHECK:       CXXMethodDecl {{.*}} used operator() 'void () const'
// CHECK:       CXXMethodDecl {{.*}} used operator() 'void () const' inline
// CHECK-NEXT:  CompoundStmt
// CHECK-NEXT:  IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT:  IntegerLiteral {{.*}} 'int' 4
// CHECK:       CXXOperatorCallExpr {{.*}} 'void':'void' '()'
// CHECK-NOT:   IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NOT:   IntegerLiteral {{.*}} 'int' 4
// CHECK:       CXXDestructorDecl
