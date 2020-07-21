// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checkes template parameter support for 'intel_reqd_sub_group_size' attribute on sycl device.

template <int SIZE>
class KernelFunctor {
public:
  // expected-warning@+3{{attribute 'intel_reqd_sub_group_size' is deprecated}}
  // expected-note@+2 {{did you mean to use 'intel::reqd_sub_group_size' instead?}}
  // expected-error@+1{{'intel_reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
  [[cl::intel_reqd_sub_group_size(SIZE)]] void operator()() {}
};

int main() {
  //expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  // no error expected
  KernelFunctor<10>();
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}
