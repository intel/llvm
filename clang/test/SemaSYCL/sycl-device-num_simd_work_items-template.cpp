// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checkes template parameter support for 'num_simd_work_items' attribute on sycl device.

template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[intel::num_simd_work_items(SIZE)]] void operator()() {}
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
// CHECK: SYCLIntelNumSimdWorkItemsAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}
