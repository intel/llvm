// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checkes template parameter support for 'scheduler_target_fmax_mhz' attribute on sycl device.

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::scheduler_target_fmax_mhz(Ty{})]] void func() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func<float>' requested here}}
  func<float>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::scheduler_target_fmax_mhz(foo() + 1)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::scheduler_target_fmax_mhz(bar() + 2)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'scheduler_target_fmax_mhz' attribute requires a non-negative integral compile time constant expression}}
  [[intel::scheduler_target_fmax_mhz(SIZE)]] void operator()() {}
};

int main() {
  //expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  // no error expected
  KernelFunctor<2>();
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: TemplateArgument integral 2
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}

// Test that checks template parameter support on function.
template <int N>
// expected-error@+1{{'scheduler_target_fmax_mhz' attribute requires a non-negative integral compile time constant expression}}
[[intel::scheduler_target_fmax_mhz(N)]] void func3() {}

template <int N>
[[intel::scheduler_target_fmax_mhz(4)]] void func4(); // expected-note {{previous attribute is here}}

template <int N>
[[intel::scheduler_target_fmax_mhz(N)]] void func4() {} // expected-warning {{attribute 'scheduler_target_fmax_mhz' is already applied with different arguments}}

int check() {
  // no error expected
  func3<3>();
  //expected-note@+1{{in instantiation of function template specialization 'func3<-1>' requested here}}
  func3<-1>();
  //expected-note@+1 {{in instantiation of function template specialization 'func4<6>' requested here}}
  func4<6>(); 
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::scheduler_target_fmax_mhz(8)]]
[[intel::scheduler_target_fmax_mhz(8)]] void func5() {}

// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: TemplateArgument integral 3
// CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 3
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}

// CHECK: FunctionDecl {{.*}} {{.*}} func5 'void ()'
// CHECK: SYCLIntelSchedulerTargetFmaxMhzAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
