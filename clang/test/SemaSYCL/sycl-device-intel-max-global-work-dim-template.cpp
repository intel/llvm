// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checkes template parameter support for 'max_global_work_dim' attribute on sycl device.

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::max_global_work_dim(Ty{})]] void func() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func<float>' requested here}}
  func<float>();
  // no error expected
  func<int>(); // OK
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::max_global_work_dim(foo() + 1)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::max_global_work_dim(bar() + 2)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'max_global_work_dim' attribute requires a non-negative integral compile time constant expression}}
  [[intel::max_global_work_dim(SIZE)]] void operator()() {}
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
// CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}

// Test that checks template parameter support on function.
template <int N>
// expected-error@+1{{'max_global_work_dim' attribute requires a non-negative integral compile time constant expression}}
[[intel::max_global_work_dim(N)]] void func3() {}

int check() {
  // no error expected
  func3<3>();
  //expected-note@+1{{in instantiation of function template specialization 'func3<-1>' requested here}}
  func3<-1>();
  return 0;
}

// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: TemplateArgument integral 3
// CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 3
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}
