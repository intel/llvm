// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks template parameter support for 'sycl_esimd_vectorize' attribute on sycl device.

// Test wrong function template instantiation and ensure that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+3{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
// expected-error@+1{{'sycl_esimd_vectorize' attribute argument must be 8, 16, or 32}}
[[intel::sycl_esimd_vectorize(Ty{})]] void func() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func<float>' requested here}}
  func<float>();
  //expected-note@+1{{in instantiation of function template specialization 'func<int>' requested here}}
  func<int>();
}

// Test a non-constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::sycl_esimd_vectorize(foo() + 12)]] void func1();

// Test a constant expression.
constexpr int bar() { return 0; }
[[intel::sycl_esimd_vectorize(bar() + 16)]] void func2(); // OK

// Test template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'sycl_esimd_vectorize' attribute argument must be 8, 16, or 32}}
  [[intel::sycl_esimd_vectorize(SIZE)]] void operator()() {}
};

int main() {
  //expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  // no error expected
  KernelFunctor<8>();
  return 0;
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelESimdVectorizeAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}

// Test template parameter support on function.
template <int N>
// expected-error@+1{{'sycl_esimd_vectorize' attribute argument must be 8, 16, or 32}}
[[intel::sycl_esimd_vectorize(N)]] void func3() {}

template <int N>
[[intel::sycl_esimd_vectorize(32)]] void func4(); // expected-note {{previous attribute is here}}

template <int N>
[[intel::sycl_esimd_vectorize(N)]] void func4() {} // expected-warning {{attribute 'sycl_esimd_vectorize' is already applied with different arguments}}

int check() {
  // no error expected.
  func3<8>();
  //expected-note@+1{{in instantiation of function template specialization 'func3<-1>' requested here}}
  func3<-1>();
  //expected-note@+1 {{in instantiation of function template specialization 'func4<16>' requested here}}
  func4<16>();
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::sycl_esimd_vectorize(8)]]
[[intel::sycl_esimd_vectorize(8)]] void func5() {}

// CHECK: FunctionTemplateDecl {{.*}} {{.*}} func3
// CHECK: NonTypeTemplateParmDecl {{.*}} {{.*}} referenced 'int' depth 0 index 0 N
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: SYCLIntelESimdVectorizeAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}

// CHECK: FunctionDecl {{.*}} {{.*}} func5 'void ()'
// CHECK: SYCLIntelESimdVectorizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
