// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks template parameter support for 'fpga_pipeline' attribute on sycl device.

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+1{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
[[intel::fpga_pipeline(Ty{})]] void func() {}

struct S {};
void var() {
  //expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::fpga_pipeline(foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::fpga_pipeline(bar() + 12)]] void func2(); // OK

// Test that checks template parameter suppport on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  [[intel::fpga_pipeline(SIZE)]] void operator()() {}
};

int main() {
  KernelFunctor<1>();
}

// No diagnostic is thrown since arguments match. Silently ignore duplicate attribute.
[[intel::fpga_pipeline]] void func3 ();
[[intel::fpga_pipeline(1)]] void func3() {} // OK

[[intel::fpga_pipeline(0)]] void func4(); // expected-note {{previous attribute is here}}
[[intel::fpga_pipeline]] void func4();    // expected-warning{{attribute 'fpga_pipeline' is already applied with different arguments}}

// No diagnostic is emitted because the arguments match.
[[intel::fpga_pipeline(1)]] void func5();
[[intel::fpga_pipeline(1)]] void func5() {} // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::fpga_pipeline(0)]] void func6(); // expected-note {{previous attribute is here}}
[[intel::fpga_pipeline(1)]] void func6(); // expected-warning{{attribute 'fpga_pipeline' is already applied with different arguments}}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelFpgaPipelineAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}

// Test that checks template parameter suppport on function.
template <int N>
[[intel::fpga_pipeline(N)]] void func6() {}

template <int N>
[[intel::fpga_pipeline(0)]] void func7();   // expected-note {{previous attribute is here}}
template <int N>
[[intel::fpga_pipeline(N)]] void func7() {} // expected-warning {{attribute 'fpga_pipeline' is already applied with different arguments}}

int check() {
  func6<1>();
  func7<1>(); //expected-note {{in instantiation of function template specialization 'func7<1>' requested here}}
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::fpga_pipeline(1)]]
[[intel::fpga_pipeline(1)]] void func8() {}

// CHECK: FunctionDecl {{.*}} {{.*}} func6 'void ()'
// CHECK: TemplateArgument integral 1
// CHECK: SYCLIntelFpgaPipelineAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}

// CHECK: FunctionDecl {{.*}} {{.*}} func8 'void ()'
// CHECK: SYCLIntelFpgaPipelineAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
