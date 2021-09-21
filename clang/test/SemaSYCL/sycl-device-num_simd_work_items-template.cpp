// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checkes template parameter support for 'num_simd_work_items' attribute on sycl device.

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+3{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
// expected-error@+1{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
[[intel::num_simd_work_items(Ty{})]] void func() {}

struct S {};
void test() {
  //expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
  //expected-note@+1{{in instantiation of function template specialization 'func<float>' requested here}}
  func<float>();
  //expected-note@+1{{in instantiation of function template specialization 'func<int>' requested here}}
  func<int>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::num_simd_work_items(foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::num_simd_work_items(bar() + 12)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
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
  return 0;
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelNumSimdWorkItemsAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}

// Test that checks template parameter support on function.
template <int N>
// expected-error@+1{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
[[intel::num_simd_work_items(N)]] void func3() {}

template <int N>
[[intel::num_simd_work_items(4)]] void func4(); // expected-note {{previous attribute is here}}

template <int N>
[[intel::num_simd_work_items(N)]] void func4() {} // expected-warning {{attribute 'num_simd_work_items' is already applied with different arguments}}

int check() {
  // no error expected.
  func3<8>();
  //expected-note@+1{{in instantiation of function template specialization 'func3<-1>' requested here}}
  func3<-1>();
  //expected-note@+1 {{in instantiation of function template specialization 'func4<6>' requested here}}
  func4<6>();
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::num_simd_work_items(2)]]
[[intel::num_simd_work_items(2)]] void func5() {}

// CHECK: FunctionTemplateDecl {{.*}} {{.*}} func3
// CHECK: NonTypeTemplateParmDecl {{.*}} {{.*}} referenced 'int' depth 0 index 0 N
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: SYCLIntelNumSimdWorkItemsAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}

// CHECK: FunctionDecl {{.*}} {{.*}} func5 'void ()'
// CHECK: SYCLIntelNumSimdWorkItemsAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}

// Tests for num_simd_work_items and reqd_work_group_size arguments check.
template <int N>
__attribute__((reqd_work_group_size(8, 6, 3))) void func6(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'reqd_work_group_size' is deprecated}} expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}
template <int N>
[[intel::num_simd_work_items(N)]] void func6(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[cl::reqd_work_group_size(8, 4, 5)]] void func7(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
template <int N>
[[intel::num_simd_work_items(N)]] void func7(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func8(); // expected-note{{conflicting attribute is here}}
template <int N>
[[intel::num_simd_work_items(3)]] void func8(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int X, int Y, int Z, int N>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func9(); // expected-note{{conflicting attribute is here}}
template <int X, int Y, int Z, int N>
[[intel::num_simd_work_items(N)]] void func9(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func10(); // expected-note{{conflicting attribute is here}}
template <int X, int Y, int Z>
[[intel::num_simd_work_items(3)]] void func10(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func11(); // expected-note{{conflicting attribute is here}}
template <int X, int Y, int Z>
[[intel::num_simd_work_items(2)]] void func11(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func12(); // expected-note{{conflicting attribute is here}}
template <int N>
[[intel::num_simd_work_items(2)]] void func12(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

int check1() {
  func6<3>(); // OK
  func6<2>(); // expected-note {{in instantiation of function template specialization 'func6<2>' requested here}}
  func7<4>(); // expected-note {{in instantiation of function template specialization 'func7<4>' requested here}}
  func7<5>(); // OK
  func8<5>(); // expected-note {{in instantiation of function template specialization 'func8<5>' requested here}}
  func8<3>(); // OK
  func9<6, 3, 5, 3>(); // expected-note {{in instantiation of function template specialization 'func9<6, 3, 5, 3>' requested here}}
  func9<9, 6, 3, 3>(); // OK
  func10<6, 3, 5>(); // expected-note {{in instantiation of function template specialization 'func10<6, 3, 5>' requested here}}
  func10<9, 6, 3>(); // OK
  func11<6, 4, 5>(); // expected-note {{in instantiation of function template specialization 'func11<6, 4, 5>' requested here}} 
  func11<8, 6, 2>(); // OK
  func12<3>(); // expected-note {{in instantiation of function template specialization 'func12<3>' requested here}} 
  func12<2>(); // OK
  return 0;
}
