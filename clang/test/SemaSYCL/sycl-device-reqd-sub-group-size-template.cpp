// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checkes template parameter support for 'reqd_sub_group_size' attribute on sycl device.

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+3{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::reqd_sub_group_size(Ty{})]] void func() {}

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
[[intel::reqd_sub_group_size(foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::reqd_sub_group_size(bar() + 12)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() {}
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
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}10{{$}}

// Test that checks template parameter support on function.
template <int N>
// expected-error@+1{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
[[intel::reqd_sub_group_size(N)]] void func3() {}

template <int N>
[[intel::reqd_sub_group_size(4)]] void func4(); // expected-note {{previous attribute is here}}

template <int N>
[[intel::reqd_sub_group_size(N)]] void func4() {} // expected-warning {{attribute 'reqd_sub_group_size' is already applied with different arguments}}

int check() {
  // no error expected
  func3<12>();
  //expected-note@+1{{in instantiation of function template specialization 'func3<-1>' requested here}}
  func3<-1>();
  func4<6>(); //expected-note {{in instantiation of function template specialization 'func4<6>' requested here}}
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::reqd_sub_group_size(8)]]
[[intel::reqd_sub_group_size(8)]] void func5() {}

// CHECK: FunctionTemplateDecl {{.*}} {{.*}} func3
// CHECK: NonTypeTemplateParmDecl {{.*}} {{.*}} referenced 'int' depth 0 index 0 N
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}12{{$}}

// CHECK: FunctionDecl {{.*}} {{.*}} func5 'void ()'
// CHECK: IntelReqdSubGroupSizeAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 8
// CHECK-NEXT: IntegerLiteral{{.*}}8{{$}}
