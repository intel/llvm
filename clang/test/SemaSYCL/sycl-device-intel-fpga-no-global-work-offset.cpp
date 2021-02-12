// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks template parameter support for 'no_global_work_offset' attribute on sycl device.

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+1{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
[[intel::no_global_work_offset(Ty{})]] void func() {}

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
[[intel::no_global_work_offset(foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::no_global_work_offset(bar() + 12)]] void func2(); // OK

// Test that checks template parameter suppport on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  [[intel::no_global_work_offset(SIZE)]] void operator()() {}
};

int main() {
  KernelFunctor<1>();
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}

// Test that checks template parameter suppport on function.
template <int N>
[[intel::no_global_work_offset(N)]] void func3() {}

int check() {
  func3<1>();
  return 0;
}

// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: TemplateArgument integral 1
// CHECK: SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 1
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
