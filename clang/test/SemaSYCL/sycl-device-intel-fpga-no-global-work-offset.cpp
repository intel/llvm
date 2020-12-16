// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -ast-dump -verify -pedantic %s | FileCheck %s

// Test that checks template parameter support for 'no_global_work_offset' attribute on sycl device.

// Test that checks wrong template instantiation.
template <typename Ty>
[[intel::no_global_work_offset(Ty{})]] void func() {}

struct S {};
// expected-error@+2{{template specialization requires 'template<>'}}
// expected-error@+1{{C++ requires a type specifier for all declarations}}
func<S>();

// Test that checks expression is not a constant expression.
int foo();
// expected-error@+1{{'no_global_work_offset' attribute requires an integer constant}}
[[intel::no_global_work_offset(foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::no_global_work_offset(bar() + 12)]] void func2(); // OK

// Test that checks template parameter suppport on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'no_global_work_offset' attribute requires a non-negative integral compile time constant expression}}
  [[intel::no_global_work_offset(SIZE)]] void operator()() {}
};

int main() {
  //expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  // no error expected
  KernelFunctor<1>();
}

// CHECK: ClassTemplateDecl {{.*}} {{.*}} KernelFunctor
// CHECK: ClassTemplateSpecializationDecl {{.*}} {{.*}} class KernelFunctor definition
// CHECK: CXXRecordDecl {{.*}} {{.*}} implicit class KernelFunctor
// CHECK: SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}

// Test that checks template parameter suppport on function.
template <int N>
[[intel::no_global_work_offset(N)]] void func3() {}

int check() {
  func3<1>();
  return 0;
}

// CHECK: FunctionTemplateDecl {{.*}} {{.*}} func3
// CHECK: NonTypeTemplateParmDecl {{.*}} {{.*}} referenced 'int' depth 0 index 0 N
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: SYCLIntelNoGlobalWorkOffsetAttr {{.*}}
// CHECK: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}1{{$}}
