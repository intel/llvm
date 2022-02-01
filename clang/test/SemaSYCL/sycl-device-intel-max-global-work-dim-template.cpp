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
  // expected-error@+1{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
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
// expected-error@+1{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
[[intel::max_global_work_dim(N)]] void func3() {}

// Test that checks template instantiations for different argument values.
template <int size>
[[intel::max_global_work_dim(1)]] void func4(); // expected-note {{previous attribute is here}}

template <int size>
[[intel::max_global_work_dim(size)]] void func4() {} // expected-warning {{attribute 'max_global_work_dim' is already applied with different arguments}}

// Checks correctness of mutual usage of different work_group_size attributes:
// reqd_work_group_size, max_work_group_size, and max_global_work_dim.
// In case the value of 'max_global_work_dim' attribute equals to 0 we shall
// ensure that if max_work_group_size and reqd_work_group_size attributes exist,
// they hold equal values (1, 1, 1).
template <int N>
[[intel::max_work_group_size(N, N, N)]] void func5(); // expected-error {{all 'max_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
template <int N>
[[intel::max_global_work_dim(0)]] void func5();

template <int N>
[[sycl::reqd_work_group_size(N)]] void func6(); // expected-error {{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
template <int N>
[[intel::max_global_work_dim(0)]] void func6();

template <int N>
[[sycl::reqd_work_group_size(N, N)]] void func7(); // expected-error {{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
template <int N>
[[intel::max_global_work_dim(0)]] void func7();

template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func8(); // expected-error {{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
template <int N>
[[intel::max_global_work_dim(0)]] void func8();

template <int N>
[[intel::max_work_group_size(N, N, N)]] void func9();
template <int N>
[[intel::max_global_work_dim(0)]] void func9();

template <int N>
[[sycl::reqd_work_group_size(N)]] void func10();
template <int N>
[[intel::max_global_work_dim(0)]] void func10();

template <int N>
[[sycl::reqd_work_group_size(N, N)]] void func11();
template <int N>
[[intel::max_global_work_dim(0)]] void func11();

template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func12();
template <int N>
[[intel::max_global_work_dim(0)]] void func12();

template <int N>
[[intel::max_global_work_dim(0)]] void func13();
template <int N>
[[intel::max_work_group_size(N, N, N)]] void func13(); // expected-error {{all 'max_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}

template <int N>
[[intel::max_global_work_dim(0)]] void func14();
template <int N>
[[intel::max_work_group_size(N, N, N)]] void func14();

int check() {
  func3<3>();  // OK
  func3<-1>(); // expected-note {{in instantiation of function template specialization 'func3<-1>' requested here}}
  func4<2>();  // expected-note {{in instantiation of function template specialization 'func4<2>' requested here}}
  func5<2>();  // expected-note {{in instantiation of function template specialization 'func5<2>' requested here}}
  func6<2>();  // expected-note {{in instantiation of function template specialization 'func6<2>' requested here}}
  func7<2>();  // expected-note {{in instantiation of function template specialization 'func7<2>' requested here}}
  func8<2>();  // expected-note {{in instantiation of function template specialization 'func8<2>' requested here}}
  func9<1>();  // OK
  func10<1>(); // OK
  func11<1>(); // OK
  func12<1>(); // OK
  func13<6>(); // expected-note {{in instantiation of function template specialization 'func13<6>' requested here}}
  func14<1>(); // OK
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::max_global_work_dim(2)]] [[intel::max_global_work_dim(2)]] void func15() {}
// CHECK: FunctionDecl {{.*}} {{.*}} func3 'void ()'
// CHECK: TemplateArgument integral 3
// CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 3
// CHECK-NEXT: SubstNonTypeTemplateParmExpr {{.*}}
// CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}
// CHECK-NEXT: IntegerLiteral{{.*}}3{{$}}

// CHECK: FunctionDecl {{.*}} {{.*}} func15 'void ()'
// CHECK: SYCLIntelMaxGlobalWorkDimAttr {{.*}}
// CHECK-NEXT: ConstantExpr {{.*}} 'int'
// CHECK-NEXT: value: Int 2
// CHECK-NEXT: IntegerLiteral{{.*}}2{{$}}
