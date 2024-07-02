// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -fsyntax-only -verify -pedantic %s

// The test checks functionality of [[intel::reqd_sub_group_size()]] attribute on SYCL kernel.


#include "sycl.hpp" //clang/test/SemaSYCL/Inputs/sycl.hpp

sycl::queue q;

// Kernel defined as a named function object
class KernelFunctor1 {
 public:
  [[sycl::reqd_work_group_size(16)]] void operator()() const {};
};

// Kernel defined as a named function object
class KernelFunctor2 {
 public:
  void operator() [[sycl::reqd_work_group_size(16)]] () const {};
};

int main() {
  // Kernel defined as a lambda
  q.submit([&](sycl::handler& h) {
    KernelFunctor1 kf1;
    KernelFunctor2 kf2;
    h.single_task(kf1);
    h.single_task(kf2);
    h.single_task<class kernel_name>(
      []()[[sycl::reqd_work_group_size(16)]]{}
    );
  });
  return 0;
}
[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B();
[[intel::reqd_sub_group_size(16)]] void A() {
}

[[intel::reqd_sub_group_size(16)]] SYCL_EXTERNAL void B() {
  A();
}
// expected-note@+1 {{conflicting attribute is here}}
[[intel::reqd_sub_group_size(2)]] void sg_size2() {}

// expected-note@+2 {{conflicting attribute is here}}
// expected-error@+1 {{conflicting attributes applied to a SYCL kernel}}
[[intel::reqd_sub_group_size(4)]] __attribute__((sycl_device)) void sg_size4() {
  sg_size2();
}

// Test that checks support and functionality of reqd_sub_group_size attribute support on function.

// Tests for incorrect argument values for Intel reqd_sub_group_size attribute.
[[intel::reqd_sub_group_size]] void one() {}         // expected-error {{'reqd_sub_group_size' attribute takes one argument}}
[[intel::reqd_sub_group_size(5)]] int a;             // expected-error{{'reqd_sub_group_size' attribute only applies to functions}}
[[intel::reqd_sub_group_size("foo")]] void func() {} // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'const char[4]'}}
[[intel::reqd_sub_group_size(-1)]] void func1() {}   // expected-error{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
[[intel::reqd_sub_group_size(0, 1)]] void arg() {}   // expected-error{{'reqd_sub_group_size' attribute takes one argument}}

// Diagnostic is emitted because the arguments mismatch.
[[intel::reqd_sub_group_size(12)]] void quux();  // expected-note {{previous attribute is here}}
[[intel::reqd_sub_group_size(100)]] void quux(); // expected-warning {{attribute 'reqd_sub_group_size' is already applied with different arguments}} expected-note {{previous attribute is here}}
[[sycl::reqd_sub_group_size(200)]] void quux();  // expected-warning {{attribute 'reqd_sub_group_size' is already applied with different arguments}}

// Make sure there's at least one argument passed.
[[sycl::reqd_sub_group_size]] void quibble(); // expected-error {{'reqd_sub_group_size' attribute takes one argument}}

// No diagnostic is emitted because the arguments match.
[[intel::reqd_sub_group_size(12)]] void same();
[[intel::reqd_sub_group_size(12)]] void same() {} // OK

// No diagnostic because the attributes are synonyms with identical behavior.
[[sycl::reqd_sub_group_size(12)]] void same(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+3{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::reqd_sub_group_size(Ty{})]] void func() {}

struct S {};
void test() {
  // expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
  // expected-note@+1{{in instantiation of function template specialization 'func<float>' requested here}}
  func<float>();
  // expected-note@+1{{in instantiation of function template specialization 'func<int>' requested here}}
  func<int>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo1();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo1' cannot be used in a constant expression}}
[[intel::reqd_sub_group_size(foo1() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar1() { return 0; }
[[intel::reqd_sub_group_size(bar1() + 12)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() {}
};

int check() {
  // expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  return 0;
}

// Test that checks template parameter support on function.
template <int N>
// expected-error@+1{{'reqd_sub_group_size' attribute requires a positive integral compile time constant expression}}
[[intel::reqd_sub_group_size(N)]] void func3() {}

template <int N>
[[intel::reqd_sub_group_size(4)]] void func4(); // expected-note {{previous attribute is here}}

template <int N>
[[intel::reqd_sub_group_size(N)]] void func4() {} // expected-warning {{attribute 'reqd_sub_group_size' is already applied with different arguments}}

int check1() {
  // no error expected
  func3<12>();
  // expected-note@+1{{in instantiation of function template specialization 'func3<-1>' requested here}}
  func3<-1>();
  func4<6>(); // expected-note {{in instantiation of function template specialization 'func4<6>' requested here}}
  return 0;
}
