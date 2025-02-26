// RUN: %clang_cc1 %s -fsyntax-only -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2020 -triple spir64 -verify

// The test checks support and functionality of [[intel::max_global_work_dim()]] attribute.

#include "sycl.hpp"

using namespace sycl;
queue q;

struct Func {
  // expected-warning@+1 {{unknown attribute 'max_global_work_dim' ignored}}
  [[intelfpga::max_global_work_dim(2)]] void operator()() const {}
};

// No diagnostic is emitted because the arguments match.
[[intel::max_global_work_dim(1)]] void bar();
[[intel::max_global_work_dim(1)]] void bar() {}

// Checking of different argument values.
[[intel::max_global_work_dim(2)]] void baz();  // expected-note {{previous attribute is here}}
[[intel::max_global_work_dim(1)]] void baz();  // expected-warning {{attribute 'max_global_work_dim' is already applied with different arguments}}

struct TRIFuncObj {
  [[intel::max_global_work_dim(0)]] void operator()() const; // expected-note {{previous attribute is here}}
};
[[intel::max_global_work_dim(1)]] void TRIFuncObj::operator()() const {} // expected-warning {{attribute 'max_global_work_dim' is already applied with different arguments}}

struct TRIFuncObjBad1 {
  [[sycl::reqd_work_group_size(4, 4, 4)]] void // expected-error {{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  operator()() const;
};

[[intel::max_global_work_dim(0)]]
void TRIFuncObjBad1::operator()() const {}

// Checks correctness of mutual usage of different work_group_size attributes:
// reqd_work_group_size, max_work_group_size and max_global_work_dim.
// In case the value of 'max_global_work_dim' attribute equals to 0 we shall
// ensure that if max_work_group_size and reqd_work_group_size attributes exist,
// they hold equal values (1, 1, 1).

struct TRIFuncObjBad2 {
  [[intel::max_global_work_dim(0)]]
  [[intel::max_work_group_size(8, 8, 8)]] // expected-error{{all 'max_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  [[sycl::reqd_work_group_size(4, 4, 4)]] // expected-error{{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  void
  operator()() const {}
};

struct TRIFuncObjBad3 {
  [[intel::max_work_group_size(8, 8, 8)]] // expected-error{{all 'max_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad4 {
  [[sycl::reqd_work_group_size(4, 4, 4)]]   // expected-error{{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad5 {
  [[sycl::reqd_work_group_size(4)]]   // expected-error{{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad6 {
  [[intel::max_global_work_dim(0)]] void
  operator()() const;
};

[[sycl::reqd_work_group_size(4, 4, 4)]] // expected-error{{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
void
TRIFuncObjBad6::operator()() const {}

struct TRIFuncObjBad7 {
  [[intel::max_global_work_dim(0)]] void
  operator()() const;
};

[[sycl::reqd_work_group_size(4, 4, 4)]] // expected-error{{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
void
TRIFuncObjBad7::operator()() const {}

struct TRIFuncObjBad8 {
  [[intel::max_global_work_dim(0)]] void
  operator()() const;
};

[[intel::max_work_group_size(4, 4, 4)]] // expected-error{{all 'max_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
void
TRIFuncObjBad8::operator()() const {}

// Tests for incorrect argument values for Intel FPGA function attributes:
// reqd_work_group_size, max_work_group_size and max_global_work_dim.

struct TRIFuncObjBad9 {
  // expected-error@+1{{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(-4, 1)]]
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad10 {
  [[intel::max_work_group_size(4, 4, 4.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad11 {
  [[sycl::reqd_work_group_size(0, 4, 4)]] // expected-error{{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  [[intel::max_global_work_dim(0)]] void
  operator()() const {}
};

struct TRIFuncObjBad12 {
  [[sycl::reqd_work_group_size(4)]]
  [[intel::max_global_work_dim(-2)]] // expected-error{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
  void operator()() const {}
};

struct TRIFuncObjBad13 {
  [[intel::max_work_group_size(4, 4, 4)]]
  [[intel::max_global_work_dim(4.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void operator()() const {}
};

struct TRIFuncObjBad14 {
  [[intel::max_work_group_size(4, 4, 4)]] void // expected-error{{all 'max_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}
  operator()() const;
};

[[intel::max_global_work_dim(0)]] void TRIFuncObjBad14::operator()() const {}

int main() {
  q.submit([&](handler &h) {
    h.single_task<class test_kernel12>(TRIFuncObjBad1());
    h.single_task<class test_kernel13>(TRIFuncObjBad2());
    [[intel::max_global_work_dim(1)]] int Var = 0; // expected-error{{'max_global_work_dim' attribute only applies to functions}}

    h.single_task<class test_kernel15>(
        []() [[intel::max_global_work_dim(-8)]]{}); // expected-error{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}

    h.single_task<class test_kernell6>(
        []() [[intel::max_global_work_dim(3),      // expected-note {{previous attribute is here}}
               intel::max_global_work_dim(2)]]{}); // expected-warning{{attribute 'max_global_work_dim' is already applied with different arguments}}

    h.single_task<class test_kernel17>(TRIFuncObjBad3());
    h.single_task<class test_kernel18>(TRIFuncObjBad4());
    h.single_task<class test_kernel19>(TRIFuncObjBad5());
    h.single_task<class test_kernel20>(TRIFuncObjBad6());
    h.single_task<class test_kernel21>(TRIFuncObjBad7());
    h.single_task<class test_kernel22>(TRIFuncObjBad8());
    h.single_task<class test_kernel23>(TRIFuncObjBad9());
    h.single_task<class test_kernel24>(TRIFuncObjBad10());
    h.single_task<class test_kernel25>(TRIFuncObjBad11());
    h.single_task<class test_kernel26>(TRIFuncObjBad12());
    h.single_task<class test_kernel27>(TRIFuncObjBad13());
    h.single_task<class test_kernel28>(TRIFuncObjBad14());

    h.single_task<class test_kernel28>(
        []() [[intel::max_global_work_dim(4)]] {}); // expected-error{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
  });
  return 0;
}

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+1 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
[[intel::max_global_work_dim(Ty{})]] void func() {}

struct S {};
void test() {
  // expected-note@+1{{in instantiation of function template specialization 'func<S>' requested here}}
  func<S>();
  // expected-note@+1{{in instantiation of function template specialization 'func<float>' requested here}}
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
constexpr int bars() { return 0; }
[[intel::max_global_work_dim(bars() + 2)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'max_global_work_dim' attribute requires integer constant between 0 and 3 inclusive}}
  [[intel::max_global_work_dim(SIZE)]] void operator()() {}
};

int kernel() {
  // expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  // no error expected
  KernelFunctor<2>();
}

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

template <int N>
[[intel::max_global_work_dim(0)]] void func15();
template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func15(); // expected-error {{all 'reqd_work_group_size' attribute arguments must be '1' when the 'max_global_work_dim' attribute argument is '0'}}

template <int N>
[[intel::max_global_work_dim(0)]] void func16();
template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func16();

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
  func15<6>(); // expected-note {{in instantiation of function template specialization 'func15<6>' requested here}}
  func16<1>(); // OK
  return 0;
}

// No diagnostic is emitted because the arguments match.
[[intel::max_global_work_dim(2)]] [[intel::max_global_work_dim(2)]] void func17() {}
