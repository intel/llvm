// RUN: %clang_cc1 %s -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -fsyntax-only -sycl-std=2020 -verify

// The test checks support and functionality of [[intel::num_simd_work_items()]] attribute.

#include "sycl.hpp"

using namespace sycl;
queue q;

// expected-warning@+1 {{unknown attribute 'num_simd_work_items' ignored}}
[[intelfpga::num_simd_work_items(22)]] void RemoveSpelling();

// No diagnostic is emitted because the arguments match.
[[intel::num_simd_work_items(12)]] void bar();
[[intel::num_simd_work_items(12)]] void bar() {} // OK

// Diagnostic is emitted because the arguments mismatch.
[[intel::num_simd_work_items(12)]] void baz();  // expected-note {{previous attribute is here}}
[[intel::num_simd_work_items(100)]] void baz(); // expected-warning {{attribute 'num_simd_work_items' is already applied with different arguments}}

// If the declaration has a [[sycl::reqd_work_group_size]]
// or [[cl::reqd_work_group_size]] attribute, tests that check
// if the work group size attribute argument (the last argument)
// can be evenly divided by the [[intel::num_simd_work_items()]] attribute.
struct TRIFuncObjBad1 {
  [[intel::num_simd_work_items(3)]]       // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(3, 6, 5)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

struct TRIFuncObjBad2 {
  [[sycl::reqd_work_group_size(3, 6, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]       // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

// Tests for the default values of [[sycl::reqd_work_group_size()]].

struct TRIFuncObjGood3 {
  [[intel::num_simd_work_items(3)]]
  [[sycl::reqd_work_group_size(3)]]
  void
  operator()() const {}
};

struct TRIFuncObjGood4 {
  [[sycl::reqd_work_group_size(3)]]
  [[intel::num_simd_work_items(3)]]
  void
  operator()() const {}
};

struct TRIFuncObjGood5 {
  [[intel::num_simd_work_items(4)]]
  [[sycl::reqd_work_group_size(4, 64)]]
  void
  operator()() const {}
};

struct TRIFuncObjGood6 {
  [[sycl::reqd_work_group_size(4, 64)]]
  [[intel::num_simd_work_items(4)]]
  void
  operator()() const {}
};

struct TRIFuncObjBad7 {
  [[sycl::reqd_work_group_size(6, 3, 5)]] // expected-note{{conflicting attribute is here}}
  [[intel::num_simd_work_items(3)]]       // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  void
  operator()() const {}
};

struct TRIFuncObjBad8 {
  [[intel::num_simd_work_items(3)]]       // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  [[sycl::reqd_work_group_size(6, 3, 5)]] // expected-note{{conflicting attribute is here}}
  void
  operator()() const {}
};

[[intel::num_simd_work_items(2)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
[[sycl::reqd_work_group_size(4, 2, 3)]] void
func1(); // expected-note@-1{{conflicting attribute is here}}

[[sycl::reqd_work_group_size(4, 2, 3)]] // expected-note{{conflicting attribute is here}}
[[intel::num_simd_work_items(2)]] void
func2(); // expected-error@-1{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

[[intel::num_simd_work_items(2)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
[[cl::reqd_work_group_size(4, 2, 3)]] void
func3(); // expected-note@-1{{conflicting attribute is here}} expected-warning@-1 {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note@-1 {{did you mean to use 'sycl::reqd_work_group_size' instead?}}

[[cl::reqd_work_group_size(4, 2, 3)]] // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
[[intel::num_simd_work_items(2)]] void
func4(); // expected-error@-1{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

// If the declaration has a __attribute__((reqd_work_group_size()))
// attribute, tests that check if the work group size attribute argument
// (the last argument) can be evenly divided by the
// [[intel::num_simd_work_items()]] attribute.
[[intel::num_simd_work_items(2)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
__attribute__((reqd_work_group_size(4, 2, 5))) void
func5(); // expected-note@-1{{conflicting attribute is here}} expected-warning@-1 {{attribute 'reqd_work_group_size' is deprecated}} expected-note@-1 {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}

[[intel::num_simd_work_items(2)]] __attribute__((reqd_work_group_size(3, 2, 6))) void func6(); // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                             // expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}

// Tests for incorrect argument values for Intel FPGA num_simd_work_items and reqd_work_group_size function attributes
struct TRIFuncObjBad9 {
  [[sycl::reqd_work_group_size(5, 5, 5)]] [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  void
  operator()() const {}
};

struct TRIFuncObjBad10 {
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(5, 5, 5)]] void
  operator()() const {}
};

struct TRIFuncObjBad11 {
  [[intel::num_simd_work_items(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[sycl::reqd_work_group_size(64, 64, 64)]] void
  operator()() const {}
};

struct TRIFuncObjBad12 {
  [[sycl::reqd_work_group_size(64, 64, 64)]] [[intel::num_simd_work_items(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void
  operator()() const {}
};

struct TRIFuncObjBad13 {
  [[sycl::reqd_work_group_size(0)]] // expected-error{{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  [[intel::num_simd_work_items(0)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  void
  operator()() const {}
};

struct TRIFuncObjBad14 {
  [[intel::num_simd_work_items(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  [[sycl::reqd_work_group_size(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void
  operator()() const {}
};

struct TRIFuncObjBad15 {
  [[intel::num_simd_work_items(3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void
  operator()() const {}
};

struct TRIFuncObjBad16 {
  [[intel::num_simd_work_items(3)]] [[sycl::reqd_work_group_size(3, 3, 3.f)]] // expected-error{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
  void
  operator()() const {}
};

struct TRIFuncObjBad17 {
  [[intel::num_simd_work_items(-1)]] // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(-1)]] // expected-error{{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  void
  operator()() const {}
};

struct TRIFuncObjBad18 {
  [[intel::num_simd_work_items(5)]] void // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
  operator()() const;
};

[[sycl::reqd_work_group_size(10, 5, 9)]] // expected-note{{conflicting attribute is here}}
void
TRIFuncObjBad18::operator()() const {}

struct TRIFuncObjBad19 {
  [[sycl::reqd_work_group_size(10, 5, 9)]] void // expected-note{{conflicting attribute is here}}
  operator()() const;
};

[[intel::num_simd_work_items(5)]] // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
void
TRIFuncObjBad19::operator()() const {}

int main() {
  q.submit([&](handler &h) {
    [[intel::num_simd_work_items(0)]] int Var = 0; // expected-error{{'num_simd_work_items' attribute only applies to functions}}

    h.single_task<class test_kernel1>(
        []() [[intel::num_simd_work_items(0)]] {}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel2>(
        []() [[intel::num_simd_work_items(-42)]] {}); // expected-error{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}

    h.single_task<class test_kernel3>(TRIFuncObjBad1());

    h.single_task<class test_kernel4>(TRIFuncObjBad2());

    h.single_task<class test_kernel5>(TRIFuncObjGood3());

    h.single_task<class test_kernel6>(TRIFuncObjGood4());

    h.single_task<class test_kernel7>(TRIFuncObjGood5());

    h.single_task<class test_kernel8>(TRIFuncObjGood6());

    h.single_task<class test_kernel9>(TRIFuncObjBad7());

    h.single_task<class test_kernel10>(TRIFuncObjBad8());

    h.single_task<class test_kernel11>(TRIFuncObjBad9());

    h.single_task<class test_kernel12>(TRIFuncObjBad10());

    h.single_task<class test_kernel13>(TRIFuncObjBad11());

    h.single_task<class test_kernel14>(TRIFuncObjBad12());

    h.single_task<class test_kernel15>(TRIFuncObjBad13());

    h.single_task<class test_kernel16>(TRIFuncObjBad14());

    h.single_task<class test_kernel17>(TRIFuncObjBad15());

    h.single_task<class test_kernel18>(TRIFuncObjBad16());

    h.single_task<class test_kernel19>(TRIFuncObjBad17());

    h.single_task<class test_kernel20>(TRIFuncObjBad18());

    h.single_task<class test_kernel21>(
        []() [[intel::num_simd_work_items(1), intel::num_simd_work_items(2)]] {}); // expected-warning{{attribute 'num_simd_work_items' is already applied with different arguments}}  // expected-note {{previous attribute is here}}

    h.single_task<class test_kernel22>(TRIFuncObjBad19());
  });
  return 0;
}

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.
template <typename Ty>
// expected-error@+3{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
// expected-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
// expected-error@+1{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
[[intel::num_simd_work_items(Ty{})]] void func7() {}

struct S {};
void test() {
  // expected-note@+1{{in instantiation of function template specialization 'func7<S>' requested here}}
  func7<S>();
  // expected-note@+1{{in instantiation of function template specialization 'func7<float>' requested here}}
  func7<float>();
  // expected-note@+1{{in instantiation of function template specialization 'func7<int>' requested here}}
  func7<int>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1{{declared here}}
int foo();
// expected-error@+2{{expression is not an integral constant expression}}
// expected-note@+1{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::num_simd_work_items(foo() + 12)]] void func8();

// Test that checks expression is a constant expression.
constexpr int barr() { return 0; }
[[intel::num_simd_work_items(barr() + 12)]] void func9(); // OK

// Tests for num_simd_work_items and reqd_work_group_size arguments check.
template <int N>
__attribute__((reqd_work_group_size(8, 6, 3))) void func10(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'reqd_work_group_size' is deprecated}} expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}
template <int N>
[[intel::num_simd_work_items(N)]] void func10(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[cl::reqd_work_group_size(8, 4, 5)]] void func11(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
template <int N>
[[intel::num_simd_work_items(N)]] void func11(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func12(); // expected-note{{conflicting attribute is here}}
template <int N>
[[intel::num_simd_work_items(3)]] void func12(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int X, int Y, int Z, int N>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func13(); // expected-note{{conflicting attribute is here}}
template <int X, int Y, int Z, int N>
[[intel::num_simd_work_items(N)]] void func13(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func14(); // expected-note{{conflicting attribute is here}}
template <int X, int Y, int Z>
[[intel::num_simd_work_items(3)]] void func14(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func15(); // expected-note{{conflicting attribute is here}}
template <int X, int Y, int Z>
[[intel::num_simd_work_items(2)]] void func15(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func16(); // expected-note{{conflicting attribute is here}}
template <int N>
[[intel::num_simd_work_items(2)]] void func16(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}

template <int N>
[[intel::num_simd_work_items(N)]] void func17(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int N>
__attribute__((reqd_work_group_size(8, 6, 3))) void func17(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'reqd_work_group_size' is deprecated}} expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}

template <int N>
[[intel::num_simd_work_items(N)]] void func18(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int N>
[[cl::reqd_work_group_size(8, 4, 5)]] void func18(); // expected-note{{conflicting attribute is here}} expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}

template <int N>
[[intel::num_simd_work_items(3)]] void func19(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func19(); // expected-note{{conflicting attribute is here}}

template <int X, int Y, int Z, int N>
[[intel::num_simd_work_items(N)]] void func20(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int X, int Y, int Z, int N>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func20(); // expected-note{{conflicting attribute is here}}

template <int X, int Y, int Z>
[[intel::num_simd_work_items(3)]] void func21(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func21(); // expected-note{{conflicting attribute is here}}

template <int X, int Y, int Z>
[[intel::num_simd_work_items(2)]] void func22(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void func22(); // expected-note{{conflicting attribute is here}}

template <int N>
[[intel::num_simd_work_items(2)]] void func23(); // expected-error{{'num_simd_work_items' attribute must evenly divide the work-group size for the 'reqd_work_group_size' attribute}}
template <int N>
[[sycl::reqd_work_group_size(N, N, N)]] void func23(); // expected-note{{conflicting attribute is here}}

int check1() {
  func10<3>();          // OK
  func10<2>();          // expected-note {{in instantiation of function template specialization 'func10<2>' requested here}}
  func11<4>();          // expected-note {{in instantiation of function template specialization 'func11<4>' requested here}}
  func11<5>();          // OK
  func12<5>();          // expected-note {{in instantiation of function template specialization 'func12<5>' requested here}}
  func12<3>();          // OK
  func13<6, 3, 5, 3>(); // expected-note {{in instantiation of function template specialization 'func13<6, 3, 5, 3>' requested here}}
  func13<9, 6, 3, 3>(); // OK
  func14<6, 3, 5>();    // expected-note {{in instantiation of function template specialization 'func14<6, 3, 5>' requested here}}
  func14<9, 6, 3>();    // OK
  func15<6, 4, 5>();    // expected-note {{in instantiation of function template specialization 'func15<6, 4, 5>' requested here}}
  func15<8, 6, 2>();    // OK
  func16<3>();          // expected-note {{in instantiation of function template specialization 'func16<3>' requested here}}
  func16<2>();          // OK
  func17<3>();          // OK
  func17<2>();          // expected-note {{in instantiation of function template specialization 'func17<2>' requested here}}
  func18<4>();          // expected-note {{in instantiation of function template specialization 'func18<4>' requested here}}
  func18<5>();          // OK
  func19<5>();          // expected-note {{in instantiation of function template specialization 'func19<5>' requested here}}
  func19<3>();          // OK
  func20<6, 3, 5, 3>(); // expected-note {{in instantiation of function template specialization 'func20<6, 3, 5, 3>' requested here}}
  func20<9, 6, 3, 3>(); // OK
  func21<6, 3, 5>();    // expected-note {{in instantiation of function template specialization 'func21<6, 3, 5>' requested here}}
  func21<9, 6, 3>();    // OK
  func22<6, 4, 5>();    // expected-note {{in instantiation of function template specialization 'func22<6, 4, 5>' requested here}}
  func22<8, 6, 2>();    // OK
  func23<3>();          // expected-note {{in instantiation of function template specialization 'func23<3>' requested here}}
  func23<2>();          // OK
  return 0;
}

// Test that checks template parameter support on member function of class template.
template <int SIZE>
class KernelFunctor {
public:
  // expected-error@+1{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
  [[intel::num_simd_work_items(SIZE)]] void operator()() {}
};

int kernel() {
  // expected-note@+1{{in instantiation of template class 'KernelFunctor<-1>' requested here}}
  KernelFunctor<-1>();
  // no error expected
  KernelFunctor<10>();
  return 0;
}

// Test that checks template parameter support on function.
template <int N>
// expected-error@+1{{'num_simd_work_items' attribute requires a positive integral compile time constant expression}}
[[intel::num_simd_work_items(N)]] void func24() {}

template <int N>
[[intel::num_simd_work_items(4)]] void func25(); // expected-note {{previous attribute is here}}

template <int N>
[[intel::num_simd_work_items(N)]] void func25() {} // expected-warning {{attribute 'num_simd_work_items' is already applied with different arguments}}

int ver() {
  // no error expected.
  func24<8>();
  // expected-note@+1{{in instantiation of function template specialization 'func24<-1>' requested here}}
  func24<-1>();
  // expected-note@+1 {{in instantiation of function template specialization 'func25<6>' requested here}}
  func25<6>();
  return 0;
}

// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::num_simd_work_items(2)]] [[intel::num_simd_work_items(2)]] void func26() {}
