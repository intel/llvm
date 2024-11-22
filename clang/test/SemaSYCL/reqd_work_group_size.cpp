// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// The test checks support and functionality of [[sycl::reqd_work_group_size()]] attribute.

// Check the basics.
[[sycl::reqd_work_group_size]] void f();                  // expected-error {{'reqd_work_group_size' attribute takes at least 1 argument}}
[[sycl::reqd_work_group_size(12, 12, 12, 12)]] void f0(); // expected-error {{'reqd_work_group_size' attribute takes no more than 3 arguments}}
[[sycl::reqd_work_group_size("derp", 1, 2)]] void f1();   // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'const char[5]'}}
[[sycl::reqd_work_group_size(1, 1, 1)]] int i;            // expected-error {{'reqd_work_group_size' attribute only applies to functions}}

class Functor33 {
public:
  // expected-error@+1{{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(32, -4)]] void operator()() const {}
};

class Functor30 {
public:
  // expected-error@+1 2{{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  [[sycl::reqd_work_group_size(30, -30, -30)]] void operator()() const {}
};

// Tests for 'reqd_work_group_size' attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
// expected-warning@+1 {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
[[sycl::reqd_work_group_size(6, 6, 6)]] [[sycl::reqd_work_group_size(6, 6, 6)]] void f2() {}

// No diagnostic is emitted because the arguments match.
[[sycl::reqd_work_group_size(32, 32, 32)]] void f3();
[[sycl::reqd_work_group_size(32, 32, 32)]] void f3(); // OK

// Produce a conflicting attribute warning when the args are different.
[[sycl::reqd_work_group_size(6, 6, 6)]]         // expected-note {{previous attribute is here}} // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
[[sycl::reqd_work_group_size(16, 16, 16)]] void // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}
f4() {}

// Catch the easy case where the attributes are all specified at once with
// different arguments.
struct TRIFuncObjGood1 {
  // expected-note@+3 {{previous attribute is here}}
  // expected-error@+2 {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-warning@+1 {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}} 
  [[sycl::reqd_work_group_size(64)]] [[sycl::reqd_work_group_size(128)]] void operator()() const {}
};

struct TRIFuncObjGood2 {
  // expected-note@+3 {{previous attribute is here}}
  // expected-error@+2 {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-warning@+1 {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
  [[sycl::reqd_work_group_size(64, 64)]] [[sycl::reqd_work_group_size(128, 128)]] void operator()() const {}
};

struct TRIFuncObjGood3 {
  [[sycl::reqd_work_group_size(8, 8)]] void // expected-note {{previous attribute is here}}
  operator()() const;
};

[[sycl::reqd_work_group_size(4, 4)]] // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}} \
// expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
void
TRIFuncObjGood3::operator()() const {}

// Show that the attribute works on member functions.
class Functor {
public:
  [[sycl::reqd_work_group_size(16, 16, 16)]] [[sycl::reqd_work_group_size(16, 16, 16)]] void operator()() const;
  [[sycl::reqd_work_group_size(16, 16, 16)]] [[sycl::reqd_work_group_size(32, 32, 32)]] void operator()(int) const; // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}} expected-note {{previous attribute is here}}
};

class FunctorC {
public:
  [[intel::max_work_group_size(64, 64, 64)]] [[sycl::reqd_work_group_size(64, 64, 64)]] void operator()() const;
  [[intel::max_work_group_size(16, 16, 16)]]      // expected-note {{conflicting attribute is here}}
  [[sycl::reqd_work_group_size(64, 64, 64)]] void // expected-error {{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}}
  operator()(int) const;
};

class Functor32 {
public:
  [[sycl::reqd_work_group_size(32, 1, 1)]]      // expected-note {{previous attribute is here}} // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
  [[sycl::reqd_work_group_size(1, 1, 32)]] void // expected-error{{attribute 'reqd_work_group_size' is already applied with different arguments}}
  operator()() const {}
};

// Ensure that template arguments behave appropriately based on instantiations.
template <int N>
[[sycl::reqd_work_group_size(N, 1, 1)]] void f6(); // #f6

// Test that template redeclarations also get diagnosed properly.
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(1, 1, 1)]] void f7(); // #f7prev

template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void f7() {} // #f7

// Test that a template redeclaration where the difference is known up front is
// diagnosed immediately, even without instantiation.
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, 1, Z)]] void f8(); // expected-note {{previous attribute is here}}
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, 2, Z)]] void f8(); // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}

void instantiate() {
  f6<1>(); // OK
  // expected-error@#f6 {{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  f6<-1>(); // expected-note {{in instantiation}}
  // expected-error@#f6 {{'reqd_work_group_size' attribute requires a positive integral compile time constant expression}}
  f6<0>();       // expected-note {{in instantiation}}
  f7<1, 1, 1>(); // OK, args are the same on the redecl.
  // expected-error@#f7 {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-note@#f7prev {{previous attribute is here}}
  // expected-warning@#f7prev {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
  // expected-warning@#f7prev {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
  f7<2, 2, 2>(); // expected-note {{in instantiation}}
}

// Tests for 'reqd_work_group_size' attribute duplication.

[[sycl::reqd_work_group_size(8)]]            // expected-note {{previous attribute is here}} // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
[[sycl::reqd_work_group_size(1, 1, 8)]] void // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}
f8(){};

[[sycl::reqd_work_group_size(32, 32, 1)]]            // expected-note {{previous attribute is here}} // expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
[[sycl::reqd_work_group_size(32, 32)]] void f9() {}  // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}

// Test that template redeclarations also get diagnosed properly.
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(64, 1, 1)]] void f10(); // #f10prev
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void f10() {} // #f10err

void test() {
  f10<64, 1, 1>(); // OK, args are the same on the redecl.
  // expected-error@#f10err {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-note@#f10prev {{previous attribute is here}}
  // expected-warning@#f10prev {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
  // expected-warning@#f10prev {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
  f10<1, 1, 64>(); // expected-note {{in instantiation}}
}

struct TRIFuncObjBad {
  [[sycl::reqd_work_group_size(32, 1, 1)]] void // expected-note {{previous attribute is here}}
  operator()() const;
};

[[sycl::reqd_work_group_size(1, 1, 32)]] // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}} \
// expected-warning {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
void
TRIFuncObjBad::operator()() const {}

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.

template <typename Ty, typename Ty1, typename Ty2>
// expected-error@+1 3{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
[[sycl::reqd_work_group_size(Ty{}, Ty1{}, Ty2{})]] void func() {}

struct S {};
void var() {
  // expected-note@+1{{in instantiation of function template specialization 'func<S, S, S>' requested here}}
  func<S, S, S>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1 3{{declared here}}
int foo();
// expected-error@+2 3{{expression is not an integral constant expression}}
// expected-note@+1 3{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[sycl::reqd_work_group_size(foo() + 12, foo() + 12, foo() + 12)]] void func1();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[sycl::reqd_work_group_size(bar() + 12, bar() + 12, bar() + 12)]] void func2(); // OK

// Test that checks template parameter support on member function of class template.
template <int SIZE, int SIZE1, int SIZE2>
class KernelFunctor {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() {}
};

int main() {
  KernelFunctor<16, 1, 1>();
}
// Test that checks template parameter support on function.
// expected-warning@+2 {{'reqd_work_group_size' attribute can only be applied to a SYCL kernel function}}
template <int N, int N1, int N2>
[[sycl::reqd_work_group_size(N, N1, N2)]] void func3() {}

int check() {
  func3<8, 8, 8>();
  return 0;
}
// The GNU and [[cl::reqd_work_group_size]] spellings are deprecated in SYCL
// mode, and still requires all three arguments.
__attribute__((reqd_work_group_size(4, 4, 4))) void four_once_more(); // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                                      // expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}
[[cl::reqd_work_group_size(4, 4, 4)]] void four_with_feeling();       // expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} \
                                                                 // expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}

__attribute__((reqd_work_group_size(4))) void four_yet_again(); // expected-error {{'reqd_work_group_size' attribute requires exactly 3 arguments}} \
                                                                // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                                // expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}

[[cl::reqd_work_group_size(4)]] void four_with_more_feeling(); // expected-error {{'reqd_work_group_size' attribute requires exactly 3 arguments}} \
                                                               // expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} \
                                                               // expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
