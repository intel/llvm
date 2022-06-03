// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Check the basics.
[[intel::max_work_group_size]] void f();                  // expected-error {{'max_work_group_size' attribute requires exactly 3 arguments}}
[[intel::max_work_group_size(12, 12, 12, 12)]] void f0(); // expected-error {{'max_work_group_size' attribute requires exactly 3 arguments}}
[[intel::max_work_group_size("derp", 1, 2)]] void f1();   // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'const char[5]'}}
[[intel::max_work_group_size(1, 1, 1)]] int i;            // expected-error {{'max_work_group_size' attribute only applies to functions}}
[[intel::max_work_group_size(-8, 8, -8)]] void neg();     // expected-error 2{{'max_work_group_size' attribute requires a positive integral compile time constant expression}}

// Tests for Intel FPGA 'max_work_group_size' attribute duplication.
// No diagnostic is emitted because the arguments match. Duplicate attribute is silently ignored.
[[intel::max_work_group_size(6, 6, 6)]] [[intel::max_work_group_size(6, 6, 6)]] void f2() {}

// No diagnostic is emitted because the arguments match.
[[intel::max_work_group_size(32, 32, 32)]] void f3();
[[intel::max_work_group_size(32, 32, 32)]] void f3(); // OK

// Produce a conflicting attribute warning when the args are different.
[[intel::max_work_group_size(6, 6, 6)]]         // expected-note {{previous attribute is here}}
[[intel::max_work_group_size(16, 16, 16)]] void // expected-warning {{attribute 'max_work_group_size' is already applied with different arguments}}
f4() {}

// Catch the easy case where the attributes are all specified at once with
// different arguments.
[[intel::max_work_group_size(16, 16, 16), intel::max_work_group_size(2, 2, 2)]] void f5(); // expected-warning {{attribute 'max_work_group_size' is already applied with different arguments}} expected-note {{previous attribute is here}}

// Show that the attribute works on member functions.
class Functor {
public:
  [[intel::max_work_group_size(16, 16, 16)]] [[intel::max_work_group_size(16, 16, 16)]] void operator()() const;
  [[intel::max_work_group_size(16, 16, 16)]] [[intel::max_work_group_size(32, 32, 32)]] void operator()(int) const; // expected-warning {{attribute 'max_work_group_size' is already applied with different arguments}} expected-note {{previous attribute is here}}
};

class FunctorC {
public:
  [[sycl::reqd_work_group_size(64, 64, 64)]] [[intel::max_work_group_size(64, 64, 64)]] void operator()() const;
  [[sycl::reqd_work_group_size(64, 64, 64)]] [[intel::max_work_group_size(16, 16, 16)]] void operator()(int) const; // expected-error {{'max_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}} expected-note {{conflicting attribute is here}}
};
// Ensure that template arguments behave appropriately based on instantiations.
template <int N>
[[intel::max_work_group_size(N, 1, 1)]] void f6(); // #f6

// Test that template redeclarations also get diagnosed properly.
template <int X, int Y, int Z>
[[intel::max_work_group_size(1, 1, 1)]] void f7(); // #f7prev

template <int X, int Y, int Z>
[[intel::max_work_group_size(X, Y, Z)]] void f7() {} // #f7

// Test that a template redeclaration where the difference is known up front is
// diagnosed immediately, even without instantiation.
template <int X, int Y, int Z>
[[intel::max_work_group_size(X, 1, Z)]] void f8(); // expected-note {{previous attribute is here}}
template <int X, int Y, int Z>
[[intel::max_work_group_size(X, 2, Z)]] void f8(); // expected-warning {{attribute 'max_work_group_size' is already applied with different arguments}}

void instantiate() {
  f6<1>(); // OK
  // expected-error@#f6 {{'max_work_group_size' attribute requires a positive integral compile time constant expression}}
  f6<-1>(); // expected-note {{in instantiation}}
  // expected-error@#f6 {{'max_work_group_size' attribute requires a positive integral compile time constant expression}}
  f6<0>();       // expected-note {{in instantiation}}
  f7<1, 1, 1>(); // OK, args are the same on the redecl.
  // expected-warning@#f7 {{attribute 'max_work_group_size' is already applied with different arguments}}
  // expected-note@#f7prev {{previous attribute is here}}
  f7<2, 2, 2>(); // expected-note {{in instantiation}}
}

// If the [[intel::max_work_group_size(X, Y, Z)]] attribute is specified on
// a declaration along with [[sycl::reqd_work_group_size(X1, Y1, Z1)]]
// attribute, check to see if values of reqd_work_group_size arguments are
// equal or less than values coming from max_work_group_size attribute.
[[sycl::reqd_work_group_size(64, 64, 64)]] // expected-note {{conflicting attribute is here}}
[[intel::max_work_group_size(64, 16, 64)]] // expected-error {{'max_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}}
void
f9() {}

[[intel::max_work_group_size(4, 4, 4)]] void f10();
[[sycl::reqd_work_group_size(2, 2, 2)]] void f10(); // OK

[[sycl::reqd_work_group_size(2, 2, 2)]] [[intel::max_work_group_size(4, 4, 4)]] void f11() {} // OK

[[sycl::reqd_work_group_size(64, 64, 64)]] void f12(); // expected-note {{conflicting attribute is here}}
[[intel::max_work_group_size(16, 16, 16)]] void f12(); // expected-error {{'max_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}}

[[intel::max_work_group_size(16, 16, 16)]]      // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(16, 64, 16)]] void // expected-error {{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}}
f13() {}

[[intel::max_work_group_size(16, 16, 16)]] void f14(); // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(64, 64, 64)]] void f14(); // expected-error{{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}}

[[cl::reqd_work_group_size(1, 2, 3)]] // expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} \
		                      // expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}
[[intel::max_work_group_size(1, 2, 3)]] void
f15() {} // OK

[[intel::max_work_group_size(2, 3, 7)]] void f16(); // expected-note {{conflicting attribute is here}}
[[sycl::reqd_work_group_size(7, 3, 2)]] void f16(); // expected-error{{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}}

[[intel::max_work_group_size(1, 2, 3)]] [[sycl::reqd_work_group_size(1, 2, 3)]] void f17(){}; // OK

[[sycl::reqd_work_group_size(16)]]            // expected-note {{conflicting attribute is here}}
[[intel::max_work_group_size(1, 1, 16)]] void // expected-error {{'max_work_group_size' attribute conflicts with 'reqd_work_group_size' attribute}}
f18();

[[intel::max_work_group_size(16, 16, 1)]] void f19();
[[sycl::reqd_work_group_size(16, 16)]] void f19(); // OK

// Test that checks wrong function template instantiation and ensures that the type
// is checked properly when instantiating from the template definition.

template <typename Ty, typename Ty1, typename Ty2>
// expected-error@+1 3{{integral constant expression must have integral or unscoped enumeration type, not 'S'}}
[[intel::max_work_group_size(Ty{}, Ty1{}, Ty2{})]] void f20() {}

struct S {};
void var() {
  // expected-note@+1 {{in instantiation of function template specialization 'f20<S, S, S>' requested here}}
  f20<S, S, S>();
}

// Test that checks expression is not a constant expression.
// expected-note@+1 3{{declared here}}
int foo();
// expected-error@+2 3{{expression is not an integral constant expression}}
// expected-note@+1 3{{non-constexpr function 'foo' cannot be used in a constant expression}}
[[intel::max_work_group_size(foo() + 12, foo() + 12, foo() + 12)]] void f21();

// Test that checks expression is a constant expression.
constexpr int bar() { return 0; }
[[intel::max_work_group_size(bar() + 12, bar() + 12, bar() + 12)]] void f22(); // OK

struct DAFuncObj {
  [[intel::max_work_group_size(4, 4, 4)]] // expected-note {{conflicting attribute is here}}
  [[cl::reqd_work_group_size(8, 8, 4)]]   // expected-error{{'reqd_work_group_size' attribute conflicts with 'max_work_group_size' attribute}} \
                                          // expected-warning{{attribute 'cl::reqd_work_group_size' is deprecated}} \
                                          // expected-note{{did you mean to use 'sycl::reqd_work_group_size' instead?}}
  void
  operator()() const {}
};

struct Func {
  // expected-warning@+1 {{unknown attribute 'max_work_group_size' ignored}}
  [[intelfpga::max_work_group_size(1, 1, 1)]] void operator()() const {}
};
