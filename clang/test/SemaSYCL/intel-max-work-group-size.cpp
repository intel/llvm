// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Check the basics.
[[intel::max_work_group_size]] void f();                  // expected-error {{'max_work_group_size' attribute requires exactly 3 arguments}}
[[intel::max_work_group_size(12, 12, 12, 12)]] void f0(); // expected-error {{'max_work_group_size' attribute requires exactly 3 arguments}}
[[intel::max_work_group_size("derp", 1, 2)]] void f1();   // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'const char[5]'}}
[[intel::max_work_group_size(1, 1, 1)]] int i;            // expected-error {{'max_work_group_size' attribute only applies to functions}}

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
