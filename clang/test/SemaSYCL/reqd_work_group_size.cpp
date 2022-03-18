// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

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
[[sycl::reqd_work_group_size(6, 6, 6)]] [[sycl::reqd_work_group_size(6, 6, 6)]] void f2() {}

// No diagnostic is emitted because the arguments match.
[[sycl::reqd_work_group_size(32, 32, 32)]] void f3();
[[sycl::reqd_work_group_size(32, 32, 32)]] void f3(); // OK

// Produce a conflicting attribute warning when the args are different.
[[sycl::reqd_work_group_size(6, 6, 6)]]         // expected-note {{previous attribute is here}}
[[sycl::reqd_work_group_size(16, 16, 16)]] void // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}
f4() {}

// Catch the easy case where the attributes are all specified at once with
// different arguments.
struct TRIFuncObjGood1 {
  // expected-note@+2 {{previous attribute is here}}
  // expected-error@+1 {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  [[sycl::reqd_work_group_size(64)]] [[sycl::reqd_work_group_size(128)]] void operator()() const {}
};

struct TRIFuncObjGood2 {
  // expected-note@+2 {{previous attribute is here}}
  // expected-error@+1 {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  [[sycl::reqd_work_group_size(64, 64)]] [[sycl::reqd_work_group_size(128, 128)]] void operator()() const {}
};

struct TRIFuncObjGood3 {
  [[sycl::reqd_work_group_size(8, 8)]] void // expected-note {{previous attribute is here}}
  operator()() const;
};

[[sycl::reqd_work_group_size(4, 4)]] // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}
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
  f7<2, 2, 2>(); // expected-note {{in instantiation}}
}

// Tests for 'reqd_work_group_size' attribute duplication.

[[sycl::reqd_work_group_size(8)]]            // expected-note {{previous attribute is here}}
[[sycl::reqd_work_group_size(1, 1, 8)]] void // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}
f8(){};

[[sycl::reqd_work_group_size(32, 32, 1)]] [[sycl::reqd_work_group_size(32, 32)]] void f9() {} // OK

// Test that template redeclarations also get diagnosed properly.
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(64, 1, 1)]] void f10(); // #f10prev
template <int X, int Y, int Z>
[[sycl::reqd_work_group_size(X, Y, Z)]] void f10() {} // #f10err

void test() {
  f10<64, 1, 1>(); // OK, args are the same on the redecl.
  // expected-error@#f10err {{attribute 'reqd_work_group_size' is already applied with different arguments}}
  // expected-note@#f10prev {{previous attribute is here}}
  f10<1, 1, 64>(); // expected-note {{in instantiation}}
}

struct TRIFuncObjBad {
  [[sycl::reqd_work_group_size(32, 1, 1)]] void // expected-note {{previous attribute is here}}
  operator()() const;
};

[[sycl::reqd_work_group_size(1, 1, 32)]] // expected-error {{attribute 'reqd_work_group_size' is already applied with different arguments}}
void
TRIFuncObjBad::operator()() const {}
