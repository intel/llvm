// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Check the basics.
[[sycl::work_group_size_hint]] void f0(); // expected-error {{'work_group_size_hint' attribute requires exactly 3 arguments}}
[[sycl::work_group_size_hint(12, 12, 12, 12)]] void f1(); // expected-error {{'work_group_size_hint' attribute requires exactly 3 arguments}}
[[sycl::work_group_size_hint("derp", 1, 2)]] void f2(); // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'const char [5]'}}
[[sycl::work_group_size_hint(1, 1, 1)]] int i; // expected-error {{'work_group_size_hint' attribute only applies to functions}}

// Produce a conflicting attribute warning when the args are different.
[[sycl::work_group_size_hint(4, 1, 1)]] void f3(); // expected-note {{previous attribute is here}}
[[sycl::work_group_size_hint(32, 1, 1)]] void f3() {} // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

// FIXME: the attribute is like reqd_work_group_size in that it has a one, two,
// and three arg form that needs to be supported.
[[sycl::work_group_size_hint(1)]] void f4(); // expected-error {{'work_group_size_hint' attribute requires exactly 3 arguments}}
[[sycl::work_group_size_hint(1, 1)]] void f5(); // expected-error {{'work_group_size_hint' attribute requires exactly 3 arguments}}

// The GNU spelling is deprecated in SYCL mode, but otherwise these attributes
// have the same semantics.
[[sycl::work_group_size_hint(4, 1, 1)]] void f6();
__attribute__((work_group_size_hint(4, 1, 1))) void f6(); // expected-warning {{attribute 'work_group_size_hint' is deprecated}} \
                                                          // expected-note {{did you mean to use '[[sycl::work_group_size_hint]]' instead?}}

// Catch the easy case where the attributes are all specified at once with
// different arguments.
[[sycl::work_group_size_hint(4, 1, 1), sycl::work_group_size_hint(32, 1, 1)]] void f7(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}} expected-note {{previous attribute is here}}

// Show that the attribute works on member functions.
class Functor {
public:
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(16, 1, 1)]] void operator()() const;
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(32, 1, 1)]] void operator()(int) const; // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}} expected-note {{previous attribute is here}}
};

// Ensure that template arguments behave appropriately based on instantiations.
template <int N>
[[sycl::work_group_size_hint(N, 1, 1)]] void f8(); // #f8

// Test that template redeclarations also get diagnosed properly.
template <int X, int Y, int Z>
[[sycl::work_group_size_hint(1, 1, 1)]] void f9(); // #f9prev

template <int X, int Y, int Z>
[[sycl::work_group_size_hint(X, Y, Z)]] void f9() {} // #f9

// Test that a template redeclaration where the difference is known up front is
// diagnosed immediately, even without instantiation.
template <int X, int Y, int Z>
[[sycl::work_group_size_hint(X, 1, Z)]] void f10(); // expected-note {{previous attribute is here}}
template <int X, int Y, int Z>
[[sycl::work_group_size_hint(X, 2, Z)]] void f10(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

void instantiate() {
  f8<1>(); // OK
  // expected-error@#f8 {{'work_group_size_hint' attribute requires a positive integral compile time constant expression}}
  f8<-1>(); // expected-note {{in instantiation}}
  // expected-error@#f8 {{'work_group_size_hint' attribute requires a positive integral compile time constant expression}}
  f8<0>(); // expected-note {{in instantiation}}

  f9<1, 1, 1>(); // OK, args are the same on the redecl.

  // expected-warning@#f9 {{attribute 'work_group_size_hint' is already applied with different arguments}}
  // expected-note@#f9prev {{previous attribute is here}}
  f9<1, 2, 3>(); // expected-note {{in instantiation}}
}
