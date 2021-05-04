// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Check the basics.
[[sycl::work_group_size_hint]] void f0(); // expected-error {{'work_group_size_hint' attribute takes at least 1 argument}}
[[sycl::work_group_size_hint(12, 12, 12, 12)]] void f1(); // expected-error {{'work_group_size_hint' attribute takes no more than 3 arguments}}
[[sycl::work_group_size_hint("derp")]] void f2(); // expected-error {{'work_group_size_hint' attribute requires parameter 0 to be an integer constant}}
[[sycl::work_group_size_hint(1)]] int i; // expected-error {{'work_group_size_hint' attribute only applies to functions}}

// FIXME: this should produce a conflicting attribute warning but doesn't. It
// is missing a merge method (and is also missing template instantiation logic).
[[sycl::work_group_size_hint(4, 1, 1)]] void f3();
[[sycl::work_group_size_hint(32, 1, 1)]] void f3() {}

// The default values are 1, so these are equivalent
[[sycl::work_group_size_hint(4)]] void f4();
[[sycl::work_group_size_hint(4, 1)]] void f4(); // OK
[[sycl::work_group_size_hint(4, 1, 1)]] void f4(); // OK

// The GNU spelling is deprecated in SYCL mode, but otherwise these attributes
// have the same semantics.
[[sycl::work_group_size_hint(4, 1, 1)]] void f5();
__attribute__((work_group_size_hint(4, 1, 1))) void f5(); // expected-warning {{attribute 'work_group_size_hint' is deprecated}} \
                                                          // expected-note {{did you mean to use '[[sycl::work_group_size_hint]]' instead?}}

// Catch the easy case where the attributes are all specified at once with
// different arguments.
[[sycl::work_group_size_hint(4, 1, 1), sycl::work_group_size_hint(32, 1, 1)]] void f6(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

// Show that the attribute works on member functions.
class Functor {
public:
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(16)]] void operator()() const;
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(32)]] void operator()(int) const; // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}
};
