// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s

// Check the basics.
[[sycl::work_group_size_hint]] void f0(); // expected-error {{'work_group_size_hint' attribute requires exactly 3 arguments}}
[[sycl::work_group_size_hint(12, 12, 12, 12)]] void f1(); // expected-error {{'work_group_size_hint' attribute requires exactly 3 arguments}}
[[sycl::work_group_size_hint("derp", 1, 2)]] void f2(); // expected-error {{'work_group_size_hint' attribute requires parameter 0 to be an integer constant}}
[[sycl::work_group_size_hint(1, 1, 1)]] int i; // expected-error {{'work_group_size_hint' attribute only applies to functions}}

// FIXME: this should produce a conflicting attribute warning but doesn't. It
// is missing a merge method (and is also missing template instantiation logic).
[[sycl::work_group_size_hint(4, 1, 1)]] void f3();
[[sycl::work_group_size_hint(32, 1, 1)]] void f3() {}

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
[[sycl::work_group_size_hint(4, 1, 1), sycl::work_group_size_hint(32, 1, 1)]] void f7(); // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}

// Show that the attribute works on member functions.
class Functor {
public:
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(16, 1, 1)]] void operator()() const;
  [[sycl::work_group_size_hint(16, 1, 1)]] [[sycl::work_group_size_hint(32, 1, 1)]] void operator()(int) const; // expected-warning {{attribute 'work_group_size_hint' is already applied with different arguments}}
};
