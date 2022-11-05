// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

// Host checks for the work_group_size_hint attribute.
class Functor16x2x1 {
public:
  [[sycl::work_group_size_hint(16, 2, 1)]] void operator()() const {};
};

// The GNU spelling is deprecated in SYCL mode, but otherwise these attributes
// have the same semantics.
[[sycl::work_group_size_hint(4, 1, 1)]] void f4x1x1();
__attribute__((work_group_size_hint(4, 1, 1))) void f4x1x1(); // expected-warning {{attribute 'work_group_size_hint' is deprecated}} \
                                                          // expected-note {{did you mean to use '[[sycl::work_group_size_hint]]' instead?}}
