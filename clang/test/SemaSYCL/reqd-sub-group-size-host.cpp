// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s
// expected-no-diagnostics

[[sycl::reqd_sub_group_size(8)]] void fun() {}

class Functor {
public:
  [[sycl::reqd_sub_group_size(16)]] void operator()() {}
};
