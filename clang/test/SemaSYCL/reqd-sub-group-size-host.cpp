// RUN: %clang_cc1 -fsycl -fsycl-is-host -fsyntax-only -verify %s
// expected-no-diagnostics

[[cl::intel_reqd_sub_group_size(8)]] void fun() {}

class Functor {
public:
  [[cl::intel_reqd_sub_group_size(16)]] void operator()() {}
};
