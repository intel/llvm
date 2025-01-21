// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s
// expected-no-diagnostics

// The test checks that work_group_size_hint is allowed in host code.

[[sycl::work_group_size_hint(4)]] void f4x1x1() {}

[[sycl::work_group_size_hint(16)]] void f16x1x1() {}

[[sycl::work_group_size_hint(32, 32, 32)]] void f32x32x32() {}

class Functor64 {
public:
  [[sycl::work_group_size_hint(64, 64, 64)]] void operator()() const {}
};
