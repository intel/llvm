// RUN: %clang_cc1 -fsycl -fsycl-is-host -fsyntax-only -Wno-sycl-2017-compat -verify %s
// expected-no-diagnostics

[[intelfpga::num_simd_work_items(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intelfpga::num_simd_work_items(42)]] void operator()() const {}
};
