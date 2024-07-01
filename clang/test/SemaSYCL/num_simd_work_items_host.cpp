// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s
// expected-no-diagnostics

[[intel::num_simd_work_items(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::num_simd_work_items(42)]] void operator()() const {}
};
