// RUN: %clang_cc1 -fsycl-is-host -Wno-sycl-2017-compat -fsyntax-only -verify %s
// expected-no-diagnostics

[[intel::max_global_work_dim(2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::max_global_work_dim(1)]] void operator()() const {}
};
