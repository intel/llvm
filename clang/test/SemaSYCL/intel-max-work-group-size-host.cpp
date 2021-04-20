// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -Wno-sycl-2017-compat -verify %s
// expected-no-diagnostics

[[intel::max_work_group_size(2, 2, 2)]] void func_do_not_ignore() {}

struct FuncObj {
  [[intel::max_work_group_size(4, 4, 4)]] void operator()() const {}
};
