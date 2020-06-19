// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fsycl -fsycl-is-device -verify %s

[[intelfpga::no_global_work_offset]] void indirect_func() {} //expected-warning {{'no_global_work_offset' attribute ignored}}

void func() { indirect_func(); }

template <typename Name, typename Type>
[[clang::sycl_kernel]] void __my_kernel__(Type bar) {
  bar();
  func();
}

template <typename Name, typename Type>
void parallel_for(Type lambda) {
  __my_kernel__<Name>(lambda);
}

void invoke_foo2() {
  parallel_for<class KernelName>([]() {});
}
