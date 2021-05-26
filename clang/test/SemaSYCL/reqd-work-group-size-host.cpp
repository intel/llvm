// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

class Functor {
public:
  [[sycl::reqd_work_group_size(4, 1, 1)]] void operator()() {}
};

template <typename name, typename Func>
void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor f;
  kernel<class kernel_name>(f);
}

[[sycl::reqd_work_group_size(4, 1, 1)]] void f4() {}

[[cl::reqd_work_group_size(4, 1, 1)]] void f5() {} // expected-warning {{attribute 'cl::reqd_work_group_size' is deprecated}} \
                                                   // expected-note {{did you mean to use 'sycl::reqd_work_group_size' instead?}}

__attribute__((reqd_work_group_size(4, 1, 1))) void f6() {} // expected-warning {{attribute 'reqd_work_group_size' is deprecated}} \
                                                            // expected-note {{did you mean to use '[[sycl::reqd_work_group_size]]' instead?}}
