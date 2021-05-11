// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s
// expected-no-diagnostics

class Functor {
public:
  [[cl::reqd_work_group_size(4, 1, 1)]] void operator()() {}
};

template <typename name, typename Func>
void kernel(Func kernelFunc) {
  kernelFunc();
}

void bar() {
  Functor f;
  kernel<class kernel_name>(f);
}

[[cl::reqd_work_group_size(4, 1, 1)]] void f4() {}
