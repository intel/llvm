// RUN: %clang_cc1 -fsycl-is-device -Wno-return-type -verify -Wno-sycl-2017-compat -fsyntax-only -std=c++17 %s
void bar();

template <typename T>
void usage(T func) {
  bar();
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

SYCL_EXTERNAL void foo();
// expected-error@+2 {{SYCL kernel cannot call a recursive function}}
// expected-note@+1 2{{function implemented using recursion declared here}}
void fum() { return fum(); };
int main() {
  kernel_single_task<class fake_kernel>([]() { usage(foo); });
}
template <typename T>
void templ_func() {
  // expected-error@+1 {{SYCL kernel cannot call a recursive function}}
  fum();
  foo();
}
void bar() { templ_func<int>(); }
