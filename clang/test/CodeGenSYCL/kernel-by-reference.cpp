// RUN: %clang_cc1 -triple spir64 -fsycl -fsycl-is-device -verify -fsyntax-only -sycl-std=2017 -DSYCL2017 %s
// RUN: %clang_cc1 -triple spir64 -fsycl -fsycl-is-device -verify -fsyntax-only -sycl-std=2020 -DSYCL2020 %s
// RUN: %clang_cc1 -triple spir64 -fsycl -fsycl-is-device -verify -fsyntax-only -Wno-sycl-strict -DNODIAG %s
// RUN: %clang_cc1 -triple spir64 -fsycl -fsycl-is-device -verify -fsyntax-only -sycl-std=2020 -Wno-sycl-strict -DNODIAG %s

// SYCL 1.2/2017 - kernel functions passed directly. (Also no const requirement, though mutable lambdas never supported)
template <typename name, typename Func>
#if defined(SYCL2020)
// expected-warning@+2 {{Passing kernel functions by value is deprecated in SYCL 2020}}
#endif
__attribute__((sycl_kernel)) void sycl_2017_single_task(Func kernelFunc) {
  kernelFunc();
}

// SYCL 2020 - kernel functions are passed by reference.
template <typename name, typename Func>
#if defined(SYCL2017)
// expected-warning@+2 {{Passing of kernel functions by reference is a SYCL 2020 extension}}
#endif
__attribute__((sycl_kernel)) void sycl_2020_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int do_nothing(int i) {
  return i + 1;
}

// ensure both compile.
int main() {
  sycl_2017_single_task<class sycl12>([]() {
    do_nothing(10);
  });

  sycl_2020_single_task<class sycl2020>([]() {
    do_nothing(11);
  });

  return 0;
}
#if defined(NODIAG)
// expected-no-diagnostics
#endif