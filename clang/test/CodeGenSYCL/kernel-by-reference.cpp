// RUN: %clang_cc1 -triple spir64 -fsycl -fsycl-is-device -verify -fsyntax-only %s

// SYCL 1.2 - kernel functions passed directly. (Also no const requirement, though mutable lambdas never supported)
template <typename name, typename Func>
// expected-warning@+1 {{Older version of SYCL headers encountered.}}
__attribute__((sycl_kernel)) void sycl_12_single_task(Func kernelFunc) {
  kernelFunc();
}

// SYCL 2020 - kernel functions are passed by reference.
template <typename name, typename Func>
__attribute__((sycl_kernel)) void sycl_2020_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int do_nothing(int i) {
  return i + 1;
}

// ensure both compile.
int main() {
  sycl_12_single_task<class sycl12>([]() {
    do_nothing(10);
  });

  sycl_2020_single_task<class sycl2020>([]() {
    do_nothing(11);
  });

  return 0;
}