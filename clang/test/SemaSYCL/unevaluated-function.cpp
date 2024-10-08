// RUN: %clang_cc1 -fsycl-is-device -fcxx-exceptions -verify -fsyntax-only %s

// Check that a function used in an unevaluated context is not subject
// to delayed device diagnostics.

bool foo1() {
  // Throw exception which is not allowed on device. Error is expected
  // only when the function is called in evaluated context.
  // expected-error@+1 1{{SYCL kernel cannot use exceptions}}
  throw 10;

  return false;
}

template <typename T>
T foo2(T t) {
  throw t;
  return t;
}

bool foo3() {
  __float128 a;
  return false;
}

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  // expected-note@+1 1{{called by}}
  kernelFunc();
}

int main() {
  using T1 = decltype(foo1());
  kernel_single_task<class fake_kernel>([]() {
    using T2 = decltype(foo1());

    // expected-note@+1 1{{called by}}
    auto S1 = foo1();
    auto S2 = sizeof(foo1());

    using T3 = decltype(foo2(decltype(foo1()){}));
    using T4 = decltype(foo3());

    T1 f1;
    T2 f2;
    T3 f3;
    T4 f4;
  });
  return 0;
}
