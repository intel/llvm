// RUN: %clang_cc1 -fsycl-is-device -fsycl-new-kernel-param-requirements -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// a struct with non-trivially copyable type as SYCL kernel parameter.

struct A { int i; };

struct B {
  int i;
  B (int _i) : i(_i) {}
  B (const B& x) : i(x.i) {}
};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

void test() {
  A IamGood;
  IamGood.i = 0;
  B IamBad(1);
  kernel_single_task<class kernel_capture_refs>([=] {
    int a = IamGood.i;
    // expected-error@+1 {{kernel parameter has non-trivially copyable class/struct type}}
    int b = IamBad.i;
  });
}
