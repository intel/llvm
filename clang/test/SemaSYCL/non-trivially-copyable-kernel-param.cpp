// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-2017-compat -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// a struct with non-trivially copyable type as SYCL kernel parameter.

// expected-no-diagnostics
// FIXME: the reason no diagnostics are expected is because checking for non-
// trivially-copyable kernel names is done via the integration footer, which is
// only run when doing a host compilation. The host compilation has not yet
// begun to include the integration footer. The cases with
// missing-expected-error comments are the ones expected to be caught by the
// integration footer.

struct A { int i; };

struct B {
  int i;
  B (int _i) : i(_i) {}
  B (const B& x) : i(x.i) {}
};

struct C : A {
  const A C2;
  C() : A{0}, C2{2}{}
};

struct D {
  int i;
  ~D();
};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

void test() {
  A IamGood;
  IamGood.i = 0;
  B IamBad(1);
  C IamAlsoGood;
  D IamAlsoBad{0};
  kernel_single_task<class kernel_capture_refs>([=] {
    int a = IamGood.i;
    // missing-expected-error@+1 {{kernel parameter has non-trivially copy constructible class/struct type}}
    int b = IamBad.i;
    int c = IamAlsoGood.i;
    // missing-expected-error@+1 {{kernel parameter has non-trivially destructible class/struct type}}
    int d = IamAlsoBad.i;
  });
}
