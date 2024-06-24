// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// a reference as SYCL kernel parameter.

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

void test_capture_explicit_ref() {
  int p = 0;
  double q = 0;
  float s = 0;
  kernel_single_task<class kernel_capture_single_ref>([
    // expected-error@+1 {{'int &' cannot be used as the type of a kernel parameter}}
      &p,
      q,
    // expected-error@+1 {{'float &' cannot be used as the type of a kernel parameter}}
      &s] {
    (void) q;
    (void) p;
    (void) s;
  });
}

void test_capture_implicit_refs() {
  int p = 0;
  double q = 0;
  kernel_single_task<class kernel_capture_refs>([&] {
    // expected-error@+1 {{'int &' cannot be used as the type of a kernel parameter}}
    (void) p;
    // expected-error@+1 {{'double &' cannot be used as the type of a kernel parameter}}
    (void) q;
  });
}
