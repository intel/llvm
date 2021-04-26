// RUN: %clang_cc1 -fsycl-is-device -verify -Wno-sycl-2017-compat -fsyntax-only %s
// RUN: %clang_cc1 -fsycl-is-device -Wno-sycl-2017-compat -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// non-standard layout struct object as SYCL kernel parameter.

// expected-no-diagnostics
// NOTE: the reason no diagnostics are expected is because checking for non-
// trivially-copyable kernel names is done via the integration footer, which is
// only run when doing a host compilation. The host compilation has not yet
// begun to include the integration footer. The cases with
// missing-expected-error comments are the ones expected to be caught by the
// integration footer.

struct Base {
  int X;
};

// This struct has non-standard layout, because both C (the most derived class)
// and Base have non-static data members.
struct C : public Base {
  int Y;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

void test() {
  C C0;
  C0.Y=0;
  kernel_single_task<class MyKernel>([=] {
    // missing-expected-error@+1 {{kernel parameter has non-standard layout class/struct type 'C'}}
    (void)C0.Y;
  });
}

struct Kernel {
  void operator()() const {
    (void) c1;
    (void) c2;
    (void) p;
    (void) q;
  }

  int p;
  // missing-expected-error@+1 {{kernel parameter has non-standard layout class/struct type 'C'}}
  C c1;

  int q;

  // missing-expected-error@+1 {{kernel parameter has non-standard layout class/struct type 'C'}}
  C c2;
};

void test_struct_field() {
  Kernel k{};

  kernel_single_task<class kernel_object>(k);
}
