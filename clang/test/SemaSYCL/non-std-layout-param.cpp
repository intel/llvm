// RUN: %clang_cc1 -fsycl-is-device -verify -fsyntax-only -std=c++11 %s

// This test checks if compiler reports compilation error on an attempt to pass
// non-standard layout struct object as SYCL kernel parameter.

struct Base {
  int X;
};

// This struct has non-standard layout, because both C (the most derived class)
// and Base have non-static data members.
struct C : public Base {
  int Y;
};

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}


void test() {
  C C0;
  C0.Y=0;
  kernel_single_task<class MyKernel>([=] {
    // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
    (void)C0.Y;
  });
}

void test_capture_explicit_ref() {
  int p = 0;
  double q = 0;
  float s = 0;
  kernel_single_task<class kernel_capture_single_ref>([
    // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
      &p,
      q,
    // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
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
    // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
    (void) p;
    // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
    (void) q;
  });
}

struct Kernel {
  void operator()() {
    (void) c1;
    (void) c2;
    (void) p;
    (void) q;
  }

  int p;
  // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
  C c1;

  int q;

  // expected-error@+1 {{kernel parameter has non-standard layout class/struct type}}
  C c2;
};

void test_struct_field() {
  Kernel k{};

  kernel_single_task<class kernel_object>(k);
}
