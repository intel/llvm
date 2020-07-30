// RUN: %clang_cc1 -fsycl -fsycl-is-device -fcxx-exceptions -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// an array of non-trivially copyable structs as SYCL kernel parameter or
// a non-constant size array.

struct B {
  int i;
  B(int _i) : i(_i) {}
  B(const B &x) : i(x.i) {}
};

struct D {
  int i;
  ~D();
};

class E {
  // expected-error@+1 {{kernel parameter is not a constant size array}}
  int i[];

public:
  int operator()() { return i[0]; }
};

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

void test() {
  B nsl1[4] = {1, 2, 3, 4};
  D nsl2[5];
  E es;
  kernel_single_task<class kernel_capture_refs>([=] {
    // expected-error@+1 {{kernel parameter has non-trivially copy constructible class/struct type}}
    int b = nsl1[2].i;
    // expected-error@+1 {{kernel parameter has non-trivially destructible class/struct type}}
    int d = nsl2[4].i;
  });

  kernel_single_task<class kernel_bad_array>(es);
}
