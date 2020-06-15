// RUN: %clang_cc1 -fsycl -fsycl-is-device -fcxx-exceptions -verify -fsyntax-only %s

// This test checks if compiler reports compilation error on an attempt to pass
// an array of non-trivially copyable structs as SYCL kernel parameter or
// a non-constant size array.

struct A {
  int i;
};

struct B {
  int i;
  B(int _i) : i(_i) {}
  B(const B &x) : i(x.i) {}
};

struct C : A {
  const A C2;
  C() : A{0}, C2{2} {}
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
  A cs1[10];
  B nsl1[4] = {1, 2, 3, 4};
  C cs2[6];
  D nsl2[5];
  E es;
  kernel_single_task<class kernel_capture_refs>([=] {
    int a = cs1[6].i;
    // expected-error@+1 {{kernel parameter has non-trivially copy constructible class/struct type}}
    int b = nsl1[2].i;
    int c = cs2[0].i;
    // expected-error@+1 {{kernel parameter has non-trivially destructible class/struct type}}
    int d = nsl2[4].i;
  });

  kernel_single_task<class kernel_bad_array>(es);
}
