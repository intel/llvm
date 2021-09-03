// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN: %s -I %sycl_include 2>&1 | FileCheck %s

// This test checks if compiler reports compilation error on an attempt to pass
// a struct with type that is not device copyable as SYCL kernel parameter.

#include <CL/sycl.hpp>

using namespace cl::sycl;

struct A {
  int i;
  A(int _i) : i(_i) {}
  A(const A &x) : i(x.i) {}
};

struct B {
  int i;
  ~B();
};

// Copy of B, needed to have crisper error diagnostic checks
struct C {
  int i;
  ~C();
};

// Copy of B, needed to have crisper error diagnostic checks
struct D {
  int i;
  ~D();
};

// Not copyable type, but it is declared as device copyable.
struct E {
  int i;
  E(int _i) : i(_i) {}
  E(const E &x) : i(x.i) {}
};
template <> struct is_device_copyable<E> : std::true_type {};

struct FunctorA {
  FunctorA() {}
  void operator()() const {
    (void)c;
    (void)i;
  }

  C c;
  int i;
};

struct FunctorB : public D {
  FunctorB() {}
  void operator()() const {
    (void)i;
    (void)j;
  }

  int j;
};

void test() {
  A IamBad(1);
  B IamAlsoBad{0};
  queue Q;
  Q.single_task<class TestA>([=] {
    int A = IamBad.i;
    int B = IamAlsoBad.i;
  });

  FunctorA FA;
  Q.single_task<class TestB>(FA);

  FunctorB FB;
  Q.single_task<class TestC>(FB);

  // FIXME: the type is marked as device copyable, but range rounding
  // optimization wraps a kernel lambda, which causes compilation errors.
  E IamGood(0);
  Q.submit([=](sycl::handler& cgh){
    const sycl::range<2> range(1026, 1026);
    cgh.parallel_for(range,[=](sycl::item<2> item) {
      int A = IamGood.i;
    });
  });
}

// CHECK: static_assert failed due to requirement 'is_device_copyable<A, void>
// CHECK: is_device_copyable_neg.cpp:67:5: note: in instantiation of function

// CHECK: static_assert failed due to requirement 'is_device_copyable<B, void>
// CHECK: is_device_copyable_neg.cpp:67:5: note: in instantiation of function

// CHECK: static_assert failed due to requirement 'is_device_copyable<C, void>
// CHECK: is_device_copyable_neg.cpp:73:5: note: in instantiation of function

// CHECK: static_assert failed due to requirement 'is_device_copyable<D, void>
// CHECK: is_device_copyable_neg.cpp:76:5: note: in instantiation of function

// CHECK: static_assert failed due to requirement 'is_device_copyable<(lambda at {{.*}}is_device_copyable_neg.cpp:83:28
// CHECK: is_device_copyable_neg.cpp:83:9: note: in instantiation of function
