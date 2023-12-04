// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only \
// RUN: %s -I %sycl_include 2>&1 | FileCheck %s

// This test checks if compiler reports compilation error on an attempt to pass
// a struct with type that is not device copyable as SYCL kernel parameter.

#include <sycl/sycl.hpp>

using namespace sycl;

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
  marray<B, 2> MarrayForNotCopyable;
  queue Q;
  Q.single_task<class TestA>([=] {
    int A = IamBad.i;
    int B = IamAlsoBad.i;
    int MB = MarrayForNotCopyable[0].i;
  });

  FunctorA FA;
  Q.single_task<class TestB>(FA);

  FunctorB FB;
  Q.single_task<class TestC>(FB);
}

// CHECK: static assertion failed due to requirement 'is_device_copyable_v<A>
// CHECK: is_device_copyable_neg.cpp:60:5: note: in instantiation of function

// CHECK: static assertion failed due to requirement 'is_device_copyable_v<B>
// CHECK: is_device_copyable_neg.cpp:60:5: note: in instantiation of function

// CHECK: static assertion failed due to requirement 'is_device_copyable_v<sycl::marray<B, 2>>
// CHECK: is_device_copyable_neg.cpp:60:5: note: in instantiation of function

// CHECK: static assertion failed due to requirement 'is_device_copyable_v<C>
// CHECK: is_device_copyable_neg.cpp:67:5: note: in instantiation of function

// CHECK: static assertion failed due to requirement 'is_device_copyable_v<D>
// CHECK: is_device_copyable_neg.cpp:70:5: note: in instantiation of function
