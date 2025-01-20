// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s

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
  // expected-error@*:* {{static assertion failed due to requirement 'is_device_copyable_v<A>': The specified type is not device copyable}}
  Q.single_task<class TestA>([=] {
    int A = IamBad.i;
    int B = IamAlsoBad.i;
    int MB = MarrayForNotCopyable[0].i;
  });

  FunctorA FA;
  // expected-error@*:* {{static assertion failed due to requirement 'is_device_copyable_v<C>': The specified type is not device copyable}}
  Q.single_task<class TestB>(FA);

  FunctorB FB;
  // expected-error@*:* {{static assertion failed due to requirement 'is_device_copyable_v<D>': The specified type is not device copyable}}
  Q.single_task<class TestC>(FB);
}
