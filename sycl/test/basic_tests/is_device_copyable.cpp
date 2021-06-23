// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify \
// RUN: %s -I %sycl_include

// This test checks that compiler does not report any errors
// for the device copyable types being passed from HOST to DEVICE.
// It also checks that the types not copyable in general, but declared as
// device copyable (by using specialization of is_device_copyable class)
// are allowed.

// expected-no-diagnostics

#include <CL/sycl.hpp>

using namespace cl::sycl;

// Trivially copyable type.
struct A {
  int i;
};

// Not copyable type, but it will be declared as device copyable.
struct BCopyable {
  int i;
  BCopyable(int _i) : i(_i) {}
  BCopyable(const BCopyable &x) : i(x.i) {}
};

// Not trivially copyable, but trivially copy constructible/destructible.
// Such types are passed to kernels to stay compatible with deprecated
// sycl 1.2.1 rules.
struct C : A {
  const A C2;
  C() : A{0}, C2{2} {}
};

// Not copyable type, but it will be declared as device copyable.
struct DCopyable {
  int i;
  ~DCopyable();
};

struct FunctorA {
  FunctorA() : b(7) {}
  void operator()() const {
    (void)b.i;
    (void)i;
  }

  BCopyable b;
  int i;
};

struct FunctorB : public DCopyable {
  FunctorB() {}
  void operator()() const {
    (void)i;
    (void)j;
  }

  int j;
};

template <> struct is_device_copyable<BCopyable> : std::true_type {};
template <> struct is_device_copyable<DCopyable> : std::true_type {};

void test() {
  A IamGood;
  IamGood.i = 0;
  BCopyable IamBadButCopyable(1);
  C IamAlsoGood;
  DCopyable IamAlsoBadButCopyable{0};
  queue Q;
  Q.single_task<class TestA>([=] {
    int A = IamGood.i;
    int B = IamBadButCopyable.i;
    int C = IamAlsoBadButCopyable.i;
    int D = IamAlsoGood.i;
  });

  Q.single_task<class TestB>(FunctorA{});
  Q.single_task<class TestC>(FunctorB{});
}
