// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify \
// RUN: %s -I %sycl_include

// This test checks that compiler does not report any errors
// for the device copyable types being passed from HOST to DEVICE.
// It also checks that the types not copyable in general, but declared as
// device copyable (by using specialization of is_device_copyable class)
// are allowed.

// expected-no-diagnostics

#include <sycl/sycl.hpp>

using namespace sycl;

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
  DCopyable IamAlsoBadButCopyable{0};
  marray<int, 5> MarrayForCopyableIsCopyable(0);
  range<2> Range{1,2};
  id<3> Id{1,2,3};
  vec<int, 3> VecForCopyableIsCopyable{3};
  queue Q;
  Q.single_task<class TestA>([=] {
    int A = IamGood.i;
    int B = IamBadButCopyable.i;
    int C = IamAlsoBadButCopyable.i;
    int E = MarrayForCopyableIsCopyable[0];
    int F = Range[1];
    int G = Id[2];
    int H = VecForCopyableIsCopyable[0];
  });

  Q.single_task<class TestB>(FunctorA{});
  Q.single_task<class TestC>(FunctorB{});

  Q.submit([=](sycl::handler &cgh) {
    const sycl::range<2> range(1026, 1026);
    cgh.parallel_for(range,
                     [=](sycl::item<2> item) { int A = IamBadButCopyable.i; });
  });
}
