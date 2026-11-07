// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify \
// RUN: %s -I %sycl_include
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -fpreview-breaking-changes -Xclang -verify \
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

// Legal but not device copyable
struct E {
  sycl::accessor<int, 1> acc;
  sycl::unsampled_image_accessor<sycl::float4, 1, sycl::access_mode::read>
      uiacc;
  sycl::sampled_image_accessor<sycl::float4, 1> siacc;
  int i;
  float f;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<E>);
E SomeE();

struct F {
  sycl::accessor<int, 1> acc[2];
  int i;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<F>);

struct G {
  std::array<sycl::accessor<int, 1>, 2> acc;
  int i;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<G>);

struct H {
  const sycl::accessor<int, 1> acc;
  int i;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<H>);

struct Derived : sycl::accessor<int, 1> {
  int i;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<Derived>);

struct Nested {
  H h;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<Nested>);

struct S {
  sycl::stream Stream;
  int i;
};
static_assert(sycl::detail::is_legal_kernel_parameter_v<S>);

struct KnownLimitation {
  sycl::accessor<int, 1> acc;
  int i;
  KnownLimitation();
  KnownLimitation(const KnownLimitation &other);
};
// Not actually legal, but included to demonstrate known limitation.
static_assert(sycl::detail::is_legal_kernel_parameter_v<KnownLimitation>);

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

static_assert(sycl::detail::is_legal_kernel_parameter_v<BCopyable>);
static_assert(sycl::detail::is_legal_kernel_parameter_v<DCopyable>);

sycl::unsampled_image_accessor<sycl::float4, 1, sycl::access_mode::read>
SomeUnsampledImageAccessor();
sycl::sampled_image_accessor<sycl::float4, 1> SomeSampledImageAccessor();

void test() {
  A IamGood;
  IamGood.i = 0;
  BCopyable IamBadButCopyable(1);
  C IamAlsoGood;
  DCopyable IamAlsoBadButCopyable{0};
  E IamGoodButSpecial = SomeE();
  F IamGoodButSpecial2;
  G IamGoodButSpecial3;
  H IamGoodButSpecial4;
  Derived IamDerived;
  Nested IamNested;
  marray<int, 5> MarrayForCopyableIsCopyable(0);
  range<2> Range{1,2};
  id<3> Id{1,2,3};
  vec<int, 3> VecForCopyableIsCopyable{3};
  auto UIAcc = SomeUnsampledImageAccessor();
  auto SIAcc = SomeSampledImageAccessor();
  S IamStream;
  KnownLimitation IamKnownLimitation;

  queue Q;
  Q.single_task<class TestA>([=] {
    int A = IamGood.i;
    int B = IamBadButCopyable.i;
    int C = IamAlsoBadButCopyable.i;
    int D = IamAlsoGood.i;
    int E = IamGoodButSpecial.i;
    int F = IamGoodButSpecial2.i;
    int G = IamGoodButSpecial3.i;
    int H = IamGoodButSpecial4.i;
    int I = MarrayForCopyableIsCopyable[0];
    int J = Range[1];
    int K = Id[2];
    int L = VecForCopyableIsCopyable[0];
    int M = IamDerived.i;
    sycl::float4 N = UIAcc.read(0);
    sycl::float4 O = SIAcc.read(0.f);
    int P = IamNested.h.i;
    int Q = IamStream.i;
    int R = IamKnownLimitation.i;
  });

  Q.single_task<class TestB>(FunctorA{});
  Q.single_task<class TestC>(FunctorB{});

  Q.submit([=](sycl::handler &cgh) {
    const sycl::range<2> range(1026, 1026);
    cgh.parallel_for(range,
                     [=](sycl::item<2> item) { int A = IamBadButCopyable.i; });
  });
}
