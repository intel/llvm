// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s
// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -fpreview-breaking-changes -Xclang -verify -Xclang -verify-ignore-unexpected=warning,note %s

// This test checks if compiler reports compilation error on an attempt to pass
// a struct with type that is not device copyable as SYCL kernel parameter.

#include <string>
#include <sycl/sycl.hpp>
#include <vector>

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

// Edge case with special type and invalid type
struct E {
  sycl::accessor<int, 1> acc;
  std::vector<int> vec;
};

static_assert(!sycl::detail::check_if_device_copyable_v<E>);
static_assert(!sycl::detail::is_legal_kernel_parameter_v<E>);
static_assert(
    sycl::detail::legal_kernel_parameters::contains_special_sycl_types_v<E>);
static_assert(
    !sycl::detail::legal_kernel_parameters::check_fields_and_bases_v<E>);

struct NotDC {
  int a;
  ~NotDC(); // Needed because of sycl 1.2.1 deprecated rules
};

template <> struct sycl::is_device_copyable<NotDC> : std::false_type {};
static_assert(!sycl::detail::check_if_device_copyable_v<NotDC>);
static_assert(!sycl::detail::is_legal_kernel_parameter_v<NotDC>);

struct F {
  sycl::accessor<int, 1> acc;
  NotDC notDC;
};
static_assert(!sycl::detail::check_if_device_copyable_v<F>);
static_assert(!sycl::detail::is_legal_kernel_parameter_v<F>);

struct G : sycl::accessor<int, 1> {
  std::string str;
};
static_assert(!sycl::detail::check_if_device_copyable_v<G>);
static_assert(!sycl::detail::is_legal_kernel_parameter_v<G>);

struct WouldBeGood {
  sycl::accessor<int, 1> acc;
  int i;
};

struct H {
  WouldBeGood wbg;
  std::vector<int> vec;
};
static_assert(!sycl::detail::check_if_device_copyable_v<H>);
static_assert(!sycl::detail::is_legal_kernel_parameter_v<H>);

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
  E IamEdgeCase;
  F IamNotCopyable;
  G IamDerivedFromAccessor;
  H IamContainingWouldBeGood;
  marray<B, 2> MarrayForNotCopyable;
  queue Q;
  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<A>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestA>([=] { int A = IamBad.i; });

  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<B>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestB>([=] { int B = IamAlsoBad.i; });

  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<sycl::marray<B, 2>>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestMB>([=] { int MB = MarrayForNotCopyable[0].i; });

  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<E>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestE>([=] { int ECAcc = IamEdgeCase.acc[0]; });

  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<F>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestF>([=] { int FAcc = IamNotCopyable.acc[0]; });

  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<G>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestG>(
      [=] { int GStrSize = IamDerivedFromAccessor.str.size(); });

  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<H>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestH>(
      [=] { int HVecSize = IamContainingWouldBeGood.vec.size(); });

  FunctorA FA;
  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<C>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestFA>(FA);

  FunctorB FB;
  // expected-error@*:* {{static assertion failed due to requirement 'is_legal_kernel_parameter_v<D>': The specified type is not a legal kernel parameter}}
  Q.single_task<class TestFB>(FB);
}
