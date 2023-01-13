// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

// clang-format on

using namespace sycl;
using namespace ext::oneapi::experimental;

struct B {};

struct A : public B {
  int x;
  A() {}
  A(int x_) : x(x_) {}

  int &operator[](std::ptrdiff_t idx) { return x; }
  int &operator[](std::ptrdiff_t idx) const { return const_cast<int &>(x); }
};

using annotated_arg_t1 =
    annotated_arg<A, decltype(properties(conduit, stable))>;

using annotated_arg_t3 = annotated_arg<A, decltype(properties(conduit))>;

struct MyIP {
  int *a;

  annotated_arg_t1 b;

  MyIP(int *a_, const A &b_) : a(a_), b(b_) {}

  void operator()() const {
    b[0] = 10;

    const A &tmp = b;
    A tmp2 = b;

    for (int i = 0; i < tmp.x; i++) {
      *a += 1;
    }
    *a += b[0];
  }
};

template <typename T> T foo() {
  A obj(5);
  return annotated_arg(obj, conduit);
}

void TestVectorAddWithAnnotatedMMHosts() {
  // Create the SYCL device queue
  queue q(sycl::ext::intel::fpga_selector_v);
  auto raw = malloc_shared<int>(1, q);

  A obj(0);
  // default ctor
  annotated_arg_t3 a1(obj);

  // copy ctor
  auto a2(a1);
  auto a3(foo<annotated_arg_t3>());
  // // assign ctor
  auto a4 = a3;

  // Construct from A instance
  auto tmp11 = annotated_arg(obj); // empty property list

  // Construct from A instance and a property list
  // auto tmp12 = annotated_arg(obj, properties{conduit});
  auto tmp12 = annotated_arg(obj, conduit);
  static_assert(std::is_same<decltype(tmp12), annotated_arg_t3>::value,
                "deduction guide failed 1");

  // Construct from A instance and variadic properties
  auto tmp13 = annotated_arg(obj, stable, conduit); // deduction guide
  static_assert(std::is_same<decltype(tmp13), annotated_arg_t1>::value,
                "deduction guide failed 2");

  // property list contains invalid property
  // auto tmp14 = annotated_arg(obj, awidth<32>);  // ERR

  // Construct from another annotated_arg
  // templated copy constructor
  annotated_arg<A, decltype(properties{conduit, stable})> arg11(tmp11);
  annotated_arg<B, decltype(properties{stable, conduit})> arg14(
      tmp11); // convertible type
  auto arg12 = annotated_arg<A, decltype(properties{stable, conduit})>(tmp11);

  // default copy constructor
  auto arg13 = annotated_arg(tmp12);
  static_assert(std::is_same<decltype(arg13), annotated_arg_t3>::value,
                "deduction guide failed 3");

  // Construct from another annotated_arg and a property list
  // annotated_arg<A, decltype(properties{conduit, stable})> arg21(tmp11,
  // properties{stable});   // ERR:  the type properties should be the union of
  // the inputs
  annotated_arg<A, decltype(properties{conduit, stable})> arg22(
      tmp12, properties{stable});
  auto arg23 = annotated_arg(tmp12, properties{stable}); // deduction guide
  static_assert(std::is_same<decltype(arg22), annotated_arg_t1>::value,
                "deduction guide failed 4");
  static_assert(std::is_same<decltype(arg23), decltype(arg22)>::value,
                "deduction guide failed 5");
  annotated_arg<B, decltype(properties{stable, conduit})> arg24(
      tmp12, properties{stable}); // convertible type

  // Property merge
  auto arg31 = annotated_arg_t3(obj, conduit);                    // OK
  auto arg32 = annotated_arg(arg31, properties{stable});          // OK
  auto arg33 = annotated_arg(arg32, properties{stable, conduit}); // OK
  // auto arg34 = annotated_arg(arg32, properties{conduit, latency<22>});  //
  // ERR: invalid property
  static_assert(std::is_same<decltype(arg32), annotated_arg_t1>::value,
                "deduction guide failed 6");
  static_assert(std::is_same<decltype(arg33), annotated_arg_t1>::value,
                "deduction guide failed 7");
  // auto arg35 = annotated_arg(arg32, properties{conduit, dwidth<22>});  //
  // ERR: two input property lists are conflict
  // annotated_arg<A, decltype(properties{conduit, stable})>
  //    arg36(arg31, properties{latency<32>, stable}); // ERR: input
  // property list is conflict with the declared type

  // Implicit Conversion
  const A &x13 = arg32; // OK
  A x14 = arg32;        // OK
  // A& x11 = arg32;   // ERR: non-const lvalue reference to type 'A' cannot
  // bind to a value of unrelated type

  // operator[]
  a1[0] = 5;

  // has/get property
  static_assert(annotated_arg_t1::has_property<conduit_key>(),
                "has property 1");
  static_assert(annotated_arg_t1::get_property<conduit_key>() == conduit,
                "get property 1");
  static_assert(annotated_arg_t1::has_property<latency_key>() == false,
                "has property 2");

  static_assert(annotated_arg_t3::has_property<stable_key>() == false,
                "has property 3");
  // auto stable_prop = annotated_arg_t3::get_property<stable_key>();   // ERR:
  // can't get non-existing property

  *raw = 0;
  q.submit([&](handler &h) { h.single_task(MyIP{raw, a1}); }).wait();

  std::cout << raw[0] << std::endl;
  free(raw, q);
}

int main() {
  TestVectorAddWithAnnotatedMMHosts();
  return 0;
}
