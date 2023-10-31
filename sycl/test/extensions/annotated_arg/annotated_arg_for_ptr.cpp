// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// expected-no-diagnostics

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

// clang-format on

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using annotated_arg_t1 =
    annotated_arg<int *, decltype(properties(buffer_location<0>, awidth<32>,
                                             dwidth<32>))>;

using annotated_arg_t2 = annotated_arg<int, decltype(properties(register_map))>;

using annotated_arg_t3 =
    annotated_arg<int *, decltype(properties(buffer_location<0>, awidth<32>))>;

struct MyIP {
  annotated_arg<int *, decltype(properties(buffer_location<0>, awidth<32>,
                                           dwidth<32>))>
      a;

  int b;

  MyIP(int *a_, int b_) : a(a_), b(b_) {}

  void operator()() const {
    int *p = a;
    const int *p2 = a;

    for (int i = 0; i < b; i++) {
      p[i] = i;
      a[i] += 1;
    }
  }
};

template <typename T> T foo() {
  auto raw = new int;
  return annotated_arg(raw, buffer_location<0>, awidth<32>);
}

void TestVectorAddWithAnnotatedMMHosts() {
  // Create the SYCL device queue
  queue q(sycl::ext::intel::fpga_selector_v);
  auto raw = malloc_shared<int>(5, q);

  // default ctor
  annotated_arg_t3 a1;
  // copy ctor
  auto a2(a1);
  auto a3(foo<annotated_arg_t3>());
  // // assign ctor
  auto a4 = a1;

  // Construct from raw pointers
  auto tmp11 = annotated_arg(raw); // empty property list
  // Construct from raw pointers and a property list
  auto tmp12 =
      annotated_arg<int *,
                    decltype(properties{buffer_location<0>, awidth<32>})>(
          raw, properties{buffer_location<0>, awidth<32>});
  auto tmp14 = annotated_arg(
      raw, properties{buffer_location<0>, awidth<32>}); // deduction guide
  static_assert(std::is_same<decltype(tmp14), annotated_arg_t3>::value,
                "deduction guide failed 1");
  // Construct from raw pointers and variadic properties
  auto tmp13 = annotated_arg(raw, buffer_location<0>, dwidth<32>,
                             awidth<32>); // deduction guide
  static_assert(std::is_same<decltype(tmp13), annotated_arg_t1>::value,
                "deduction guide failed 2");
  auto tmp15 = annotated_arg(raw, buffer_location<0>, awidth<32>);
  static_assert(std::is_same<decltype(tmp15), annotated_arg_t3>::value,
                "deduction guide failed 1");

  auto tmp16 = annotated_arg(raw, properties{alignment<16>}); // deduction guide

  // Property list can't have duplicated properties
  // auto tmp16 = annotated_arg(raw, awidth<32>, awidth<32>);   // ERR
  // auto tmp17 = annotated_arg(raw, awidth<32>, awidth<22>);  // ERR

  // auto tmp18 = annotated_arg(raw, properties{awidth<32>, dwidth<32>,
  // awidth<32>}); // ERR: Duplicate properties in property list auto tmp19 =
  // annotated_arg(raw, properties{awidth<32>, awidth<22>});  // ERR

  // Construct from another annotated_arg
  // templated copy constructor
  annotated_arg<int *, decltype(properties{buffer_location<0>, awidth<32>,
                                           dwidth<32>})>
      arg11(tmp11);
  auto arg12 =
      annotated_arg<int *, decltype(properties{buffer_location<0>, dwidth<32>,
                                               awidth<32>})>(tmp11);

  // default copy constructor
  auto arg13 = annotated_arg(tmp12);
  static_assert(std::is_same<decltype(arg13), annotated_arg_t3>::value,
                "deduction guide failed 3");

  // Construct from another annotated_arg and a property list
  // annotated_arg<int*, decltype(properties{awidth<32>, dwidth<32>})>
  // arg21(tmp11, properties{dwidth<32>});   // ERR:  the type properties should
  // be the union of the inputs
  annotated_arg<int *, decltype(properties{buffer_location<0>, awidth<32>,
                                           dwidth<32>})>
      arg22(tmp12, properties{dwidth<32>});
  auto arg23 = annotated_arg(
      tmp12, properties{buffer_location<0>, dwidth<32>}); // deduction guide
  static_assert(std::is_same<decltype(arg22), annotated_arg_t1>::value,
                "deduction guide failed 4");
  static_assert(std::is_same<decltype(arg23), decltype(arg22)>::value,
                "deduction guide failed 5");

  // Construct from inconvertible type
  // annotated_arg<int> tmp21;
  // annotated_arg<int*, decltype(properties{dwidth<32>})> arg24(tmp21,
  // properties{dwidth<32>});   // ERR

  // Property merge
  auto arg31 = annotated_arg_t3(raw, buffer_location<0>, awidth<32>); // OK
  auto arg32 =
      annotated_arg(arg31, properties{buffer_location<0>, dwidth<32>}); // OK
  auto arg33 = annotated_arg(
      arg32, properties{buffer_location<0>, dwidth<32>, awidth<32>}); // OK
  auto arg34 = annotated_arg(
      arg32, properties{buffer_location<0>, awidth<32>, latency<22>}); // OK
  static_assert(std::is_same<decltype(arg32), annotated_arg_t1>::value,
                "deduction guide failed 6");
  static_assert(std::is_same<decltype(arg33), annotated_arg_t1>::value,
                "deduction guide failed 7");
  // auto arg34 = annotated_arg(arg32, properties{awidth<32>, dwidth<22>});  //
  // ERR: two input property lists are conflict
  // annotated_arg<int*, decltype(properties{awidth<32>, dwidth<32>})>
  //    arg35(arg31, properties{latency<32>, dwidth<32>}); // ERR: input
  // property list is conflict with the declared type

  // Implicit Conversion
  int *x11 = arg13;
  const int *x13 = arg32;

  // operator[]
  arg31[0] = 1;
  for (int i = 1; i < 5; i++) {
    arg31[i] = arg31[i - 1];
  }

  // has/get property
  static_assert(annotated_arg_t1::has_property<awidth_key>(), "has property 1");
  static_assert(annotated_arg_t1::get_property<awidth_key>() == awidth<32>,
                "get property 1");
  static_assert(annotated_arg_t1::has_property<latency_key>() == false,
                "has property 2");

  static_assert(annotated_arg_t3::has_property<dwidth_key>() == false,
                "has property 3");
  // auto dwidth_prop = annotated_arg_t3::get_property<dwidth_key>();   // ERR

  q.submit([&](handler &h) { h.single_task(MyIP{raw, 5}); }).wait();

  for (int i = 0; i < 5; i++) {
    std::cout << raw[i] << std::endl;
  }

  free(raw, q);
}

int main() {
  TestVectorAddWithAnnotatedMMHosts();
  return 0;
}
