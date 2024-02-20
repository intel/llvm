// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsyntax-only -Xclang -verify %s

#include "sycl/sycl.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

#include <iostream>

// clang-format on

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;

using annotated_ptr_t1 =
    annotated_ptr<int, decltype(properties(buffer_location<0>, awidth<32>,
                                           dwidth<32>))>;

using annotated_ptr_t2 =
    annotated_ptr<int, decltype(properties(buffer_location<0>, register_map,
                                           alignment<8>))>;

using annotated_ptr_t3 =
    annotated_ptr<int, decltype(properties(buffer_location<0>, awidth<32>))>;

struct MyIP {
  annotated_ptr<int, decltype(properties(buffer_location<0>, awidth<32>,
                                         dwidth<32>))>
      a;

  int b;

  MyIP(int *a_, int b_) : a(a_), b(b_) {}

  void operator()() const {
    // const int *p = a;  // ERR: converting to raw pointer not allowed
    for (int i = 0; i < b - 2; i++) {
      a[i + 2] = a[i + 1] + a[i];
    }
    *(a + 1) *= 5;
  }
};

template <typename T> T foo() {
  auto raw = new int;
  return annotated_ptr(raw, buffer_location<0>, awidth<32>);
}

void TestVectorAddWithAnnotatedMMHosts() {
  // Create the SYCL device queue
  queue q(sycl::ext::intel::fpga_selector_v);
  auto raw = malloc_shared<int>(5, q);
  for (int i = 0; i < 5; i++) {
    *raw = i;
  }

  // default ctor
  annotated_ptr_t3 a1;
  // copy ctor
  auto a2(a1);
  auto a3(foo<annotated_ptr_t3>());
  // // assign ctor
  auto a4 = a1;

  // Construct from raw pointers
  auto tmp11 = annotated_ptr(raw); // empty property list
  // Construct from raw pointers and a property list
  auto tmp12 =
      annotated_ptr<int, decltype(properties{buffer_location<0>, awidth<32>})>(
          raw, properties{buffer_location<0>, awidth<32>});
  auto tmp14 = annotated_ptr(
      raw, properties{buffer_location<0>, awidth<32>}); // deduction guide
  static_assert(std::is_same<decltype(tmp14), annotated_ptr_t3>::value,
                "deduction guide failed 1");
  // Construct from raw pointers and variadic properties
  auto tmp13 = annotated_ptr(raw, buffer_location<0>, dwidth<32>,
                             awidth<32>); // deduction guide
  static_assert(std::is_same<decltype(tmp13), annotated_ptr_t1>::value,
                "deduction guide failed 2");
  auto tmp15 = annotated_ptr(raw, buffer_location<0>, awidth<32>);
  static_assert(std::is_same<decltype(tmp15), annotated_ptr_t3>::value,
                "deduction guide failed 1");

  // Construct from another annotated_ptr
  // templated copy constructor
  annotated_ptr<int, decltype(properties{buffer_location<0>, awidth<32>,
                                         dwidth<32>})>
      arg11(tmp11);
  auto arg12 =
      annotated_ptr<int, decltype(properties{buffer_location<0>, dwidth<32>,
                                             awidth<32>})>(tmp11);

  // default copy constructor
  auto arg13 = annotated_ptr(tmp12);
  static_assert(std::is_same<decltype(arg13), annotated_ptr_t3>::value,
                "deduction guide failed 3");

  // Construct from another annotated_ptr and a property list
  annotated_ptr<int, decltype(properties{buffer_location<0>, awidth<32>,
                                         dwidth<32>})>
      arg22(tmp12, properties{dwidth<32>});
  auto arg23 = annotated_ptr(tmp12, properties{dwidth<32>}); // deduction guide
  static_assert(std::is_same<decltype(arg22), annotated_ptr_t1>::value,
                "deduction guide failed 4");
  static_assert(std::is_same<decltype(arg23), decltype(arg22)>::value,
                "deduction guide failed 5");

  // Construct from inconvertible type
  // annotated_ptr<float> tmp21;
  // annotated_ptr<int, decltype(properties{dwidth<32>})> arg24(tmp21,
  // properties{dwidth<32>});   // ERR

  // Removed
  // Assignment / implicit conversion
  // expected-note@sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp:* {{candidate function not viable: no known conversion from 'int *' to 'const annotated_ptr}}
  // expected-error@+1 {{no viable overloaded '='}}
  a1 = raw;

  // Property merge
  auto arg31 = annotated_ptr_t3(raw, buffer_location<0>, awidth<32>);     // OK
  auto arg32 = annotated_ptr(arg31, properties{dwidth<32>});              // OK
  auto arg33 = annotated_ptr(arg32, properties{dwidth<32>, awidth<32>});  // OK
  auto arg34 = annotated_ptr(arg32, properties{awidth<32>, latency<22>}); // OK
  static_assert(std::is_same<decltype(arg32), annotated_ptr_t1>::value,
                "deduction guide failed 6");
  static_assert(std::is_same<decltype(arg33), annotated_ptr_t1>::value,
                "deduction guide failed 7");

  // operator[]
  arg31[0] = 1;
  for (int i = 1; i < 5; i++) {
    arg31[i] = arg31[i - 1];
  }

  // prefix/postfix increment/decrement
  for (int i = 0; i < 5; i++) {
    *arg31 = i;
    arg31++;
    --arg31;
  }

  // has/get property
  static_assert(annotated_ptr_t1::has_property<awidth_key>(), "has_property 1");
  static_assert(annotated_ptr_t1::get_property<awidth_key>() == awidth<32>,
                "get_property 1");
  static_assert(annotated_ptr_t2::has_property<latency_key>() == false,
                "has_property 2");

  static_assert(annotated_ptr_t2::has_property<alignment_key>(),
                "has_property 3");

  static_assert(annotated_ptr_t2::get_property<alignment_key>() == alignment<8>,
                "get_property 3");
  // auto dwidth_prop = annotated_ptr_t3::get_property<dwidth_key>();   // ERR

  q.submit([&](handler &h) { h.single_task(MyIP{raw, 5}); }).wait();

  for (int i = 0; i < 5; i++) {
    std::cout << raw[i] << std::endl;
  }

  class test {
    int n;

  public:
    test(int n_) : n(n_) {}
    test(const test &t) { n = t.n; }
  };
  // expected-error@sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp:* {{annotated_ptr can only encapsulate either a trivially-copyable type or void!}}
  // expected-note@+1 {{in instantiation of template class 'sycl::ext::oneapi::experimental::annotated_ptr<test>'}}
  annotated_ptr<test> non_trivially_copyable;

  annotated_ptr<void> void_type;

  struct g {
    int a;
  };
  g g0, g1;
  // TODO: these notes shouldn't be emitted
  // expected-note@sycl/types.hpp:* {{candidate template ignored: could not match 'vec<T, Num>'}}
  // expected-note@sycl/types.hpp:* {{candidate template ignored: could not match 'detail::SwizzleOp}}
  // expected-note@sycl/types.hpp:* {{candidate template ignored: could not match 'vec<T, Num>'}}
  // expected-error@+1 {{invalid operands to binary expression}}
  auto g2 = g0 + g1;

  annotated_ptr gp{&g0};
  auto g3 = *gp;

  free(raw, q);
}

int main() {
  TestVectorAddWithAnnotatedMMHosts();
  return 0;
}
