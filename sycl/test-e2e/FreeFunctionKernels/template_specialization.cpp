// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

struct TestStruct {
  int x;
  float y;

  TestStruct(int a, float b) : x(a), y(b) {}
};

namespace A::B::C {
class TestClass {
  int a;
  float b;

public:
  TestClass(int x, float y) : a(x), b(y) {}

  void setA(int x) { a = x; }
  void setB(float y) { b = y; }
};
} // namespace A::B::C

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum(T arg) {}

template <>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum<int>(int arg) {
  arg = 42;
}

template <> void sum<float>(float arg) { arg = 3.14f; }

template <> void sum<TestStruct>(TestStruct arg) {
  arg.x = 100;
  arg.y = 2.0f;
}

template <> void sum<A::B::C::TestClass>(A::B::C::TestClass arg) {
  arg.setA(10);
  arg.setB(5.0f);
}

template <typename T> void test_func() {
  queue Q;
  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id id = ext::oneapi::experimental::get_kernel_id<sum<T>>();
  kernel Kernel = bundle.get_kernel(id);
  Q.submit([&](handler &h) {
    h.set_args(static_cast<T>(4));
    h.parallel_for(nd_range{{1}, {1}}, Kernel);
  });
}

template <typename T> void test_func_custom_type() {
  queue Q;
  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id id = ext::oneapi::experimental::get_kernel_id<sum<T>>();
  kernel Kernel = bundle.get_kernel(id);
  Q.submit([&](handler &h) {
    h.set_args(T(1, 2.0f));
    h.parallel_for(nd_range{{1}, {1}}, Kernel);
  });
}

int main() {
  test_func<int>();
  test_func<float>();
  test_func<uint32_t>();
  test_func<char>();
  test_func_custom_type<TestStruct>();
  test_func_custom_type<A::B::C::TestClass>();
  return 0;
}
