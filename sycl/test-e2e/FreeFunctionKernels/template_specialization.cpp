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

template <>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum<int *>(int *arg) {
  *arg = 42;
}

template <int, typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum1(T arg) {}

template <>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum1<3, sycl::accessor<int, 1>>(sycl::accessor<int, 1> arg) {
  arg[0] = 42;
}

template <>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::nd_range_kernel<1>))
void sum1<3, float>(float arg) {
  arg = 3.14f + static_cast<float>(3);
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

template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void F(int X) {
  volatile T Y = static_cast<T>(X);
}

template <>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void F<float>(int X) {
  volatile float Y = static_cast<float>(X);
}

template <typename... Args>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void variadic_templated(Args... args) {}

template <>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (ext::oneapi::experimental::single_task_kernel))
void variadic_templated<double>(double b) {
  b = 20.0f;
}

template <auto *Func, typename T> void test_func() {
  queue Q;
  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id id = ext::oneapi::experimental::get_kernel_id<Func>();
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

void test_accessor() {
  sycl::queue Q;
  constexpr size_t N = 4;
  int data[N] = {0, 1, 2, 3};
  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id id = ext::oneapi::experimental::get_kernel_id<
      sum1<3, sycl::accessor<int, 1>>>();
  kernel Kernel = bundle.get_kernel(id);
  sycl::buffer<int, 1> buf(data, sycl::range<1>(N));
  Q.submit([&](handler &h) {
    auto acc = buf.get_access<sycl::access::mode::write>(h);
    h.set_args(acc);
    h.parallel_for(nd_range{{1}, {1}}, Kernel);
  });

  auto acc = buf.get_host_access();
  assert(acc[0] == 42);
}

void test_shared() {
  sycl::queue Q;
  int *data = sycl::malloc_shared<int>(4, Q);

  kernel_bundle bundle =
      get_kernel_bundle<bundle_state::executable>(Q.get_context());
  kernel_id id = ext::oneapi::experimental::get_kernel_id<sum<int *>>();
  kernel Kernel = bundle.get_kernel(id);
  Q.submit([&](handler &h) {
    h.set_args(data);
    h.parallel_for(nd_range{{1}, {1}}, Kernel);
  });
  sycl::free(data, Q);
}

int main() {
  test_func<sum<int>, int>();
  test_func<sum<float>, float>();
  test_func<sum<uint32_t>, uint32_t>();
  test_func<sum<char>, char>();
  test_func_custom_type<TestStruct>();
  test_func_custom_type<A::B::C::TestClass>();
  test_func<F<float>, float>();
  test_func<F<uint32_t>, uint32_t>();
  test_func<variadic_templated<double>, double>();
  test_func<sum1<3, float>, float>();
  test_accessor();
  test_shared();
  return 0;
}
