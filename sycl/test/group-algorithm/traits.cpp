// RUN: %clangxx -std=c++17 -fsyntax-only -Xclang -verify %s -I %sycl_include -Xclang -verify-ignore-unexpected=note,warning
// expected-no-diagnostics
#include <CL/sycl.hpp>

using namespace sycl::ONEAPI;

struct CustomType {};

template <typename T> struct CustomFunctor {};

int main() {
  static_assert(is_native_function_object_v<plus<>>);
  static_assert(is_native_function_object_v<multiplies<>>);
  static_assert(is_native_function_object_v<bit_or<>>);
  static_assert(is_native_function_object_v<bit_xor<>>);
  static_assert(is_native_function_object_v<bit_and<>>);
  static_assert(is_native_function_object_v<plus<float>>);
  static_assert(is_native_function_object_v<plus<sycl::vec<float, 4>>>);
  static_assert(!is_native_function_object_v<plus<CustomType>>);
  static_assert(!is_native_function_object_v<CustomFunctor<float>>);
}
