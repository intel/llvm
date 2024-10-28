// RUN: %clangxx -fsycl -fsyntax-only %s
//==-------------- type_traits.cpp - SYCL type_traits test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <type_traits>

using namespace std;

namespace s = sycl;
namespace d = sycl::detail;

template <typename T, bool Expected = true> void test_is_integral() {
  static_assert(d::is_integral<T>::value == Expected, "");
}

template <typename T, bool Expected = true> void test_is_floating_point() {
  static_assert(d::is_floating_point<T>::value == Expected, "");
}

template <typename T, bool Expected = true> void test_is_arithmetic() {
  static_assert(d::is_arithmetic<T>::value == Expected, "");
}

template <typename T, typename T2, typename CheckedT, bool Expected = true>
void test_change_base_type_t() {
  static_assert(
      is_same<d::change_base_type_t<T, T2>, CheckedT>::value == Expected, "");
}

template <typename T, typename CheckedT, bool Expected = true>
void test_vector_element_t() {
  static_assert(is_same<d::vector_element_t<T>, CheckedT>::value == Expected,
                "");
}

template <typename T, typename CheckedT, bool Expected = true>
void test_make_unsigned_t() {
  static_assert(is_same<d::make_unsigned_t<T>, CheckedT>::value == Expected,
                "");
}

template <typename T, bool Expected = true> void test_is_pointer() {
  static_assert(d::is_pointer<T>::value == Expected, "");
}

template <typename T, typename CheckedT, bool Expected = true>
void test_remove_pointer_t() {
  static_assert(is_same<d::remove_pointer_t<T>, CheckedT>::value == Expected,
                "");
}

int main() {
  test_is_pointer<int *>();
  test_is_pointer<float *>();
  test_is_pointer<s::constant_ptr<int>>();
  test_is_pointer<s::constant_ptr<float>>();
  test_is_pointer<s::int2 *>();
  test_is_pointer<s::float2 *>();
  test_is_pointer<s::constant_ptr<s::int2>>();
  test_is_pointer<s::constant_ptr<s::float2>>();

  test_remove_pointer_t<int *, int>();
  test_remove_pointer_t<float *, float>();
  test_remove_pointer_t<s::constant_ptr<int>, int>();
  test_remove_pointer_t<s::constant_ptr<float>, float>();
  test_remove_pointer_t<s::int2 *, s::int2>();
  test_remove_pointer_t<s::float2 *, s::float2>();
  test_remove_pointer_t<s::constant_ptr<s::int2>, s::int2>();
  test_remove_pointer_t<s::constant_ptr<s::float2>, s::float2>();

  test_is_integral<int>();
  test_is_integral<s::int2>();
  test_is_integral<float, false>();
  test_is_integral<s::float2, false>();
  test_is_integral<s::half, false>();
  test_is_integral<s::half2, false>();

  test_is_floating_point<int, false>();
  test_is_floating_point<s::int2, false>();
  test_is_floating_point<float>();
  test_is_floating_point<s::float2>();
  test_is_floating_point<s::half>();
  test_is_floating_point<s::half2>();

  test_is_arithmetic<int>();
  test_is_arithmetic<s::int2>();
  test_is_arithmetic<float>();
  test_is_arithmetic<s::float2>();
  test_is_arithmetic<s::half>();
  test_is_arithmetic<s::half2>();

  test_change_base_type_t<int, float, float>();
  test_change_base_type_t<s::int2, float, s::float2>();
  test_change_base_type_t<long, float, float>();
  test_change_base_type_t<s::long2, float, s::float2>();

  test_vector_element_t<int, int>();
  test_vector_element_t<const int, const int>();
  test_vector_element_t<volatile int, volatile int>();
  test_vector_element_t<const volatile int, const volatile int>();
  test_vector_element_t<s::int2, int>();
  test_vector_element_t<const s::int2, const int>();
  test_vector_element_t<volatile s::int2, volatile int>();
  test_vector_element_t<const volatile s::int2, const volatile int>();

  test_make_unsigned_t<int, unsigned int>();
  test_make_unsigned_t<const int, const unsigned int>();
  test_make_unsigned_t<unsigned int, unsigned int>();
  test_make_unsigned_t<const unsigned int, const unsigned int>();
  test_make_unsigned_t<s::int2, s::uint2>();
  test_make_unsigned_t<const s::int2, const s::uint2>();
  test_make_unsigned_t<s::uint2, s::uint2>();
  test_make_unsigned_t<const s::uint2, const s::uint2>();

#ifdef __SYCL_DEVICE_ONLY__
  static_assert(
      std::is_same_v<
          s::remove_decoration_t<const __attribute__((opencl_global)) int>,
          const int>);
  static_assert(
      std::is_same_v<s::remove_decoration_t<const volatile
                                            __attribute__((opencl_global)) int>,
                     const volatile int>);
  static_assert(
      std::is_same_v<
          s::remove_decoration_t<const __attribute__((opencl_global)) int *>,
          const int *>);
  static_assert(std::is_same_v<s::remove_decoration_t<const __attribute__((
                                   opencl_global)) int *const>,
                               const int *const>);
#endif

  return 0;
}
