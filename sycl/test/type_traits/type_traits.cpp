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

template <typename T, typename TL, typename CheckedT, bool Expected = true>
void test_make_type_t() {
  static_assert(is_same<d::make_type_t<T, TL>, CheckedT>::value == Expected,
                "");
}

template <typename T, typename CheckedT, bool Expected = true>
void test_make_larger_t() {
  static_assert(is_same<d::make_larger_t<T>, CheckedT>::value == Expected, "");
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

template <typename T> void test_nan_types() {
  static_assert(sizeof(d::vector_element_t<d::nan_return_t<T>>) ==
                sizeof(d::nan_argument_base_t<T>));
}

template <typename T, typename CheckedT, bool Expected = true>
void test_make_signed_t() {
  static_assert(is_same<d::make_signed_t<T>, CheckedT>::value == Expected, "");
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

template <typename T, typename SpaceList, bool Expected = true>
void test_is_address_space_compliant() {
  static_assert(d::is_address_space_compliant<T, SpaceList>::value == Expected,
                "");
}

template <typename T, int Checked, bool Expected = true>
void test_vector_size() {
  static_assert((d::vector_size<T>::value == Checked) == Expected, "");
}

template <bool Expected, typename... Args> void test_is_same_vector_size() {
  static_assert(d::is_same_vector_size<Args...>::value == Expected, "");
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

  test_is_address_space_compliant<int *, d::gvl::nonconst_address_space_list>();
  test_is_address_space_compliant<float *,
                                  d::gvl::nonconst_address_space_list>();
  test_is_address_space_compliant<s::constant_ptr<int>,
                                  d::gvl::nonconst_address_space_list, false>();
  test_is_address_space_compliant<s::constant_ptr<float>,
                                  d::gvl::nonconst_address_space_list, false>();
  test_is_address_space_compliant<s::int2 *,
                                  d::gvl::nonconst_address_space_list>();
  test_is_address_space_compliant<s::float2 *,
                                  d::gvl::nonconst_address_space_list>();
  test_is_address_space_compliant<s::constant_ptr<s::int2>,
                                  d::gvl::nonconst_address_space_list, false>();
  test_is_address_space_compliant<s::constant_ptr<s::float2>,
                                  d::gvl::nonconst_address_space_list, false>();

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

  test_make_type_t<int, d::gtl::scalar_unsigned_int_list, unsigned int>();
  test_make_type_t<s::opencl::cl_int, d::gtl::scalar_float_list,
                   s::opencl::cl_float>();
  test_make_type_t<s::vec<s::opencl::cl_int, 3>,
                   d::gtl::scalar_unsigned_int_list,
                   s::vec<s::opencl::cl_uint, 3>>();
  test_make_type_t<s::vec<s::opencl::cl_int, 3>, d::gtl::scalar_float_list,
                   s::vec<s::opencl::cl_float, 3>>();

  test_make_larger_t<s::half, float>();
  test_make_larger_t<s::half3, s::float3>();
  test_make_larger_t<float, double>();
  test_make_larger_t<s::float3, s::double3>();
  test_make_larger_t<double, void>();
  test_make_larger_t<s::double3, void>();
  test_make_larger_t<int32_t, int64_t>();
  test_make_larger_t<s::vec<int32_t, 8>, s::vec<int64_t, 8>>();
  test_make_larger_t<int64_t, void>();
  test_make_larger_t<s::vec<int64_t, 8>, void>();

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

  test_nan_types<unsigned short>();
  test_nan_types<unsigned int>();
  test_nan_types<unsigned long>();
  test_nan_types<unsigned long long>();
  test_nan_types<s::ushort2>();
  test_nan_types<s::uint2>();
  test_nan_types<s::ulong2>();

  test_make_signed_t<int, int>();
  test_make_signed_t<const int, const int>();
  test_make_signed_t<unsigned int, int>();
  test_make_signed_t<const unsigned int, const int>();
  test_make_signed_t<s::int2, s::int2>();
  test_make_signed_t<const s::int2, const s::int2>();
  test_make_signed_t<s::uint2, s::int2>();
  test_make_signed_t<const s::uint2, const s::int2>();

  test_make_unsigned_t<int, unsigned int>();
  test_make_unsigned_t<const int, const unsigned int>();
  test_make_unsigned_t<unsigned int, unsigned int>();
  test_make_unsigned_t<const unsigned int, const unsigned int>();
  test_make_unsigned_t<s::int2, s::uint2>();
  test_make_unsigned_t<const s::int2, const s::uint2>();
  test_make_unsigned_t<s::uint2, s::uint2>();
  test_make_unsigned_t<const s::uint2, const s::uint2>();

  test_vector_size<int, 1>();
  test_vector_size<float, 1>();
  test_vector_size<double, 1>();
  test_vector_size<s::int2, 2>();
  test_vector_size<s::float3, 3>();
  test_vector_size<s::double4, 4>();
  test_vector_size<s::vec<int, 1>, 1>();

  test_is_same_vector_size<true, int>();
  test_is_same_vector_size<true, s::int2>();
  test_is_same_vector_size<true, int, float>();
  test_is_same_vector_size<false, int, s::float2>();
  test_is_same_vector_size<true, s::int2, s::float2>();
  test_is_same_vector_size<false, s::int2, float>();
  test_is_same_vector_size<true, s::constant_ptr<int>>();
  test_is_same_vector_size<true, s::constant_ptr<s::int2>>();
  test_is_same_vector_size<true, s::constant_ptr<s::int2>, s::int2>();
  test_is_same_vector_size<false, s::constant_ptr<s::int2>, float>();

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
