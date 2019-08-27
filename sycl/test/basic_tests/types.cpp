// RUN: %clangxx -fsycl %s -o %t.out
//==--------------- types.cpp - SYCL types test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cstdint>
#include <limits>
#include <type_traits>

using namespace std;
namespace s = cl::sycl;

template <typename T, int N> inline void checkVectorSizeAndAlignment() {
  using VectorT = s::vec<T, N>;
  constexpr auto RealLength = (N != 3 ? N : 4);
  static_assert(sizeof(VectorT) == (sizeof(T) * RealLength), "");
  static_assert(alignof(VectorT) == (alignof(T) * RealLength), "");
}

template <typename T> inline void checkVectorsWithN() {
  checkVectorSizeAndAlignment<T, 1>();
  checkVectorSizeAndAlignment<T, 2>();
  checkVectorSizeAndAlignment<T, 3>();
  checkVectorSizeAndAlignment<T, 4>();
  checkVectorSizeAndAlignment<T, 8>();
  checkVectorSizeAndAlignment<T, 16>();
}

inline void checkVectors() {
  checkVectorsWithN<half>();
  checkVectorsWithN<float>();
  checkVectorsWithN<double>();
  checkVectorsWithN<char>();
  checkVectorsWithN<signed char>();
  checkVectorsWithN<unsigned char>();
  checkVectorsWithN<signed short>();
  checkVectorsWithN<unsigned short>();
  checkVectorsWithN<signed int>();
  checkVectorsWithN<unsigned int>();
  checkVectorsWithN<signed long>();
  checkVectorsWithN<unsigned long>();
  checkVectorsWithN<signed long long>();
  checkVectorsWithN<unsigned long long>();
  checkVectorsWithN<s::cl_char>();
  checkVectorsWithN<s::cl_uchar>();
  checkVectorsWithN<s::cl_short>();
  checkVectorsWithN<s::cl_ushort>();
  checkVectorsWithN<s::cl_int>();
  checkVectorsWithN<s::cl_uint>();
  checkVectorsWithN<s::cl_long>();
  checkVectorsWithN<s::cl_ulong>();
  checkVectorsWithN<s::cl_half>();
  checkVectorsWithN<s::cl_float>();
  checkVectorsWithN<s::cl_double>();
}

template <typename T, int ExpectedSize>
inline void checkSizeForSignedIntegral() {
  static_assert(is_integral<T>::value, "");
  static_assert(is_signed<T>::value, "");
  static_assert(sizeof(T) == ExpectedSize, "");
}

template <typename T, int ExpectedSize>
inline void checkSizeForUnsignedIntegral() {
  static_assert(is_integral<T>::value, "");
  static_assert(is_unsigned<T>::value, "");
  static_assert(sizeof(T) == ExpectedSize, "");
}

template <typename T, int ExpectedSize>
inline void checkSizeForFloatingPoint() {
  static_assert(is_floating_point<T>::value, "");
  static_assert(numeric_limits<T>::is_iec559, "");
  static_assert(sizeof(T) == ExpectedSize, "");
}

template <>
inline void checkSizeForFloatingPoint<s::cl_half, sizeof(std::int16_t)>() {
  // TODO is_floating_point does not support sycl half now.
  // static_assert(is_floating_point<T>::is_iec559, "");
  // TODO numeric_limits does not support sycl half now.
  // static_assert(numeric_limits<T>::is_iec559, "");
  static_assert(sizeof(s::cl_half) == sizeof(std::int16_t), "");
}

int main() {
  // Check the size and alignment of the SYCL vectors.
  checkVectors();

  // Table 4.93: Additional scalar data types supported by SYCL.
  static_assert(sizeof(s::byte) == sizeof(std::int8_t), "");

  // Table 4.94: Scalar data type aliases supported by SYCL
  static_assert(is_same<s::cl_bool, decltype(0 != 1)>::value, "");
  checkSizeForSignedIntegral<s::cl_char, sizeof(std::int8_t)>();
  checkSizeForUnsignedIntegral<s::cl_uchar, sizeof(std::uint8_t)>();
  checkSizeForSignedIntegral<s::cl_short, sizeof(std::int16_t)>();
  checkSizeForUnsignedIntegral<s::cl_ushort, sizeof(std::uint16_t)>();
  checkSizeForSignedIntegral<s::cl_int, sizeof(std::int32_t)>();
  checkSizeForUnsignedIntegral<s::cl_uint, sizeof(std::uint32_t)>();
  checkSizeForSignedIntegral<s::cl_long, sizeof(std::int64_t)>();
  checkSizeForUnsignedIntegral<s::cl_ulong, sizeof(std::uint64_t)>();
  checkSizeForFloatingPoint<s::cl_half, sizeof(std::int16_t)>();
  checkSizeForFloatingPoint<s::cl_float, sizeof(std::int32_t)>();
  checkSizeForFloatingPoint<s::cl_double, sizeof(std::int64_t)>();

  // Other OpenCL aliases which can be found in the SYCL specification
  checkSizeForUnsignedIntegral<s::cl_size_t, sizeof(std::int64_t)>();
  checkSizeForSignedIntegral<s::cl_ptrdiff_t, sizeof(std::int64_t)>();
  checkSizeForSignedIntegral<s::cl_intptr_t, sizeof(std::int64_t)>();
  checkSizeForUnsignedIntegral<s::cl_uintptr_t, sizeof(std::int64_t)>();

  return 0;
}
