// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only -D__NO_EXT_VECTOR_TYPE_ON_HOST__
//==--------------- types.cpp - SYCL types test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <cfloat>
#include <cstdint>
#include <type_traits>

using namespace std;
namespace s = sycl;

template <typename T, int N> inline void checkVectorSizeAndAlignment() {
  using VectorT = s::vec<T, N>;
  constexpr auto RealLength = (N != 3 ? N : 4);
  static_assert(sizeof(VectorT) == (sizeof(T) * RealLength), "");
#if defined(_WIN32) && (_MSC_VER) &&                                           \
    defined(__NO_EXT_VECTOR_TYPE_ON_HOST__) && !defined(__SYCL_DEVICE_ONLY__)
  // See comments around __SYCL_ALIGNED_VAR macro definition in types.hpp
  // We can't enforce proper alignment of "huge" vectors (>64 bytes) on Windows
  // and the test exposes this limitation.
  if constexpr (alignof(T) * RealLength < 64)
#endif
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
  checkVectorsWithN<bool>();
  checkVectorsWithN<s::half>();
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
  checkVectorsWithN<::cl_char>();
  checkVectorsWithN<::cl_uchar>();
  checkVectorsWithN<::cl_short>();
  checkVectorsWithN<::cl_ushort>();
  checkVectorsWithN<::cl_int>();
  checkVectorsWithN<::cl_uint>();
  checkVectorsWithN<::cl_long>();
  checkVectorsWithN<::cl_ulong>();
  checkVectorsWithN<::cl_half>();
  checkVectorsWithN<::cl_float>();
  checkVectorsWithN<::cl_double>();
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

template <> inline void checkSizeForFloatingPoint<s::half, sizeof(int16_t)>() {
  // TODO is_floating_point does not support sycl half now.
  // static_assert(is_floating_point<T>::is_iec559, "");
  // TODO numeric_limits does not support sycl half now.
  // static_assert(numeric_limits<T>::is_iec559, "");
  static_assert(sizeof(s::half) == sizeof(int16_t), "");
}

int main() {
  // Test for creating constexpr expressions
  constexpr sycl::specialization_id<sycl::vec<sycl::half, 2>> id(1.0);
  constexpr sycl::marray<sycl::half, 2> MH(3);
  // Check the size and alignment of the SYCL vectors.
  checkVectors();

  // Table 4.93: Additional scalar data types supported by SYCL.
  static_assert(sizeof(s::byte) == sizeof(int8_t), "");

  // SYCL 2020: Table 193. Scalar data type aliases supported by SYCL OpenCL
  // backend
  static_assert(is_same<s::opencl::cl_bool, decltype(0 != 1)>::value, "");
  checkSizeForSignedIntegral<s::opencl::cl_char, sizeof(int8_t)>();
  checkSizeForUnsignedIntegral<s::opencl::cl_uchar, sizeof(uint8_t)>();
  checkSizeForSignedIntegral<s::opencl::cl_short, sizeof(int16_t)>();
  checkSizeForUnsignedIntegral<s::opencl::cl_ushort, sizeof(uint16_t)>();
  checkSizeForSignedIntegral<s::opencl::cl_int, sizeof(int32_t)>();
  checkSizeForUnsignedIntegral<s::opencl::cl_uint, sizeof(uint32_t)>();
  checkSizeForSignedIntegral<s::opencl::cl_long, sizeof(int64_t)>();
  checkSizeForUnsignedIntegral<s::opencl::cl_ulong, sizeof(uint64_t)>();
  checkSizeForFloatingPoint<s::opencl::cl_half, sizeof(int16_t)>();
  checkSizeForFloatingPoint<s::opencl::cl_float, sizeof(int32_t)>();
  checkSizeForFloatingPoint<s::opencl::cl_double, sizeof(int64_t)>();

  return 0;
}
