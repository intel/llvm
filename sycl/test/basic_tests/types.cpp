// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes -fsycl-targets=%sycl_triple %s -fsyntax-only %}

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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES

  // SYCL 2020 spec says that alignment is supposed to be same as size,
  // but MSVC won't allow an alignment of anything larger than 64 for
  // a direct parameter. The math built-ins use direct param calls.
  // It has been decided to change the spec to have a max alignment of
  // 64.
  if constexpr (alignof(T) * RealLength <= 64)
    static_assert(alignof(VectorT) == (alignof(T) * RealLength), "");
  else
    static_assert(alignof(VectorT) == 64,
                  "huge vectors should have a maximum alignment of 64");

#else // __INTEL_PREVIEW_BREAKING_CHANGES

#if defined(_WIN32) && (_MSC_VER) &&                                           \
    defined(__NO_EXT_VECTOR_TYPE_ON_HOST__) && !defined(__SYCL_DEVICE_ONLY__)
  // See comments around __SYCL_ALIGNED_VAR macro definition in types.hpp
  // We can't enforce proper alignment of "huge" vectors (>64 bytes) on Windows
  // and the test exposes this limitation.
  if constexpr (alignof(T) * RealLength < 64)
#endif
    static_assert(alignof(VectorT) == (alignof(T) * RealLength), "");

#endif // __INTEL_PREVIEW_BREAKING_CHANGES
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

template <typename vecType, int numOfElems>
std::string vec2string(const sycl::vec<vecType, numOfElems> &vec) {
  std::string str = "";
  for (size_t i = 0; i < numOfElems - 1; ++i) {
    str += std::to_string(vec[i]) + ",";
  }
  str = "{" + str + std::to_string(vec[numOfElems - 1]) + "}";
  return str;
}

// vec::operator! might return a different type as described in Table 143 of the
// SYCL 2020 specification. This function checks that the result type matches
// the expected type.
template <typename T, typename Expected> inline void checkVecNotReturnType() {
  constexpr int N = 4;
  using Vector = sycl::vec<T, N>;
#if defined(__INTEL_PREVIEW_BREAKING_CHANGES)
  using ExpectedVector = sycl::vec<Expected, N>;
  using OpNotResult = decltype(operator!(std::declval<Vector>()));
#else
  using ExpectedVector = sycl::vec<T, N>;
  using OpNotResult = decltype(std::declval<Vector>().operator!());
#endif
  static_assert(std::is_same_v<OpNotResult, ExpectedVector>,
                "Incorrect vec::operator! return type");
}

// the math built-in testing ensures that the vec binary ops get tested,
// but the unary ops are only tested by the CTS tests. Here we do some
// basic testing of the unary ops, ensuring they compile correctly.
template <typename T> void checkVecUnaryOps(T &v) {

  std::cout << vec2string(v) << std::endl;

  T d = +v;
  std::cout << vec2string(d) << std::endl;

  T e = -v;
  std::cout << vec2string(e) << std::endl;

  // ~ only supported by integral types.
  if constexpr (std::is_integral_v<T>) {
    T g = ~v;
    std::cout << vec2string(g) << std::endl;
  }

  auto f = !v;
  std::cout << vec2string(f) << std::endl;

  // Check operator! return type
  checkVecNotReturnType<int8_t, int8_t>();
  checkVecNotReturnType<uint8_t, int8_t>();
  checkVecNotReturnType<int16_t, int16_t>();
  checkVecNotReturnType<uint16_t, int16_t>();
  checkVecNotReturnType<sycl::half, int16_t>();
  checkVecNotReturnType<int32_t, int32_t>();
  checkVecNotReturnType<uint32_t, int32_t>();
  checkVecNotReturnType<float, int32_t>();
  checkVecNotReturnType<int64_t, int64_t>();
  checkVecNotReturnType<uint64_t, int64_t>();
  checkVecNotReturnType<double, int64_t>();
}

void checkVariousVecUnaryOps() {
  sycl::vec<int, 1> vi1{1};
  checkVecUnaryOps(vi1);
  sycl::vec<int, 16> vi{1, 2, 0, -4, 1, 2, 0, -4, 1, 2, 0, -4, 1, 2, 0, -4};
  checkVecUnaryOps(vi);

  sycl::vec<long, 1> vl1{1};
  checkVecUnaryOps(vl1);
  sycl::vec<long, 16> vl{2, 3, 0, -5, 2, 3, 0, -5, 2, 3, 0, -5, 2, 3, 0, -5};
  checkVecUnaryOps(vl);

  sycl::vec<long long, 1> vll1{1};
  checkVecUnaryOps(vll1);
  sycl::vec<long long, 16> vll{0, 3, 4, -6, 0, 3, 4, -6,
                               0, 3, 4, -6, 0, 3, 4, -6};
  checkVecUnaryOps(vll);

  sycl::vec<float, 1> vf1{1};
  checkVecUnaryOps(vf1);
  sycl::vec<float, 16> vf{0, 4, 5, -9, 0, 4, 5, -9, 0, 4, 5, -9, 0, 4, 5, -9};
  checkVecUnaryOps(vf);

  sycl::vec<double, 1> vd1{1};
  checkVecUnaryOps(vd1);
  sycl::vec<double, 16> vd{0, 4, 5, -9, 0, 4, 5, -9, 0, 4, 5, -9, 0, 4, 5, -9};
  checkVecUnaryOps(vd);

  sycl::vec<sycl::half, 1> vh1{1};
  checkVecUnaryOps(vh1);
  sycl::vec<sycl::half, 16> vh{0, 4, 5, -9, 0, 4, 5, -9,
                               0, 4, 5, -9, 0, 4, 5, -9};
  checkVecUnaryOps(vh);

  sycl::vec<sycl::ext::oneapi::bfloat16, 1> vbf1{1};
  checkVecUnaryOps(vbf1);
  sycl::vec<sycl::ext::oneapi::bfloat16, 16> vbf{0, 4, 5, -9, 0, 4, 5, -9,
                                                 0, 4, 5, -9, 0, 4, 5, -9};
  checkVecUnaryOps(vbf);
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

  checkVariousVecUnaryOps();

  return 0;
}
