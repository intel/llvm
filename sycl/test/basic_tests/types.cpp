// RUN: %clangxx -fsycl %s -o %t.out -lOpenCL
//==--------------- types.cpp - SYCL types test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

template <typename T, int N> inline void checkVectorSizeAndAlignment() {
  using VectorT = cl::sycl::vec<T, N>;
  constexpr auto RealLength = (N != 3 ? N : 4);
  static_assert(sizeof(VectorT) == (sizeof(T) * RealLength),
                "Wrong size of vec<T, N>");
  static_assert(alignof(VectorT) == (alignof(T) * RealLength),
                "Wrong alignment of vec<T, N>");
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

int main() {
  checkVectors();
  return 0;
}
