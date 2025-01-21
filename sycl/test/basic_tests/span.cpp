// RUN: %clangxx -fsycl -fsyntax-only %s

//==--------------- span.cpp - SYCL span test ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

int main() {
  // test the various span declarations, especially unspecialized ones.
  // should compile
  int arr[]{1, 2, 3, 4};
  const int constArr[]{8, 7, 6};
  std::vector<int> vec(4);

  // unspecialized
  sycl::span fromArray{arr};
  sycl::span fromConstArray{constArr};
  sycl::span fromVec{vec};

  // partly specialized
  sycl::span<int> fromIntArray{arr};
  sycl::span<const int> fromIntConstArray{constArr};
  sycl::span<int> fromIntVec{vec};

  // fully specialized
  sycl::span<int,4> fullSpecArray{arr};
  sycl::span<const int,3> fullSpecConstArray{constArr};
  sycl::span<int, 4> fullSpecVecArray{vec};

  // check that the extent is deduced correctly
  static_assert(decltype(fromArray)::extent == decltype(fullSpecArray)::extent,
                "extent doesn't match between unspecialized and fully "
                "specialized span from array");
  static_assert(decltype(fromConstArray)::extent ==
                    decltype(fullSpecConstArray)::extent,
                "extent doesn't match between unspecialized and fully "
                "specialized span from const array");

  return 0;
}
