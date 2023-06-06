//===-- sycl_range.hpp - Provides support for sycl::range usage in tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides all neccessary helpers and wrappers to use with tests with
/// sycl::range
///
//===----------------------------------------------------------------------===//

#pragma once

#include "logger.hpp"

namespace esimd_test::api::functional {

template <int Dims> sycl::range<Dims> get_sycl_range(size_t count);

template <> sycl::range<1> get_sycl_range<1>(size_t count) {
  return sycl::range<1>(count);
}
template <> sycl::range<2> get_sycl_range<2>(size_t count) {
  if (count % 2 != 0) {
    return sycl::range<2>(count, 1);
  }
  return sycl::range<2>(count / 2, 2);
}
template <> sycl::range<3> get_sycl_range<3>(size_t count) {
  if (count % 2 != 0) {
    return sycl::range<3>(count, 1, 1);
  }
  return sycl::range<3>(count / 2, 1, 2);
}

namespace log {
// Specialization of StringMaker for sycl::range logging purposes
template <int Dims> struct StringMaker<sycl::range<Dims>> {
  static std::string stringify(const sycl::range<Dims> &range) {
    std::ostringstream stream;
    stream << "sycl::range<" << std::to_string(Dims);
    stream << ">{" << range[0];
    if constexpr (Dims >= 2) {
      stream << ", " << range[1];
    }
    if constexpr (Dims >= 3) {
      stream << ", " << range[2];
    }
    stream << "}";
    return stream.str();
  }
};
} // namespace log

} // namespace esimd_test::api::functional
