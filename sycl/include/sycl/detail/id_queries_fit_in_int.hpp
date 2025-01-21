//==-------------------- id_queries_fit_in_int.hpp -------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Our SYCL implementation has a special mode (introduced for performance
// reasons) in which it assume that all result of all id queries (i.e. global
// sizes, work-group sizes, local id, global id, etc.) fit within MAX_INT.
//
// This header contains corresponding helper functions related to this mode.
//
//===----------------------------------------------------------------------===//

#pragma once

// We only use those helpers to throw an exception if user selected a range that
// would violate the assumption. That can only happen on host and therefore to
// optimize our headers, the helpers below are only available for host
// compilation.
#ifndef __SYCL_DEVICE_ONLY__

#include <sycl/exception.hpp>
#include <sycl/nd_range.hpp>
#include <sycl/range.hpp>

#include <limits>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

#if __SYCL_ID_QUERIES_FIT_IN_INT__
template <typename T> struct NotIntMsg;

template <int Dims> struct NotIntMsg<range<Dims>> {
  constexpr static const char *Msg =
      "Provided range is out of integer limits. Pass "
      "`-fno-sycl-id-queries-fit-in-int' to disable range check.";
};

template <int Dims> struct NotIntMsg<id<Dims>> {
  constexpr static const char *Msg =
      "Provided offset is out of integer limits. Pass "
      "`-fno-sycl-id-queries-fit-in-int' to disable offset check.";
};

template <typename T, typename ValT>
typename std::enable_if_t<std::is_same<ValT, size_t>::value ||
                          std::is_same<ValT, unsigned long long>::value>
checkValueRangeImpl(ValT V) {
  static constexpr size_t Limit =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  if (V > Limit)
    throw sycl::exception(make_error_code(errc::nd_range), NotIntMsg<T>::Msg);
}
#endif

template <int Dims, typename T>
typename std::enable_if_t<std::is_same_v<T, range<Dims>> ||
                          std::is_same_v<T, id<Dims>>>
checkValueRange([[maybe_unused]] const T &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  for (size_t Dim = 0; Dim < Dims; ++Dim)
    checkValueRangeImpl<T>(V[Dim]);

  {
    unsigned long long Product = 1;
    for (size_t Dim = 0; Dim < Dims; ++Dim) {
      Product *= V[Dim];
      // check value now to prevent product overflow in the end
      checkValueRangeImpl<T>(Product);
    }
  }
#endif
}

template <int Dims>
void checkValueRange([[maybe_unused]] const range<Dims> &R,
                     [[maybe_unused]] const id<Dims> &O) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  checkValueRange<Dims>(R);
  checkValueRange<Dims>(O);

  for (size_t Dim = 0; Dim < Dims; ++Dim) {
    unsigned long long Sum = R[Dim] + O[Dim];

    checkValueRangeImpl<range<Dims>>(Sum);
  }
#endif
}

template <int Dims, typename T>
typename std::enable_if_t<std::is_same_v<T, nd_range<Dims>>>
checkValueRange([[maybe_unused]] const T &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  checkValueRange<Dims>(V.get_global_range());
  checkValueRange<Dims>(V.get_local_range());
  checkValueRange<Dims>(V.get_offset());

  checkValueRange<Dims>(V.get_global_range(), V.get_offset());
#endif
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif
