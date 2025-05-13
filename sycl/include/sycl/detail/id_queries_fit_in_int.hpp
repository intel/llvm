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
constexpr static const char *Msg =
    "Provided range and/or offset does not fit in int. Pass "
    "`-fno-sycl-id-queries-fit-in-int' to remove this limit.";

template <typename ValT>
typename std::enable_if_t<std::is_same<ValT, size_t>::value ||
                          std::is_same<ValT, unsigned long long>::value>
checkValueRangeImpl(ValT V) {
  static constexpr size_t Limit =
      static_cast<size_t>((std::numeric_limits<int>::max)());
  if (V > Limit)
    throw sycl::exception(make_error_code(errc::nd_range), Msg);
}

inline void checkMulOverflow(size_t a, size_t b) {
#ifndef _MSC_VER
  int Product;
  // Since we must fit in SIGNED int, we can ignore the upper 32 bits.
  if (__builtin_mul_overflow(unsigned(a), unsigned(b), &Product)) {
    throw sycl::exception(make_error_code(errc::nd_range), Msg);
  }
#else
  checkValueRangeImpl(a);
  checkValueRangeImpl(b);
  size_t Product = a * b;
  checkValueRangeImpl(Product);
#endif
}

inline void checkMulOverflow(size_t a, size_t b, size_t c) {
#ifndef _MSC_VER
  int Product;
  // Since we must fit in SIGNED int, we can ignore the upper 32 bits.
  if (__builtin_mul_overflow(unsigned(a), unsigned(b), &Product) ||
      __builtin_mul_overflow(Product, unsigned(c), &Product)) {
    throw sycl::exception(make_error_code(errc::nd_range), Msg);
  }
#else
  checkValueRangeImpl(a);
  checkValueRangeImpl(b);
  size_t Product = a * b;
  checkValueRangeImpl(Product);

  checkValueRangeImpl(c);
  Product *= c;
  checkValueRangeImpl(Product);
#endif
}

// TODO: Remove this function when offsets are removed.
template <int Dims>
inline bool hasNonZeroOffset(const sycl::nd_range<Dims> &V) {
  size_t Product = 1;
  for (int Dim = 0; Dim < Dims; ++Dim) {
    Product *= V.get_offset()[Dim];
  }
  return (Product != 0);
}
#endif //__SYCL_ID_QUERIES_FIT_IN_INT__

template <int Dims>
void checkValueRange([[maybe_unused]] const sycl::range<Dims> &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  if constexpr (Dims == 1) {
    // For 1D range, just check the value against MAX_INT.
    checkValueRangeImpl(V[0]);
  } else if constexpr (Dims == 2) {
    // For 2D range, check if computing the linear range overflows.
    checkMulOverflow(V[0], V[1]);
  } else if constexpr (Dims == 3) {
    // For 3D range, check if computing the linear range overflows.
    checkMulOverflow(V[0], V[1], V[2]);
  }
#endif
}

template <int Dims>
void checkValueRange([[maybe_unused]] const sycl::id<Dims> &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  // An id cannot be linearized without a range, so check each component.
  for (int Dim = 0; Dim < Dims; ++Dim) {
    checkValueRangeImpl(V[Dim]);
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
    checkValueRangeImpl(Sum);
  }
#endif
}

template <int Dims>
void checkValueRange([[maybe_unused]] const sycl::nd_range<Dims> &V) {
#if __SYCL_ID_QUERIES_FIT_IN_INT__
  // In an ND-range, we only need to check the global linear size, because:
  // - The linear size must be greater than any of the dimensions.
  // - Each dimension of the global range is larger than the local range.
  // TODO: Remove this branch when offsets are removed.
  if (hasNonZeroOffset(V)) /*[[unlikely]]*/ {
    checkValueRange<Dims>(V.get_global_range(), V.get_offset());
  } else {
    checkValueRange<Dims>(V.get_global_range());
  }
#endif
}

} // namespace detail
} // namespace _V1
} // namespace sycl

#endif
