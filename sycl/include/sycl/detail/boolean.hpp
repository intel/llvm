//==----------- boolean.hpp - SYCL boolean type ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/generic_type_traits.hpp> // for is_sgeninteger, msbIsSet
#include <sycl/detail/vector_traits.hpp>       // for vector_alignment

#include <initializer_list> // for initializer_list
#include <stddef.h>         // for size_t
#include <stdint.h>         // for int8_t
#include <type_traits>      // for is_same

namespace sycl {
inline namespace _V1 {
namespace detail {

template <int Num> struct Assigner {
  template <typename R, typename T> static void assign(R &r, const T x) {
    Assigner<Num - 1>::assign(r, x);
    r.template swizzle<Num>() = x.value[Num];
  }

  template <typename R, typename T, typename ET>
  static void init(R &r, const T x) {
    Assigner<Num - 1>::template init<R, T, ET>(r, x);
    ET v = x.template swizzle<Num>();
    r.value[Num] = msbIsSet(v) * (-1);
  }
};

template <> struct Assigner<0> {
  template <typename R, typename T> static void assign(R &r, const T x) {
    r.template swizzle<0>() = x.value[0];
  }
  template <typename R, typename T, typename ET>
  static void init(R &r, const T x) {
    ET v = x.template swizzle<0>();
    r.value[0] = msbIsSet(v) * (-1);
  }
};

template <int N> struct Boolean {
  static_assert(((N == 2) || (N == 3) || (N == 4) || (N == 8) || (N == 16)),
                "Invalid size");

  using element_type = int8_t;

#ifdef __SYCL_DEVICE_ONLY__
  using DataType = element_type __attribute__((ext_vector_type(N)));
  using vector_t = DataType;
#else
  using DataType = element_type[N];
#endif

  Boolean() : value{0} {}

  Boolean(std::initializer_list<element_type> l) {
    for (size_t I = 0; I < N; ++I) {
      value[I] = *(l.begin() + I) ? -1 : 0;
    }
  }

  Boolean(const Boolean &rhs) {
    for (size_t I = 0; I < N; ++I) {
      value[I] = rhs.value[I];
    }
  }

  template <typename T> Boolean(const T rhs) {
    static_assert(is_vgeninteger_v<T>, "Invalid constructor");
    Assigner<N - 1>::template init<Boolean<N>, T, typename T::element_type>(
        *this, rhs);
  }

#ifdef __SYCL_DEVICE_ONLY__
  // TODO change this to the vectors assignment when the assignment will be
  // fixed on Intel GPU NEO OpenCL runtime
  Boolean(const vector_t rhs) {
    for (size_t I = 0; I < N; ++I) {
      value[I] = rhs[I];
    }
  }

  operator vector_t() const { return value; }
#endif

  template <typename T> operator T() const {
    static_assert(is_vgeninteger_v<T>, "Invalid conversion");
    T r;
    Assigner<N - 1>::assign(r, *this);
    return r;
  }

private:
  template <int Num> friend struct Assigner;
  alignas(detail::vector_alignment<element_type, N>::value) DataType value;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
