//==----------- boolean.hpp - SYCL boolean type ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/generic_type_traits.hpp>
#include <CL/sycl/types.hpp>

#include <initializer_list>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
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
    static_assert(is_vgeninteger<T>::value, "Invalid constructor");
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
    static_assert(is_vgeninteger<T>::value, "Invalid conversion");
    T r;
    Assigner<N - 1>::assign(r, *this);
    return r;
  }

private:
  template <int Num> friend struct Assigner;
  alignas(detail::vector_alignment<element_type, N>::value) DataType value;
};

template <> struct Boolean<1> {
  Boolean() = default;

  // Build from a signed interger type
  template <typename T> Boolean(T val) : value(val) {
    static_assert(is_sgeninteger<T>::value, "Invalid constructor");
  }

  // Cast to a signed interger type
  template <typename T> operator T() const {
    static_assert(is_sgeninteger<T>::value, "Invalid conversion");
    return value;
  }

#ifdef __SYCL_DEVICE_ONLY__
  // Build from a boolean type
  Boolean(bool f) : value(f) {}
  // Cast to a boolean type
  operator bool() const { return value; }
#endif

private:
  alignas(1) bool value = false;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
