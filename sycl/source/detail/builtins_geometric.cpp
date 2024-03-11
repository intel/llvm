//==--------- builtins_geometric.cpp - SYCL built-in geometric functions ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines the host versions of functions defined
// in SYCL SPEC section - 4.13.6 Geometric functions.

#include "builtins_helper.hpp"
#include <sycl/detail/stl_type_traits.hpp>

#include <cmath>

namespace s = sycl;
namespace d = s::detail;
namespace cl = s::opencl;

namespace __host_std {

__SYCL_EXPORT cl::cl_float sycl_host_Dot(s::vec<float, 1>, s::vec<float, 1>);
__SYCL_EXPORT cl::cl_float sycl_host_Dot(s::vec<cl::cl_float, 2>,
                                         s::vec<cl::cl_float, 2>);
__SYCL_EXPORT cl::cl_float sycl_host_Dot(s::vec<cl::cl_float, 3>,
                                         s::vec<cl::cl_float, 3>);
__SYCL_EXPORT cl::cl_float sycl_host_Dot(s::vec<cl::cl_float, 4>,
                                         s::vec<cl::cl_float, 4>);
__SYCL_EXPORT cl::cl_double sycl_host_Dot(s::vec<double, 1>, s::vec<double, 1>);
__SYCL_EXPORT cl::cl_double sycl_host_Dot(s::vec<cl::cl_double, 2>,
                                          s::vec<cl::cl_double, 2>);
__SYCL_EXPORT cl::cl_double sycl_host_Dot(s::vec<cl::cl_double, 3>,
                                          s::vec<cl::cl_double, 3>);
__SYCL_EXPORT cl::cl_double sycl_host_Dot(s::vec<cl::cl_double, 4>,
                                          s::vec<cl::cl_double, 4>);
__SYCL_EXPORT cl::cl_half sycl_host_Dot(s::vec<s::half, 1>, s::vec<s::half, 1>);
__SYCL_EXPORT cl::cl_half sycl_host_Dot(s::vec<cl::cl_half, 2>,
                                        s::vec<cl::cl_half, 2>);
__SYCL_EXPORT cl::cl_half sycl_host_Dot(s::vec<cl::cl_half, 3>,
                                        s::vec<cl::cl_half, 3>);
__SYCL_EXPORT cl::cl_half sycl_host_Dot(s::vec<cl::cl_half, 4>,
                                        s::vec<cl::cl_half, 4>);

__SYCL_EXPORT cl::cl_int sycl_host_All(s::vec<int, 1>);
__SYCL_EXPORT cl::cl_int sycl_host_All(s::vec<cl::cl_int, 2>);
__SYCL_EXPORT cl::cl_int sycl_host_All(s::vec<cl::cl_int, 3>);
__SYCL_EXPORT cl::cl_int sycl_host_All(s::vec<cl::cl_int, 4>);

namespace {

template <typename T> inline T __cross(T p0, T p1) {
  T result(0);
  result.x() = p0.y() * p1.z() - p0.z() * p1.y();
  result.y() = p0.z() * p1.x() - p0.x() * p1.z();
  result.z() = p0.x() * p1.y() - p0.y() * p1.x();
  return result;
}

template <typename T> inline void __FMul_impl(T &r, T p0, T p1) {
  r += p0 * p1;
}

template <typename T> inline T __FMul(T p0, T p1) {
  T result = 0;
  __FMul_impl(result, p0, p1);
  return result;
}

template <typename T>
inline typename std::enable_if_t<d::is_sgengeo_v<T>, T> __length(T t) {
  return std::sqrt(__FMul(t, t));
}

template <typename T>
inline typename std::enable_if_t<d::is_vgengeo_v<T>, typename T::element_type>
__length(T t) {
  return std::sqrt(sycl_host_Dot(t, t));
}

template <typename T>
inline typename std::enable_if_t<d::is_sgengeo_v<T>, T> __normalize(T t) {
  T r = __length(t);
  return t / T(r);
}

template <typename T>
inline typename std::enable_if_t<d::is_vgengeo_v<T>, T> __normalize(T t) {
  typename T::element_type r = __length(t);
  return t / T(r);
}

template <typename T>
inline typename std::enable_if_t<d::is_sgengeo_v<T>, T> __fast_length(T t) {
  return std::sqrt(__FMul(t, t));
}

template <typename T>
inline typename std::enable_if_t<d::is_vgengeo_v<T>, typename T::element_type>
__fast_length(T t) {
  return std::sqrt(sycl_host_Dot(t, t));
}

template <typename T>
inline typename std::enable_if_t<d::is_vgengeo_v<T>, T> __fast_normalize(T t) {
  if (sycl_host_All(t == T(0.0f)))
    return t;
  typename T::element_type r = std::sqrt(sycl_host_Dot(t, t));
  return t / T(r);
}

} // namespace

// --------------- 4.13.6 Geometric functions. Host implementations ------------
// cross
__SYCL_EXPORT s::vec<cl::cl_float, 3>
sycl_host_cross(s::vec<cl::cl_float, 3> p0,
                s::vec<cl::cl_float, 3> p1) __NOEXC {
  return __cross(p0, p1);
}
__SYCL_EXPORT s::vec<cl::cl_float, 4>
sycl_host_cross(s::vec<cl::cl_float, 4> p0,
                s::vec<cl::cl_float, 4> p1) __NOEXC {
  return __cross(p0, p1);
}
__SYCL_EXPORT s::vec<cl::cl_double, 3>
sycl_host_cross(s::vec<cl::cl_double, 3> p0,
                s::vec<cl::cl_double, 3> p1) __NOEXC {
  return __cross(p0, p1);
}
__SYCL_EXPORT s::vec<cl::cl_double, 4>
sycl_host_cross(s::vec<cl::cl_double, 4> p0,
                s::vec<cl::cl_double, 4> p1) __NOEXC {
  return __cross(p0, p1);
}
__SYCL_EXPORT s::vec<cl::cl_half, 3>
sycl_host_cross(s::vec<cl::cl_half, 3> p0, s::vec<cl::cl_half, 3> p1) __NOEXC {
  return __cross(p0, p1);
}
__SYCL_EXPORT s::vec<cl::cl_half, 4>
sycl_host_cross(s::vec<cl::cl_half, 4> p0, s::vec<cl::cl_half, 4> p1) __NOEXC {
  return __cross(p0, p1);
}

// FMul
__SYCL_EXPORT cl::cl_float sycl_host_FMul(cl::cl_float p0, cl::cl_float p1) {
  return __FMul(p0, p1);
}
__SYCL_EXPORT cl::cl_double sycl_host_FMul(cl::cl_double p0, cl::cl_double p1) {
  return __FMul(p0, p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_FMul(cl::cl_half p0, cl::cl_half p1) {
  return __FMul(p0, p1);
}

// Dot
MAKE_GEO_1V_2V_RS(sycl_host_Dot, __FMul_impl, s::cl_float, s::cl_float,
                  s::cl_float)
MAKE_GEO_1V_2V_RS(sycl_host_Dot, __FMul_impl, s::cl_double, s::cl_double,
                  s::cl_double)
MAKE_GEO_1V_2V_RS(sycl_host_Dot, __FMul_impl, s::cl_half, s::cl_half,
                  s::cl_half)

// length
__SYCL_EXPORT cl::cl_float sycl_host_length(cl::cl_float p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_length(cl::cl_double p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_half sycl_host_length(cl::cl_half p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_length(s::vec<float, 1> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_length(s::vec<cl::cl_float, 2> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_length(s::vec<cl::cl_float, 3> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_length(s::vec<cl::cl_float, 4> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_length(s::vec<double, 1> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_length(s::vec<cl::cl_double, 2> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_length(s::vec<cl::cl_double, 3> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_length(s::vec<cl::cl_double, 4> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_half sycl_host_length(s::vec<s::half, 1> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_half sycl_host_length(s::vec<cl::cl_half, 2> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_half sycl_host_length(s::vec<cl::cl_half, 3> p) {
  return __length(p);
}
__SYCL_EXPORT cl::cl_half sycl_host_length(s::vec<cl::cl_half, 4> p) {
  return __length(p);
}

// distance
__SYCL_EXPORT cl::cl_float sycl_host_distance(cl::cl_float p0,
                                              cl::cl_float p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_distance(s::vec<float, 1> p0,
                                              s::vec<float, 1> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_distance(s::vec<cl::cl_float, 2> p0,
                                              s::vec<cl::cl_float, 2> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_distance(s::vec<cl::cl_float, 3> p0,
                                              s::vec<cl::cl_float, 3> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_distance(s::vec<cl::cl_float, 4> p0,
                                              s::vec<cl::cl_float, 4> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_double sycl_host_distance(cl::cl_double p0,
                                               cl::cl_double p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_distance(s::vec<double, 1> p0,
                                              s::vec<double, 1> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_double sycl_host_distance(s::vec<cl::cl_double, 2> p0,
                                               s::vec<cl::cl_double, 2> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_double sycl_host_distance(s::vec<cl::cl_double, 3> p0,
                                               s::vec<cl::cl_double, 3> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_double sycl_host_distance(s::vec<cl::cl_double, 4> p0,
                                               s::vec<cl::cl_double, 4> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_half sycl_host_distance(cl::cl_half p0, cl::cl_half p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_distance(s::vec<s::half, 1> p0,
                                              s::vec<s::half, 1> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_half sycl_host_distance(s::vec<cl::cl_half, 2> p0,
                                             s::vec<cl::cl_half, 2> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_half sycl_host_distance(s::vec<cl::cl_half, 3> p0,
                                             s::vec<cl::cl_half, 3> p1) {
  return sycl_host_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_half sycl_host_distance(s::vec<cl::cl_half, 4> p0,
                                             s::vec<cl::cl_half, 4> p1) {
  return sycl_host_length(p0 - p1);
}

// normalize
__SYCL_EXPORT cl::cl_float sycl_host_normalize(cl::cl_float p) {
  return __normalize(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_normalize(s::vec<float, 1> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_float, 2>
sycl_host_normalize(s::vec<cl::cl_float, 2> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_float, 3>
sycl_host_normalize(s::vec<cl::cl_float, 3> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_float, 4>
sycl_host_normalize(s::vec<cl::cl_float, 4> p) {
  return __normalize(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_normalize(cl::cl_double p) {
  return __normalize(p);
}
__SYCL_EXPORT cl::cl_double sycl_host_normalize(s::vec<double, 1> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_double, 2>
sycl_host_normalize(s::vec<cl::cl_double, 2> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_double, 3>
sycl_host_normalize(s::vec<cl::cl_double, 3> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_double, 4>
sycl_host_normalize(s::vec<cl::cl_double, 4> p) {
  return __normalize(p);
}
__SYCL_EXPORT cl::cl_half sycl_host_normalize(cl::cl_half p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_half, 2>
sycl_host_normalize(s::vec<cl::cl_half, 2> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_half, 3>
sycl_host_normalize(s::vec<cl::cl_half, 3> p) {
  return __normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_half, 4>
sycl_host_normalize(s::vec<cl::cl_half, 4> p) {
  return __normalize(p);
}

// fast_length
__SYCL_EXPORT cl::cl_float sycl_host_fast_length(cl::cl_float p) {
  return __fast_length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_length(s::vec<float, 1> p) {
  return __fast_length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_length(s::vec<cl::cl_float, 2> p) {
  return __fast_length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_length(s::vec<cl::cl_float, 3> p) {
  return __fast_length(p);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_length(s::vec<cl::cl_float, 4> p) {
  return __fast_length(p);
}

// fast_normalize
__SYCL_EXPORT cl::cl_float sycl_host_fast_normalize(cl::cl_float p) {
  if (p == 0.0f)
    return p;
  cl::cl_float r = std::sqrt(sycl_host_FMul(p, p));
  return p / r;
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_normalize(s::vec<float, 1> p) {
  return __fast_normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_float, 2>
sycl_host_fast_normalize(s::vec<cl::cl_float, 2> p) {
  return __fast_normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_float, 3>
sycl_host_fast_normalize(s::vec<cl::cl_float, 3> p) {
  return __fast_normalize(p);
}
__SYCL_EXPORT s::vec<cl::cl_float, 4>
sycl_host_fast_normalize(s::vec<cl::cl_float, 4> p) {
  return __fast_normalize(p);
}

// fast_distance
__SYCL_EXPORT cl::cl_float sycl_host_fast_distance(cl::cl_float p0,
                                                   cl::cl_float p1) {
  return sycl_host_fast_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_distance(s::vec<float, 1> p0,
                                                   s::vec<float, 1> p1) {
  return sycl_host_fast_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_distance(s::vec<cl::cl_float, 2> p0,
                                                   s::vec<cl::cl_float, 2> p1) {
  return sycl_host_fast_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_distance(s::vec<cl::cl_float, 3> p0,
                                                   s::vec<cl::cl_float, 3> p1) {
  return sycl_host_fast_length(p0 - p1);
}
__SYCL_EXPORT cl::cl_float sycl_host_fast_distance(s::vec<cl::cl_float, 4> p0,
                                                   s::vec<cl::cl_float, 4> p1) {
  return sycl_host_fast_length(p0 - p1);
}

} // namespace __host_std
