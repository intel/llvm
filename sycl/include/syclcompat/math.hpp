/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL compatibility extension
 *
 *  math.hpp
 *
 *  Description:
 *    math utilities for the SYCL compatibility extension.
 **************************************************************************/

// The original source was under the license below:
//==---- math.hpp ---------------------------------*- C++ -*----------------==//
//
// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>

#ifndef SYCL_EXT_ONEAPI_COMPLEX
#define SYCL_EXT_ONEAPI_COMPLEX
#endif

#include <sycl/ext/oneapi/experimental/complex/complex.hpp>

namespace syclcompat {
namespace detail {

namespace complex_namespace = sycl::ext::oneapi::experimental;

template <typename ValueT>
using complex_type = detail::complex_namespace::complex<ValueT>;

template <typename ValueT>
inline ValueT clamp(ValueT val, ValueT min_val, ValueT max_val) {
  return sycl::clamp(val, min_val, max_val);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
template <>
inline sycl::ext::oneapi::bfloat16 clamp(sycl::ext::oneapi::bfloat16 val,
                                         sycl::ext::oneapi::bfloat16 min_val,
                                         sycl::ext::oneapi::bfloat16 max_val) {
  if (val < min_val)
    return min_val;
  if (val > max_val)
    return max_val;
  return val;
}
#endif
template <typename ValueT>
inline sycl::marray<ValueT, 2> clamp(sycl::marray<ValueT, 2> val,
                                     sycl::marray<ValueT, 2> min_val,
                                     sycl::marray<ValueT, 2> max_val) {
  return {clamp(val[0], min_val[0], max_val[0]),
          clamp(val[1], min_val[1], max_val[1])};
}

template <typename VecT, class BinaryOperation, class = void>
class vectorized_binary {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    VecT v4;
    for (size_t i = 0; i < v4.size(); ++i) {
      v4[i] = binary_op(a[i], b[i]);
    }
    return v4;
  }
};

template <typename ValueT> inline bool isnan(const ValueT a) {
  return sycl::isnan(a);
}
#ifdef SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS
inline bool isnan(const sycl::ext::oneapi::bfloat16 a) {
  return sycl::ext::oneapi::experimental::isnan(a);
}
#endif

} // namespace detail

/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \returns The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return sycl::fast_length(a[0]);
  case 2:
    return sycl::fast_length(sycl::float2(a[0], a[1]));
  case 3:
    return sycl::fast_length(sycl::float3(a[0], a[1], a[2]));
  case 4:
    return sycl::fast_length(sycl::float4(a[0], a[1], a[2], a[3]));
  case 0:
    return 0;
  default:
    float f = 0;
    for (int i = 0; i < len; ++i)
      f += a[i] * a[i];
    return sycl::sqrt(f);
  }
}

/// Calculate the square root of the input array.
/// \param [in] a The array pointer
/// \param [in] len Length of the array
/// \returns The square root
template <typename ValueT>
inline ValueT length(const ValueT *a, const int len) {
  switch (len) {
  case 1:
    return a[0];
  case 2:
    return sycl::length(sycl::vec<ValueT, 2>(a[0], a[1]));
  case 3:
    return sycl::length(sycl::vec<ValueT, 3>(a[0], a[1], a[2]));
  case 4:
    return sycl::length(sycl::vec<ValueT, 4>(a[0], a[1], a[2], a[3]));
  default:
    ValueT ret = 0;
    for (int i = 0; i < len; ++i)
      ret += a[i] * a[i];
    return sycl::sqrt(ret);
  }
}

/// Compute vectorized max for two values, with each value treated as a vector
/// type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
template <typename S, typename T> inline T vectorized_max(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  v2 = sycl::max(v2, v3);
  v0 = v2.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized min for two values, with each value treated as a vector
/// type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
template <typename S, typename T> inline T vectorized_min(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  v2 = sycl::min(v2, v3);
  v0 = v2.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized unary operation for a value, with the value treated as a
/// vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] UnaryOperation The unary operation class
/// \param [in] a The input value
/// \returns The vectorized unary operation value of the input value
template <typename VecT, class UnaryOperation>
inline unsigned vectorized_unary(unsigned a, const UnaryOperation unary_op) {
  sycl::vec<unsigned, 1> v0{a};
  auto v1 = v0.as<VecT>();
  auto v2 = unary_op(v1);
  v0 = v2.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Compute vectorized absolute difference for two values without modulo
/// overflow, with each value treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized absolute difference of the two values
template <typename VecT>
inline unsigned vectorized_sum_abs_diff(unsigned a, unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 = sycl::abs_diff(v2, v3);
  unsigned sum = 0;
  for (size_t i = 0; i < v4.size(); ++i) {
    sum += v4[i];
  }
  return sum;
}

/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type \p S
/// \param [in] S The type of the vector
/// \param [in] T The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <typename S, typename T> inline T vectorized_isgreater(T a, T b) {
  sycl::vec<T, 1> v0{a}, v1{b};
  auto v2 = v0.template as<S>();
  auto v3 = v1.template as<S>();
  auto v4 = sycl::isgreater(v2, v3);
  v0 = v4.template as<sycl::vec<T, 1>>();
  return v0;
}

/// Compute vectorized isgreater for two unsigned int values, with each value
/// treated as a vector of two unsigned short
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <>
inline unsigned vectorized_isgreater<sycl::ushort2, unsigned>(unsigned a,
                                                              unsigned b) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.template as<sycl::ushort2>();
  auto v3 = v1.template as<sycl::ushort2>();
  sycl::ushort2 v4;
  v4[0] = v2[0] > v3[0];
  v4[1] = v2[1] > v3[1];
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

/// Returns min(max(val, min_val), max_val)
/// \param [in] val The input value
/// \param [in] min_val The minimum value
/// \param [in] max_val The maximum value
/// \returns the value between min_val and max_val
template <typename ValueT>
inline ValueT clamp(ValueT val, ValueT min_val, ValueT max_val) {
  return detail::clamp(val, min_val, max_val);
}

/// Determine whether 2 element value is NaN.
/// \param [in] a The input value
/// \returns the comparison result
template <typename ValueT>
inline std::enable_if_t<ValueT::size() == 2, ValueT> isnan(const ValueT a) {
  return {detail::isnan(a[0]), detail::isnan(a[1])};
}

/// cbrt function wrapper.
template <typename ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        ValueT>
cbrt(ValueT val) {
  return sycl::cbrt(static_cast<ValueT>(val));
}

// min/max function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
// sycl::half supported as well.
template <typename T1, typename T2>
std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                 std::common_type_t<T1, T2>>
min(T1 a, T2 b) {
  return sycl::min<std::common_type_t<T1, T2>>(a, b);
}
template <typename T1, typename T2>
std::enable_if_t<std::is_floating_point_v<T1> && std::is_floating_point_v<T2>,
                 std::common_type_t<T1, T2>>
min(T1 a, T2 b) {
  return sycl::fmin<std::common_type_t<T1, T2>>(a, b);
}
sycl::half min(sycl::half a, sycl::half b) {
  return sycl::fmin<sycl::half>(a, b);
}

template <typename T1, typename T2>
std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                 std::common_type_t<T1, T2>>
max(T1 a, T2 b) {
  return sycl::max<std::common_type_t<T1, T2>>(a, b);
}
template <typename T1, typename T2>
std::enable_if_t<std::is_floating_point_v<T1> && std::is_floating_point_v<T2>,
                 std::common_type_t<T1, T2>>
max(T1 a, T2 b) {
  return sycl::fmax<std::common_type_t<T1, T2>>(a, b);
}
sycl::half max(sycl::half a, sycl::half b) {
  return sycl::fmax<sycl::half>(a, b);
}

/// Performs 2 elements comparison and returns the bigger one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the bigger value
template <typename ValueT, typename ValueU>
inline std::common_type_t<ValueT, ValueU> fmax_nan(const ValueT a,
                                                   const ValueU b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmax<std::common_type_t<ValueT, ValueU>>(a, b);
}
template <typename ValueT, typename ValueU>
inline sycl::vec<std::common_type_t<ValueT, ValueU>, 2>
fmax_nan(const sycl::vec<ValueT, 2> a, const sycl::vec<ValueU, 2> b) {
  return {fmax_nan(a[0], b[0]), fmax_nan(a[1], b[1])};
}

/// Performs 2 elements comparison and returns the smaller one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the smaller value
template <typename ValueT, typename ValueU>
inline std::common_type_t<ValueT, ValueU> fmin_nan(const ValueT a,
                                                   const ValueU b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmin<std::common_type_t<ValueT, ValueU>>(a, b);
}
template <typename ValueT, typename ValueU>
inline sycl::vec<std::common_type_t<ValueT, ValueU>, 2>
fmin_nan(const sycl::vec<ValueT, 2> a, const sycl::vec<ValueU, 2> b) {
  return {fmin_nan(a[0], b[0]), fmin_nan(a[1], b[1])};
}

// pow functions overload.
inline float pow(const float a, const int b) { return sycl::pown(a, b); }
inline double pow(const double a, const int b) { return sycl::pown(a, b); }

template <typename ValueT, typename ValueU>
inline typename std::enable_if_t<std::is_floating_point_v<ValueT>, ValueT>
pow(const ValueT a, const ValueU b) {
  return sycl::pow(a, static_cast<ValueT>(b));
}

// FIXME: non-floating point values default to double, requires fp64. Affects
// testing and documentation as well.
template <typename ValueT, typename ValueU>
inline typename std::enable_if_t<!std::is_floating_point_v<ValueT>, double>
pow(const ValueT a, const ValueU b) {
  return sycl::pow(static_cast<double>(a), static_cast<double>(b));
}

/// Performs relu saturation.
/// \param [in] a The input value
/// \returns the relu saturation result
template <typename ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        ValueT>
relu(const ValueT a) {
  if (!detail::isnan(a) && a < 0.f)
    return 0.f;
  return a;
}
template <class ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        sycl::vec<ValueT, 2>>
relu(const sycl::vec<ValueT, 2> a) {
  return {relu(a[0]), relu(a[1])};
}
template <class ValueT>
inline std::enable_if_t<std::is_floating_point_v<ValueT> ||
                            std::is_same_v<sycl::half, ValueT>,
                        sycl::marray<ValueT, 2>>
relu(const sycl::marray<ValueT, 2> a) {
  return {relu(a[0]), relu(a[1])};
}

/// Computes the multiplication of two complex numbers.
/// \tparam T Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> cmul(sycl::vec<T, 2> x, sycl::vec<T, 2> y) {
  sycl::ext::oneapi::experimental::complex<T> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 * t2;
  return sycl::vec<T, 2>(t1.real(), t1.imag());
}

/// Computes the division of two complex numbers.
/// \tparam T Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename T>
sycl::vec<T, 2> cdiv(sycl::vec<T, 2> x, sycl::vec<T, 2> y) {
  sycl::ext::oneapi::experimental::complex<T> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 / t2;
  return sycl::vec<T, 2>(t1.real(), t1.imag());
}

/// Computes the magnitude of a complex number.
/// \tparam T Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename T> T cabs(sycl::vec<T, 2> x) {
  sycl::ext::oneapi::experimental::complex<T> t(x[0], x[1]);
  return abs(t);
}

/// Computes the complex conjugate of a complex number.
/// \tparam T Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename T> sycl::vec<T, 2> conj(sycl::vec<T, 2> x) {
  sycl::ext::oneapi::experimental::complex<T> t(x[0], x[1]);
  t = conj(t);
  return sycl::vec<T, 2>(t.real(), t.imag());
}

/// Performs complex number multiply addition.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns the operation result
template <typename ValueT>
inline sycl::vec<ValueT, 2> cmul_add(const sycl::vec<ValueT, 2> a,
                                     const sycl::vec<ValueT, 2> b,
                                     const sycl::vec<ValueT, 2> c) {
  sycl::ext::oneapi::experimental::complex<ValueT> t(a[0], a[1]);
  sycl::ext::oneapi::experimental::complex<ValueT> u(b[0], b[1]);
  sycl::ext::oneapi::experimental::complex<ValueT> v(c[0], c[1]);
  t = t * u + v;
  return sycl::vec<ValueT, 2>{t.real(), t.imag()};
}
template <typename ValueT>
inline sycl::marray<ValueT, 2> cmul_add(const sycl::marray<ValueT, 2> a,
                                        const sycl::marray<ValueT, 2> b,
                                        const sycl::marray<ValueT, 2> c) {
  sycl::ext::oneapi::experimental::complex<ValueT> t(a[0], a[1]);
  sycl::ext::oneapi::experimental::complex<ValueT> u(b[0], b[1]);
  sycl::ext::oneapi::experimental::complex<ValueT> v(c[0], c[1]);
  t = t * u + v;
  return sycl::marray<ValueT, 2>{t.real(), t.imag()};
}

/// A sycl::abs wrapper functors.
struct abs {
  template <typename ValueT> auto operator()(const ValueT x) const {
    return sycl::abs(x);
  }
};

/// A sycl::abs_diff wrapper functors.
struct abs_diff {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::abs_diff(x, y);
  }
};

/// A sycl::add_sat wrapper functors.
struct add_sat {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::add_sat(x, y);
  }
};

/// A sycl::rhadd wrapper functors.
struct rhadd {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::rhadd(x, y);
  }
};

/// A sycl::hadd wrapper functors.
struct hadd {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::hadd(x, y);
  }
};

/// A sycl::max wrapper functors.
struct maximum {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::max(x, y);
  }
};

/// A sycl::min wrapper functors.
struct minimum {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::min(x, y);
  }
};

/// A sycl::sub_sat wrapper functors.
struct sub_sat {
  template <typename ValueT>
  auto operator()(const ValueT x, const ValueT y) const {
    return sycl::sub_sat(x, y);
  }
};

/// Compute vectorized binary operation value for two values, with each value
/// treated as a vector type \p VecT.
/// \tparam [in] VecT The type of the vector
/// \tparam [in] BinaryOperation The binary operation class
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized binary operation value of the two values
template <typename VecT, class BinaryOperation>
inline unsigned vectorized_binary(unsigned a, unsigned b,
                                  const BinaryOperation binary_op) {
  sycl::vec<unsigned, 1> v0{a}, v1{b};
  auto v2 = v0.as<VecT>();
  auto v3 = v1.as<VecT>();
  auto v4 =
      detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
  v0 = v4.template as<sycl::vec<unsigned, 1>>();
  return v0;
}

} // namespace syclcompat
