
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

#include <limits>
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
template <typename VecT, class BinaryOperation>
class vectorized_binary<
    VecT, BinaryOperation,
    std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>> {
public:
  inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op) {
    return binary_op(a, b).template as<VecT>();
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

/// Extend the 'val' to 'bit' size, zero extend for unsigned int and signed
/// extend for signed int.
template <typename ValueT>
inline int64_t zero_or_signed_extent(ValueT val, unsigned bit) {
  if constexpr (std::is_signed_v<ValueT>) {
    return int64_t(val) << (64 - bit) >> (64 - bit);
  }
  return val;
}

template <typename RetT, bool needSat, typename AT, typename BT,
          typename BinaryOperation>
inline constexpr RetT extend_binary(AT a, BT b, BinaryOperation binary_op) {
  const int64_t extend_a = zero_or_signed_extent(a, 33);
  const int64_t extend_b = zero_or_signed_extent(b, 33);
  const int64_t ret = binary_op(extend_a, extend_b);
  if constexpr (needSat)
    return detail::clamp<int64_t>(ret, std::numeric_limits<RetT>::min(),
                                  std::numeric_limits<RetT>::max());
  return ret;
}

template <typename RetT, bool needSat, typename AT, typename BT, typename CT,
          typename BinaryOperation1, typename BinaryOperation2>
inline constexpr RetT extend_binary(AT a, BT b, CT c,
                                    BinaryOperation1 binary_op,
                                    BinaryOperation2 second_op) {
  const int64_t extend_a = zero_or_signed_extent(a, 33);
  const int64_t extend_b = zero_or_signed_extent(b, 33);
  int64_t extend_temp =
      zero_or_signed_extent(binary_op(extend_a, extend_b), 34);
  if constexpr (needSat)
    extend_temp =
        detail::clamp<int64_t>(extend_temp, std::numeric_limits<RetT>::min(),
                               std::numeric_limits<RetT>::max());
  const int64_t extend_c = zero_or_signed_extent(c, 33);
  return second_op(extend_temp, extend_c);
}
} // namespace detail

/// Compute fast_length for variable-length array
/// \param [in] a The array
/// \param [in] len Length of the array
/// \returns The computed fast_length
inline float fast_length(const float *a, int len) {
  switch (len) {
  case 1:
    return a[0];
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

/// Returns min(max(val, min_val), max_val)
/// \param [in] val The input value
/// \param [in] min_val The minimum value
/// \param [in] max_val The maximum value
/// \returns the value between min_val and max_val
template <typename ValueT>
inline ValueT clamp(ValueT val, ValueT min_val, ValueT max_val) {
  return detail::clamp(val, min_val, max_val);
}

/// Performs comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, ValueT, ValueT>, bool>,
    bool>
compare(const ValueT a, const ValueT b, const BinaryOperation binary_op) {
  return binary_op(a, b);
}
template <typename ValueT>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<std::not_equal_to<>, ValueT, ValueT>,
                   bool>,
    bool>
compare(const ValueT a, const ValueT b, const std::not_equal_to<> binary_op) {
  return !detail::isnan(a) && !detail::isnan(b) && binary_op(a, b);
}
/// Performs 2 element comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, ValueT>
compare(const ValueT a, const ValueT b, const BinaryOperation binary_op) {
  return {compare(a[0], b[0], binary_op), compare(a[1], b[1], binary_op)};
}

/// Performs unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<
    std::is_same_v<std::invoke_result_t<BinaryOperation, ValueT, ValueT>, bool>,
    bool>
unordered_compare(const ValueT a, const ValueT b,
                  const BinaryOperation binary_op) {
  return detail::isnan(a) || detail::isnan(b) || binary_op(a, b);
}

/// Performs 2 element unordered comparison.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, ValueT>
unordered_compare(const ValueT a, const ValueT b,
                  const BinaryOperation binary_op) {
  return {unordered_compare(a[0], b[0], binary_op),
          unordered_compare(a[1], b[1], binary_op)};
}

/// Performs 2 element comparison and return true if both results are true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, bool>
compare_both(const ValueT a, const ValueT b, const BinaryOperation binary_op) {
  return compare(a[0], b[0], binary_op) && compare(a[1], b[1], binary_op);
}

/// Performs 2 element unordered comparison and return true if both results are
/// true.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline std::enable_if_t<ValueT::size() == 2, bool>
unordered_compare_both(const ValueT a, const ValueT b,
                       const BinaryOperation binary_op) {
  return unordered_compare(a[0], b[0], binary_op) &&
         unordered_compare(a[1], b[1], binary_op);
}

/// Performs 2 elements comparison, compare result of each element is 0 (false)
/// or 0xffff (true), returns an unsigned int by composing compare result of two
/// elements.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline unsigned compare_mask(const sycl::vec<ValueT, 2> a,
                             const sycl::vec<ValueT, 2> b,
                             const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-compare(a[0], b[0], binary_op),
                             -compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}

/// Performs 2 elements unordered comparison, compare result of each element is
/// 0 (false) or 0xffff (true), returns an unsigned int by composing compare
/// result of two elements.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] binary_op functor that implements the binary operation
/// \returns the comparison result
template <typename ValueT, class BinaryOperation>
inline unsigned unordered_compare_mask(const sycl::vec<ValueT, 2> a,
                                       const sycl::vec<ValueT, 2> b,
                                       const BinaryOperation binary_op) {
  return sycl::vec<short, 2>(-unordered_compare(a[0], b[0], binary_op),
                             -unordered_compare(a[1], b[1], binary_op))
      .as<sycl::vec<unsigned, 1>>();
}

/// Determine whether 2 element value is NaN.
/// \param [in] a The input value
/// \returns the comparison result
template <typename ValueT>
inline std::enable_if_t<ValueT::size() == 2, ValueT> isnan(const ValueT a) {
  return {detail::isnan(a[0]), detail::isnan(a[1])};
}

/// cbrt function wrapper.
template <typename ValueT> inline ValueT cbrt(ValueT val) {
  return sycl::cbrt((ValueT)val);
}

// min function overloads.
// For floating-point types, `float` or `double` arguments are acceptable.
// For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
// `std::int64_t` type arguments are acceptable.
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

// pow functions overload.
inline float pow(const float a, const int b) { return sycl::pown(a, b); }
inline double pow(const double a, const int b) { return sycl::pown(a, b); }
inline float pow(const float a, const float b) { return sycl::pow(a, b); }
inline double pow(const double a, const double b) { return sycl::pow(a, b); }
template <typename ValueT, typename U>
inline typename std::enable_if_t<std::is_floating_point_v<ValueT>, ValueT>
pow(const ValueT a, const U b) {
  return sycl::pow(a, static_cast<ValueT>(b));
}
template <typename ValueT, typename U>
inline typename std::enable_if_t<!std::is_floating_point_v<ValueT>, double>
pow(const ValueT a, const U b) {
  return sycl::pow(static_cast<double>(a), static_cast<double>(b));
}

/// Performs relu saturation.
/// \param [in] a The input value
/// \returns the relu saturation result
template <typename ValueT> inline ValueT relu(const ValueT a) {
  if (!detail::isnan(a) && a < 0.f)
    return 0.f;
  return a;
}
template <class ValueT>
inline sycl::vec<ValueT, 2> relu(const sycl::vec<ValueT, 2> a) {
  return {relu(a[0]), relu(a[1])};
}
template <class ValueT>
inline sycl::marray<ValueT, 2> relu(const sycl::marray<ValueT, 2> a) {
  return {relu(a[0]), relu(a[1])};
}

/// Computes the multiplication of two complex numbers.
/// \tparam ValueT Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename ValueT>
sycl::vec<ValueT, 2> cmul(sycl::vec<ValueT, 2> x, sycl::vec<ValueT, 2> y) {
  detail::complex_type<ValueT> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 * t2;
  return sycl::vec<ValueT, 2>(t1.real(), t1.imag());
}

/// Computes the division of two complex numbers.
/// \tparam ValueT Complex element type
/// \param [in] x The first input complex number
/// \param [in] y The second input complex number
/// \returns The result
template <typename ValueT>
sycl::vec<ValueT, 2> cdiv(sycl::vec<ValueT, 2> x, sycl::vec<ValueT, 2> y) {
  detail::complex_type<ValueT> t1(x[0], x[1]), t2(y[0], y[1]);
  t1 = t1 / t2;
  return sycl::vec<ValueT, 2>(t1.real(), t1.imag());
}

/// Computes the magnitude of a complex number.
/// \tparam ValueT Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename ValueT> ValueT cabs(sycl::vec<ValueT, 2> x) {
  detail::complex_type<ValueT> t(x[0], x[1]);
  return detail::complex_namespace::abs(t);
}

/// Computes the complex conjugate of a complex number.
/// \tparam ValueT Complex element type
/// \param [in] x The input complex number
/// \returns The result
template <typename ValueT> sycl::vec<ValueT, 2> conj(sycl::vec<ValueT, 2> x) {
  detail::complex_type<ValueT> t(x[0], x[1]);
  t = conj(t);
  return sycl::vec<ValueT, 2>(t.real(), t.imag());
}

/// Performs complex number multiply addition.
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \returns the operation result
template <typename ValueT>
inline sycl::vec<ValueT, 2> complex_mul_add(const sycl::vec<ValueT, 2> a,
                                            const sycl::vec<ValueT, 2> b,
                                            const sycl::vec<ValueT, 2> c) {
  return sycl::vec<ValueT, 2>{a[0] * b[0] - a[1] * b[1] + c[0],
                              a[0] * b[1] + a[1] * b[0] + c[1]};
}
template <typename ValueT>
inline sycl::marray<ValueT, 2>
complex_mul_add(const sycl::marray<ValueT, 2> a,
                const sycl::marray<ValueT, 2> b,
                const sycl::marray<ValueT, 2> c) {
  return sycl::marray<ValueT, 2>{a[0] * b[0] - a[1] * b[1] + c[0],
                                 a[0] * b[1] + a[1] * b[0] + c[1]};
}

/// Performs 2 elements comparison and returns the bigger one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the bigger value
template <typename ValueT>
inline ValueT fmax_nan(const ValueT a, const ValueT b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmax(a, b);
}
template <typename ValueT>
inline sycl::vec<ValueT, 2> fmax_nan(const sycl::vec<ValueT, 2> a,
                                     const sycl::vec<ValueT, 2> b) {
  return {fmax_nan(a[0], b[0]), fmax_nan(a[1], b[1])};
}

/// Performs 2 elements comparison and returns the smaller one. If either of
/// inputs is NaN, then return NaN.
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns the smaller value
template <typename ValueT>
inline ValueT fmin_nan(const ValueT a, const ValueT b) {
  if (detail::isnan(a) || detail::isnan(b))
    return NAN;
  return sycl::fmin(a, b);
}
template <typename ValueT>
inline sycl::vec<ValueT, 2> fmin_nan(const sycl::vec<ValueT, 2> a,
                                     const sycl::vec<ValueT, 2> b) {
  return {fmin_nan(a[0], b[0]), fmin_nan(a[1], b[1])};
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

/// Compute vectorized isgreater for two values, with each value treated as a
/// vector type \p ValueU
/// \param [in] ValueU The type of the vector
/// \param [in] ValueT The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized greater than of the two values
template <typename ValueU, typename ValueT>
inline ValueT vectorized_isgreater(ValueT a, ValueT b) {
  sycl::vec<ValueT, 1> v0{a}, v1{b};
  auto v2 = v0.template as<ValueU>();
  auto v3 = v1.template as<ValueU>();
  auto v4 = sycl::isgreater(v2, v3);
  v0 = v4.template as<sycl::vec<ValueT, 1>>();
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

/// Compute vectorized max for two values, with each value treated as a vector
/// type \p ValueU.
/// \tparam [in] ValueU The type of the vector
/// \tparam [in] ValueT The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized max of the two values
template <typename ValueU, typename ValueT>
inline ValueT vectorized_max(ValueT a, ValueT b) {
  sycl::vec<ValueT, 1> v0{a}, v1{b};
  auto v2 = v0.template as<ValueU>();
  auto v3 = v1.template as<ValueU>();
  auto v4 = sycl::max(v2, v3);
  v0 = v4.template as<sycl::vec<ValueT, 1>>();
  return v0;
}

/// Compute vectorized min for two values, with each value treated as a vector
/// type \p ValueU.
/// \tparam [in] ValueU The type of the vector
/// \tparam [in] ValueT The type of the original values
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The vectorized min of the two values
template <typename ValueU, typename ValueT>
inline ValueT vectorized_min(ValueT a, ValueT b) {
  sycl::vec<ValueT, 1> v0{a}, v1{b};
  auto v2 = v0.template as<ValueU>();
  auto v3 = v1.template as<ValueU>();
  auto v4 = sycl::min(v2, v3);
  v0 = v4.template as<sycl::vec<ValueT, 1>>();
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

/// Extend \p a and \p b to 33 bit and add them.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend addition of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, std::plus());
}

/// Extend Inputs to 33 bit, add \p a, \p b, then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend addition of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, std::plus(), second_op);
}

/// Extend \p a and \p b to 33 bit and add them with saturation.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend addition of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_add_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, std::plus());
}

/// Extend Inputs to 33 bit, add \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend addition of \p a, \p b with saturation and \p second_op
/// with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_add_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, std::plus(), second_op);
}

/// Extend \p a and \p b to 33 bit and minus them.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend subtraction of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, std::minus());
}

/// Extend Inputs to 33 bit, minus \p a, \p b, then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend subtraction of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, std::minus(), second_op);
}

/// Extend \p a and \p b to 33 bit and minus them with saturation.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend subtraction of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_sub_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, std::minus());
}

/// Extend Inputs to 33 bit, minus \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend subtraction of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_sub_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, std::minus(), second_op);
}

/// Extend \p a and \p b to 33 bit and do abs_diff.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend abs_diff of the two values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, abs_diff());
}

/// Extend Inputs to 33 bit, abs_diff \p a, \p b, then do \p second_op with \p
/// c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend abs_diff of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, abs_diff(), second_op);
}

/// Extend \p a and \p b to 33 bit and do abs_diff with saturation.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The extend abs_diff of the two values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_absdiff_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, abs_diff());
}

/// Extend Inputs to 33 bit, abs_diff \p a, \p b with saturation, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The extend abs_diff of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_absdiff_sat(AT a, BT b, CT c,
                                         BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, abs_diff(), second_op);
}

/// Extend \p a and \p b to 33 bit and return smaller one.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The smaller one of the two extended values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, minimum());
}

/// Extend Inputs to 33 bit, find the smaller one in \p a, \p b, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The smaller one of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, minimum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return smaller one with saturation.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The smaller one of the two extended values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_min_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, minimum());
}

/// Extend Inputs to 33 bit, find the smaller one in \p a, \p b with saturation,
/// then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The smaller one of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_min_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, minimum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return bigger one.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The bigger one of the two extended values
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max(AT a, BT b) {
  return detail::extend_binary<RetT, false>(a, b, maximum());
}

/// Extend Inputs to 33 bit, find the bigger one in \p a, \p b, then do \p
/// second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The bigger one of \p a, \p b and \p second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max(AT a, BT b, CT c, BinaryOperation second_op) {
  return detail::extend_binary<RetT, false>(a, b, c, maximum(), second_op);
}

/// Extend \p a and \p b to 33 bit and return bigger one with saturation.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \param [in] a The first value
/// \param [in] b The second value
/// \returns The bigger one of the two extended values with saturation
template <typename RetT, typename AT, typename BT>
inline constexpr RetT extend_max_sat(AT a, BT b) {
  return detail::extend_binary<RetT, true>(a, b, maximum());
}

/// Extend Inputs to 33 bit, find the bigger one in \p a, \p b with saturation,
/// then do \p second_op with \p c.
/// \tparam [in] RetT The type of the return value
/// \tparam [in] AT The type of the first value
/// \tparam [in] BT The type of the second value
/// \tparam [in] CT The type of the third value
/// \tparam [in] BinaryOperation The type of the second operation
/// \param [in] a The first value
/// \param [in] b The second value
/// \param [in] c The third value
/// \param [in] second_op The operation to do with the third value
/// \returns The bigger one of \p a, \p b with saturation and \p
/// second_op with \p c
template <typename RetT, typename AT, typename BT, typename CT,
          typename BinaryOperation>
inline constexpr RetT extend_max_sat(AT a, BT b, CT c,
                                     BinaryOperation second_op) {
  return detail::extend_binary<RetT, true>(a, b, c, maximum(), second_op);
}

} // namespace syclcompat
