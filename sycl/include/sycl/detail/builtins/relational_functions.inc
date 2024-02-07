//==------------------- relational_functions.hpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Intentionally insufficient set of includes and no "#pragma once".

#include <sycl/detail/builtins/helper_macros.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T>
struct bitselect_elem_type
    : std::bool_constant<check_type_in_v<
          get_elem_type_t<T>, float, double, half, char, signed char, short,
          int, long, long long, unsigned char, unsigned short, unsigned int,
          unsigned long, unsigned long long>> {};

template <typename T>
struct rel_ret_traits
    : std::conditional<is_scalar_arithmetic_v<T>, bool,
                       std::conditional_t<
                           is_marray_v<T>, marray<bool, num_elements<T>::value>,
                           same_size_signed_int_t<simplify_if_swizzle_t<T>>>> {
};
} // namespace detail

BUILTIN_CREATE_ENABLER(builtin_enable_bitselect, default_ret_type,
                       bitselect_elem_type, any_shape, same_elem_type)
BUILTIN_CREATE_ENABLER(builtin_enable_rel, rel_ret_traits, fp_elem_type,
                       non_scalar_only, same_elem_type)

namespace detail {
#ifdef __SYCL_DEVICE_ONLY__
template <typename FuncTy, typename... Ts>
auto builtin_device_rel_impl(FuncTy F, const Ts &...xs) {
  using T = typename first_type<Ts...>::type;
  if constexpr (detail::is_vec_or_swizzle_v<T>) {
    // decltype(ret) is signed char ext_vector_type(N). Convert it to
    // sycl::vec<signed char, N> first and then to the required return type of
    // the relation builtin (vector of int16_t/int32_t/int64_t depending on the
    // arguments' element type).
    auto ret = F(builtins::convert_arg(xs)...);
    vec<signed char, num_elements<T>::value> tmp{ret};
    using res_elem_type =
        make_type_t<get_elem_type_t<T>, type_list<int16_t, int32_t, int64_t>>;
    static_assert(is_scalar_arithmetic_v<res_elem_type>);
    return tmp.template convert<res_elem_type>();
  } else if constexpr (std::is_same_v<T, half>) {
    return bool{F(builtins::convert_arg(xs)...)};
  } else {
    static_assert(!detail::is_swizzle_v<T>);
    return F(builtins::convert_arg(xs)...);
  }
}
#endif

template <typename FuncTy, typename... Ts>
auto builtin_delegate_rel_impl(FuncTy F, const Ts &...x) {
  using T = typename first_type<Ts...>::type;
  if constexpr ((... || is_swizzle_v<Ts>)) {
    return F(simplify_if_swizzle_t<T>{x}...);
  } else if constexpr (is_vec_v<T>) {
    // TODO: using Res{} to avoid Werror. Not sure if ok.
    vec<same_size_signed_int_t<get_elem_type_t<T>>, T::size()> Res{};
    detail::loop<T::size()>(
        [&](auto idx) { Res[idx] = F(x[idx]...) ? -1 : 0; });
    return Res;
  } else {
    // marray.
    marray<bool, T::size()> Res;
    // TODO: Can we optimize this? Note that using vector version isn't
    // straightforward as it doesn't return booleans.
    detail::loop<T::size()>([&](auto idx) { Res[idx] = F(x[idx]...); });
    return Res;
  }
}
} // namespace detail

#ifdef __SYCL_DEVICE_ONLY__
#define BUILTIN_REL(NUM_ARGS, NAME, SPIRV_IMPL)                                \
  bool NAME(NUM_ARGS##_TYPE_ARG(float)) { return SPIRV_IMPL(NUM_ARGS##_ARG); } \
  bool NAME(NUM_ARGS##_TYPE_ARG(double)) {                                     \
    return SPIRV_IMPL(NUM_ARGS##_ARG);                                         \
  }                                                                            \
  bool NAME(NUM_ARGS##_TYPE_ARG(half)) {                                       \
    return SPIRV_IMPL(NUM_ARGS##_CONVERTED_ARG);                               \
  }                                                                            \
  template <NUM_ARGS##_TYPENAME_TYPE>                                          \
  detail::builtin_enable_rel_t<NUM_ARGS##_TEMPLATE_TYPE> NAME(                 \
      NUM_ARGS##_TEMPLATE_TYPE_ARG) {                                          \
    if constexpr (detail::is_marray_v<T0>)                                     \
      return detail::builtin_delegate_rel_impl(                                \
          [&](NUM_ARGS##_AUTO_ARG) { return NAME(NUM_ARGS##_ARG); },           \
          NUM_ARGS##_ARG);                                                     \
    else                                                                       \
      return detail::builtin_device_rel_impl(                                  \
          [&](NUM_ARGS##_AUTO_ARG) { return SPIRV_IMPL(NUM_ARGS##_ARG); },     \
          NUM_ARGS##_ARG);                                                     \
  }
#else
#define BUILTIN_REL(NUM_ARGS, NAME, SPIRV_IMPL)                                \
  FOR_EACH3(HOST_IMPL_SCALAR_RET_TYPE, NUM_ARGS, NAME, bool, FP_TYPES)         \
  HOST_IMPL_TEMPLATE_CUSTOM_DELEGATOR(NUM_ARGS, NAME, builtin_enable_rel_t,    \
                                      rel, rel_ret_traits,                     \
                                      builtin_delegate_rel_impl)
#endif

BUILTIN_REL(TWO_ARGS, isequal, __spirv_FOrdEqual)
BUILTIN_REL(TWO_ARGS, isnotequal, __spirv_FUnordNotEqual)
BUILTIN_REL(TWO_ARGS, isgreater, __spirv_FOrdGreaterThan)
BUILTIN_REL(TWO_ARGS, isgreaterequal, __spirv_FOrdGreaterThanEqual)
BUILTIN_REL(TWO_ARGS, isless, __spirv_FOrdLessThan)
BUILTIN_REL(TWO_ARGS, islessequal, __spirv_FOrdLessThanEqual)
BUILTIN_REL(TWO_ARGS, islessgreater, __spirv_FOrdNotEqual)
BUILTIN_REL(ONE_ARG, isfinite, __spirv_IsFinite)
BUILTIN_REL(ONE_ARG, isinf, __spirv_IsInf)
BUILTIN_REL(ONE_ARG, isnan, __spirv_IsNan)
BUILTIN_REL(ONE_ARG, isnormal, __spirv_IsNormal)
BUILTIN_REL(TWO_ARGS, isordered, __spirv_Ordered)
BUILTIN_REL(TWO_ARGS, isunordered, __spirv_Unordered)
BUILTIN_REL(ONE_ARG, signbit, __spirv_SignBitSet)

#undef BUILTIN_REL

#ifdef __SYCL_DEVICE_ONLY__
DEVICE_IMPL_TEMPLATE(
    THREE_ARGS, bitselect, builtin_enable_bitselect_t, [](auto... xs) {
      using ret_ty = detail::builtin_enable_bitselect_t<THREE_ARGS_TEMPLATE_TYPE>;
      using detail::builtins::convert_result;
      return convert_result<ret_ty>(__spirv_ocl_bitselect(xs...));
    })
#else
HOST_IMPL_TEMPLATE(THREE_ARGS, bitselect, builtin_enable_bitselect_t, rel,
                   default_ret_type)
#endif

namespace detail {
template <typename T>
struct builtin_enable_rel_all_any
    : std::enable_if<(is_marray_v<T> &&
                      std::is_same_v<get_elem_type_t<T>, bool>) ||
                         (is_vec_or_swizzle_v<T> &&
                          check_type_in_v<get_elem_type_t<T>, int8_t, int16_t,
                                          int32_t, int64_t>),
                     std::conditional_t<is_marray_v<T>, bool, int>> {};

template <typename T>
struct builtin_enable_rel_all_any_deprecated
    : std::enable_if<((is_scalar_arithmetic_v<T> || is_marray_v<T>)) &&
                         check_type_in_v<get_elem_type_t<T>, signed char, short,
                                         int, long, long long>,
                     bool> {};
} // namespace detail

template <typename T>
typename detail::builtin_enable_rel_all_any<T>::type any(T x) {
  if constexpr (detail::is_marray_v<T>) {
    return std::any_of(x.begin(), x.end(), [](bool x) { return x; });
  } else {
    for (size_t i = 0; i < detail::num_elements<T>::value; ++i)
      if (detail::msbIsSet(x[i]))
        return true;
    return false;
  }
}

template <typename T>
__SYCL_DEPRECATED("This overload is deprecated in SYCL 2020.")
typename detail::builtin_enable_rel_all_any_deprecated<T>::type any(T x) {
  if constexpr (detail::is_marray_v<T>) {
    return std::any_of(x.begin(), x.end(),
                       [](auto x) { return detail::msbIsSet(x); });
  } else {
    return detail::msbIsSet(x);
  }
}

template <typename T>
typename detail::builtin_enable_rel_all_any<T>::type all(T x) {
  if constexpr (detail::is_marray_v<T>) {
    return std::all_of(x.begin(), x.end(), [](bool x) { return x; });
  } else {
    for (size_t i = 0; i < detail::num_elements<T>::value; ++i)
      if (!detail::msbIsSet(x[i]))
        return false;
    return true;
  }
}

template <typename T>
__SYCL_DEPRECATED("This overload is deprecated in SYCL 2020.")
typename detail::builtin_enable_rel_all_any_deprecated<T>::type all(T x) {
  if constexpr (detail::is_marray_v<T>) {
    return std::all_of(x.begin(), x.end(),
                       [](auto x) { return detail::msbIsSet(x); });
  } else {
    return detail::msbIsSet(x);
  }
}
namespace detail {
template <typename T>
inline constexpr bool is_rel_generic_scalar_v =
    check_type_in_v<T, char, signed char, short, int, long, long long,
                    unsigned char, unsigned short, unsigned int, unsigned long,
                    unsigned long long, float, double, half>;
template <typename T>
inline constexpr bool is_rel_vector_elem_type_v =
    check_type_in_v<T, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                    uint32_t, uint64_t, float, double, half>;

template <typename T0, typename T1, typename T2>
struct rel_enable_select_marray_helper {
  static constexpr bool check_T0 =
      is_marray_v<T0> && is_rel_generic_scalar_v<get_elem_type_t<T0>>;
  static constexpr bool check_T1 = std::is_same_v<T0, T1>;
  static constexpr bool check_T2 =
      is_marray_v<T2> && std::is_same_v<get_elem_type_t<T2>, bool> &&
      num_elements<T0>::value == num_elements<T2>::value;

  static constexpr bool value = check_T0 && check_T1 && check_T2;
};

template <typename T0, typename T1, typename T2>
struct rel_enable_select_vec_helper {
  using T0_simplified = simplify_if_swizzle_t<T0>;
  using T1_simplified = simplify_if_swizzle_t<T1>;
  using T2_simplified = simplify_if_swizzle_t<T2>;
  using T0_elem_type = get_elem_type_t<T0>;
  using T2_elem_type = get_elem_type_t<T2>;

  static constexpr bool check_T0 =
      is_vec_or_swizzle_v<T0> && is_rel_vector_elem_type_v<get_elem_type_t<T0>>;
  static constexpr bool check_T1 = std::is_same_v<T0_simplified, T1_simplified>;
  static constexpr bool check_T2 =
      is_vec_or_swizzle_v<T2> &&
      num_elements<T0>::value == num_elements<T2>::value &&
      std ::is_integral_v<T2_elem_type> &&
      sizeof(T0_elem_type) == sizeof(T2_elem_type);
  static constexpr bool value = check_T0 && check_T1 && check_T2;
};

template <typename T0, typename T1, typename T2>
inline constexpr bool rel_enable_select_v =
    detail::rel_enable_select_marray_helper<T0, T1, T2>::value ||
    detail::rel_enable_select_vec_helper<T0, T1, T2>::value;
} // namespace detail

// __spirv_ocl_select doesn't behave as required by SYCL/OpenCL spec for vector
// data types (MSB-related stuff).
template <typename T>
std::enable_if_t<detail::is_rel_generic_scalar_v<T>, T> select(T a, T b,
                                                               bool c) {
  return (c ? b : a);
}

template <typename T0, typename T1, typename T2>
std::enable_if_t<detail::rel_enable_select_v<T0, T1, T2>,
                 detail::simplify_if_swizzle_t<T0>>
select(T0 a, T1 b, T2 c) {
  if constexpr (detail::is_marray_v<T0>) {
    T0 ret;
    for (size_t i = 0; i < T0::size(); ++i)
      ret[i] = (c[i] ? b[i] : a[i]);
    return ret;
  } else {
    detail::simplify_if_swizzle_t<T0> ret;
    for (size_t i = 0; i < ret.size(); ++i)
      ret[i] = (detail::msbIsSet(c[i]) ? b[i] : a[i]);
    return ret;
  }
}
} // namespace _V1
} // namespace sycl
