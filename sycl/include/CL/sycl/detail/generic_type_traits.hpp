//==----------- generic_type_traits - SYCL type traits ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/aliases.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/generic_type_lists.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/half_type.hpp>

#include <limits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

template <typename T> using is_floatn = is_contained<T, gtl::vector_float_list>;

template <typename T> using is_genfloatf = is_contained<T, gtl::float_list>;

template <typename T>
using is_doublen = is_contained<T, gtl::vector_double_list>;

template <typename T> using is_genfloatd = is_contained<T, gtl::double_list>;

template <typename T> using is_halfn = is_contained<T, gtl::vector_half_list>;

template <typename T> using is_genfloath = is_contained<T, gtl::half_list>;

template <typename T> using is_genfloat = is_contained<T, gtl::floating_list>;

template <typename T>
using is_sgenfloat = is_contained<T, gtl::scalar_floating_list>;

template <typename T>
using is_vgenfloat = is_contained<T, gtl::vector_floating_list>;

template <typename T>
using is_gengeofloat = is_contained<T, gtl::geo_float_list>;

template <typename T>
using is_gengeodouble = is_contained<T, gtl::geo_double_list>;

template <typename T> using is_gengeohalf = is_contained<T, gtl::geo_half_list>;

template <typename T>
using is_vgengeofloat = is_contained<T, gtl::vector_geo_float_list>;

template <typename T>
using is_vgengeodouble = is_contained<T, gtl::vector_geo_double_list>;

template <typename T>
using is_vgengeohalf = is_contained<T, gtl::vector_geo_half_list>;

template <typename T> using is_sgengeo = is_contained<T, gtl::scalar_geo_list>;

template <typename T> using is_vgengeo = is_contained<T, gtl::vector_geo_list>;

template <typename T>
using is_gencrossfloat = is_contained<T, gtl::cross_float_list>;

template <typename T>
using is_gencrossdouble = is_contained<T, gtl::cross_double_list>;

template <typename T>
using is_gencrosshalf = is_contained<T, gtl::cross_half_list>;

template <typename T>
using is_gencross = is_contained<T, gtl::cross_floating_list>;

template <typename T>
using is_charn = is_contained<T, gtl::vector_default_char_list>;

template <typename T>
using is_scharn = is_contained<T, gtl::vector_signed_char_list>;

template <typename T>
using is_ucharn = is_contained<T, gtl::vector_unsigned_char_list>;

template <typename T>
using is_igenchar = is_contained<T, gtl::signed_char_list>;

template <typename T>
using is_ugenchar = is_contained<T, gtl::unsigned_char_list>;

template <typename T> using is_genchar = is_contained<T, gtl::char_list>;

template <typename T>
using is_shortn = is_contained<T, gtl::vector_signed_short_list>;

template <typename T>
using is_genshort = is_contained<T, gtl::signed_short_list>;

template <typename T>
using is_ushortn = is_contained<T, gtl::vector_unsigned_short_list>;

template <typename T>
using is_ugenshort = is_contained<T, gtl::unsigned_short_list>;

template <typename T>
using is_uintn = is_contained<T, gtl::vector_unsigned_int_list>;

template <typename T>
using is_ugenint = is_contained<T, gtl::unsigned_int_list>;

template <typename T>
using is_intn = is_contained<T, gtl::vector_signed_int_list>;

template <typename T> using is_genint = is_contained<T, gtl::signed_int_list>;

template <typename T>
using is_ulongn = is_contained<T, gtl::vector_unsigned_long_list>;

template <typename T>
using is_ugenlong = is_contained<T, gtl::unsigned_long_list>;

template <typename T>
using is_longn = is_contained<T, gtl::vector_signed_long_list>;

template <typename T> using is_genlong = is_contained<T, gtl::signed_long_list>;

template <typename T>
using is_ulonglongn = is_contained<T, gtl::vector_unsigned_longlong_list>;

template <typename T>
using is_ugenlonglong = is_contained<T, gtl::unsigned_longlong_list>;

template <typename T>
using is_longlongn = is_contained<T, gtl::vector_signed_longlong_list>;

template <typename T>
using is_genlonglong = is_contained<T, gtl::signed_longlong_list>;

template <typename T>
using is_igenlonginteger = is_contained<T, gtl::signed_long_integer_list>;

template <typename T>
using is_ugenlonginteger = is_contained<T, gtl::unsigned_long_integer_list>;

template <typename T> using is_geninteger = is_contained<T, gtl::integer_list>;

template <typename T>
using is_igeninteger = is_contained<T, gtl::signed_integer_list>;

template <typename T>
using is_ugeninteger = is_contained<T, gtl::unsigned_integer_list>;

template <typename T>
using is_sgeninteger = is_contained<T, gtl::scalar_integer_list>;

template <typename T>
using is_vgeninteger = is_contained<T, gtl::vector_integer_list>;

template <typename T>
using is_sigeninteger = is_contained<T, gtl::scalar_signed_integer_list>;

template <typename T>
using is_sugeninteger = is_contained<T, gtl::scalar_unsigned_integer_list>;

template <typename T>
using is_vigeninteger = is_contained<T, gtl::vector_signed_integer_list>;

template <typename T>
using is_vugeninteger = is_contained<T, gtl::vector_unsigned_integer_list>;

template <typename T> using is_gentype = is_contained<T, gtl::basic_list>;

template <typename T>
using is_vgentype = is_contained<T, gtl::vector_basic_list>;

template <typename T>
using is_sgentype = is_contained<T, gtl::scalar_basic_list>;

template <typename T>
using is_igeninteger8bit = is_gen_based_on_type_sizeof<T, 1, is_igeninteger>;

template <typename T>
using is_igeninteger16bit = is_gen_based_on_type_sizeof<T, 2, is_igeninteger>;

template <typename T>
using is_igeninteger32bit = is_gen_based_on_type_sizeof<T, 4, is_igeninteger>;

template <typename T>
using is_igeninteger64bit = is_gen_based_on_type_sizeof<T, 8, is_igeninteger>;

template <typename T>
using is_ugeninteger8bit = is_gen_based_on_type_sizeof<T, 1, is_ugeninteger>;

template <typename T>
using is_ugeninteger16bit = is_gen_based_on_type_sizeof<T, 2, is_ugeninteger>;

template <typename T>
using is_ugeninteger32bit = is_gen_based_on_type_sizeof<T, 4, is_ugeninteger>;

template <typename T>
using is_ugeninteger64bit = is_gen_based_on_type_sizeof<T, 8, is_ugeninteger>;

template <typename T>
using is_geninteger8bit = is_gen_based_on_type_sizeof<T, 1, is_geninteger>;

template <typename T>
using is_geninteger16bit = is_gen_based_on_type_sizeof<T, 2, is_geninteger>;

template <typename T>
using is_geninteger32bit = is_gen_based_on_type_sizeof<T, 4, is_geninteger>;

template <typename T>
using is_geninteger64bit = is_gen_based_on_type_sizeof<T, 8, is_geninteger>;

template <typename T>
using is_genintptr = bool_constant<
    is_pointer<T>::value && is_genint<remove_pointer_t<T>>::value &&
    is_address_space_compliant<T, gvl::nonconst_address_space_list>::value>;

template <typename T>
using is_genfloatptr = bool_constant<
    is_pointer<T>::value && is_genfloat<remove_pointer_t<T>>::value &&
    is_address_space_compliant<T, gvl::nonconst_address_space_list>::value>;

template <typename T>
using is_genptr = bool_constant<
    is_pointer<T>::value && is_gentype<remove_pointer_t<T>>::value &&
    is_address_space_compliant<T, gvl::nonconst_address_space_list>::value>;

template <typename T> using is_nan_type = is_contained<T, gtl::nan_list>;

// nan_types
template <typename T, typename Enable = void> struct nan_types;

template <typename T>
struct nan_types<
    T, enable_if_t<is_contained<T, gtl::unsigned_short_list>::value, T>> {
  using ret_type = change_base_type_t<T, half>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_short_list, half>;
};

template <typename T>
struct nan_types<
    T, enable_if_t<is_contained<T, gtl::unsigned_int_list>::value, T>> {
  using ret_type = change_base_type_t<T, float>;
  using arg_type = find_same_size_type_t<gtl::scalar_unsigned_int_list, float>;
};

template <typename T>
struct nan_types<
    T,
    enable_if_t<is_contained<T, gtl::unsigned_long_integer_list>::value, T>> {
  using ret_type = change_base_type_t<T, double>;
  using arg_type =
      find_same_size_type_t<gtl::scalar_unsigned_long_integer_list, double>;
};

template <typename T> using nan_return_t = typename nan_types<T, T>::ret_type;

template <typename T>
using nan_argument_base_t = typename nan_types<T, T>::arg_type;

template <typename T>
using make_floating_point_t = make_type_t<T, gtl::scalar_floating_list>;

template <typename T>
using make_singed_integer_t = make_type_t<T, gtl::scalar_signed_integer_list>;

template <typename T>
using make_unsinged_integer_t =
    make_type_t<T, gtl::scalar_unsigned_integer_list>;

template <typename T, typename B, typename Enable = void>
struct convert_data_type_impl;

template <typename T, typename B>
struct convert_data_type_impl<T, B, enable_if_t<is_sgentype<T>::value, T>> {
  B operator()(T t) { return static_cast<B>(t); }
};

template <typename T, typename B>
struct convert_data_type_impl<T, B, enable_if_t<is_vgentype<T>::value, T>> {
  vec<B, T::size()> operator()(T t) { return t.template convert<B>(); }
};

template <typename T, typename B>
using convert_data_type = convert_data_type_impl<T, B, T>;

// Try to get pointer_t, otherwise T
template <typename T> class TryToGetPointerT {
  static T check(...);
  template <typename A> static typename A::pointer_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value =
      std::is_pointer<T>::value || !std::is_same<T, type>::value;
};

// Try to get element_type, otherwise T
template <typename T> class TryToGetElementType {
  static T check(...);
  template <typename A> static typename A::element_type check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get vector_t, otherwise T
template <typename T> class TryToGetVectorT {
  static T check(...);
  template <typename A> static typename A::vector_t check(const A &);

public:
  using type = decltype(check(T()));
  static constexpr bool value = !std::is_same<T, type>::value;
};

// Try to get pointer_t (if pointer_t indicates on the type with_remainder
// vector_t creates a pointer type on vector_t), otherwise T
template <typename T> class TryToGetPointerVecT {
  static T check(...);
  template <typename A>
  static typename DecoratedType<
      typename TryToGetVectorT<typename TryToGetElementType<A>::type>::type,
      A::address_space>::type *
  check(const A &);
  template <typename A>
  static typename TryToGetVectorT<A>::type *check(const A *);

public:
  using type = decltype(check(T()));
};

template <typename T, typename = typename detail::enable_if_t<
                          TryToGetPointerT<T>::value, std::true_type>>
typename TryToGetPointerVecT<T>::type TryToGetPointer(T &t) {
  // TODO find the better way to get the pointer to underlying data from vec
  // class
  return reinterpret_cast<typename TryToGetPointerVecT<T>::type>(t.get());
}

template <typename T>
typename TryToGetPointerVecT<T *>::type TryToGetPointer(T *t) {
  // TODO find the better way to get the pointer to underlying data from vec
  // class
  return reinterpret_cast<typename TryToGetPointerVecT<T *>::type>(t);
}

template <typename T, typename = typename detail::enable_if_t<
                          !TryToGetPointerT<T>::value, std::false_type>>
T TryToGetPointer(T &t) {
  return t;
}

// select_apply_cl_scalar_t selects from T8/T16/T32/T64 basing on
// sizeof(IN).  expected to handle scalar types.
template <typename T, typename T8, typename T16, typename T32, typename T64>
using select_apply_cl_scalar_t =
    conditional_t<sizeof(T) == 1, T8,
                  conditional_t<sizeof(T) == 2, T16,
                                conditional_t<sizeof(T) == 4, T32, T64>>>;

// Shortcuts for selecting scalar int/unsigned int/fp type.
template <typename T>
using select_cl_scalar_integral_signed_t =
    select_apply_cl_scalar_t<T, sycl::cl_char, sycl::cl_short, sycl::cl_int,
                             sycl::cl_long>;

template <typename T>
using select_cl_scalar_integral_unsigned_t =
    select_apply_cl_scalar_t<T, sycl::cl_uchar, sycl::cl_ushort, sycl::cl_uint,
                             sycl::cl_ulong>;

template <typename T>
using select_cl_scalar_float_t =
    select_apply_cl_scalar_t<T, std::false_type, sycl::cl_half, sycl::cl_float,
                             sycl::cl_double>;

template <typename T>
using select_cl_scalar_integral_t =
    conditional_t<std::is_signed<T>::value,
                  select_cl_scalar_integral_signed_t<T>,
                  select_cl_scalar_integral_unsigned_t<T>>;

// select_cl_scalar_t picks corresponding cl_* type for input
// scalar T or returns T if T is not scalar.
template <typename T>
using select_cl_scalar_t = conditional_t<
    std::is_integral<T>::value, select_cl_scalar_integral_t<T>,
    conditional_t<
        std::is_floating_point<T>::value, select_cl_scalar_float_t<T>,
        // half is a special case: it is implemented differently on host and
        // device and therefore, might lower to different types
        conditional_t<std::is_same<T, half>::value,
                      cl::sycl::detail::half_impl::BIsRepresentationT, T>>>;

// select_cl_vector_or_scalar does cl_* type selection for element type of
// a vector type T and does scalar type substitution.  If T is not
// vector or scalar unmodified T is returned.
template <typename T, typename Enable = void> struct select_cl_vector_or_scalar;

template <typename T>
struct select_cl_vector_or_scalar<
    T, typename detail::enable_if_t<is_vgentype<T>::value>> {
  using type =
      // select_cl_scalar_t returns _Float16, so, we try to instantiate vec
      // class with _Float16 DataType, which is not expected there
      // So, leave vector<half, N> as-is
      vec<conditional_t<std::is_same<typename T::element_type, half>::value,
                        typename T::element_type,
                        select_cl_scalar_t<typename T::element_type>>,
          T::size()>;
};

template <typename T>
struct select_cl_vector_or_scalar<
    T, typename detail::enable_if_t<!is_vgentype<T>::value>> {
  using type = select_cl_scalar_t<T>;
};

// select_cl_mptr_or_vector_or_scalar does cl_* type selection for type
// pointed by multi_ptr or for element type of a vector type T and does
// scalar type substitution.  If T is not mutlti_ptr or vector or scalar
// unmodified T is returned.
template <typename T, typename Enable = void>
struct select_cl_mptr_or_vector_or_scalar;

template <typename T>
struct select_cl_mptr_or_vector_or_scalar<
    T, typename detail::enable_if_t<is_genptr<T>::value &&
                                    !std::is_pointer<T>::value>> {
  using type = multi_ptr<
      typename select_cl_vector_or_scalar<typename T::element_type>::type,
      T::address_space>;
};

template <typename T>
struct select_cl_mptr_or_vector_or_scalar<
    T, typename detail::enable_if_t<!is_genptr<T>::value ||
                                    std::is_pointer<T>::value>> {
  using type = typename select_cl_vector_or_scalar<T>::type;
};

// All types converting shortcut.
template <typename T>
using SelectMatchingOpenCLType_t =
    typename select_cl_mptr_or_vector_or_scalar<T>::type;

// Converts T to OpenCL friendly
//
template <typename T>
using ConvertToOpenCLType_t = conditional_t<
    TryToGetVectorT<SelectMatchingOpenCLType_t<T>>::value,
    typename TryToGetVectorT<SelectMatchingOpenCLType_t<T>>::type,
    conditional_t<
        TryToGetPointerT<SelectMatchingOpenCLType_t<T>>::value,
        typename TryToGetPointerVecT<SelectMatchingOpenCLType_t<T>>::type,
        SelectMatchingOpenCLType_t<T>>>;

// convertDataToType() function converts data from FROM type to TO type using
// 'as' method for vector type and copy otherwise.
template <typename FROM, typename TO>
typename detail::enable_if_t<is_vgentype<FROM>::value &&
                                 is_vgentype<TO>::value &&
                                 sizeof(TO) == sizeof(FROM),
                             TO>
convertDataToType(FROM t) {
  return t.template as<TO>();
}

template <typename FROM, typename TO>
typename detail::enable_if_t<!(is_vgentype<FROM>::value &&
                               is_vgentype<TO>::value) &&
                                 sizeof(TO) == sizeof(FROM),
                             TO>
convertDataToType(FROM t) {
  return TryToGetPointer(t);
}

// Used for all, any and select relational built-in functions
template <typename T> inline constexpr T msbMask(T) {
  using UT = make_unsigned_t<T>;
  return T(UT(1) << (sizeof(T) * 8 - 1));
}

template <typename T> inline constexpr bool msbIsSet(const T x) {
  return (x & msbMask(x));
}

template <typename T>
using common_rel_ret_t =
    conditional_t<is_vgentype<T>::value, make_singed_integer_t<T>, int>;

// forward declaration
template <int N> struct Boolean;

// Try to get vector element count or 1 otherwise
template <typename T, typename Enable = void> struct TryToGetNumElements;

template <typename T>
struct TryToGetNumElements<
    T, typename detail::enable_if_t<TryToGetVectorT<T>::value>> {
  static constexpr int value = T::size();
};
template <typename T>
struct TryToGetNumElements<
    T, typename detail::enable_if_t<!TryToGetVectorT<T>::value>> {
  static constexpr int value = 1;
};

// Used for relational comparison built-in functions
template <typename T> struct RelationalReturnType {
#ifdef __SYCL_DEVICE_ONLY__
  using type = Boolean<TryToGetNumElements<T>::value>;
#else
  using type = common_rel_ret_t<T>;
#endif
};

template <typename T> using rel_ret_t = typename RelationalReturnType<T>::type;

// Used for any and all built-in functions
template <typename T> struct RelationalTestForSignBitType {
#ifdef __SYCL_DEVICE_ONLY__
  using return_type = detail::Boolean<1>;
  using argument_type = detail::Boolean<TryToGetNumElements<T>::value>;
#else
  using return_type = int;
  using argument_type = T;
#endif
};

template <typename T>
using rel_sign_bit_test_ret_t =
    typename RelationalTestForSignBitType<T>::return_type;

template <typename T>
using rel_sign_bit_test_arg_t =
    typename RelationalTestForSignBitType<T>::argument_type;

template <typename T, typename Enable = void> struct RelConverter;

template <typename T>
struct RelConverter<
    T, typename detail::enable_if_t<TryToGetElementType<T>::value>> {
  static const int N = T::size();
#ifdef __SYCL_DEVICE_ONLY__
  using bool_t = typename Boolean<N>::vector_t;
  using ret_t = common_rel_ret_t<T>;
#else
  using bool_t = Boolean<N>;
  using ret_t = rel_ret_t<T>;
#endif

  static ret_t apply(bool_t value) {
#ifdef __SYCL_DEVICE_ONLY__
    typename ret_t::vector_t result(0);
    for (size_t I = 0; I < N; ++I) {
      result[I] = value[I];
    }
    return result;
#else
    return value;
#endif
  }
};

template <typename T>
struct RelConverter<
    T, typename detail::enable_if_t<!TryToGetElementType<T>::value>> {
  using R = rel_ret_t<T>;
#ifdef __SYCL_DEVICE_ONLY__
  using value_t = bool;
#else
  using value_t = R;
#endif

  static R apply(value_t value) { return value; }
};

template <typename T> static constexpr T max_v() {
  return (std::numeric_limits<T>::max)();
}

template <typename T> static constexpr T min_v() {
  return (std::numeric_limits<T>::min)();
}

template <typename T> static constexpr T quiet_NaN() {
  return std::numeric_limits<T>::quiet_NaN();
}

// is_same_vector_size
template <int FirstSize, typename... Args> class is_same_vector_size_impl;

template <int FirstSize, typename T, typename... Args>
class is_same_vector_size_impl<FirstSize, T, Args...> {
  using CurrentT = detail::remove_pointer_t<T>;
  static constexpr int Size = vector_size<CurrentT>::value;
  static constexpr bool IsSizeEqual = (Size == FirstSize);

public:
  static constexpr bool value =
      IsSizeEqual ? is_same_vector_size_impl<FirstSize, Args...>::value : false;
};

template <int FirstSize>
class is_same_vector_size_impl<FirstSize> : public std::true_type {};

template <typename T, typename... Args> class is_same_vector_size {
  using CurrentT = remove_pointer_t<T>;
  static constexpr int Size = vector_size<CurrentT>::value;

public:
  static constexpr bool value = is_same_vector_size_impl<Size, Args...>::value;
};

// check_vector_size
template <typename... Args> inline void check_vector_size() {
  static_assert(is_same_vector_size<Args...>::value,
                "The built-in function arguments must [point to|have] types "
                "with the same number of elements.");
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
