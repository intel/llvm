//==----------- scalar_infrastructure.hpp ---------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/detail/type_traits/vec_marray_traits.hpp>
#include <sycl/half_type.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifdef __FAST_MATH__
template <typename T>
struct use_fast_math
    : std::is_same<std::remove_cv_t<get_elem_type_t<T>>, float> {};
#else
template <typename> struct use_fast_math : std::false_type {};
#endif
template <typename T> constexpr bool use_fast_math_v = use_fast_math<T>::value;

// Utility trait for getting the decoration of a multi_ptr.
template <typename T> struct get_multi_ptr_decoration;
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct get_multi_ptr_decoration<
    multi_ptr<ElementType, Space, DecorateAddress>> {
  static constexpr access::decorated value = DecorateAddress;
};

template <typename T>
constexpr access::decorated get_multi_ptr_decoration_v =
    get_multi_ptr_decoration<T>::value;

// Utility trait for checking if a multi_ptr has a "writable" address space,
// i.e. global, local, private or generic.
template <typename T> struct has_writeable_addr_space : std::false_type {};
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct has_writeable_addr_space<multi_ptr<ElementType, Space, DecorateAddress>>
    : std::bool_constant<Space == access::address_space::global_space ||
                         Space == access::address_space::local_space ||
                         Space == access::address_space::private_space ||
                         Space == access::address_space::generic_space> {};

template <typename T>
constexpr bool has_writeable_addr_space_v = has_writeable_addr_space<T>::value;

// Utility trait for changing the element type of a type T. If T is a scalar,
// the new type replaces T completely.
template <typename NewElemT, typename T, typename = void>
struct change_elements {
  using type = NewElemT;
};
template <typename NewElemT, typename T>
struct change_elements<NewElemT, T, std::enable_if_t<is_marray_v<T>>> {
  using type =
      marray<typename change_elements<NewElemT, typename T::value_type>::type,
             T::size()>;
};
template <typename NewElemT, typename T>
struct change_elements<NewElemT, T, std::enable_if_t<is_vec_or_swizzle_v<T>>> {
  using type =
      vec<typename change_elements<NewElemT, typename T::element_type>::type,
          T::size()>;
};

template <typename NewElemT, typename T>
using change_elements_t = typename change_elements<NewElemT, T>::type;

template <typename... Ts>
inline constexpr bool builtin_same_shape_v =
    ((... && is_scalar_arithmetic_v<Ts>) || (... && is_marray_v<Ts>) ||
     (... && is_vec_or_swizzle_v<Ts>)) &&
    (... && (num_elements<Ts>::value ==
             num_elements<typename first_type<Ts...>::type>::value));

template <typename... Ts>
inline constexpr bool builtin_same_or_swizzle_v =
    // Use builtin_same_shape_v to filter out types unrelated to builtins.
    builtin_same_shape_v<Ts...> && all_same_v<simplify_if_swizzle_t<Ts>...>;

template <typename T>
struct fp_elem_type
    : std::bool_constant<
          check_type_in_v<get_elem_type_t<T>, float, double, half>> {};
template <typename T>
struct float_elem_type
    : std::bool_constant<check_type_in_v<get_elem_type_t<T>, float>> {};

template <typename... Ts>
struct same_basic_shape : std::bool_constant<builtin_same_shape_v<Ts...>> {};

template <typename... Ts>
struct same_elem_type : std::bool_constant<same_basic_shape<Ts...>::value &&
                                           all_same_v<get_elem_type_t<Ts>...>> {
};

template <typename> struct any_shape : std::true_type {};

template <typename T>
struct scalar_only : std::bool_constant<is_scalar_arithmetic_v<T>> {};

template <typename T>
struct non_scalar_only : std::bool_constant<!is_scalar_arithmetic_v<T>> {};

template <typename T> struct default_ret_type {
  using type = T;
};

template <typename T> struct scalar_ret_type {
  using type = get_elem_type_t<T>;
};

template <template <typename> typename RetTypeTrait,
          template <typename> typename ElemTypeChecker,
          template <typename> typename ShapeChecker,
          template <typename...> typename ExtraConditions, typename... Ts>
struct builtin_enable
    : std::enable_if<
          ElemTypeChecker<typename first_type<Ts...>::type>::value &&
              ShapeChecker<typename first_type<Ts...>::type>::value &&
              ExtraConditions<Ts...>::value,
          typename RetTypeTrait<
              simplify_if_swizzle_t<typename first_type<Ts...>::type>>::type> {
};

#define BUILTIN_CREATE_ENABLER(NAME, RET_TYPE_TRAIT, ELEM_TYPE_CHECKER,        \
                               SHAPE_CHECKER, EXTRA_CONDITIONS)                \
  namespace detail {                                                           \
  template <typename... Ts>                                                    \
  using NAME##_t =                                                             \
      typename builtin_enable<RET_TYPE_TRAIT, ELEM_TYPE_CHECKER,               \
                              SHAPE_CHECKER, EXTRA_CONDITIONS, Ts...>::type;   \
  }
} // namespace detail
} // namespace _V1
} // namespace sycl