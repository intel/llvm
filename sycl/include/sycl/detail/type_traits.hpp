//==----------------- type_traits.hpp - SYCL type traits -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>             // for decorated, address_space
#include <sycl/detail/generic_type_lists.hpp> // for vec, marray, integer_list
#include <sycl/detail/type_list.hpp>          // for is_contained, find_twi...
#include <sycl/half_type.hpp>                 // for half

#include <array>       // for array
#include <cstddef>     // for size_t
#include <tuple>       // for tuple
#include <type_traits> // for true_type, false_type

namespace sycl {
inline namespace _V1 {
namespace detail {
template <class T> struct is_fixed_size_group : std::false_type {};

template <class T>
inline constexpr bool is_fixed_size_group_v = is_fixed_size_group<T>::value;

template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
class SwizzleOp;
} // namespace detail

template <int Dimensions> class group;
struct sub_group;
namespace ext::oneapi {
struct sub_group;

namespace experimental {
template <typename Group, std::size_t Extent> class group_with_scratchpad;

template <class T> struct is_fixed_topology_group : std::false_type {};

template <class T>
inline constexpr bool is_fixed_topology_group_v =
    is_fixed_topology_group<T>::value;

template <int Dimensions> class root_group;
template <int Dimensions>
struct is_fixed_topology_group<root_group<Dimensions>> : std::true_type {};

template <int Dimensions>
struct is_fixed_topology_group<sycl::group<Dimensions>> : std::true_type {};

template <>
struct is_fixed_topology_group<sycl::ext::oneapi::sub_group> : std::true_type {
};
template <> struct is_fixed_topology_group<sycl::sub_group> : std::true_type {};

template <class T> struct is_user_constructed_group : std::false_type {};

template <class T>
inline constexpr bool is_user_constructed_group_v =
    is_user_constructed_group<T>::value;

namespace detail {
template <typename T> struct is_group_helper : std::false_type {};

template <typename Group, std::size_t Extent>
struct is_group_helper<group_with_scratchpad<Group, Extent>> : std::true_type {
};
} // namespace detail
} // namespace experimental
} // namespace ext::oneapi

namespace detail {

template <typename T> struct is_group : std::false_type {};

template <int Dimensions>
struct is_group<group<Dimensions>> : std::true_type {};

template <typename T> struct is_sub_group : std::false_type {};

template <> struct is_sub_group<ext::oneapi::sub_group> : std::true_type {};
template <> struct is_sub_group<sycl::sub_group> : std::true_type {};

template <typename T>
struct is_generic_group
    : std::integral_constant<bool,
                             is_group<T>::value || is_sub_group<T>::value> {};

namespace half_impl {
class half;
}
} // namespace detail
using half = detail::half_impl::half;

// Forward declaration
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;

template <class T>
struct is_group : std::bool_constant<detail::is_group<T>::value ||
                                     detail::is_sub_group<T>::value> {};

template <class T> inline constexpr bool is_group_v = is_group<T>::value;

namespace ext::oneapi::experimental {
template <class T>
inline constexpr bool is_group_helper_v =
    detail::is_group_helper<std::decay_t<T>>::value;
} // namespace ext::oneapi::experimental

namespace detail {
// Type for Intel device UUID extension.
// For details about this extension, see
// sycl/doc/extensions/supported/sycl_ext_intel_device_info.md
using uuid_type = std::array<unsigned char, 16>;

template <typename T, typename R> struct copy_cv_qualifiers;

template <typename T, typename R>
using copy_cv_qualifiers_t = typename copy_cv_qualifiers<T, R>::type;

template <int V> using int_constant = std::integral_constant<int, V>;
// vector_size
// scalars are interpreted as a vector of 1 length.
template <typename T> struct vector_size_impl : int_constant<1> {};
template <typename T, int N>
struct vector_size_impl<vec<T, N>> : int_constant<N> {};
template <typename T>
struct vector_size
    : vector_size_impl<std::remove_cv_t<std::remove_reference_t<T>>> {};

// vector_element
template <typename T> struct vector_element_impl;
template <typename T>
using vector_element_impl_t = typename vector_element_impl<T>::type;
template <typename T> struct vector_element_impl {
  using type = T;
};
template <typename T, int N> struct vector_element_impl<vec<T, N>> {
  using type = T;
};
template <typename T> struct vector_element {
  using type =
      copy_cv_qualifiers_t<T, vector_element_impl_t<std::remove_cv_t<T>>>;
};
template <class T> using vector_element_t = typename vector_element<T>::type;

template <class T> using marray_element_t = typename T::value_type;

// get_elem_type
// Get the element type of T. If T is a scalar, the element type is considered
// the type of the scalar.
template <typename T, typename = void> struct get_elem_type {
  using type = T;
};
template <typename T, size_t N> struct get_elem_type<marray<T, N>> {
  using type = T;
};
template <typename T, int N> struct get_elem_type<vec<T, N>> {
  using type = T;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct get_elem_type<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                               OperationCurrentT, Indexes...>> {
  using type = typename get_elem_type<std::remove_cv_t<VecT>>::type;
};

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
struct get_elem_type<multi_ptr<ElementType, Space, DecorateAddress>> {
  using type = ElementType;
};

template <typename T, typename = void>
struct is_ext_vector : std::false_type {};

template <typename T>
struct is_ext_vector<
    T, std::void_t<decltype(__builtin_reduce_max(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_ext_vector_v = is_ext_vector<T>::value;

template <typename T>
struct get_elem_type<T, std::enable_if_t<is_ext_vector_v<T>>> {
  using type = decltype(__builtin_reduce_max(std::declval<T>()));
};

template <typename T> using get_elem_type_t = typename get_elem_type<T>::type;

// change_base_type_t
template <typename T, typename B> struct change_base_type {
  using type = B;
};

template <typename T, int N, typename B> struct change_base_type<vec<T, N>, B> {
  using type = vec<B, N>;
};

template <typename T, typename B>
using change_base_type_t = typename change_base_type<T, B>::type;

// Applies the same the cv-qualifiers from T type to R type
template <typename T, typename R> struct copy_cv_qualifiers_impl {
  using type = R;
};

template <typename T, typename R> struct copy_cv_qualifiers_impl<const T, R> {
  using type = const R;
};

template <typename T, typename R>
struct copy_cv_qualifiers_impl<volatile T, R> {
  using type = volatile R;
};

template <typename T, typename R>
struct copy_cv_qualifiers_impl<const volatile T, R> {
  using type = const volatile R;
};

template <typename T, typename R> struct copy_cv_qualifiers {
  using type = typename copy_cv_qualifiers_impl<T, std::remove_cv_t<R>>::type;
};

// make_signed with support SYCL vec class
template <typename T> struct make_signed {
  using type = std::make_signed_t<T>;
};
template <typename T> using make_signed_t = typename make_signed<T>::type;
template <class T> struct make_signed<const T> {
  using type = const make_signed_t<T>;
};
template <class T, int N> struct make_signed<vec<T, N>> {
  using type = vec<make_signed_t<T>, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct make_signed<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                             OperationCurrentT, Indexes...>> {
  using type = make_signed_t<std::remove_cv_t<VecT>>;
};
template <class T, std::size_t N> struct make_signed<marray<T, N>> {
  using type = marray<make_signed_t<T>, N>;
};

// make_unsigned with support SYCL vec class
template <typename T> struct make_unsigned {
  using type = std::make_unsigned_t<T>;
};
template <typename T> using make_unsigned_t = typename make_unsigned<T>::type;
template <class T> struct make_unsigned<const T> {
  using type = const make_unsigned_t<T>;
};
template <class T, int N> struct make_unsigned<vec<T, N>> {
  using type = vec<make_unsigned_t<T>, N>;
};
template <typename VecT, typename OperationLeftT, typename OperationRightT,
          template <typename> class OperationCurrentT, int... Indexes>
struct make_unsigned<SwizzleOp<VecT, OperationLeftT, OperationRightT,
                               OperationCurrentT, Indexes...>> {
  using type = make_unsigned_t<std::remove_cv_t<VecT>>;
};
template <class T, std::size_t N> struct make_unsigned<marray<T, N>> {
  using type = marray<make_unsigned_t<T>, N>;
};

// Checks that sizeof base type of T equal N and T satisfies S<T>::value
template <typename T, int N, template <typename> class S>
inline constexpr bool is_gen_based_on_type_sizeof_v =
    S<T>::value && (sizeof(vector_element_t<T>) == N);

template <typename> struct is_vec : std::false_type {};
template <typename T, int N> struct is_vec<sycl::vec<T, N>> : std::true_type {};

template <typename T> constexpr bool is_vec_v = is_vec<T>::value;

template <typename> struct get_vec_size {
  static constexpr int size = 1;
};

template <typename T, int N> struct get_vec_size<sycl::vec<T, N>> {
  static constexpr int size = N;
};

// is_marray
template <typename> struct is_marray : std::false_type {};
template <typename T, size_t N>
struct is_marray<sycl::marray<T, N>> : std::true_type {};

template <typename T> constexpr bool is_marray_v = is_marray<T>::value;

// is_integral
template <typename T>
struct is_integral : std::is_integral<vector_element_t<T>> {};

// is_floating_point
template <typename T>
struct is_floating_point_impl : std::is_floating_point<T> {};

template <> struct is_floating_point_impl<half> : std::true_type {};

template <typename T>
struct is_floating_point
    : is_floating_point_impl<std::remove_cv_t<vector_element_t<T>>> {};

// is_arithmetic
template <typename T>
struct is_arithmetic
    : std::bool_constant<is_integral<T>::value || is_floating_point<T>::value> {
};

template <typename T>
struct is_scalar_arithmetic
    : std::bool_constant<!is_vec<T>::value && is_arithmetic<T>::value> {};

template <typename T>
inline constexpr bool is_scalar_arithmetic_v = is_scalar_arithmetic<T>::value;

template <typename T>
struct is_vector_arithmetic
    : std::bool_constant<is_vec<T>::value && is_arithmetic<T>::value> {};

// is_bool
template <typename T>
struct is_scalar_bool
    : std::bool_constant<std::is_same_v<std::remove_cv_t<T>, bool>> {};

template <typename T>
struct is_vector_bool
    : std::bool_constant<is_vec<T>::value &&
                         is_scalar_bool<vector_element_t<T>>::value> {};

template <typename T>
struct is_bool
    : std::bool_constant<is_scalar_bool<vector_element_t<T>>::value> {};

// is_pointer
template <typename T> struct is_pointer_impl : std::false_type {};

template <typename T> struct is_pointer_impl<T *> : std::true_type {};

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
struct is_pointer_impl<multi_ptr<T, Space, DecorateAddress>> : std::true_type {
};

template <typename T>
struct is_pointer : is_pointer_impl<std::remove_cv_t<T>> {};

template <typename T> inline constexpr bool is_pointer_v = is_pointer<T>::value;

// is_multi_ptr
template <typename T> struct is_multi_ptr : std::false_type {};

template <typename ElementType, access::address_space Space,
          access::decorated IsDecorated>
struct is_multi_ptr<multi_ptr<ElementType, Space, IsDecorated>>
    : std::true_type {};

template <class T>
inline constexpr bool is_multi_ptr_v = is_multi_ptr<T>::value;

// is_non_legacy_multi_ptr
template <typename T> struct is_non_legacy_multi_ptr : std::false_type {};

template <typename ElementType, access::address_space Space>
struct is_non_legacy_multi_ptr<
    multi_ptr<ElementType, Space, access::decorated::yes>> : std::true_type {};

template <typename ElementType, access::address_space Space>
struct is_non_legacy_multi_ptr<
    multi_ptr<ElementType, Space, access::decorated::no>> : std::true_type {};

template <class T>
inline constexpr bool is_non_legacy_multi_ptr_v =
    is_non_legacy_multi_ptr<T>::value;

// is_legacy_multi_ptr
template <typename T> struct is_legacy_multi_ptr : std::false_type {};

template <typename ElementType, access::address_space Space>
struct is_legacy_multi_ptr<
    multi_ptr<ElementType, Space, access::decorated::legacy>> : std::true_type {
};

template <class T>
inline constexpr bool is_legacy_multi_ptr_v = is_legacy_multi_ptr<T>::value;

// remove_pointer_t
template <typename T> struct remove_pointer_impl {
  using type = T;
};

template <typename T> struct remove_pointer_impl<T *> {
  using type = T;
};

template <typename T, access::address_space Space,
          access::decorated DecorateAddress>
struct remove_pointer_impl<multi_ptr<T, Space, DecorateAddress>> {
  using type = T;
};

template <typename T>
struct remove_pointer : remove_pointer_impl<std::remove_cv_t<T>> {};

template <typename T> using remove_pointer_t = typename remove_pointer<T>::type;

// is_address_space_compliant
template <typename T, typename SpaceList>
struct is_address_space_compliant_impl : std::false_type {};

template <typename T, typename SpaceList>
struct is_address_space_compliant_impl<T *, SpaceList> : std::true_type {};

template <typename T, typename SpaceList, access::address_space Space,
          access::decorated DecorateAddress>
struct is_address_space_compliant_impl<multi_ptr<T, Space, DecorateAddress>,
                                       SpaceList>
    : std::bool_constant<is_one_of_spaces<Space, SpaceList>::value> {};

template <typename T, typename SpaceList>
struct is_address_space_compliant
    : is_address_space_compliant_impl<std::remove_cv_t<T>, SpaceList> {};

template <typename T, typename SpaceList>
inline constexpr bool is_address_space_compliant_v =
    is_address_space_compliant<T, SpaceList>::value;

// make_type_t
template <typename T, typename TL> struct make_type_impl {
  using type = find_same_size_type_t<TL, T>;
};

template <typename T, int N, typename TL> struct make_type_impl<vec<T, N>, TL> {
  using scalar_type = typename make_type_impl<T, TL>::type;
  using type = vec<scalar_type, N>;
};

template <typename T, typename TL>
using make_type_t = typename make_type_impl<T, TL>::type;

// make_larger_t
template <typename T, typename Enable = void> struct make_larger_impl;
template <typename T>
struct make_larger_impl<
    T, std::enable_if_t<is_contained<T, gtl::scalar_floating_list>::value, T>> {
  using type = find_twice_as_large_type_t<gtl::scalar_floating_list, T>;
};

template <typename T>
struct make_larger_impl<
    T, std::enable_if_t<is_contained<T, gtl::scalar_signed_integer_list>::value,
                        T>> {
  using type = find_twice_as_large_type_t<gtl::scalar_signed_integer_list, T>;
};

template <typename T>
struct make_larger_impl<
    T, std::enable_if_t<
           is_contained<T, gtl::scalar_unsigned_integer_list>::value, T>> {
  using type = find_twice_as_large_type_t<gtl::scalar_unsigned_integer_list, T>;
};

template <typename T, int N> struct make_larger_impl<vec<T, N>, vec<T, N>> {
  using base_type = vector_element_t<vec<T, N>>;
  using upper_type = typename make_larger_impl<base_type, base_type>::type;
  using new_type = vec<upper_type, N>;
  static constexpr bool found = !std::is_same_v<upper_type, void>;
  using type = std::conditional_t<found, new_type, void>;
};

template <typename T, size_t N>
struct make_larger_impl<marray<T, N>, marray<T, N>> {
  using base_type = marray_element_t<marray<T, N>>;
  using upper_type = typename make_larger_impl<base_type, base_type>::type;
  using new_type = marray<upper_type, N>;
  static constexpr bool found = !std::is_same_v<upper_type, void>;
  using type = std::conditional_t<found, new_type, void>;
};

template <typename T> struct make_larger {
  using type = typename make_larger_impl<T, T>::type;
};

template <typename T> using make_larger_t = typename make_larger<T>::type;

#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
template <access::address_space AS, class DataT>
using const_if_const_AS =
    typename std::conditional<AS == access::address_space::constant_space,
                              const DataT, DataT>::type;
#else
template <access::address_space AS, class DataT>
using const_if_const_AS = DataT;
#endif

template <typename T> struct function_traits {};

template <typename Ret, typename... Args> struct function_traits<Ret(Args...)> {
  using ret_type = Ret;
  using args_type = std::tuple<Args...>;
};

// No first_type_t due to
// https://open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#1430.
template <typename T, typename... Ts> struct first_type {
  using type = T;
};

template <typename T0, typename... Ts>
inline constexpr bool all_same_v = (... && std::is_same_v<T0, Ts>);

// Example usage:
//   using mapped = map_type<type_to_map, from0, /*->*/ to0,
//                                        from1, /*->*/ to1,
//                                        ...>
template <typename...> struct map_type {
  using type = void;
};

template <typename T, typename From, typename To, typename... Rest>
struct map_type<T, From, To, Rest...> {
  using type = std::conditional_t<std::is_same_v<From, T>, To,
                                  typename map_type<T, Rest...>::type>;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
