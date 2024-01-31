//==--- builtins_utils_scalar.hpp - SYCL built-in function utilities -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>              // for address_space, decorated
#include <sycl/aliases.hpp>                    // for half
#include <sycl/detail/boolean.hpp>             // for Boolean
#include <sycl/detail/builtins.hpp>            // for __invoke_select, __in...
#include <sycl/detail/defines_elementary.hpp>  // for __SYCL_ALWAYS_INLINE
#include <sycl/detail/generic_type_traits.hpp> // for is_svgenfloat, is_sge...
#include <sycl/detail/type_list.hpp>           // for is_contained, type_list
#include <sycl/detail/type_traits.hpp>         // for make_larger_t, marray...
#include <sycl/half_type.hpp>                  // for half, intel
#include <sycl/multi_ptr.hpp>                  // for address_space_cast

#include <algorithm>
#include <cstring>

namespace sycl {
inline namespace _V1 {

#ifdef __SYCL_DEVICE_ONLY__
#define __sycl_std
#else
namespace __sycl_std = __host_std;
#endif

namespace detail {
#ifdef __FAST_MATH__
template <typename T>
struct use_fast_math
    : std::is_same<std::remove_cv_t<get_elem_type_t<T>>, float> {};
#else
template <typename> struct use_fast_math : std::false_type {};
#endif
template <typename T> constexpr bool use_fast_math_v = use_fast_math<T>::value;

// sycl::select(sgentype a, sgentype b, bool c) calls OpenCL built-in
// select(sgentype a, sgentype b, igentype c). This type trait makes the
// proper conversion for argument c from bool to igentype, based on sgentype
// == T.
// TODO: Consider unifying this with select_cl_scalar_integral_signed_t.
template <typename T>
using get_select_opencl_builtin_c_arg_type = typename std::conditional_t<
    sizeof(T) == 1, char,
    std::conditional_t<
        sizeof(T) == 2, short,
        std::conditional_t<
            (detail::is_contained<
                 T, detail::type_list<long, unsigned long>>::value &&
             (sizeof(T) == 4 || sizeof(T) == 8)),
            long, // long and ulong are 32-bit on
                  // Windows and 64-bit on Linux
            std::conditional_t<
                sizeof(T) == 4, int,
                std::conditional_t<sizeof(T) == 8, long long, void>>>>>;

// Common utility for selecting a type based on the specified size.
template <size_t Size, typename T8, typename T16, typename T32, typename T64>
using select_scalar_by_size_t = std::conditional_t<
    Size == 1, T8,
    std::conditional_t<
        Size == 2, T16,
        std::conditional_t<Size == 4, T32,
                           std::conditional_t<Size == 8, T64, void>>>>;

template <size_t N, size_t... Ns> constexpr bool CheckSizeIn() {
  constexpr bool SameSize[] = {(N == Ns)...};
  // Replace with std::any_of with C++20.
  for (size_t I = 0; I < sizeof...(Ns); ++I)
    if (SameSize[I])
      return true;
  return false;
}

// Checks if the type of the operation is the same. For scalars and marray that
// requires the types to be exact matches. For vector and swizzles it requires
// that the corresponding vector conversion is the same.
template <typename T1, typename T2, typename = void>
struct is_same_op : std::is_same<T1, T2> {};

template <typename T1, typename T2>
constexpr bool is_same_op_v = is_same_op<T1, T2>::value;

// Constexpr function for checking that all types are the same, considering
// swizzles and vectors the same if they have the same number of elements and
// the same element type.
template <typename T, typename... Ts> constexpr bool CheckAllSameOpType() {
  constexpr bool SameType[] = {
      is_same_op_v<std::remove_cv_t<T>, std::remove_cv_t<Ts>>...};
  // Replace with std::all_of with C++20.
  for (size_t I = 0; I < sizeof...(Ts); ++I)
    if (!SameType[I])
      return false;
  return true;
}

// NOTE: We need a constexpr variable definition for the constexpr functions
//       as MSVC thinks function definitions are the same otherwise.
template <typename... Ts>
constexpr bool check_all_same_op_type_v = CheckAllSameOpType<Ts...>();
// NOTE: We need a constexpr variable definition for the constexpr functions
//       as MSVC thinks function definitions are the same otherwise.
template <size_t... Ns> constexpr bool check_size_in_v = CheckSizeIn<Ns...>();

// Utility traits for getting a signed integer type with the specified size.
template <size_t Size> struct get_signed_int_by_size {
  using type = select_scalar_by_size_t<Size, int8_t, int16_t, int32_t, int64_t>;
};
template <typename T> struct same_size_signed_int {
  using type = typename get_signed_int_by_size<sizeof(T)>::type;
};

template <typename T>
using same_size_signed_int_t = typename same_size_signed_int<T>::type;

// Utility traits for getting a unsigned integer type with the specified size.
template <size_t Size> struct get_unsigned_int_by_size {
  using type =
      select_scalar_by_size_t<Size, uint8_t, uint16_t, uint32_t, uint64_t>;
};
template <typename T> struct same_size_unsigned_int {
  using type = typename get_unsigned_int_by_size<sizeof(T)>::type;
};

// Utility trait for getting an upsampled integer type.
// NOTE: For upsampling we look for an integer of double the size of the
// specified type.
template <typename T> struct upsampled_int {
  using type =
      std::conditional_t<std::is_unsigned_v<T>,
                         typename get_unsigned_int_by_size<sizeof(T) * 2>::type,
                         typename get_signed_int_by_size<sizeof(T) * 2>::type>;
};

template <typename T> using upsampled_int_t = typename upsampled_int<T>::type;

// Utility for converting a swizzle to a vector or preserve the type if it isn't
// a swizzle.
template <typename T> struct simplify_if_swizzle {
  using type = T;
};

template <typename T>
using simplify_if_swizzle_t = typename simplify_if_swizzle<T>::type;

// Utility trait for checking if T's element type is in Ts.
template <typename T, typename... Ts>
struct is_valid_elem_type : std::false_type {};

template <typename T, typename... Ts>
constexpr bool is_valid_elem_type_v = is_valid_elem_type<T, Ts...>::value;

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

// Wrapper trait around nan_return to allow propagation through swizzles and
// marrays.
template <typename T> struct nan_return_unswizzled {
  using type = typename nan_types<T, T>::ret_type;
};

template <typename T>
using nan_return_unswizzled_t = typename nan_return_unswizzled<T>::type;
} // namespace detail
} // namespace _V1
} // namespace sycl
