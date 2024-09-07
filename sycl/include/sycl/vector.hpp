//==---------------- vector.hpp --- Implements sycl::vec -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Check if Clang's ext_vector_type attribute is available. Host compiler
// may not be Clang, and Clang may not be built with the extension.
#ifdef __clang__
#ifndef __has_extension
#define __has_extension(x) 0
#endif
#ifdef __HAS_EXT_VECTOR_TYPE__
#error "Undefine __HAS_EXT_VECTOR_TYPE__ macro"
#endif
#if __has_extension(attribute_ext_vector_type)
#define __HAS_EXT_VECTOR_TYPE__
#endif
#endif // __clang__

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ostream>
#include <type_traits>
#include <utility>

#ifdef __SYCL_VEC_STANDALONE
#define __SYCL_EBO
#define __SYCL2020_DEPRECATED(...)

namespace sycl {
inline namespace _V1 {
namespace access {
enum class address_space { global_space };
enum class decorated { yes, no };
enum class placeholder;
enum class mode;
enum class target;
} // namespace access

namespace detail {
template <access::target accessTarget> struct TargetToAS {
  constexpr static access::address_space AS =
      access::address_space::global_space;
};
} // namespace detail

template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress>
class multi_ptr;

template <typename To, typename From>
constexpr std::enable_if_t<sizeof(To) == sizeof(From) &&
                               std::is_trivially_copyable<From>::value &&
                               std::is_trivially_copyable<To>::value,
                           To>
bit_cast(const From &from) noexcept {
  return __builtin_bit_cast(To, from);
}
template <typename DataT, int NumElements> class __SYCL_EBO vec;

namespace detail::half_impl {
class half;
#ifdef __SYCL_DEVICE_ONLY__
using StorageT = _Float16;
using BIsRepresentationT = _Float16;
using VecElemT = _Float16;
#else  // SYCL_DEVICE_ONLY
using StorageT = uint16_t;
// No need to extract underlying data type for built-in functions operating on
// host
using BIsRepresentationT = half;
using VecElemT = half;
#endif // SYCL_DEVICE_ONLY
} // namespace detail::half_impl
using half = detail::half_impl::half;

namespace ext::oneapi {
class bfloat16;
namespace detail {
using Bfloat16StorageT = uint16_t;
}
} // namespace ext::oneapi

namespace detail {
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
class __SYCL_EBO Swizzle;

template <typename> struct is_vec : std::false_type {};
template <typename T, int N> struct is_vec<sycl::vec<T, N>> : std::true_type {};

template <typename T> constexpr bool is_vec_v = is_vec<T>::value;

template <typename> struct is_swizzle : std::false_type {};
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
struct is_swizzle<Swizzle<IsConstVec, DataT, VecSize, Indexes...>>
    : std::true_type {};

template <typename T> constexpr bool is_swizzle_v = is_swizzle<T>::value;

template <typename T>
constexpr bool is_vec_or_swizzle_v = is_vec_v<T> || is_swizzle_v<T>;

template <typename T, typename = void>
struct is_ext_vector : std::false_type {};

template <typename T>
struct is_ext_vector<
    T, std::void_t<decltype(__builtin_reduce_max(std::declval<T>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_ext_vector_v = is_ext_vector<T>::value;

template <typename T, typename = void> struct get_elem_type {
  using type = T;
};
template <typename T>
struct get_elem_type<T, std::enable_if_t<is_vec_or_swizzle_v<T>>> {
  using type = typename T::element_type;
};
template <typename T> using get_elem_type_t = typename get_elem_type<T>::type;

template <typename T>
using select_cl_scalar_integral_signed_t = std::conditional_t<
    sizeof(T) == 1, int8_t,
    std::conditional_t<sizeof(T) == 2, int16_t,
                       std::conditional_t<sizeof(T) == 4, int32_t, int64_t>>>;

template <typename T>
using select_cl_scalar_integral_unsigned_t = std::conditional_t<
    sizeof(T) == 1, uint8_t,
    std::conditional_t<sizeof(T) == 2, uint16_t,
                       std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

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

template <typename T> auto convertToOpenCLType(T &&x) {
  using no_ref = std::remove_reference_t<T>;
  if constexpr (is_vec_v<no_ref>) {
    using ElemTy = typename no_ref::element_type;
    // sycl::half may convert to _Float16, and we would try to instantiate
    // vec class with _Float16 DataType, which is not expected there. As
    // such, leave vector<half, N> as-is.
    using MatchingVec =
        vec<std::conditional_t<std::is_same_v<ElemTy, half>, ElemTy,
                               decltype(convertToOpenCLType(
                                   std::declval<ElemTy>()))>,
            no_ref::size()>;
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::bit_cast<typename MatchingVec::vector_t>(x);
#else
    return x.template as<MatchingVec>();
#endif
  } else if constexpr (is_vec_v<no_ref>) {
    using ElemTy = typename no_ref::element_type;
    // sycl::half may convert to _Float16, and we would try to instantiate
    // vec class with _Float16 DataType, which is not expected there. As
    // such, leave vector<half, N> as-is.
    using MatchingVec =
        vec<std::conditional_t<std::is_same_v<ElemTy, half>, ElemTy,
                               decltype(convertToOpenCLType(
                                   std::declval<ElemTy>()))>,
            no_ref::size()>;
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::bit_cast<typename MatchingVec::vector_t>(x);
#else
    return x.template as<MatchingVec>();
#endif
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
  } else if constexpr (std::is_same_v<no_ref, std::byte>) {
    return static_cast<uint8_t>(x);
#endif
  } else if constexpr (std::is_integral_v<no_ref>) {
    using OpenCLType =
        std::conditional_t<std::is_signed_v<no_ref>,
                           select_cl_scalar_integral_signed_t<no_ref>,
                           select_cl_scalar_integral_unsigned_t<no_ref>>;
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  } else if constexpr (std::is_same_v<no_ref, half>) {
    // Make it a dependent type.
    using OpenCLType =
        std::conditional_t<false, T,
                           sycl::detail::half_impl::BIsRepresentationT>;
    static_assert(std::is_same_v<OpenCLType,
                                 sycl::detail::half_impl::BIsRepresentationT>);
    static_assert(sizeof(OpenCLType) == sizeof(T));
    return static_cast<OpenCLType>(x);
  } else if constexpr (std::is_same_v<ext::oneapi::bfloat16, no_ref>) {
    // On host, don't interpret BF16 as uint16.
#ifdef __SYCL_DEVICE_ONLY__
    using OpenCLType = sycl::ext::oneapi::detail::Bfloat16StorageT;
    return sycl::bit_cast<OpenCLType>(x);
#else
    return std::forward<T>(x);
#endif
  } else if constexpr (std::is_floating_point_v<no_ref>) {
    static_assert(std::is_same_v<no_ref, float> ||
                      std::is_same_v<no_ref, double>,
                  "Other FP types are not expected/supported (yet?)");
    return std::forward<T>(x);
  } else {
    static_assert(std::is_same_v<T, void>, "Something is wrong");
    return std::forward<T>(x);
  }
}

template <typename T>
using ConvertToOpenCLType_t = decltype(convertToOpenCLType(std::declval<T>()));

template <typename To, typename From> auto convertFromOpenCLTypeFor(From &&x) {
  if constexpr (std::is_same_v<To, bool> &&
                std::is_same_v<std::remove_reference_t<From>, bool>) {
    // FIXME: Something seems to be wrong elsewhere...
    return x;
  } else {
    using OpenCLType = decltype(convertToOpenCLType(std::declval<To>()));
    static_assert(std::is_same_v<std::remove_reference_t<From>, OpenCLType>);
    static_assert(sizeof(OpenCLType) == sizeof(To));
    using To_noref = std::remove_reference_t<To>;
    using From_noref = std::remove_reference_t<From>;
    if constexpr (is_vec_v<To_noref> && is_vec_v<From_noref>)
      return x.template as<To_noref>();
    else if constexpr (is_vec_v<To_noref> && is_ext_vector_v<From_noref>)
      return To_noref{bit_cast<typename To_noref::vector_t>(x)};
    else
      return static_cast<To>(x);
  }
}

// Helper function for concatenating two std::array.
template <typename T, std::size_t... Is1, std::size_t... Is2>
constexpr std::array<T, sizeof...(Is1) + sizeof...(Is2)>
ConcatArrays(const std::array<T, sizeof...(Is1)> &A1,
             const std::array<T, sizeof...(Is2)> &A2,
             std::index_sequence<Is1...>, std::index_sequence<Is2...>) {
  return {A1[Is1]..., A2[Is2]...};
}
template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> ConcatArrays(const std::array<T, N1> &A1,
                                              const std::array<T, N2> &A2) {
  return ConcatArrays(A1, A2, std::make_index_sequence<N1>(),
                      std::make_index_sequence<N2>());
}

// Utility for creating an std::array from the results of flattening the
// arguments using a flattening functor.
template <typename DataT, template <typename, typename> typename FlattenF,
          typename... ArgTN>
struct ArrayCreator;
template <typename DataT, template <typename, typename> typename FlattenF,
          typename ArgT, typename... ArgTN>
struct ArrayCreator<DataT, FlattenF, ArgT, ArgTN...> {
  static constexpr auto Create(const ArgT &Arg, const ArgTN &...Args) {
    auto ImmArray = FlattenF<DataT, ArgT>()(Arg);
    // Due to a bug in MSVC narrowing size_t to a bool in an if constexpr causes
    // warnings. To avoid this we add the comparison to 0.
    if constexpr (sizeof...(Args) > 0)
      return ConcatArrays(
          ImmArray, ArrayCreator<DataT, FlattenF, ArgTN...>::Create(Args...));
    else
      return ImmArray;
  }
};
template <typename DataT, template <typename, typename> typename FlattenF>
struct ArrayCreator<DataT, FlattenF> {
  static constexpr auto Create() { return std::array<DataT, 0>{}; }
};

// Helper function for creating an arbitrary sized array with the same value
// repeating.
template <typename T, size_t... Is>
static constexpr std::array<T, sizeof...(Is)>
RepeatValueHelper(const T &Arg, std::index_sequence<Is...>) {
  auto ReturnArg = [&](size_t) { return Arg; };
  return {ReturnArg(Is)...};
}
template <size_t N, typename T>
static constexpr std::array<T, N> RepeatValue(const T &Arg) {
  return RepeatValueHelper(Arg, std::make_index_sequence<N>());
}

#define __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES                                      \
  /* __swizzled_vec__ XYZW_ACCESS() const; */                                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N <= 4, x, 0)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2 || N == 3 || N == 4, y, 1)                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3 || N == 4, z, 2)                          \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, w, 3)                                    \
                                                                               \
  /* __swizzled_vec__ RGBA_ACCESS() const; */                                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, r, 0)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, g, 1)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, b, 2)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, a, 3)                                    \
                                                                               \
  /* __swizzled_vec__ INDEX_ACCESS() const; */                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 0, s0, 0)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 1, s1, 1)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 2, s2, 2)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 2, s3, 3)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 4, s4, 4)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 4, s5, 5)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 4, s6, 6)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N > 4, s7, 7)                                    \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, s8, 8)                                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, s9, 9)                                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, sA, 10)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, sB, 11)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, sC, 12)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, sD, 13)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, sE, 14)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, sF, 15)                                 \
                                                                               \
  /* __swizzled_vec__ lo()/hi() const; */                                      \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, lo, 0)                                   \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, lo, 0, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, lo, 0, 1)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, lo, 0, 1, 2, 3)                          \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, lo, 0, 1, 2, 3, 4, 5, 6, 7)             \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, hi, 1)                                   \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, hi, 2, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, hi, 2, 3)                                \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, hi, 4, 5, 6, 7)                          \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, hi, 8, 9, 10, 11, 12, 13, 14, 15)       \
  /* __swizzled_vec__ odd()/even() const; */                                   \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, odd, 1)                                  \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, odd, 1, 3)                               \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, odd, 1, 3)                               \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, odd, 1, 3, 5, 7)                         \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, odd, 1, 3, 5, 7, 9, 11, 13, 15)         \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 2, even, 0)                                 \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 3, even, 0, 2)                              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 4, even, 0, 2)                              \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 8, even, 0, 2, 4, 6)                        \
  __SYCL_SWIZZLE_MIXIN_METHOD(N == 16, even, 0, 2, 4, 6, 8, 10, 12, 14)        \
  /* Omitted SYCL_SIMPLE_SWIZZLES */

#define __SYCL_SWIZZLE_MIXIN_METHOD_NON_CONST(COND, NAME, ...)                 \
  template <int N = NumElements, typename Self_ = Self>                        \
  std::enable_if_t<                                                            \
      (COND), decltype(std::declval<Self_>().template swizzle<__VA_ARGS__>())> \
  NAME() {                                                                     \
    return static_cast<Self_ *>(this)->template swizzle<__VA_ARGS__>();        \
  }

#define __SYCL_SWIZZLE_MIXIN_METHOD_CONST(COND, NAME, ...)                     \
  template <int N = NumElements, typename Self_ = Self>                        \
  std::enable_if_t<(COND), decltype(std::declval<const Self_>()                \
                                        .template swizzle<__VA_ARGS__>())>     \
  NAME() const {                                                               \
    return static_cast<const Self_ *>(this)->template swizzle<__VA_ARGS__>();  \
  }

template <typename Self, int NumElements> struct NamedSwizzlesMixinConst {
#define __SYCL_SWIZZLE_MIXIN_METHOD(COND, NAME, ...)                           \
  __SYCL_SWIZZLE_MIXIN_METHOD_CONST(COND, NAME, __VA_ARGS__)

  __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES

#undef __SYCL_SWIZZLE_MIXIN_METHOD
};

template <typename Self, int NumElements> struct NamedSwizzlesMixinBoth {
#define __SYCL_SWIZZLE_MIXIN_METHOD(COND, NAME, ...)                           \
  __SYCL_SWIZZLE_MIXIN_METHOD_NON_CONST(COND, NAME, __VA_ARGS__)               \
  __SYCL_SWIZZLE_MIXIN_METHOD_CONST(COND, NAME, __VA_ARGS__)

  __SYCL_SWIZZLE_MIXIN_ALL_SWIZZLES

#undef __SYCL_SWIZZLE_MIXIN_METHOD
};

#undef __SYCL_SWIZZLE_MIXIN_METHOD_CONST
#undef __SYCL_SWIZZLE_MIXIN_METHOD_NON_CONST

} // namespace detail
} // namespace _V1
} // namespace sycl

#else
#include <sycl/access/access.hpp>
#include <sycl/aliases.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/generic_type_traits.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/half_type.hpp>

#include <sycl/detail/named_swizzles_mixin.hpp>
#endif

namespace sycl {
// TODO: Fix in the next ABI breaking windows.
enum class rounding_mode { automatic = 0, rte = 1, rtz = 2, rtp = 3, rtn = 4 };

inline namespace _V1 {
struct elem {
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};

template <typename DataT, int NumElements> class __SYCL_EBO vec;

namespace detail {
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
class __SYCL_EBO Swizzle;

template <typename Swizzle> struct is_assignable_swizzle;

template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
struct is_assignable_swizzle<Swizzle<IsConstVec, DataT, VecSize, Indexes...>> {
  static constexpr bool value = !IsConstVec && []() constexpr {
    int Idxs[] = {Indexes...};
    for (std::size_t i = 1; i < sizeof...(Indexes); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        if (Idxs[j] == Idxs[i])
          // Repeating index
          return false;
    }

    return true;
  }();
};

template <typename Swizzle>
constexpr bool is_assignable_swizzle_v = is_assignable_swizzle<Swizzle>::value;

// We need that trait when the type is still incomplete (inside mixin), so
// cannot deduce the property through the swizzle's `operator[]`.
template <typename Swizzle> struct is_over_const_vec_impl;

template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
struct is_over_const_vec_impl<Swizzle<IsConstVec, DataT, VecSize, Indexes...>>
    : std::bool_constant<IsConstVec> {};

template <typename Swizzle>
inline constexpr bool is_over_const_vec =
    is_over_const_vec_impl<Swizzle>::value;

#ifdef __SYCL_DEVICE_ONLY__
template <typename DataT>
using element_type_for_vector_t = typename detail::map_type<
    DataT,
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
    std::byte, /*->*/ std::uint8_t, //
#endif
    bool, /*->*/ std::uint8_t,                            //
    sycl::half, /*->*/ sycl::detail::half_impl::StorageT, //
    sycl::ext::oneapi::bfloat16,
    /*->*/ sycl::ext::oneapi::detail::Bfloat16StorageT, //
    char, /*->*/ detail::ConvertToOpenCLType_t<char>,   //
    DataT, /*->*/ DataT                                 //
    >::type;

// Type used for passing sycl::vec to SPIRV builtins.
// We can not use ext_vector_type(1) as it's not supported by SPIRV
// plugins (CTS fails).
template <typename DataT, int NumElements>
using vector_t =
    typename std::conditional_t<NumElements == 1,
                                element_type_for_vector_t<DataT>,
                                element_type_for_vector_t<DataT> __attribute__((
                                    ext_vector_type(NumElements)))>;
#endif // __SYCL_DEVICE_ONLY__

// Provide a class that can deduce element_type/size() from an incomplete type
// to be used in mixins like:
//
//   template <class Self>
//   struct AMixin : private from_incomplete<Self> {
//     /* `typename` is required with gcc and not clang /*
//     ... typename AMixin::element_type/AMixin::size() ...
//   };
//
// or via type alias
//
//   template <class Self>
//   class AMixin {
//     using element_type = typename from_incomplete<Self>::element_type;
//     ...
//   };
//
// NOTE: `AMixin` CANNOT use `DataT` as type alias because MSVC is buggy without
// `/permissive:-`, see https://godbolt.org/z/bMdn3hWds
//
//
// We'd like actual vec/swizle to `public`-inherit from this to avoid code
// duplication as well, but it's impossible due to `-Winaccessible-base`
// warning:
//
//   > direct base 'from_incomplete<vec<int, 2>>' is inaccessible due to
//   > ambiguity.
//
// I personally think it's meaningless, because this helper is eligible for
// Empty Bases Optimization meaning its size as a sub-object is zero and no
// members of it will ever be accessed (and `element_type`/`size()` don't result
// in an ill-formed code, meaning no errors are emitted for them).
template <typename T> struct from_incomplete;
template <typename T>
struct from_incomplete<const T> : public from_incomplete<T> {};

template <typename DataT, int NumElements>
struct from_incomplete<vec<DataT, NumElements>> {
  using element_type = DataT;
  static constexpr size_t size() { return NumElements; }

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = vector_t<DataT, size()>;
#endif
};

template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
struct from_incomplete<Swizzle<IsConstVec, DataT, VecSize, Indexes...>>
    : public from_incomplete<vec<DataT, sizeof...(Indexes)>> {};

template <class T, class U, class = void>
struct is_explicitly_convertible_to_impl : std::false_type {};

template <class T, class U>
struct is_explicitly_convertible_to_impl<
    T, U, std::void_t<decltype(static_cast<U>(std::declval<T>()))>>
    : std::true_type {};

template <class T, class U>
struct is_explicitly_convertible_to : is_explicitly_convertible_to_impl<T, U> {
};

template <class T, class U>
inline constexpr bool is_explicitly_convertible_to_v =
    is_explicitly_convertible_to<T, U>::value;

// Templated vs. non-templated conversion operator behaves differently when two
// conversions are needed as in the case below:
//
//   sycl::vec<int, 1> v;
//   std::ignore = static_cast<bool>(v);
//
// Make sure the snippet above compiles. That is important because
//
//   sycl::vec<int, 2> v;
//   if (v.x() == 42)
//     ...
//
// must go throw `v.x()` returning a swizzle, then its `operator==` returning
// vec<int, 1> and we want that code to compile.
enum class ConversionOpType {
  conv_regular,
  conv_explicit,
  conv_template,
  conv_explicit_template_convert
};
template <typename Self, typename To, ConversionOpType ConvType, bool Enable>
struct ConversionOperatorMixin {};
template <typename Self, typename To>
struct ConversionOperatorMixin<Self, To, ConversionOpType::conv_regular, true> {
  operator To() const {
    return static_cast<const Self *>(this)->template convertOperatorImpl<To>();
  }
};
template <typename Self, typename To>
struct ConversionOperatorMixin<Self, To, ConversionOpType::conv_explicit, true> {
  explicit operator To() const {
    return static_cast<const Self *>(this)->template convertOperatorImpl<To>();
  }
};
template <typename Self, typename To>
struct ConversionOperatorMixin<Self, To, ConversionOpType::conv_template, true> {
  template <class T, typename = std::enable_if_t<std::is_same_v<T, To>>>
  operator T() const {
    return static_cast<const Self *>(this)->template convertOperatorImpl<To>();
  }
};

// Only for vec/swizzle<DataT, 1>:
template <typename Self, typename To /* must be DataT */>
struct ConversionOperatorMixin<
    Self, To, ConversionOpType::conv_explicit_template_convert, true> {
  // FIXME: guard against byte and check the other is integral
  // TODO: probable remove swizzle/vec from T as well.
  template <class T,
            typename = std::enable_if_t<is_explicitly_convertible_to_v<To, T> &&
                                        !std::is_same_v<T, To>
#ifdef __SYCL_DEVICE_ONLY__
                                        && !std::is_same_v<T, vector_t<To, 1>>
#endif
                                        >>
  explicit operator T() const {
    return static_cast<const Self *>(this)->template convertOperatorImpl<T>();
  }
};

// Everything could have been much easier if we had C++20 concepts, then all the
// operators could be provided in a single mixin class with proper `requires`
// clauses on each overload. Until then, we have to have at least a separate
// mixing for each requirement (e.g. not byte, neither byte nor fp, not fp,
// etc.). Grouping like that would also be somewhat confusing, so we just create
// a separate mixin for each overload/narrow set of overloads and just "merge"
// them all back later.

template <typename SelfOperandTy, typename = void>
class IncDecMixin {};

template <typename SelfOperandTy>
class IncDecMixin<
    SelfOperandTy,
    std::enable_if_t<!std::is_same_v<
        bool, typename from_incomplete<SelfOperandTy>::element_type>>> {
  using element_type = typename from_incomplete<SelfOperandTy>::element_type;

public:
  friend SelfOperandTy &operator++(SelfOperandTy &x) {
    x += element_type{1};
    return x;
  }
  friend SelfOperandTy &operator--(SelfOperandTy &x) {
    x -= element_type{1};
    return x;
  }
  friend auto operator++(SelfOperandTy &x, int) {
    auto tmp = +x;
    x += element_type{1};
    return tmp;
  }
  friend auto operator--(SelfOperandTy &x, int) {
    auto tmp = +x;
    x -= element_type{1};
    return tmp;
  }
};

// TODO: The specification doesn't mention this specifically, but that's what
// the implementation has been doing and it seems to be a reasonable thing to
// do. Otherwise shift operators for byte element type would have to be disabled
// completely to follow C++ standard approach.
template <typename Self, typename = void>
struct ByteShiftsNonAssignMixin {};

template <typename SelfOperandTy, typename = void>
struct ByteShiftsOpAssignMixin {};

#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
template <typename Self>
struct ByteShiftsNonAssignMixin<
    Self, std::enable_if_t<std::is_same_v<
              std::byte, typename from_incomplete<Self>::element_type>>> {
  friend auto operator<<(const Self &lhs, int shift) {
    vec<typename from_incomplete<Self>::element_type,
        from_incomplete<Self>::size()>
        tmp;
    for (int i = 0; i < tmp.size(); ++i)
      tmp[i] = lhs[i] << shift;
    return tmp;
  }
  friend auto operator>>(const Self &lhs, int shift) {
    vec<typename from_incomplete<Self>::element_type,
        from_incomplete<Self>::size()>
        tmp;
    for (int i = 0; i < tmp.size(); ++i)
      tmp[i] = lhs[i] >> shift;
    return tmp;
  }
};

template <typename SelfOperandTy>
struct ByteShiftsOpAssignMixin<
    SelfOperandTy,
    std::enable_if_t<std::is_same_v<
        std::byte, typename from_incomplete<SelfOperandTy>::element_type>>> {
  friend SelfOperandTy &operator<<=(SelfOperandTy &lhs, int shift) {
    lhs = lhs << shift;
    return lhs;
  }
  friend SelfOperandTy &operator>>=(SelfOperandTy &lhs, int shift) {
    lhs = lhs >> shift;
    return lhs;
  }
};
#endif

// We use std::plus<void> and similar to "map" template parameter to an
// overloaded operator. These three below are missing from `<functional>`.
struct ShiftLeft {
  template <class T, class U>
  constexpr auto operator()(T &&lhs, U &&rhs) const
      -> decltype(std::forward<T>(lhs) << std::forward<U>(rhs)) {
    return std::forward<T>(lhs) << std::forward<U>(rhs);
  }
};
struct ShiftRight {
  template <class T, class U>
  constexpr auto operator()(T &&lhs,
                            U &&rhs) const -> decltype(std::forward<T>(lhs) >>
                                                       std::forward<U>(rhs)) {
    return std::forward<T>(lhs) >> std::forward<U>(rhs);
  }
};

struct UnaryPlus {
  template <class T>
  constexpr auto operator()(T &&arg) const -> decltype(+std::forward<T>(arg)) {
    return +std::forward<T>(arg);
  }
};

template <class T>
static constexpr bool not_fp =
    !std::is_same_v<T, float> && !std::is_same_v<T, double> &&
    !std::is_same_v<T, half> && !std::is_same_v<T, ext::oneapi::bfloat16>;

// Not using `is_byte_v` to avoid unnecessary dependencies on `half`/`bfloat16`
// headers.
template <class T>
static constexpr bool not_byte =
#if (!defined(_HAS_STD_BYTE) || _HAS_STD_BYTE != 0)
    !std::is_same_v<T, std::byte>;
#else
    true;
#endif

// To provide information about operators availability depending on vec/swizzle
// element type.
template <typename Op, typename T>
inline constexpr bool is_op_available_for_type = false;

#define __SYCL_OP_AVAILABILITY(OP, COND)                                       \
  template <typename T>                                                        \
  inline constexpr bool is_op_available_for_type<OP, T> = COND;

// clang-format off
__SYCL_OP_AVAILABILITY(std::plus<void>          , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::minus<void>         , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::multiplies<void>    , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::divides<void>       , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::modulus<void>       , not_byte<T> && not_fp<T>)

__SYCL_OP_AVAILABILITY(std::bit_and<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_or<void>        , not_fp<T>)
__SYCL_OP_AVAILABILITY(std::bit_xor<void>       , not_fp<T>)

__SYCL_OP_AVAILABILITY(std::equal_to<void>      , true)
__SYCL_OP_AVAILABILITY(std::not_equal_to<void>  , true)
__SYCL_OP_AVAILABILITY(std::less<void>          , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::greater<void>       , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::less_equal<void>    , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::greater_equal<void> , not_byte<T>)

__SYCL_OP_AVAILABILITY(std::logical_and<void>   , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::logical_or<void>    , not_byte<T>)

__SYCL_OP_AVAILABILITY(ShiftLeft                , not_byte<T> && not_fp<T>)
__SYCL_OP_AVAILABILITY(ShiftRight               , not_byte<T> && not_fp<T>)

// Unary
__SYCL_OP_AVAILABILITY(std::negate<void>        , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::logical_not<void>   , not_byte<T>)
__SYCL_OP_AVAILABILITY(std::bit_not<void>       , not_fp<T>)
__SYCL_OP_AVAILABILITY(UnaryPlus                , not_byte<T>)
// clang-format on

#undef __SYCL_OP_AVAILABILITY

template <typename Self, typename Op>
inline constexpr bool is_op_available =
    (from_incomplete<Self>::size() >= 1 &&
     is_op_available_for_type<Op,
                              typename from_incomplete<Self>::element_type>);

// Vector-specific part of the mixins' implementation.
struct VectorImpl {
private:
#ifdef __SYCL_DEVICE_ONLY__
  static constexpr bool is_host = false;
#else
  static constexpr bool is_host = true;
#endif

  template <typename T> static constexpr int num_elements() {
    if constexpr (is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

public:
  // Binop:
  template <typename T0, typename T1, typename OpTy>
  auto operator()(const T0 &Lhs, const T1 &Rhs, OpTy &&Op) {
    static_assert(std::is_same_v<get_elem_type_t<T0>, get_elem_type_t<T1>>);
    constexpr auto N = (std::max)(num_elements<T0>(), num_elements<T1>());
    using DataT = get_elem_type_t<T0>;
    constexpr bool is_logical =
        std::is_same_v<OpTy, std::equal_to<void>> ||
        std::is_same_v<OpTy, std::not_equal_to<void>> ||
        std::is_same_v<OpTy, std::less<void>> ||
        std::is_same_v<OpTy, std::greater<void>> ||
        std::is_same_v<OpTy, std::less_equal<void>> ||
        std::is_same_v<OpTy, std::greater_equal<void>> ||
        std::is_same_v<OpTy, std::logical_and<void>> ||
        std::is_same_v<OpTy, std::logical_or<void>>;
    auto Get = [](const auto &a, [[maybe_unused]] int idx) {
      if constexpr (is_vec_v<std::remove_const_t<
                        std::remove_reference_t<decltype(a)>>>)
        return a[idx];
      else
        return a;
    };
    using ResultVec =
        vec<std::conditional_t<
                is_logical, detail::select_cl_scalar_integral_signed_t<DataT>,
                DataT>,
            N>;
    if constexpr (is_host || std::is_same_v<DataT, ext::oneapi::bfloat16> ||
                  std::is_same_v<DataT, bool> || N == 1) {
      ResultVec tmp{};
      for (int i = 0; i < N; ++i)
        if constexpr (is_logical)
          tmp[i] = Op(Get(Lhs, i), Get(Rhs, i)) ? -1 : 0;
        else
          tmp[i] = Op(Get(Lhs, i), Get(Rhs, i));
      return tmp;
    } else {
      using vec_t = vec<DataT, N>;
      using vector_t = typename vec_t::vector_t;
      if constexpr (is_logical) {
        // Workaround a crash in the C++ front end, reported internally.
        constexpr bool no_crash =
            std::is_same_v<OpTy, std::logical_and<void>> ||
            std::is_same_v<OpTy, std::logical_or<void>>;
        if constexpr (no_crash) {
          auto res = Op(static_cast<vector_t>(vec_t{Lhs}),
                        static_cast<vector_t>(vec_t{Rhs}));
          // bit_cast is needed to cast between char/signed char
          // `ext_vector_type`s.
          //
          // TODO: Can we just change `vector_t`, or is that some mismatch
          // between clang/SPIR-V?
          return ResultVec{sycl::bit_cast<typename ResultVec::vector_t>(res)};
        } else {
          auto vec_lhs = static_cast<vector_t>(vec_t{Lhs});
          auto vec_rhs = static_cast<vector_t>(vec_t{Rhs});
          auto res = [&]() {
            if constexpr (std::is_same_v<OpTy, std::equal_to<void>>)
              return vec_lhs == vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::not_equal_to<void>>)
              return vec_lhs != vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::less<void>>)
              return vec_lhs < vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::greater<void>>)
              return vec_lhs > vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::less_equal<void>>)
              return vec_lhs <= vec_rhs;
            else if constexpr (std::is_same_v<OpTy, std::greater_equal<void>>)
              return vec_lhs >= vec_rhs;
            else
              static_assert(!std::is_same_v<OpTy, OpTy>, "Must be unreachable");
          }();
          // See the comment above.
          return ResultVec{sycl::bit_cast<typename ResultVec::vector_t>(res)};
        }
      } else {
        return ResultVec{Op(static_cast<vector_t>(vec_t{Lhs}),
                            static_cast<vector_t>(vec_t{Rhs}))};
      }
    }
  }

  // Unary op:
  template <typename T, typename OpTy> auto operator()(const T &X, OpTy &&Op) {
    static_assert(is_vec_v<T>);
    constexpr bool is_logical = std::is_same_v<OpTy, std::logical_not<void>>;
    if constexpr (is_logical) {
      vec<detail::select_cl_scalar_integral_signed_t<typename T::element_type>,
          T::size()>
          tmp;
      for (int i = 0; i < T::size(); ++i)
        tmp[i] = Op(X[i]) ? -1 : 0;
      return tmp;
    } else if constexpr (is_host ||
                         std::is_same_v<bool, typename T::element_type>) {
      T tmp;
      for (int i = 0; i < T::size(); ++i)
        tmp[i] = Op(X[i]);
      return tmp;
    } else {
      return T{Op(static_cast<typename T::vector_t>(X))};
    }
  }
};

// In swizzles, depending on the constness of the underlying vector and if
// swizzle indices are repeating or not, opassign operators might not be
// available for an operation even if such an operator can be overloaded (e.g.,
// `+`/`+=` vs `<`).
//
// While it's not the same in vec, we process vec mixins similarly to swizzles
// to unify the code, both between vec/swizzle, and between arithmetic/logical
// ops.

template <typename Self, bool OpAssign, typename Op, typename = void>
class SwizzleOpMixin {};

template <typename Self, bool OpAssign, typename Op, typename = void>
class VecOpMixin {};

#define __SYCL_BINARY_OP_MIXIN(OP, BINOP)                                      \
  template <typename Self>                                                     \
  class SwizzleOpMixin<Self, false, OP,                                        \
                       std::enable_if_t<is_op_available<Self, OP>>> {          \
    using element_type = typename from_incomplete<Self>::element_type;         \
    static constexpr int N = from_incomplete<Self>::size();                    \
                                                                               \
  public:                                                                      \
    template <typename T,                                                      \
              typename = std::enable_if_t<                                     \
                  std::is_convertible_v<T, element_type> && !is_swizzle_v<T>>> \
    friend auto operator BINOP(const Self &lhs, const T &rhs) {                \
      using Vec = vec<element_type, N>;                                        \
      return OP{}(Vec{lhs}, Vec{static_cast<element_type>(rhs)});              \
    }                                                                          \
    template <typename T,                                                      \
              typename = std::enable_if_t<                                     \
                  std::is_convertible_v<T, element_type> && !is_swizzle_v<T>>> \
    friend auto operator BINOP(const T &lhs, const Self &rhs) {                \
      using Vec = vec<element_type, N>;                                        \
      return OP{}(Vec{static_cast<element_type>(lhs)}, Vec{rhs});              \
    }                                                                          \
    friend auto operator BINOP(const Self &lhs,                                \
                               const vec<element_type, N> &rhs) {              \
      return OP{}(vec<element_type, N>{lhs}, rhs);                             \
    }                                                                          \
    friend auto operator BINOP(const vec<element_type, N> &lhs,                \
                               const Self &rhs) {                              \
      return OP{}(lhs, vec<element_type, N>{rhs});                             \
    }                                                                          \
    template <typename OtherSwizzle,                                           \
              typename = std::enable_if_t<is_swizzle_v<OtherSwizzle>>,         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<element_type,                                 \
                                 typename OtherSwizzle::element_type> &&       \
                  N == OtherSwizzle::size()>>                                  \
    friend auto operator BINOP(const Self &lhs, const OtherSwizzle &rhs) {     \
      using ResultVec = vec<element_type, N>;                                  \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
    /* Can't have both (Self, Swizzle) and (Swizzle, Self) enabled at the      \
     * same time if they use the same `const` as that would be ambiguous. As   \
     * such, only enable the latter if "constness" differs. */                 \
    template <typename OtherSwizzle,                                           \
              typename = std::enable_if_t<is_swizzle_v<OtherSwizzle>>,         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<element_type,                                 \
                                 typename OtherSwizzle::element_type> &&       \
                  N == OtherSwizzle::size() &&                                 \
                  is_over_const_vec<Self> != is_over_const_vec<OtherSwizzle>>> \
    friend auto operator BINOP(const OtherSwizzle &lhs, const Self &rhs) {     \
      using ResultVec = vec<element_type, N>;                                  \
      return OP{}(static_cast<ResultVec>(lhs), static_cast<ResultVec>(rhs));   \
    }                                                                          \
  };                                                                           \
  template <typename Self>                                                     \
  class VecOpMixin<Self, false, OP,                                            \
                   std::enable_if_t<is_op_available<Self, OP>>> {              \
    using element_type = typename from_incomplete<Self>::element_type;         \
    static constexpr int N = from_incomplete<Self>::size();                    \
                                                                               \
  public:                                                                      \
    template <typename T, typename = std::enable_if_t<                         \
                              std::is_convertible_v<T, element_type>>>         \
    friend auto operator BINOP(const Self &lhs, const T &rhs) {                \
      return OP{}(lhs, Self{static_cast<element_type>(rhs)});                  \
    }                                                                          \
    template <typename T, typename = std::enable_if_t<                         \
                              std::is_convertible_v<T, element_type>>>         \
    friend auto operator BINOP(const T &lhs, const Self &rhs) {                \
      return OP{}(Self{static_cast<element_type>(lhs)}, rhs);                  \
    }                                                                          \
    friend auto operator BINOP(const Self &lhs, const Self &rhs) {             \
      return VectorImpl{}(lhs, rhs, OP{});                                     \
    }                                                                          \
  };

#define __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(OP, BINOP, OPASSIGN)               \
  __SYCL_BINARY_OP_MIXIN(OP, BINOP)                                            \
  template <typename Self>                                                     \
  class SwizzleOpMixin<Self, true, OP,                                         \
                       std::enable_if_t<is_op_available<Self, OP>>> {          \
    using element_type = typename from_incomplete<Self>::element_type;         \
    static constexpr int N = from_incomplete<Self>::size();                    \
                                                                               \
  public:                                                                      \
    template <typename T,                                                      \
              typename = std::enable_if_t<                                     \
                  std::is_convertible_v<T, element_type> && !is_swizzle_v<T>>> \
    friend const Self &operator OPASSIGN(const Self & lhs, const T & rhs) {    \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
    friend const Self &operator OPASSIGN(const Self & lhs,                     \
                                         const vec<element_type, N> &rhs) {    \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
    template <typename OtherSwizzle,                                           \
              typename = std::enable_if_t<is_swizzle_v<OtherSwizzle>>,         \
              typename = std::enable_if_t<                                     \
                  std::is_same_v<element_type,                                 \
                                 typename OtherSwizzle::element_type> &&       \
                  N == OtherSwizzle::size()>>                                  \
    friend const Self &operator OPASSIGN(const Self & lhs,                     \
                                         const OtherSwizzle & rhs) {           \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
  };                                                                           \
  template <typename Self>                                                     \
  class VecOpMixin<Self, true, OP,                                             \
                   std::enable_if_t<is_op_available<Self, OP>>> {              \
    using element_type = typename from_incomplete<Self>::element_type;         \
    static constexpr int N = from_incomplete<Self>::size();                    \
                                                                               \
  public:                                                                      \
    template <typename T, typename = std::enable_if_t<                         \
                              std::is_convertible_v<T, element_type>>>         \
    friend Self &operator OPASSIGN(Self & lhs, const T & rhs) {                \
      lhs = OP{}(lhs, static_cast<element_type>(rhs));                         \
      return lhs;                                                              \
    }                                                                          \
    friend Self &operator OPASSIGN(Self & lhs, const Self & rhs) {             \
      lhs = OP{}(lhs, rhs);                                                    \
      return lhs;                                                              \
    }                                                                          \
  };

// There is no "OpAssign" version of the unary operators overloads, use "false"
// directly. That will leave "true" version without partial specialization and
// would use default empty implementation. That is important, becuase we only
// want the "false" one to provide the implementation to avoid ambiguity.
#define __SYCL_UNARY_OP_MIXIN(OP, UOP)                                         \
  template <typename Self>                                                     \
  class SwizzleOpMixin<Self, false, OP,                                        \
                       std::enable_if_t<is_op_available<Self, OP>>> {          \
    using element_type = typename from_incomplete<Self>::element_type;         \
    static constexpr int N = from_incomplete<Self>::size();                    \
                                                                               \
  public:                                                                      \
    friend auto operator UOP(const Self &x) {                                  \
      return OP{}(vec<element_type, N>{x});                                    \
    }                                                                          \
  };                                                                           \
  template <typename Self>                                                     \
  class VecOpMixin<Self, false, OP,                                            \
                   std::enable_if_t<is_op_available<Self, OP>>> {              \
    using element_type = typename from_incomplete<Self>::element_type;         \
    static constexpr int N = from_incomplete<Self>::size();                    \
                                                                               \
  public:                                                                      \
    friend auto operator UOP(const Self &x) { return VectorImpl{}(x, OP{}); }  \
  };

// clang-format off
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::plus<void>       , +, +=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::minus<void>      , -, -=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::multiplies<void> , *, *=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::divides<void>    , /, /=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::modulus<void>    , %, %=)

  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_and<void>    , &, &=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_or<void>     , |, |=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(std::bit_xor<void>    , ^, ^=)

  __SYCL_BINARY_OP_MIXIN(std::equal_to<void>                , ==)
  __SYCL_BINARY_OP_MIXIN(std::not_equal_to<void>            , !=)
  __SYCL_BINARY_OP_MIXIN(std::less<void>                    , <)
  __SYCL_BINARY_OP_MIXIN(std::greater<void>                 , >)
  __SYCL_BINARY_OP_MIXIN(std::less_equal<void>              , <=)
  __SYCL_BINARY_OP_MIXIN(std::greater_equal<void>           , >=)

  __SYCL_BINARY_OP_MIXIN(std::logical_and<void>             , &&)
  __SYCL_BINARY_OP_MIXIN(std::logical_or<void>              , ||)

  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(ShiftLeft             , <<, <<=)
  __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN(ShiftRight            , >>, >>=)

  __SYCL_UNARY_OP_MIXIN(std::negate<void>                   , -)
  __SYCL_UNARY_OP_MIXIN(std::logical_not<void>              , !)
  __SYCL_UNARY_OP_MIXIN(std::bit_not<void>                  , ~)
  __SYCL_UNARY_OP_MIXIN(UnaryPlus                           , +)
// clang-format on

#undef __SYCL_BINARY_OP_AND_OPASSIGN_MIXIN
#undef __SYCL_BINARY_OP_MIXIN
#undef __SYCL_UNARY_OP_MIXIN

// Now use individual per-operation mixins to create aggregated mixins that are
// easier to use.

// clang-format off
#define __SYCL_COMBINE_OP_MIXINS(MIXIN_TEMPLATE, ...)                    \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::plus<void>>,                   \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::minus<void>>,                  \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::multiplies<void>>,             \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::divides<void>>,                \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::modulus<void>>,                \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::bit_and<void>>,                \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::bit_or<void>>,                 \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::bit_xor<void>>,                \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::equal_to<void>>,               \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::not_equal_to<void>>,           \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::less<void>>,                   \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::greater<void>>,                \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::less_equal<void>>,             \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::greater_equal<void>>,          \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::logical_and<void>>,            \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::logical_or<void>>,             \
  public MIXIN_TEMPLATE<__VA_ARGS__, ShiftLeft>,                         \
  public MIXIN_TEMPLATE<__VA_ARGS__, ShiftRight>,                        \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::negate<void>>,                 \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::logical_not<void>>,            \
  public MIXIN_TEMPLATE<__VA_ARGS__, std::bit_not<void>>,                \
  public MIXIN_TEMPLATE<__VA_ARGS__, UnaryPlus>
// clang-format on

template <typename Self, bool EnableAssign>
struct __SYCL_EBO SwizzleOpsMixin
    : __SYCL_COMBINE_OP_MIXINS(SwizzleOpMixin, Self, EnableAssign) {};

template <typename Self>
struct __SYCL_EBO VecOpsMixin
    : __SYCL_COMBINE_OP_MIXINS(VecOpMixin, Self, false),
      __SYCL_COMBINE_OP_MIXINS(VecOpMixin, Self, true) {};

#undef __SYCL_COMBINE_OP_MIXINS

// Mixins infrastructure above is complete, now use these shared (vec/swizzle)
// mixins to define swizzle class.

template <typename Self,
#ifdef __SYCL_DEVICE_ONLY__
          typename vector_t = typename from_incomplete<Self>::vector_t,
#endif
          typename DataT = typename from_incomplete<Self>::element_type,
          int N = from_incomplete<Self>::size()>
struct __SYCL_EBO VecConversionsMixin :
#ifdef __SYCL_DEVICE_ONLY__
    public detail::ConversionOperatorMixin<
        Self, vector_t, ConversionOpType::conv_explicit,
        // if `vector_t` and `DataT` are the same, then the `operator DataT`
        // from the above is enough.
        !std::is_same_v<DataT, vector_t>>,
#endif
    public ConversionOperatorMixin<Self, DataT, ConversionOpType::conv_regular,
                                   /* Enable = */ N == 1>,
    public ConversionOperatorMixin<
        Self, DataT, ConversionOpType::conv_explicit_template_convert,
        /* Enable = */ (N == 1)> {
};

template <typename Self, bool AllowAssignOps = is_assignable_swizzle_v<Self>>
struct __SYCL_EBO SwizzleMixins
    : public NamedSwizzlesMixinConst<Self, from_incomplete<Self>::size()>,
      public SwizzleOpsMixin<Self, false>,
      public ByteShiftsNonAssignMixin<Self>,
      // Same conversions as in sycl::vec of the same size as the produced
      // swizzle.
      public VecConversionsMixin<Self>,
      // Conversion to sycl::vec, must be available only when `NumElements > 1`
      // per the SYCL 2020 specification:
      public ConversionOperatorMixin<
          Self,
          vec<typename from_incomplete<Self>::element_type,
              from_incomplete<Self>::size()>,
          ConversionOpType::conv_regular,
          /* Enable = */ true> {};

template <typename Self>
struct __SYCL_EBO SwizzleMixins<Self, true>
    : public SwizzleMixins<Self, false>,
      public SwizzleOpsMixin<Self, true>,
      public IncDecMixin<const Self>,
      public ByteShiftsOpAssignMixin<const Self> {};

template <int... Indexes>
inline constexpr bool has_repeating_indexes = []() constexpr {
  int Idxs[] = {Indexes...};
  for (std::size_t i = 1; i < sizeof...(Indexes); ++i) {
    for (std::size_t j = 0; j < i; ++j)
      if (Idxs[j] == Idxs[i])
        // Repeating index
        return true;
  }

  return false;
}();

template <typename Self, int VecSize, typename = void> class SwizzleBase {
  using DataT = typename from_incomplete<Self>::element_type;
  using VecT =
      std::conditional_t<is_over_const_vec<Self>, const vec<DataT, VecSize>,
                         vec<DataT, VecSize>>;

public:
  explicit SwizzleBase(VecT &Vec) : Vec(Vec) {}

  const Self &operator=(const Self &) = delete;

protected:
  VecT &Vec;
};

template <typename Self, int VecSize>
class SwizzleBase<Self, VecSize, std::enable_if_t<is_assignable_swizzle_v<Self>>> {
  using DataT = typename from_incomplete<Self>::element_type;
  using VecT =
      std::conditional_t<is_over_const_vec<Self>, const vec<DataT, VecSize>,
                         vec<DataT, VecSize>>;
  static constexpr int N = from_incomplete<Self>::size();

public:
  explicit SwizzleBase(VecT &Vec) : Vec(Vec) {}

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void load(size_t offset,
            multi_ptr<const DataT, AddressSpace, IsDecorated> ptr) const {
    vec<DataT, N> v;
    v.load(offset, ptr);
    *static_cast<Self *>(this) = v;
  }

  template <bool OtherIsConstVec, int OtherVecSize, int... OtherIndexes>
  std::enable_if_t<sizeof...(OtherIndexes) == N, const Self &>
  operator=(const Swizzle<OtherIsConstVec, DataT, OtherVecSize, OtherIndexes...>
                &rhs) {
    return (*this = static_cast<vec<DataT, N>>(rhs));
  }

  const Self &operator=(const vec<DataT, N> &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = rhs[i];

    return *static_cast<const Self *>(this);
  }

  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, DataT> &&
                                        !is_swizzle_v<T>>>
  const Self &operator=(const T &rhs) const {
    for (int i = 0; i < N; ++i)
      (*static_cast<const Self *>(this))[i] = static_cast<DataT>(rhs);

    return *static_cast<const Self *>(this);
  }

  // Default copy-assignment. Self's implicitly generated copy-assignment uses
  // this.
  //
  // We're templated on "Self", so each Swizzle has its own SwizzleBase and the
  // following is ok (1-to-1 bidirectional mapping between Self and its
  // SwizzleBase instantiation) even if a bit counterintuitive.
  const SwizzleBase &operator=(const SwizzleBase &rhs) const {
    const Self &self = (*static_cast<const Self *>(this));
    self = static_cast<vec<DataT, N>>(static_cast<const Self &>(rhs));
    return self;
  }

protected:
  VecT &Vec;
};

// Can't have sycl::vec anywhere in template parameters because that would bring
// its hidden friends into ADL.
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
class __SYCL_EBO Swizzle
    : public SwizzleBase<Swizzle<IsConstVec, DataT, VecSize, Indexes...>,
                         VecSize>,
      public SwizzleMixins<Swizzle<IsConstVec, DataT, VecSize, Indexes...>> {
  using Base =
      SwizzleBase<Swizzle<IsConstVec, DataT, VecSize, Indexes...>, VecSize>;
  static constexpr int NumElements = sizeof...(Indexes);
  using ResultVec = vec<DataT, NumElements>;

  // Get underlying vec index for (*this)[idx] access.
  static constexpr auto get_vec_idx(int idx) {
    int counter = 0;
    int result = -1;
    ((result = counter++ == idx ? Indexes : result), ...);
    return result;
  }

#ifdef __SYCL_DEVICE_ONLY__
public:
  using vector_t = typename vec<DataT, NumElements>::vector_t;

private:
#endif // __SYCL_DEVICE_ONLY__

  // This mixin calls `convertOperatorImpl` below so has to be a friend.
  template <typename Self, typename To, ConversionOpType ConvType, bool Enable>
  friend struct ConversionOperatorMixin;

  template <class To> To convertOperatorImpl() const {
    if constexpr (std::is_same_v<To, DataT> && NumElements == 1) {
      return (*this)[0];
    } else if constexpr (std::is_same_v<To, ResultVec>) {
      return ResultVec{this->Vec[Indexes]...};
#ifdef __SYCL_DEVICE_ONLY__
    } else if constexpr (std::is_same_v<To, vector_t>) {
      // operator ResultVec() isn't available for single-element swizzle, create
      // sycl::vec explicitly here.
      return static_cast<vector_t>(ResultVec{this->Vec[Indexes]...});
#endif
    } else {
      static_assert(is_explicitly_convertible_to_v<DataT, To> &&
                    NumElements == 1);
      return static_cast<To>((*this)[0]);
    }
  }

public:
  using Base::Base;
  using Base::operator=;

  using element_type = DataT;
  using value_type = DataT;

  Swizzle() = delete;
  Swizzle(const Swizzle &) = delete;

  static constexpr size_t byte_size() noexcept {
    return ResultVec::byte_size();
  }
  static constexpr size_t size() noexcept { return ResultVec::size(); }

  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  size_t get_size() const { return static_cast<ResultVec>(*this).get_size(); }

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const {
    return static_cast<ResultVec>(*this).get_count();
  };

  template <typename ConvertT,
            rounding_mode RoundingMode = rounding_mode::automatic>
  vec<ConvertT, NumElements> convert() const {
    return static_cast<ResultVec>(*this)
        .template convert<ConvertT, RoundingMode>();
  }

  template <typename asT> asT as() const {
    return static_cast<ResultVec>(*this).template as<asT>();
  }

  template <access::address_space AddressSpace, access::decorated IsDecorated>
  void store(size_t offset,
             multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const {
    return static_cast<ResultVec>(*this).store(offset, ptr);
  }

  template <int... swizzleIndexes> auto swizzle() const {
    return this->Vec.template swizzle<get_vec_idx(swizzleIndexes)...>();
  }

  auto &operator[](int index) const { return this->Vec[get_vec_idx(index)]; }
};
} // namespace detail

///////////////////////// class sycl::vec /////////////////////////
// Provides a cross-platform vector class template that works efficiently on
// SYCL devices as well as in host C++ code.
template <typename DataT, int NumElements>
class __SYCL_EBO vec
    : public detail::VecConversionsMixin<vec<DataT, NumElements>>,
      public detail::IncDecMixin<vec<DataT, NumElements>>,
      public detail::ByteShiftsNonAssignMixin<vec<DataT, NumElements>>,
      public detail::ByteShiftsOpAssignMixin<vec<DataT, NumElements>>,
      public detail::VecOpsMixin<vec<DataT, NumElements>>,
      public detail::NamedSwizzlesMixinBoth<vec<DataT, NumElements>,
                                            NumElements> {

  static_assert(NumElements == 1 || NumElements == 2 || NumElements == 3 ||
                    NumElements == 4 || NumElements == 8 || NumElements == 16,
                "Invalid number of elements for sycl::vec: only 1, 2, 3, 4, 8 "
                "or 16 are supported");
  static_assert(sizeof(bool) == sizeof(uint8_t), "bool size is not 1 byte");

  // https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#memory-layout-and-alignment
  // It is required by the SPEC to align vec<DataT, 3> with vec<DataT, 4>.
  static constexpr size_t AdjustedNum = (NumElements == 3) ? 4 : NumElements;

  // This represent type of underlying value. There should be only one field
  // in the class, so vec<float, 16> should be equal to float16 in memory.
  using DataType = std::array<DataT, AdjustedNum>;

#ifdef __SYCL_DEVICE_ONLY__
public:
  using vector_t = detail::vector_t<DataT, NumElements>;

private:
#endif // __SYCL_DEVICE_ONLY__

  template <typename Self, typename To, detail::ConversionOpType ConvType,
            bool Enable>
  friend struct detail::ConversionOperatorMixin;

  template <class To> To convertOperatorImpl() const {
    if constexpr (std::is_same_v<To, DataT> && NumElements == 1) {
      return m_Data[0];
#ifdef __SYCL_DEVICE_ONLY__
    } else if constexpr (std::is_same_v<To, vector_t>) {
      /* @SYCL2020
       * Available only when: compiled for the device.
       * Converts this SYCL vec instance to the underlying backend-native vector
       * type defined by vector_t.
       */
      return sycl::bit_cast<vector_t>(m_Data);
#endif
    } else {
      static_assert(detail::is_explicitly_convertible_to_v<DataT, To> &&
                    NumElements == 1);
      return static_cast<To>((*this)[0]);
    }
  }

  // Utility trait for creating an std::array from an vector argument.
  template <typename DataT_, typename T> class FlattenVecArg {
    template <std::size_t... Is>
    static constexpr auto helper(const T &V, std::index_sequence<Is...>) {
      return std::array{static_cast<DataT_>(V[Is])...};
    }

  public:
    constexpr auto operator()(const T &A) const {
      if constexpr (detail::is_vec_or_swizzle_v<T>) {
        return helper(A, std::make_index_sequence<T ::size()>());
      } else {
        return std::array{static_cast<DataT_>(A)};
      }
    }
  };

  // Alias for shortening the vec arguments to array converter.
  template <typename DataT_, typename... ArgTN>
  using VecArgArrayCreator =
      detail::ArrayCreator<DataT_, FlattenVecArg, ArgTN...>;

  template <int... Indexes>
  using Swizzle = detail::Swizzle<false, DataT, NumElements, Indexes...>;
  template <int... Indexes>
  using ConstSwizzle = detail::Swizzle<true, DataT, NumElements, Indexes...>;

  // Shortcuts for args validation in vec(const argTN &... args) ctor.
  template <typename CtorArgTy>
  static constexpr bool AllowArgTypeInVariadicCtor = []() constexpr {
    if constexpr (detail::is_vec_or_swizzle_v<CtorArgTy>) {
      return std::is_convertible_v<typename CtorArgTy::element_type, DataT>;
    } else {
      return std::is_convertible_v<CtorArgTy, DataT>;
    }
  }();

  template <typename T> static constexpr int num_elements() {
    if constexpr (detail::is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

  // Element type for relational operator return value.
  using rel_t = detail::select_cl_scalar_integral_signed_t<DataT>;

public:
  // Aliases required by SYCL 2020 to make sycl::vec consistent
  // with that of marray and buffer.
  using element_type = DataT;
  using value_type = DataT;

  /****************** Constructors **************/
  vec() = default;
  constexpr vec(const vec &Rhs) = default;
  constexpr vec(vec &&Rhs) = default;

private:
  // Implementation detail for the next public ctor. Note that for 3-elements
  // vector created from vector_t we use 4-elements array, potentially ignoring
  // the last padding element.
  template <typename Container, size_t... Is>
  constexpr vec(const Container &Arr, std::index_sequence<Is...>)
      : m_Data{Arr[Is]...} {}

  template <class T> struct type_identity {
    using type = T;
  };

public:
  // Explicit because replication isn't an obvious conversion.
  template <int N = NumElements, typename = std::enable_if_t<(N > 1)>>
  explicit constexpr vec(const DataT &arg)
      : vec{detail::RepeatValue<NumElements>(arg),
            std::make_index_sequence<NumElements>()} {}

  // Extra `void` to make this really different from the previous for the C++
  // compiler.
  template <int N = NumElements, typename = std::enable_if_t<(N == 1)>, typename = void>
  constexpr vec(const DataT &arg)
      : vec{detail::RepeatValue<NumElements>(arg),
            std::make_index_sequence<NumElements>()} {}

  // Constructor from values of base type or vec of base type. Checks that
  // base types are match and that the NumElements == sum of lengths of args.
  template <
      typename... argTN,
      typename = std::enable_if_t<
          (NumElements > 1 && ((AllowArgTypeInVariadicCtor<argTN> && ...)) &&
           ((num_elements<argTN>() + ...)) == NumElements)>>
  constexpr vec(const argTN &...args)
      : vec{VecArgArrayCreator<DataT, argTN...>::Create(args...),
            std::make_index_sequence<NumElements>()} {}

  /****************** Assignment Operators **************/
  constexpr vec &operator=(const vec &Rhs) = default;

  template <typename T,
            typename = std::enable_if_t<std::is_convertible_v<T, DataT>>>
  vec &operator=(const T &Rhs) {
    *this = vec{static_cast<DataT>(Rhs)};
    return *this;
  }

#ifdef __SYCL_DEVICE_ONLY__
 public:
   template <typename vector_t_ = vector_t,
             typename = std::enable_if_t<!std::is_same_v<vector_t_, DataT>>>
   // TODO: current draft would use non-template `vector_t` as an operand,
   // causing sycl::vec<sycl::half, N>{1} to go through different paths on
   // host/device, open question in the specification.
   explicit vec(vector_t openclVector)
       // FIXME: Doesn't work when instantiated for 3-elements vectors,
       // indetermined padding can't be used to initialize constexpr std::array
       // storage.
       : vec(bit_cast<DataType>(openclVector),
             std::make_index_sequence<AdjustedNum>()) {}
#endif // __SYCL_DEVICE_ONLY__

  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  static constexpr size_t get_count() { return size(); }
  static constexpr size_t size() noexcept { return NumElements; }
  __SYCL2020_DEPRECATED(
      "get_size() is deprecated, please use byte_size() instead")
  static constexpr size_t get_size() { return byte_size(); }
  static constexpr size_t byte_size() noexcept { return sizeof(m_Data); }

public:
  // Out-of-class definition is in `sycl/detail/vector_convert.hpp`
  template <typename convertT,
            rounding_mode roundingMode = rounding_mode::automatic>
  vec<convertT, NumElements> convert() const;

  template <typename asT> asT as() const { return sycl::bit_cast<asT>(*this); }

private:
  static constexpr bool one_elem_swizzle_return_scalar = false;

public:
  template <int... SwizzleIndexes>
  std::conditional_t<sizeof...(SwizzleIndexes) == 1 &&
                         one_elem_swizzle_return_scalar,
                     DataT &, Swizzle<SwizzleIndexes...>>
  swizzle() {
    if constexpr (sizeof...(SwizzleIndexes) == 1 &&
                  one_elem_swizzle_return_scalar)
      return this->operator[](SwizzleIndexes...);
    else
      return Swizzle<SwizzleIndexes...>{*this};
  }

  template <int... SwizzleIndexes>
  std::conditional_t<sizeof...(SwizzleIndexes) == 1 &&
                         one_elem_swizzle_return_scalar,
                     const DataT &, ConstSwizzle<SwizzleIndexes...>>
  swizzle() const {
    if constexpr (sizeof...(SwizzleIndexes) == 1 &&
                  one_elem_swizzle_return_scalar)
      return this->operator[](SwizzleIndexes...);
    else
      return ConstSwizzle<SwizzleIndexes...>{*this};
  }

  const DataT &operator[](int i) const { return m_Data[i]; }

  DataT &operator[](int i) { return m_Data[i]; }

  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    for (int I = 0; I < NumElements; I++) {
      m_Data[I] = *multi_ptr<const DataT, Space, DecorateAddress>(
          Ptr + Offset * NumElements + I);
    }
  }
  template <access::address_space Space, access::decorated DecorateAddress>
  void load(size_t Offset, multi_ptr<DataT, Space, DecorateAddress> Ptr) {
    multi_ptr<const DataT, Space, DecorateAddress> ConstPtr(Ptr);
    load(Offset, ConstPtr);
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  load(size_t Offset,
       accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
           Acc) {
    multi_ptr<const DataT, detail::TargetToAS<Target>::AS,
              access::decorated::yes>
        MultiPtr(Acc);
    load(Offset, MultiPtr);
  }
  void load(size_t Offset, const DataT *Ptr) {
    for (int I = 0; I < NumElements; ++I)
      m_Data[I] = Ptr[Offset * NumElements + I];
  }

  template <access::address_space Space, access::decorated DecorateAddress>
  void store(size_t Offset,
             multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    for (int I = 0; I < NumElements; I++) {
      *multi_ptr<DataT, Space, DecorateAddress>(Ptr + Offset * NumElements +
                                                I) = m_Data[I];
    }
  }
  template <int Dimensions, access::mode Mode,
            access::placeholder IsPlaceholder, access::target Target,
            typename PropertyListT>
  void
  store(size_t Offset,
        accessor<DataT, Dimensions, Mode, Target, IsPlaceholder, PropertyListT>
            Acc) {
    multi_ptr<DataT, detail::TargetToAS<Target>::AS, access::decorated::yes>
        MultiPtr(Acc);
    store(Offset, MultiPtr);
  }
  void store(size_t Offset, DataT *Ptr) const {
    for (int I = 0; I < NumElements; ++I)
      Ptr[Offset * NumElements + I] = m_Data[I];
  }

private:
  // fields
  // Alignment is the same as size, to a maximum size of 64. SPEC requires
  // "The elements of an instance of the SYCL vec class template are stored
  // in memory sequentially and contiguously and are aligned to the size of
  // the element type in bytes multiplied by the number of elements."
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;
  // friends
  template <typename T1, int T2> friend class __SYCL_EBO vec;
};

static_assert(sizeof(vec<int, 2>) == 2 * sizeof(int),
              "Empty Bases Optimization didn't work!");
///////////////////////// class sycl::vec /////////////////////////

#ifdef __cpp_deduction_guides
// all compilers supporting deduction guides also support fold expressions
template <class T, class... U,
          class = std::enable_if_t<(std::is_same_v<T, U> && ...)>>
vec(T, U...) -> vec<T, sizeof...(U) + 1>;
#endif

} // namespace _V1
} // namespace sycl
