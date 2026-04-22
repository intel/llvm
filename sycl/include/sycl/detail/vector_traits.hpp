//===--- vector_traits.hpp - Shared support for SYCL vec headers
//-----------===//
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

// See vec::DataType definitions for more details.
#ifndef __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
#define __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE !__SYCL_USE_LIBSYCL8_VEC_IMPL
#endif

#if !defined(__HAS_EXT_VECTOR_TYPE__) && defined(__SYCL_DEVICE_ONLY__)
#error "SYCL device compiler is built without ext_vector_type support"
#endif

#include <sycl/aliases.hpp>

#include <cstddef>
#include <type_traits>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename T> struct from_incomplete;
template <typename T>
struct from_incomplete<const T> : public from_incomplete<T> {};

template <typename DataT, int NumElements>
struct from_incomplete<vec<DataT, NumElements>> {
  using element_type = DataT;
  static constexpr size_t size() { return NumElements; }
};

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
namespace hide_swizzle_from_adl {
template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
class Swizzle;
} // namespace hide_swizzle_from_adl

template <bool IsConstVec, typename DataT, int VecSize, int... Indexes>
struct from_incomplete<
    hide_swizzle_from_adl::Swizzle<IsConstVec, DataT, VecSize, Indexes...>> {
  using element_type = DataT;
  static constexpr size_t size() { return sizeof...(Indexes); }

  using vec_ty = std::conditional_t<IsConstVec, const vec<DataT, VecSize>,
                                    vec<DataT, VecSize>>;
  using result_vec_ty = vec<DataT, size()>;
  static constexpr int vec_size = VecSize;
  static constexpr bool is_over_const_vec = IsConstVec;
  static constexpr bool has_repeating_indexes = []() constexpr {
    int Idxs[] = {Indexes...};
    for (std::size_t i = 1; i < sizeof...(Indexes); ++i) {
      for (std::size_t j = 0; j < i; ++j)
        if (Idxs[j] == Idxs[i])
          return true;
    }

    return false;
  }();
  static constexpr bool is_assignable = !IsConstVec && !has_repeating_indexes;
};
#endif

template <bool Cond, typename Mixin> struct ApplyIf {};
template <typename Mixin> struct ApplyIf<true, Mixin> : Mixin {};

template <typename Self> struct vec_ops_base {
  struct Combined {};
};
template <typename Self> struct swizzle_ops_base {
  struct Combined {};
};

template <typename DataT, int NumElements> class vec_arith;

} // namespace detail
} // namespace _V1
} // namespace sycl
