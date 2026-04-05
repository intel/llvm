//==----------------- vector_base.hpp - vec storage helpers ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/common.hpp>
#include <sycl/detail/type_traits/vec_marray_traits.hpp>

#include <algorithm>

#ifndef __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
#define __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE !__SYCL_USE_LIBSYCL8_VEC_IMPL
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename T1, int T2> class vec_base_test;

template <typename DataT, int NumElements> class vec_base {
  static constexpr size_t AdjustedNum = (NumElements == 3) ? 4 : NumElements;
  using DataType = std::conditional_t<
#if __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
      true,
#else
      sizeof(std::array<DataT, AdjustedNum>) == sizeof(DataT[AdjustedNum]) &&
          alignof(std::array<DataT, AdjustedNum>) ==
              alignof(DataT[AdjustedNum]),
#endif
      DataT[AdjustedNum], std::array<DataT, AdjustedNum>>;

  template <typename T1, int T2> friend class detail::vec_base_test;

protected:
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;

  template <size_t... Is>
  constexpr vec_base(const DataT &Val, std::index_sequence<Is...>)
      : m_Data{((void)Is, Val)...} {}

  template <size_t... Is>
  constexpr vec_base(const std::array<DataT, NumElements> &Arr,
                     std::index_sequence<Is...>)
      : m_Data{Arr[Is]...} {}

  template <typename CtorArgTy>
  static constexpr bool AllowArgTypeInVariadicCtor = []() constexpr {
    if constexpr (std::is_convertible_v<CtorArgTy, DataT>) {
      return true;
    } else if constexpr (is_vec_or_swizzle_v<CtorArgTy>) {
      if constexpr (CtorArgTy::size() == 1 &&
                    std::is_convertible_v<typename CtorArgTy::element_type,
                                          DataT>) {
        return true;
      }
      return std::is_same_v<typename CtorArgTy::element_type, DataT>;
    } else {
      return false;
    }
  }();

  template <typename T> static constexpr int num_elements() {
    if constexpr (is_vec_or_swizzle_v<T>)
      return T::size();
    else
      return 1;
  }

  template <typename DataT_, typename T> class FlattenVecArg {
    template <std::size_t... Is>
    static constexpr auto helper(const T &V, std::index_sequence<Is...>) {
#if __SYCL_USE_LIBSYCL8_VEC_IMPL
      if constexpr (is_swizzle_v<T>)
        return std::array{static_cast<DataT_>(V.getValue(Is))...};
      else
#endif
        return std::array{static_cast<DataT_>(V[Is])...};
    }

  public:
    constexpr auto operator()(const T &A) const {
      if constexpr (is_vec_or_swizzle_v<T>) {
        return helper(A, std::make_index_sequence<T::size()>());
      } else {
        return std::array{static_cast<DataT_>(A)};
      }
    }
  };

  template <typename DataT_, typename... ArgTN>
  using VecArgArrayCreator = ArrayCreator<DataT_, FlattenVecArg, ArgTN...>;

public:
  constexpr vec_base() = default;
  constexpr vec_base(const vec_base &) = default;
  constexpr vec_base(vec_base &&) = default;
  constexpr vec_base &operator=(const vec_base &) = default;
  constexpr vec_base &operator=(vec_base &&) = default;

  explicit constexpr vec_base(const DataT &arg)
      : vec_base(arg, std::make_index_sequence<NumElements>()) {}

  template <typename... argTN,
            typename = std::enable_if_t<
                ((AllowArgTypeInVariadicCtor<argTN> && ...)) &&
                ((num_elements<argTN>() + ...)) == NumElements>>
  constexpr vec_base(const argTN &...args)
      : vec_base{VecArgArrayCreator<DataT, argTN...>::Create(args...),
                 std::make_index_sequence<NumElements>()} {}
};

#if !__SYCL_USE_LIBSYCL8_VEC_IMPL
template <typename DataT> class vec_base<DataT, 1> {
  using DataType = std::conditional_t<
#if __SYCL_USE_PLAIN_ARRAY_AS_VEC_STORAGE
      true,
#else
      sizeof(std::array<DataT, 1>) == sizeof(DataT[1]) &&
          alignof(std::array<DataT, 1>) == alignof(DataT[1]),
#endif
      DataT[1], std::array<DataT, 1>>;

protected:
  static constexpr int alignment = (std::min)((size_t)64, sizeof(DataType));
  alignas(alignment) DataType m_Data;

public:
  constexpr vec_base() = default;
  constexpr vec_base(const vec_base &) = default;
  constexpr vec_base(vec_base &&) = default;
  constexpr vec_base &operator=(const vec_base &) = default;
  constexpr vec_base &operator=(vec_base &&) = default;

  constexpr vec_base(const DataT &arg) : m_Data{{arg}} {}
};
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl