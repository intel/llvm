//==------------- gentype_utilities.hpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builtins/scalar_infrastructure.hpp>
#include <sycl/detail/memcpy.hpp>

#ifndef SYCL_DETAIL_BUILTINS_SCALAR_ONLY
#include <sycl/detail/vector_convert.hpp>
#include <sycl/marray.hpp>
#include <sycl/multi_ptr.hpp>
#include <sycl/vector.hpp>
#endif

namespace sycl {
inline namespace _V1 {
namespace detail {

// Utility functions for converting to/from vec/marray.
#ifndef SYCL_DETAIL_BUILTINS_SCALAR_ONLY
template <class T, size_t N> vec<T, 2> to_vec2(marray<T, N> X, size_t Start) {
  return {X[Start], X[Start + 1]};
}
template <class T, size_t N> vec<T, N> to_vec(marray<T, N> X) {
  vec<T, N> Vec;
  for (size_t I = 0; I < N; I++) {
    Vec[I] = X[I];
  }
  return Vec;
}
template <class T, int N> marray<T, N> to_marray(vec<T, N> X) {
  marray<T, N> Marray;
  for (size_t I = 0; I < N; I++) {
    Marray[I] = X[I];
  }
  return Marray;
}
#else
template <class T, size_t N> void to_vec2(marray<T, N>, size_t) = delete;
template <class T, size_t N> void to_vec(marray<T, N>) = delete;
template <class T, int N> void to_marray(vec<T, N>) = delete;
#endif

#ifndef SYCL_DETAIL_BUILTINS_SCALAR_ONLY
template <typename FuncTy, typename... Ts>
auto builtin_marray_impl(FuncTy F, const Ts &...x) {
  using ret_elem_type = decltype(F(x[0]...));
  using T = typename first_type<Ts...>::type;
  marray<ret_elem_type, T::size()> Res;
  constexpr auto N = T::size();
  for (size_t I = 0; I < N / 2; ++I) {
    auto PartialRes = [&]() {
      using elem_ty = get_elem_type_t<T>;
      if constexpr (std::is_integral_v<elem_ty>) {
        return F(
            to_vec2(x, I * 2)
                .template as<vec<
                    std::conditional_t<std::is_signed_v<elem_ty>,
                                       fixed_width_signed<sizeof(elem_ty)>,
                                       fixed_width_unsigned<sizeof(elem_ty)>>,
                    2>>()...);
      } else {
        return F(to_vec2(x, I * 2)...);
      }
    }();
    sycl::detail::memcpy_no_adl(&Res[I * 2], &PartialRes,
                                sizeof(decltype(PartialRes)));
  }
  if (N % 2) {
    Res[N - 1] = F(x[N - 1]...);
  }
  return Res;
}
#else
template <typename FuncTy, typename... Ts>
void builtin_marray_impl(FuncTy, const Ts &...) = delete;
#endif

#ifndef SYCL_DETAIL_BUILTINS_SCALAR_ONLY
template <typename FuncTy, typename... Ts>
auto builtin_delegate_to_scalar(FuncTy F, const Ts &...x) {
  using T = typename first_type<Ts...>::type;
  static_assert(is_vec_or_swizzle_v<T> || is_marray_v<T>);

  constexpr auto Size = T::size();
  using ret_elem_type = decltype(F(x[0]...));
  std::conditional_t<is_marray_v<T>, marray<ret_elem_type, Size>,
                     vec<ret_elem_type, Size>>
      r{};

  if constexpr (is_marray_v<T>) {
    for (size_t i = 0; i < Size; ++i) {
      r[i] = F(x[i]...);
    }
  } else {
    loop<Size>([&](auto idx) { r[idx] = F(x[idx]...); });
  }

  return r;
}
#else
template <typename FuncTy, typename... Ts>
void builtin_delegate_to_scalar(FuncTy, const Ts &...) = delete;
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl