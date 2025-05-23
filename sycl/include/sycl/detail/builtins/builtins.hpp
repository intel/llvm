//==------------------- builtins.hpp ---------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implement SYCL builtin functions. This implementation is mainly driven by the
// requirement of not including <cmath> anywhere in the SYCL headers (i.e. from
// within <sycl/sycl.hpp>), because it pollutes global namespace. Note that we
// can avoid that using MSVC's STL as the pollution happens even from
// <vector>/<string> and other headers that have to be included per the SYCL
// specification. As such, an alternative approach might be to use math
// intrinsics with GCC/clang-based compilers and use <cmath> when using MSVC as
// a host compiler. That hasn't been tried/investigated.
//
// Current implementation splits builtins into several files following the SYCL
// 2020 (revision 8) split into common/math/geometric/relational/etc. functions.
// For each set, the implementation is split into a user-visible
// include/sycl/detail/builtins/*_functions.hpp providing full device-side
// implementation as well as defining user-visible APIs and defining ABI
// implemented under source/builtins/*_functions.cpp for the host side. We
// provide both scalar/vector overloads through symbols in the SYCL runtime
// library due to the <cmath> limitation above (for scalars) and due to
// performance reasons for vector overloads (to be able to benefit from
// vectorization).
//
// Providing declaration for the host side symbols contained in the library
// comes with its own challenges. One is compilation time - blindly providing
// all those declarations takes significant time (about 10% slowdown for
// "clang++ -fsycl" when compiling just "#include <sycl/sycl.hpp>"). Another
// issue is that return type for templates is part of the mangling (and as such
// SFINAE requirements too). To overcome that we structure host side
// implementation roughly like this (in most cases):
//
// math_function.cpp exports:
//   float sycl::__sin_impl(float);
//   float1 sycl::__sin_impl(float1);
//   float2 sycl::__sin_impl(float2);
//   ...
//   /* same for other types */
//
// math_functions.hpp provide an implementation based on the following idea (in
// ::sycl namespace):
//   float sin(float x) {
//     extern __sin_impl(float);
//     return __sin_impl(x);
//   }
//   template <typename T>
//   enable_if_valid_type<T> sin(T x) {
//     if constexpr (marray_or_swizzle) {
//       ...
//       call sycl::sin(vector_or_scalar)
//     } else {
//       extern T __sin_impl(T);
//       return __sin_impl(x);
//     }
//   }
// That way we avoid having the full set of explicit declaration for the symbols
// in the library and instead only pay with compile time when those template
// instantiations actually happen.

#pragma once

#include <sycl/detail/type_traits.hpp>
#include <sycl/detail/type_traits/vec_marray_traits.hpp>
#include <sycl/detail/vector_convert.hpp>
#include <sycl/marray.hpp> // for marray
#include <sycl/vector.hpp> // for vec

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

// Utility functions for converting to/from vec/marray.
template <class T, size_t N> vec<T, 2> to_vec2(marray<T, N> X, size_t Start) {
  return {X[Start], X[Start + 1]};
}
template <class T, size_t N> vec<T, N> to_vec(marray<T, N> X) {
  vec<T, N> Vec;
  for (size_t I = 0; I < N; I++)
    Vec[I] = X[I];
  return Vec;
}
template <class T, int N> marray<T, N> to_marray(vec<T, N> X) {
  marray<T, N> Marray;
  for (size_t I = 0; I < N; I++)
    Marray[I] = X[I];
  return Marray;
}

namespace builtins {
#ifdef __SYCL_DEVICE_ONLY__
template <typename T> auto convert_arg(T &&x) {
  using no_cv_ref = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (is_vec_v<no_cv_ref>) {
    using elem_type = get_elem_type_t<no_cv_ref>;
    using converted_elem_type =
        decltype(convert_arg(std::declval<elem_type>()));

    constexpr auto N = no_cv_ref::size();
    using result_type = std::conditional_t<N == 1, converted_elem_type,
                                           converted_elem_type
                                           __attribute__((ext_vector_type(N)))>;
    return bit_cast<result_type>(x);
  } else if constexpr (is_swizzle_v<no_cv_ref>) {
    return convert_arg(simplify_if_swizzle_t<no_cv_ref>{x});
  } else {
    static_assert(is_scalar_arithmetic_v<no_cv_ref> ||
                  is_multi_ptr_v<no_cv_ref> || std::is_pointer_v<no_cv_ref> ||
                  std::is_same_v<no_cv_ref, half>);
    return convertToOpenCLType(std::forward<T>(x));
  }
}
#endif
} // namespace builtins

template <typename FuncTy, typename... Ts>
auto builtin_marray_impl(FuncTy F, const Ts &...x) {
  using ret_elem_type = decltype(F(x[0]...));
  using T = typename first_type<Ts...>::type;
  marray<ret_elem_type, T::size()> Res;
  constexpr auto N = T::size();
  for (size_t I = 0; I < N / 2; ++I) {
    auto PartialRes = [&]() {
      using elem_ty = get_elem_type_t<T>;
      if constexpr (std::is_integral_v<elem_ty>)
        return F(
            to_vec2(x, I * 2)
                .template as<vec<
                    std::conditional_t<std::is_signed_v<elem_ty>,
                                       fixed_width_signed<sizeof(elem_ty)>,
                                       fixed_width_unsigned<sizeof(elem_ty)>>,
                    2>>()...);
      else
        return F(to_vec2(x, I * 2)...);
    }();
    sycl::detail::memcpy_no_adl(&Res[I * 2], &PartialRes,
                                sizeof(decltype(PartialRes)));
  }
  if (N % 2)
    Res[N - 1] = F(x[N - 1]...);
  return Res;
}

template <typename FuncTy, typename... Ts>
auto builtin_default_host_impl(FuncTy F, const Ts &...x) {
  // We implement support for marray/swizzle in the headers and export symbols
  // for scalars/vector from the library binary. The reason is that scalar
  // implementations mostly depend on <cmath> which pollutes global namespace,
  // so we can't unconditionally include it from the SYCL headers. Vector
  // overloads have to be implemented in the library next to scalar overloads in
  // order to be vectorizable.
  if constexpr ((... || is_marray_v<Ts>)) {
    return builtin_marray_impl(F, x...);
  } else {
    return F(simplify_if_swizzle_t<Ts>{x}...);
  }
}

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
    for (size_t i = 0; i < Size; ++i)
      r[i] = F(x[i]...);
  } else {
    loop<Size>([&](auto idx) { r[idx] = F(x[idx]...); });
  }

  return r;
}

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

/*
The headers below are specifically implemented without including all the
necessary headers to allow preprocessing them on their own and providing
human-friendly result. One can use a command like this to achieve that:
clang++ -[DU]__SYCL_DEVICE_ONLY__ -x c++ math_functions.inc  \
        -I <..>/llvm/sycl/include -E -o - \
    | grep -v '^#' | clang-format > math_functions.{host|device}.ii
*/

#include <sycl/detail/builtins/common_functions.inc>
#include <sycl/detail/builtins/geometric_functions.inc>
#include <sycl/detail/builtins/half_precision_math_functions.inc>
#include <sycl/detail/builtins/integer_functions.inc>
#include <sycl/detail/builtins/math_functions.inc>
#include <sycl/detail/builtins/native_math_functions.inc>
#include <sycl/detail/builtins/relational_functions.inc>
