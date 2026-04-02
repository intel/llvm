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

#include <sycl/detail/builtins/gentype_utilities.hpp>
#include <sycl/detail/builtins/scalar_infrastructure.hpp>
#include <sycl/detail/memcpy.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {
namespace detail {

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
