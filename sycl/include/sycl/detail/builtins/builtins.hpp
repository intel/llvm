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

/*
The headers below are split by builtin function category and can be
preprocessed individually to inspect the generated host/device variants. One
can use a command like this to achieve that:
clang++ -[DU]__SYCL_DEVICE_ONLY__ -x c++ math_functions.hpp  \
    -I <..>/llvm/sycl/include -E -o - \
  | grep -v '^#' | clang-format > math_functions.{host|device}.ii
*/

#include <sycl/detail/builtins/common_functions.hpp>
#include <sycl/detail/builtins/geometric_functions.hpp>
#include <sycl/detail/builtins/half_precision_math_functions.hpp>
#include <sycl/detail/builtins/integer_functions.hpp>
#include <sycl/detail/builtins/math_functions.hpp>
#include <sycl/detail/builtins/native_math_functions.hpp>
#include <sycl/detail/builtins/relational_functions.hpp>
