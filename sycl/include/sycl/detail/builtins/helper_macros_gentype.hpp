//==-- helper_macros_gentype.hpp -- Gentype helper macros for sycl builtins ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/builtins/helper_macros.hpp>

// Use (NAME)/(NS::NAME) to deal win min/max macros in windows.h throughout
// this file.

#define DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(                                  \
    NUM_ARGS, NAME, ENABLER, DELEGATOR, NS, /*SCALAR_VEC_IMPL*/...)            \
  template <NUM_ARGS##_TYPENAME_TYPE>                                          \
  detail::ENABLER<NUM_ARGS##_TEMPLATE_TYPE>(NAME)(                             \
      NUM_ARGS##_TEMPLATE_TYPE_ARG) {                                          \
    using ToTy = detail::ENABLER<NUM_ARGS##_TEMPLATE_TYPE>;                    \
    if constexpr (detail::is_marray_v<T0>) {                                   \
      return detail::DELEGATOR(                                                \
          [](NUM_ARGS##_AUTO_ARG) { return (NS::NAME)(NUM_ARGS##_ARG); },      \
          NUM_ARGS##_ARG);                                                     \
    } else if constexpr (detail::is_vec_v<ToTy>) {                             \
      if constexpr (ToTy::size() == 3) {                                       \
        /* For vectors of length 3, make sure to only copy 3 elements, not 4,  \
           to work around code generation issues, see LLVM #144454. */         \
        auto From = __VA_ARGS__(NUM_ARGS##_CONVERTED_ARG);                     \
        ToTy To;                                                               \
        constexpr auto N =                                                     \
            ToTy::size() * sizeof(detail::get_elem_type_t<ToTy>);              \
        sycl::detail::memcpy_no_adl(&To, &From, N);                            \
        return To;                                                             \
      } else {                                                                 \
        return bit_cast<ToTy>(__VA_ARGS__(NUM_ARGS##_CONVERTED_ARG));          \
      }                                                                        \
    } else {                                                                   \
      return bit_cast<ToTy>(__VA_ARGS__(NUM_ARGS##_CONVERTED_ARG));            \
    }                                                                          \
  }

#define DEVICE_IMPL_TEMPLATE(NUM_ARGS, NAME, ENABLER, /*SCALAR_VEC_IMPL*/...)  \
  DEVICE_IMPL_TEMPLATE_CUSTOM_DELEGATE(NUM_ARGS, NAME, ENABLER,                \
                                       builtin_marray_impl, sycl, __VA_ARGS__)
