//==---------------- Wrapper around corecrt.h ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MSVC's STL emits functions like _invalid_parameter and _invoke_watson
// to validate parameters passed to STL containers, like std::array, when
// compiled in debug mode.
// STL containers when used in device code fails to compile on Windows
// because these functions are not marked with SYCL_EXTERNAL attribute.

// As a workaround, this wrapper defines a custom, empty defination of
// _invalid_parameter and _invoke_watson for device code.
// MSVC picks up these definitions instead of the declarations from MSVC's
// <corecrt> header.

#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(_DEBUG)

#include <cstdint> // For uintptr_t

extern "C" inline void __cdecl _invalid_parameter(wchar_t const *,
                                                  wchar_t const *,
                                                  wchar_t const *, unsigned int,
                                                  uintptr_t) {
  // Do nothing when called in device code
}

extern "C" __declspec(noreturn) void __cdecl _invoke_watson(
    wchar_t const *const expression, wchar_t const *const function_name,
    wchar_t const *const file_name, unsigned int const line_number,
    uintptr_t const reserved) {
  // Do nothing.
}
#endif

#if defined(__has_include_next)
// GCC/clang support go through this path.
#include_next <corecrt.h>
#else
// MSVC doesn't support "#include_next", so we have to be creative.
// Our header is located in "stl_wrappers/corecrt.h" so it won't be picked by
// the aforementioned include. MSVC's installation, on the other hand, has the
// layout where the following would result in the <corecrt.h> we want. This is
// obviously hacky, but the best we can do...
#include <../ucrt/corecrt.h>
#endif
