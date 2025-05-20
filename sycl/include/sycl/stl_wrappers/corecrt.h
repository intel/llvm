//==---------------- Wrapper around corecrt.h ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When std::array is used in device code, MSVC's STL uses a _invalid_parameter
// function. This causes issue when _invalid_parameter is invoked from device
// code:

// 1. `_invalid_parameter` is provided via ucrtbased.dll at runtime: DLLs are
//    not loaded for device code, thus causing undefined symbol errors.

// 2. MSVC's STL never defined the function as SYCL_EXTERNAL, errors are thrown
//    when device code tries to invoke `_invalid_parameter`.

// As a workaround, this wrapper wraps around corecrt.h and defines a custom
// _invalid_parameter for device code compilation.

// This new SYCL_EXTERNAL definition of _invalid_parameter has to be declared
// before corecrt.h is included: Thus, we have this STL wrapper instead of
// declaring _invalid_parameter function in SYCL headers.

#pragma once

#if defined(__SYCL_DEVICE_ONLY__) && defined(_DEBUG)

#include <cstdint> // For uintptr_t

extern "C" inline void __cdecl _invalid_parameter(wchar_t const *,
                                                  wchar_t const *,
                                                  wchar_t const *, unsigned int,
                                                  uintptr_t) {
  // Do nothing when called in device code
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
