//==---------------- Wrapper around crtdbg.h ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// When std::array is used in device code, MSVC's STL uses _CrtDbgReport
// function with variable arguments, when compiled in debug mode (-D_DEBUG
// flag).

// Variable argument functions are not supported in SPIR-V, and the compilation
// fails with llvm-spirv error: UnsupportedVarArgFunction: Variadic functions
// other than 'printf' are not supported in SPIR-V.

// As a workaround, in this wrapper, we define our own variable templated
// _CrtDbgReport which overrides the variable argument _CrtDbgReport function
// declaration in crtdbg.h.

// The variable templated _CrtDbgReport function has to be declared before the
// crtdbg.h header is included, and that's why we have this STL wrapper instead
// of declaring the _CrtDbgReport function in SYCL headers.

#pragma once

#if defined(__SYCL_DEVICE_ONLY__)
template <typename... Ts> int _CrtDbgReport(Ts...) { return 0; }
#endif

#if defined(__has_include_next)
// GCC/clang support go through this path.
#include_next <crtdbg.h>
#else
// MSVC doesn't support "#include_next", so we have to be creative.
// Our header is located in "stl_wrappers/crtdbg.h" so it won't be picked by the
// following include. MSVC's installation, on the other hand, has the layout
// where the following would result in the <crtdbg.h> we want. This is obviously
// hacky, but the best we can do...
#include <../ucrt/crtdbg.h>
#endif
