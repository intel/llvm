//==---- __msvc_bit_utils.hpp wrapper around MSVC STL ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// VS2026 MSVC STL's <__msvc_bit_utils.hpp> declares `__isa_available` (a
// runtime CPU-feature global) and other STL headers (<bit>, <vector>,
// <numeric>, <bitset>, <complex>, <__msvc_int128.hpp>) include this header
// transitively. SYCL device code cannot access host runtime globals, so
// provide a device-side definition before the real header is included.
//
// The VALUE of this variable only steers the STL's runtime feature dispatch
// — both branches compile either way. We pick __ISA_AVAILABLE_X86 (== 0,
// the baseline in <isa_availability.h>), which matches a spir64 device's
// reality (no x86 ISA), and selects the STL's scalar fallback paths if the
// dispatches are ever reached.
//
// Reads of `std::__isa_available` from device code are permitted by a
// named-symbol allowlist in clang's SemaExpr (see `isMsvcSTLGlobalVar`),
// so no per-decl attribute is needed here.
//
// Mirror the source structure of MSVC STL's __msvc_bit_utils.hpp, which
// declares the symbol inside `namespace std { extern "C" { ... } }`.
#if defined(__SYCL_DEVICE_ONLY__) && defined(_MSC_VER)
namespace std {
extern "C" {
int __isa_available = 0;
}
} // namespace std
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(_MSC_VER)

// Include real STL <__msvc_bit_utils.hpp> header - the next one from the
// include search directories.
#if defined(__has_include_next)
// GCC/clang support go through this path.
#if __has_include_next(<__msvc_bit_utils.hpp>)
#include_next <__msvc_bit_utils.hpp>
#endif
#else
// MSVC doesn't support "#include_next", so we have to be creative.
// Our header is located in "stl_wrappers/__msvc_bit_utils.hpp" so it won't be
// picked by the following include. MSVC's installation, on the other hand,
// has the layout where the following would result in the
// <__msvc_bit_utils.hpp> we want. This is obviously hacky, but the best we
// can do...
#if __has_include(<../include/__msvc_bit_utils.hpp>)
#include <../include/__msvc_bit_utils.hpp>
#endif
#endif
