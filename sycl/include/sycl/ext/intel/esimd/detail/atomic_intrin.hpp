//==-------- atomic_intrin.hpp - Atomic intrinsic definition file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/exception.hpp>

// This function implements atomic update of pre-existing variable in the
// absense of C++ 20's atomic_ref.
template <typename Ty> Ty atomic_add_fetch(Ty *ptr, Ty val) {
#ifdef _WIN32
  // TODO: Windows will be supported soon
  __ESIMD_UNSUPPORTED_ON_HOST;
#else
  return __atomic_add_fetch(ptr, val, __ATOMIC_RELAXED);
#endif
}

/// @endcond ESIMD_DETAIL
