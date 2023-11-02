//==---------------- types.hpp --- SYCL types ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements vec and __swizzled_vec__ classes.

#pragma once

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
#include <sycl/detail/vec_new.hpp>
#else
#include <sycl/detail/vec_old.hpp>
#endif
