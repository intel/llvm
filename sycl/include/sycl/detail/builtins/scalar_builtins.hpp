//==--------------- scalar_builtins.hpp -----------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Internal scalar-only entry point for SYCL builtins.
#define SYCL_DETAIL_BUILTINS_SCALAR_ONLY
#include <sycl/detail/builtins/builtins.hpp>
#undef SYCL_DETAIL_BUILTINS_SCALAR_ONLY