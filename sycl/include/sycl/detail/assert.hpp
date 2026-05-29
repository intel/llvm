//==---------- assert.hpp ---- SYCL assert support ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cassert> // for assert

#ifdef __SYCL_DEVICE_ONLY__
// TODO remove this when 'assert' is supported in device code
#define __SYCL_ASSERT(x)
#else
#define __SYCL_ASSERT(x) assert(x)
#endif // #ifdef __SYCL_DEVICE_ONLY__
