//==------------ sycl.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sycl.hpp>

namespace __SYCL2020_DEPRECATED(
    "cl::sycl is deprecated, use ::sycl instead.") cl {
namespace sycl = ::sycl;
}
