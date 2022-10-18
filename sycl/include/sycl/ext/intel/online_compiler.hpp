//===------- online_compiler.hpp - Online source compilation service ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

#if !defined(_MSC_VER) || defined(__clang__)
// MSVC doesn't support #warning and we cannot use other methods to report a
// warning from inside a system header (which SYCL is considered to be).
#warning sycl/ext/intel/online_compiler.hpp usage is deprecated, \
include sycl/ext/intel/experimental/online_compiler.hpp instead
#endif

#include <sycl/ext/intel/experimental/online_compiler.hpp>
