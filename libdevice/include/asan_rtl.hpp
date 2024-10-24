//==-- sanitizer_device_utils.hpp - Declaration for sanitizer global var ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "spir_global_var.hpp"
#include <cstdint>

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header

extern SPIR_GLOBAL_VAR __SYCL_GLOBAL__ uint64_t *__SYCL_LOCAL__
    __AsanLaunchInfo;
