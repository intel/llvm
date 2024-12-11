//==-- msan_rtl.hpp - Declaration for sanitizer global var ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "sanitizer_defs.hpp"
#include "spir_global_var.hpp"

// Treat this header as system one to workaround frontend's restriction
#pragma clang system_header
