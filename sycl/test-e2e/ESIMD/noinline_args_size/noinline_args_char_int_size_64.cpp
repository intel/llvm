//===--- noinline_args_char_int_size_64.cpp  - DPC++ ESIMD on-device test -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// XFAIL: gpu && linux
// RUN: %{build} -I%S/.. -o %t.out
// RUN: env IGC_FunctionControl=3 IGC_ForceInlineStackCallWithImplArg=1 %{run} %t.out

#include <cstdint>

constexpr unsigned VL = 16;
constexpr unsigned SIZE = 64;

using a_data_t = int8_t;
using b_data_t = int32_t;
using c_data_t = int32_t;

#include "Inputs/noinline_args_size_common.hpp"
