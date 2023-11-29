//==------- fp_args_char_int_size_64.cpp  - DPC++ ESIMD on-device test -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -Xclang -fsycl-allow-func-ptr -I%S/.. -o %t.out
// RUN: %{run} %t.out

#include <cstdint>

constexpr unsigned VL = 16;
constexpr unsigned SIZE = 64;

using a_data_t = int32_t;
using b_data_t = int32_t;
using c_data_t = int32_t;

#include "Inputs/fp_args_size_common.hpp"
