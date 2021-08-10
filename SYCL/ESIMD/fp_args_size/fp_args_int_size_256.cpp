//==------- fp_args_char_int_size_256.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || rocm
// RUN: %clangxx -Xclang -fsycl-allow-func-ptr -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <cstdint>

constexpr unsigned VL = 16;
constexpr unsigned SIZE = 256;

using a_data_t = int32_t;
using b_data_t = int32_t;
using c_data_t = int32_t;

#include "Inputs/fp_args_size_common.hpp"
