//==--------------- spec_const_char.cpp  - DPC++ ESIMD on-device test -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// On Windows vector compute backend (as a part of IGC) uses llvm-7 and llvm-7
// based spirv translator. This translator doesn't have the ability to overwrite
// the default specialization constant value. That is why the support in Windows
// driver is disabled at all. This feature will start working on Windows when
// the llvm version is switched to 9.
// UNSUPPORTED: windows
// Linux Level Zero fail with assertion in SPIRV about specialization constant
// type size.
// RUN: %clangxx -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || rocm

#include <cstdint>

#define DEF_VAL -22
#define REDEF_VAL 33
#define STORE 2

using spec_const_t = int8_t;
using container_t = int8_t;

#include "Inputs/spec_const_common.hpp"
