//==--------------- spec_const_ushort.cpp  - DPC++ ESIMD on-device test ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// RUN: %clangxx -fsycl -I%S/.. %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// UNSUPPORTED: cuda || hip
// TODO online_compiler check fails for esimd_emulator
// XFAIL: esimd_emulator

#include <cstdint>

#define DEF_VAL 0xcafe
#define REDEF_VAL 0xdeaf
#define STORE 2

using spec_const_t = uint16_t;
using container_t = uint16_t;

#include "Inputs/spec-const-2020-common.hpp"
