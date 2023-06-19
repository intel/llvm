//==------------- BitonicSortKv2_emu.cpp  - DPC++ ESIMD on-device test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==----------------------------------------------------------------------==//
// REQUIRES: esimd_emulator
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#define ESIMD_EMU

#include "BitonicSortKv2.hpp"
