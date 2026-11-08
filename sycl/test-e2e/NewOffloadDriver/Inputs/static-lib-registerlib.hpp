//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
// Shared declarations for the static-lib-* registerlib tests.

#pragma once

#include <cstddef>

void run_jit(std::size_t *out, std::size_t N);
void run_dg2(std::size_t *out, std::size_t N);
void run_bmg(std::size_t *out, std::size_t N);
void run_gen_generic(std::size_t *out, std::size_t N);

inline constexpr std::size_t JITOffset = 0;
inline constexpr std::size_t DG2Offset = 100;
inline constexpr std::size_t BMGOffset = 1000;
inline constexpr std::size_t GenGenericOffset = 10000;
