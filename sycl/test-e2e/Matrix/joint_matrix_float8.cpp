//==----------- joint_matrix_float8.cpp  - DPC++ joint_matrix---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: arch-intel_gpu_cri, aspect-ext_intel_matrix

// RUN: %{build} -Xspirv-translator=spir64 --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_fp_conversions,+SPV_KHR_bfloat16 -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: target-nvidia, target-amd, spirv-backend
// UNSUPPORTED-INTENDED: only supported by backends with CRI driver, and the
// SPIR-V backend does not support the required SPIR-V extensions

#include "common.hpp"
#include "joint_matrix_float8_impl.hpp"
