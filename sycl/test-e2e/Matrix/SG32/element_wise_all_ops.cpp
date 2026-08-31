//==------------ element_wise_all_ops.cpp  - DPC++ joint_matrix-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: target-spir

// REQUIRES: aspect-ext_intel_matrix
// REQUIRES-INTEL-DRIVER: lin: 27501, win: 101.4943
// SG size = 32 is not currently supported for SYCL Joint Matrix by IGC on DG2
// UNSUPPORTED: gpu-intel-dg2
// UNSUPPORTED-TRACKER: GSD-10700

// RUN: %{build} -Xspirv-translator=spir64 --spirv-ext=+SPV_EXT_float8,+SPV_INTEL_float4,+SPV_INTEL_int4,+SPV_INTEL_fp_conversions -o %t.out
// RUN: %{run} %t.out

#include "common.hpp"

#define SG_SZ 32

#include "element_wise_all_ops_impl.hpp"
