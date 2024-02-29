//===----- UtilsSYCLNativeCPU.h - Pass pipeline for SYCL Native CPU ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility functions and constants for SYCL Native CPU.
//
//===----------------------------------------------------------------------===//
#pragma once
#include "llvm/ADT/Twine.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"

namespace llvm {
namespace sycl {
namespace utils {

void addSYCLNativeCPUBackendPasses(ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM,
                                   OptimizationLevel OptLevel);
const constexpr char NativeCPUGlobalId[] = "__dpcpp_nativecpu_get_global_id";
const constexpr char NativeCPUGlobaRange[] =
    "__dpcpp_nativecpu_get_global_range";
const constexpr char NativeCPUGlobalOffset[] =
    "__dpcpp_nativecpu_get_global_offset";
const constexpr char NativeCPULocalId[] = "__dpcpp_nativecpu_get_local_id";
const constexpr char NativeCPUNumGroups[] = "__dpcpp_nativecpu_get_num_groups";
const constexpr char NativeCPUWGSize[] = "__dpcpp_nativecpu_get_wg_size";
const constexpr char NativeCPUWGId[] = "__dpcpp_nativecpu_get_wg_id";
const constexpr char NativeCPUSetNumSubgroups[] =
    "__dpcpp_nativecpu_set_num_sub_groups";
const constexpr char NativeCPUSetSubgroupId[] =
    "__dpcpp_nativecpu_set_sub_group_id";
const constexpr char NativeCPUSetMaxSubgroupSize[] =
    "__dpcpp_nativecpu_set_max_sub_group_size";
const constexpr char NativeCPUSetLocalId[] = "__dpcpp_nativecpu_set_local_id";

constexpr char SYCLNATIVECPUSUFFIX[] = ".SYCLNCPU";
constexpr char SYCLNATIVECPUKERNEL[] = ".NativeCPUKernel";
constexpr char SYCLNATIVECPUPREFIX[] = "__dpcpp_nativecpu";
inline llvm::Twine addSYCLNativeCPUSuffix(StringRef S) {
  if (S.starts_with(SYCLNATIVECPUPREFIX) || S.ends_with(SYCLNATIVECPUKERNEL))
    return S;
  return llvm::Twine(S, SYCLNATIVECPUSUFFIX);
}

} // namespace utils
} // namespace sycl
} // namespace llvm
