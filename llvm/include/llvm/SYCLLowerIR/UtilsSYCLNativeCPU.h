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
#include "llvm/Target/TargetMachine.h"

namespace llvm {
void addSYCLNativeCPUBackendPasses(ModulePassManager &MPM,
                                   ModuleAnalysisManager &MAM);
namespace sycl {
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
} // namespace sycl
} // namespace llvm
