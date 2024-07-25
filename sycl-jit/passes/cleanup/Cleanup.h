//==--- Cleanup.h - Helper for update of metadata and kernel information ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_PASSES_CLEANUP_H
#define SYCL_FUSION_PASSES_CLEANUP_H

#include "Kernel.h"
#include "target/TargetFusionInfo.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
///
/// Perform cleanup after running a pass.
///
/// @param[in] ArgUsageInfo New argument usage info.
/// @param[in] F Function to be cleaned.
/// @param[in] AM Module analysis manager.
/// @param[in] EraseMD Keys of metadata to remove.
void fullCleanup(ArrayRef<::jit_compiler::ArgUsageUT> ArgUsageInfo, Function *F,
                 ModuleAnalysisManager &AM, TargetFusionInfo &TFI,
                 ArrayRef<StringRef> EraseMD);
} // namespace llvm

#endif // SYCL_FUSION_PASSES_CLEANUP_H
