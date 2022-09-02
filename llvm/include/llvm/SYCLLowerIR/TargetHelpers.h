//===------------ TargetHelpers.h - Helpers for SYCL kernels ------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper functions for processing SYCL kernels.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_SYCL_LOWER_IR_TARGET_HELPERS_H
#define LLVM_SYCL_SYCL_LOWER_IR_TARGET_HELPERS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace llvm {
namespace TargetHelpers {

enum class ArchType { Cuda, AMDHSA, Unsupported };

struct KernelPayload {
  KernelPayload(Function *Kernel, MDNode *MD = nullptr);
  Function *Kernel;
  MDNode *MD;
};

ArchType getArchType(const Module &M);

std::string getAnnotationString(ArchType AT);

void populateKernels(Module &M, SmallVectorImpl<KernelPayload> &Kernels,
                     TargetHelpers::ArchType AT);

} // end namespace TargetHelpers
} // end namespace llvm

#endif
