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

struct KernelCache {
  void populateKernels(Module &M);

  bool isKernel(Function &F) const;

  /// Updates cached data with a function intended as a replacement of an
  /// existing function.
  void handleReplacedWith(Function &OldF, Function &NewF);

  /// Updates cached data with a new clone of an existing function.
  /// The KernelOnly parameter updates cached data with only the information
  /// required to identify the new function as a kernel.
  void handleNewCloneOf(Function &OldF, Function &NewF, bool KernelOnly);

private:
  /// Extra data about a kernel function. Only applicable to NVPTX kernels,
  /// which have associated annotation metadata.
  struct KernelPayload {
    explicit KernelPayload() = default;
    KernelPayload(NamedMDNode *ModuleAnnotationsMD);

    bool hasAnnotations() const { return ModuleAnnotationsMD != nullptr; }

    /// ModuleAnnotationsMD - metadata conntaining the unique global list of
    /// annotations.
    NamedMDNode *ModuleAnnotationsMD = nullptr;
    SmallVector<MDNode *> DependentMDs;
  };

  /// List of kernels in original Module order
  SmallVector<Function *, 4> Kernels;
  /// Map of kernels to extra data. Also serves as a quick kernel query.
  SmallDenseMap<Function *, KernelPayload> KernelData;

public:
  using iterator = decltype(Kernels)::iterator;
  using const_iterator = decltype(Kernels)::const_iterator;

  iterator begin() { return Kernels.begin(); }
  iterator end() { return Kernels.end(); }

  const_iterator begin() const { return Kernels.begin(); }
  const_iterator end() const { return Kernels.end(); }

  bool empty() const { return Kernels.empty(); }
};

bool isSYCLDevice(const Module &M);

} // end namespace TargetHelpers
} // end namespace llvm

#endif
