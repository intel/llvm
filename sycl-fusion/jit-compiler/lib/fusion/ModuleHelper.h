//== ModuleHelper.h - Helper to prune unnecesary functions from LLVM module ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_FUSION_JIT_COMPILER_FUSION_MODULEHELPER_H
#define SYCL_FUSION_JIT_COMPILER_FUSION_MODULEHELPER_H

#include "llvm/IR/Module.h"

namespace helper {

class ModuleHelper {
public:
  ///
  /// Clone the module and prune unused functions, i.e., functions not reachable
  /// from the functions in CGRoots.
  static std::unique_ptr<llvm::Module>
  cloneAndPruneModule(llvm::Module *Mod,
                      llvm::ArrayRef<llvm::Function *> CGRoots);

private:
  ///
  /// Identify functions not reachable from the function in CGRoots.
  static void identifyUnusedFunctions(
      llvm::Module *Mod, llvm::ArrayRef<llvm::Function *> CGRoots,
      llvm::SmallPtrSetImpl<llvm::Function *> &UnusedFunctions);
};

} // namespace helper

#endif // SYCL_FUSION_JIT_COMPILER_FUSION_MODULEHELPER_H
