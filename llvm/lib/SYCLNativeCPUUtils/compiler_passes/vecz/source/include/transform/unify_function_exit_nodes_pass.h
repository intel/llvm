//===-- unify_function_exit_nodes_pass.h - Ensure fn's have one return ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is used to ensure that functions have at most one return and one
// unreachable instruction in them.
//
//===----------------------------------------------------------------------===//

#ifndef VECZ_TRANSFORM_UNIFY_FUNCTION_EXIT_NODES_PASS_H_INCLUDED
#define VECZ_TRANSFORM_UNIFY_FUNCTION_EXIT_NODES_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace vecz {

class UnifyFunctionExitNodesPass
    : public llvm::OptionalPassInfoMixin<UnifyFunctionExitNodesPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &AM);

  /// @brief Pass name.
  static llvm::StringRef name() { return "Unify function exit nodes pass"; }
};

} // namespace vecz

#endif // VECZ_TRANSFORM_UNIFY_FUNCTION_EXIT_NODES_PASS_H_INCLUDED
