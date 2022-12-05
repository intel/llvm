//===------------ SYCLUtils.h - SYCL utility functions
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"

#include <functional>
namespace llvm {
namespace sycl {
namespace utils {
using CallGraphNodeAction = std::function<void(Function *)>;

// Traverses call graph starting from given function up the call chain applying
// given action to each function met on the way. If \c ErrorOnNonCallUse
// parameter is true, then no functions' uses are allowed except calls.
// Otherwise, any function where use of the current one happened is added to the
// call graph as if the use was a call.
// Functions which are part of the visited set ('Visited' parameter) are not
// traversed.
void traverseCallgraphUp(llvm::Function *F, CallGraphNodeAction NodeF,
                         SmallPtrSetImpl<Function *> &Visited,
                         bool ErrorOnNonCallUse);

template <class CallGraphNodeActionF>
void traverseCallgraphUp(Function *F, CallGraphNodeActionF ActionF,
                         SmallPtrSetImpl<Function *> &Visited,
                         bool ErrorOnNonCallUse) {
  traverseCallgraphUp(F, CallGraphNodeAction(ActionF), Visited,
                      ErrorOnNonCallUse);
}

template <class CallGraphNodeActionF>
void traverseCallgraphUp(Function *F, CallGraphNodeActionF ActionF,
                         bool ErrorOnNonCallUse = true) {
  SmallPtrSet<Function *, 32> Visited;
  traverseCallgraphUp(F, CallGraphNodeAction(ActionF), Visited,
                      ErrorOnNonCallUse);
}
} // namespace utils
} // namespace sycl
} // namespace llvm
