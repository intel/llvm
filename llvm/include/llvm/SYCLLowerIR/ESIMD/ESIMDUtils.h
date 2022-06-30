//===------------ ESIMDUtils.h - ESIMD utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for processing ESIMD code.
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"

#include <functional>

namespace llvm {
namespace esimd {

constexpr char ATTR_DOUBLE_GRF[] = "esimd-double-grf";

using CallGraphNodeAction = std::function<void(Function *)>;
void traverseCallgraphUp(llvm::Function *F, CallGraphNodeAction NodeF,
                         bool ErrorOnNonCallUse);

// Traverses call graph starting from given function up the call chain applying
// given action to each function met on the way. If \c ErrorOnNonCallUse
// parameter is true, then no functions' uses are allowed except calls.
// Otherwise, any function where use of the current one happened is added to the
// call graph as if the use was a call.
template <class CallGraphNodeActionF>
void traverseCallgraphUp(Function *F, CallGraphNodeActionF ActionF,
                         bool ErrorOnNonCallUse = true) {
  traverseCallgraphUp(F, CallGraphNodeAction{ActionF}, ErrorOnNonCallUse);
}

// Tells whether given function is a ESIMD kernel.
bool isESIMDKernel(const Function &F);

/// Reports and error with the message \p Msg concatenated with the optional
/// \p OptMsg if \p Condition is false.
inline void assert_and_diag(bool Condition, StringRef Msg,
                            StringRef OptMsg = "") {
  if (!Condition) {
    auto T = Twine(Msg) + OptMsg;
    llvm::report_fatal_error(T, true /* crash diagnostics */);
  }
}

} // namespace esimd
} // namespace llvm
