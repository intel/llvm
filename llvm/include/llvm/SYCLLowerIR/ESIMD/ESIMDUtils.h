//===------------ ESIMDUtils.h - ESIMD utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for processing ESIMD code.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"

#include <functional>

namespace llvm {
namespace esimd {

constexpr char ATTR_DOUBLE_GRF[] = "esimd-double-grf";
constexpr char ESIMD_MARKER_MD[] = "sycl_explicit_simd";

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

// Tells whether given function is a ESIMD kernel.
bool isESIMDKernel(const Function &F);
// Tells whether given function is a ESIMD function.
bool isESIMD(const Function &F);
// Tells whether given function is a kernel.
bool isKernel(const Function &F);

/// Reports and error with the message \p Msg concatenated with the optional
/// \p OptMsg if \p Condition is false.
inline void assert_and_diag(bool Condition, StringRef Msg,
                            StringRef OptMsg = "") {
  if (!Condition) {
    auto T = Twine(Msg) + OptMsg;
    llvm::report_fatal_error(T, true /* crash diagnostics */);
  }
}

/// Tells if this value is a bit cast or address space cast.
bool isCast(const Value *V);

/// Climbs up the use-def chain of given value until a value which is not a
/// bit cast or address space cast is met.
const Value *stripCasts(const Value *V);
Value *stripCasts(Value *V);

/// Collects uses of given value "looking through" casts. I.e. if a use is a
/// cast (chain), then uses of the result of the cast (chain) are collected.
void collectUsesLookThroughCasts(const Value *V,
                                 SmallPtrSetImpl<const Use *> &Uses);

/// Unwraps a presumably simd* type to extract the native vector type encoded
/// in it. Returns nullptr if failed to do so.
Type *getVectorTyOrNull(StructType *STy);

} // namespace esimd
} // namespace llvm
