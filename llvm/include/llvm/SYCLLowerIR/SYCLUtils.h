//===------------ SYCLUtils.h - SYCL utility functions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for SYCL.
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"

#include <functional>

namespace llvm {
namespace sycl {
namespace utils {
constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";
constexpr char ATTR_SYCL_OPTLEVEL[] = "sycl-optlevel";

using CallGraphNodeAction = ::std::function<void(Function *)>;
using CallGraphFunctionFilter =
    std::function<bool(const Instruction *, const Function *)>;

// Traverses call graph starting from given function up the call chain applying
// given action to each function met on the way. If \c ErrorOnNonCallUse
// parameter is true, then no functions' uses are allowed except calls.
// Otherwise, any function where use of the current one happened is added to the
// call graph as if the use was a call.
// The 'functionFilter' parameter is a callback function that can be used to
// control which functions will be added to a call graph.
//
// The callback is invoked whenever a function being traversed is used
// by some instruction which is not a call to this instruction (e.g. storing
// function pointer to memory) - the first parameter is the using instructions,
// the second - the function being traversed. The parent function of the
// instruction is added to the call graph depending on whether the callback
// returns 'true' (added) or 'false' (not added).
// Functions which are part of the visited set ('Visited' parameter) are not
// traversed.

void traverseCallgraphUp(
    llvm::Function *F, CallGraphNodeAction NodeF,
    SmallPtrSetImpl<Function *> &Visited, bool ErrorOnNonCallUse,
    const CallGraphFunctionFilter &functionFilter =
        [](const Instruction *, const Function *) { return true; });

template <class CallGraphNodeActionF>
void traverseCallgraphUp(
    Function *F, CallGraphNodeActionF ActionF,
    SmallPtrSetImpl<Function *> &Visited, bool ErrorOnNonCallUse,
    const CallGraphFunctionFilter &functionFilter =
        [](const Instruction *, const Function *) { return true; }) {
  traverseCallgraphUp(F, CallGraphNodeAction(ActionF), Visited,
                      ErrorOnNonCallUse, functionFilter);
}

template <class CallGraphNodeActionF>
void traverseCallgraphUp(
    Function *F, CallGraphNodeActionF ActionF, bool ErrorOnNonCallUse = true,
    const CallGraphFunctionFilter &functionFilter =
        [](const Instruction *, const Function *) { return true; }) {
  SmallPtrSet<Function *, 32> Visited;
  traverseCallgraphUp(F, CallGraphNodeAction(ActionF), Visited,
                      ErrorOnNonCallUse, functionFilter);
}

/// Tells if this value is a bit cast or address space cast.
bool isCast(const Value *V);

/// Tells if this value is a GEP instructions with all zero indices.
bool isZeroGEP(const Value *V);

/// Climbs up the use-def chain of given value until a value which is not a
/// bit cast or address space cast is met.
const Value *stripCasts(const Value *V);
Value *stripCasts(Value *V);

/// Climbs up the use-def chain of given value until a value is met which is
/// neither of:
/// - bit cast
/// - address space cast
/// - GEP instruction with all zero indices
const Value *stripCastsAndZeroGEPs(const Value *V);
Value *stripCastsAndZeroGEPs(Value *V);

/// Collects uses of given value "looking through" casts. I.e. if a use is a
/// cast (chain), then uses of the result of the cast (chain) are collected.
void collectUsesLookThroughCasts(const Value *V,
                                 SmallPtrSetImpl<const Use *> &Uses);

/// Collects uses of given pointer-typed value "looking through" casts and GEPs
/// with all zero indices - those pointer transformation instructions which
/// don't change pointed-to value. E.g. if a use is a cast (chain), then uses of
/// the result of the cast (chain) are collected.
void collectUsesLookThroughCastsAndZeroGEPs(const Value *V,
                                            SmallPtrSetImpl<const Use *> &Uses);

void collectUsesLookThroughCasts(const Value *V,
                                 SmallPtrSetImpl<const Use *> &Uses);

void collectUsesLookThroughCastsAndZeroGEPs(const Value *V,
                                            SmallPtrSetImpl<const Use *> &Uses);

bool collectPossibleStoredVals(
    Value *Addr, SmallPtrSetImpl<Value *> &Vals,
    std::function<bool(const CallInst *)> EscapesIfAddrIsArgOf =
        [](const CallInst *) { return true; });

inline bool isSYCLExternalFunction(const Function *F) {
  return F->hasFnAttribute(ATTR_SYCL_MODULE_ID);
}

} // namespace utils
} // namespace sycl
} // namespace llvm
