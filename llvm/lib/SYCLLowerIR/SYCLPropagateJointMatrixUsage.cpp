//===------------------ SYCLPropagateJointMatrixUsage.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass propagates optional kernel features metadata through a module call graph
// for sycl_ext_oneapi_matrix extension
//
// The pass consists of three main steps:
//
// I. It builds Function -> string of joint matrix types and sizes values
// mapping for usage in step II
// II. Propagates all the values from step I. to the top of the call graph
// III. Generates metadata with values of joint matrix types and sizes
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLPropagateJointMatrixUsage.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"

#include <set>

using namespace llvm;

namespace {

/// Returns true if the function is a SPIRV or SYCL builtin, e.g.
/// _Z28__spirv_GlobalInvocationId_xv
/// NB! This function was copied from sycl-post-link/ModuleSplitter.cpp and the
/// definition of entry point (i.e. implementation of the function) should be in
/// sync between those two.
bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.starts_with("__spirv_") || FName.starts_with("__sycl_");
}

bool isEntryPoint(const Function &F) {
  // Skip declarations, we can't analyze them
  if (F.isDeclaration()) {
    // F.print(outs());
    return false;
  }

  // Kernels are always considered to be entry points
  if (CallingConv::SPIR_KERNEL == F.getCallingConv())
    return true;

  // FIXME: sycl-post-link allows to disable treating SYCL_EXTERNAL's as entry
  // points - do we need similar flag here?
  // SYCL_EXTERNAL functions with sycl-module-id attribute
  // are also considered as entry points (except __spirv_* and __sycl_*
  // functions)
  return F.hasFnAttribute("sycl-module-id") && !isSpirvSyclBuiltin(F.getName());
}

using CallGraphTy = DenseMap<Function *, SmallPtrSet<Function *, 8>>;

/// Updates call graph with the information from function @F
void fillCallGraph(Function *F, CallGraphTy &CG) {
  for (Instruction &I : instructions(F)) {
    if (const auto *CI = dyn_cast<CallInst>(&I)) {
      if (!CI->isIndirectCall() && CI->getCalledFunction())
        CG[F].insert(CI->getCalledFunction());
    }
  }
}

using JointMatrixValueStringTy = SmallString<40>;
using JointMatrixValuesSetTy = std::set<JointMatrixValueStringTy>;
using FunctionToJointMatrixValuesMapTy =
    DenseMap<Function *, JointMatrixValuesSetTy>;

/// Creates mapping between a function and an information about matrix types and
/// sizes of sycl::ext::oneapi::experimental::matrix::joint_matrix type
void fillFunctionToJointMatrixValuesMap(
    Function *F,
    FunctionToJointMatrixValuesMapTy &FunctionToJointMatrixValues) {
  // assume we have other sycl-joint-matrix-* attributes if
  // sycl-joint-matrix-type is present
  if (!F->hasFnAttribute("sycl-joint-matrix-type"))
    return;

  JointMatrixValueStringTy Result;
  // NB! The order of attributes must not change as it is used later in SYCL
  // RT
  // The order is:
  //   - sycl-joint-matrix-type
  //   - sycl-joint-matrix-use
  //   - sycl-joint-matrix-rows
  //   - sycl-joint-matrix-cols
  // NB! Values must be separated with a comma
  Result += F->getFnAttribute("sycl-joint-matrix-type").getValueAsString();
  Result += ",";
  Result += F->getFnAttribute("sycl-joint-matrix-use").getValueAsString();
  Result += ",";
  Result += F->getFnAttribute("sycl-joint-matrix-rows").getValueAsString();
  Result += ",";
  Result += F->getFnAttribute("sycl-joint-matrix-cols").getValueAsString();
  FunctionToJointMatrixValues[F].insert(Result);
}

/// Creates mapping between a function and an information about matrix types and
/// sizes of sycl::ext::oneapi::experimental::matrix::joint_matrix_mad()
/// function
void fillFunctionToJointMatrixMadValuesMap(
    Function *F,
    FunctionToJointMatrixValuesMapTy &FunctionToJointMatrixMapValues) {
  // assume we have other sycl-joint-matrix-mad-* attributes if
  // sycl-joint-matrix-mad-type-A is present
  if (!F->hasFnAttribute("sycl-joint-matrix-mad-type-A"))
    return;

  JointMatrixValueStringTy Result;
  // NB! The order of attributes must not change as it is used later in SYCL
  // RT
  // The order is:
  //   - sycl-joint-matrix-mad-type-A
  //   - sycl-joint-matrix-mad-type-B
  //   - sycl-joint-matrix-mad-type-C
  //   - sycl-joint-matrix-mad-type-D
  //   - sycl-joint-matrix-mad-size-M
  //   - sycl-joint-matrix-mad-size-K
  //   - sycl-joint-matrix-mad-size-N
  // NB! Values must be separated with a comma
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-type-A").getValueAsString();
  Result += ",";
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-type-B").getValueAsString();
  Result += ",";
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-type-C").getValueAsString();
  Result += ",";
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-type-D").getValueAsString();
  Result += ",";
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-size-M").getValueAsString();
  Result += ",";
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-size-K").getValueAsString();
  Result += ",";
  Result +=
      F->getFnAttribute("sycl-joint-matrix-mad-size-N").getValueAsString();
  FunctionToJointMatrixMapValues[F].insert(Result);
}

/// Propagates joint matrix values from leaves up to the top of call graph.
/// NB! Call graph corresponds to call graph of SYCL code which
/// can't contain recursive calls. So there can't be loops in
/// a call graph. But there can be path's intersections.
void propagateJointMatrixValuesThroughCG(
    Function *F, CallGraphTy &CG,
    FunctionToJointMatrixValuesMapTy &FunctionToJointMatrixValues,
    FunctionToJointMatrixValuesMapTy &FunctionToJointMatrixMadValues,
    SmallPtrSet<const Function *, 16> &Visited) {
  const auto It = CG.find(F);
  if (It == CG.end())
    return;

  JointMatrixValuesSetTy LocalJointMatrixValues;
  JointMatrixValuesSetTy LocalJointMatrixMadValues;
  for (Function *Callee : It->second) {
    if (Visited.insert(Callee).second)
      propagateJointMatrixValuesThroughCG(
          Callee, CG, FunctionToJointMatrixValues,
          FunctionToJointMatrixMadValues, Visited);

    const auto &CalleeJointMatrixValues = FunctionToJointMatrixValues[Callee];
    LocalJointMatrixValues.insert(CalleeJointMatrixValues.begin(),
                                  CalleeJointMatrixValues.end());
    const auto &CalleeJointMatrixMadValues =
        FunctionToJointMatrixMadValues[Callee];
    LocalJointMatrixMadValues.insert(CalleeJointMatrixMadValues.begin(),
                                     CalleeJointMatrixMadValues.end());
  }
  FunctionToJointMatrixValues[F].insert(LocalJointMatrixValues.begin(),
                                        LocalJointMatrixValues.end());
  FunctionToJointMatrixMadValues[F].insert(LocalJointMatrixMadValues.begin(),
                                           LocalJointMatrixMadValues.end());
}

void setSyclJointMatrixMetadata(StringRef MetadataName, Module *M, Function *F,
                                FunctionToJointMatrixValuesMapTy ValuesMap) {
  JointMatrixValuesSetTy Values = ValuesMap[F];
  SmallString<256> StringValue;
  for (auto It = Values.begin(); It != Values.end(); It++) {
    StringValue += *It;
    // NB! Each information about joint_matrix type and joint_matrix_mad
    // function should be separated by a semicolon
    if (std::next(It) != Values.end())
      StringValue += ";";
  }
  if (StringValue.empty())
    return;

  MDString *MDStringValue = MDString::get(M->getContext(), StringValue);
  MDNode *MDN = MDNode::get(M->getContext(), MDStringValue);
  F->setMetadata(MetadataName, MDN);
}

} // anonymous namespace

PreservedAnalyses
SYCLPropagateJointMatrixUsagePass::run(Module &M, ModuleAnalysisManager &MAM) {
  FunctionToJointMatrixValuesMapTy FunctionToJointMatrixValues;
  FunctionToJointMatrixValuesMapTy FunctionToJointMatrixMadValues;
  SmallVector<Function *, 16> EntryPoints;
  CallGraphTy CG;
  for (Function &F : M.functions()) {
    fillFunctionToJointMatrixValuesMap(&F, FunctionToJointMatrixValues);
    fillFunctionToJointMatrixMadValuesMap(&F, FunctionToJointMatrixMadValues);
    fillCallGraph(&F, CG);

    if (isEntryPoint(F))
      EntryPoints.push_back(&F);
  }

  SmallPtrSet<const Function *, 16> Visited;
  for (const auto F : EntryPoints) {
    propagateJointMatrixValuesThroughCG(F, CG, FunctionToJointMatrixValues,
                                        FunctionToJointMatrixMadValues,
                                        Visited);
  }

  for (Function *F : EntryPoints) {
    setSyclJointMatrixMetadata("sycl_joint_matrix", &M, F,
                               FunctionToJointMatrixValues);
    setSyclJointMatrixMetadata("sycl_joint_matrix_mad", &M, F,
                               FunctionToJointMatrixMadValues);
  }

  return PreservedAnalyses::all();
}
