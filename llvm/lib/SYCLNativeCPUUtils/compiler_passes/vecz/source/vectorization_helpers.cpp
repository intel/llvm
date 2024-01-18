// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "vectorization_helpers.h"

#include <compiler/utils/attributes.h>
#include <compiler/utils/metadata.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/TypeSize.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <optional>

#include "debugging.h"
#include "vectorization_context.h"
#include "vectorization_unit.h"
#include "vecz/vecz_choices.h"

using namespace llvm;
using namespace vecz;

namespace {

Function *declareFunction(const VectorizationUnit &VU) {
  Module &Module = VU.context().module();
  const Function *const ScalarFn = VU.scalarFunction();
  const ElementCount SimdWidth = VU.width();

  // For kernels, the vectorized function type is is the same as the original
  // scalar function type, since function arguments are uniform. We no longer
  // use Vectorization Units for builtins.
  FunctionType *VectorizedFnType = VU.scalarFunction()->getFunctionType();
  VECZ_FAIL_IF(!VectorizedFnType);
  const std::string VectorizedName =
      getVectorizedFunctionName(ScalarFn->getName(), SimdWidth, VU.choices());
  Module.getOrInsertFunction(VectorizedName, VectorizedFnType);
  auto *const VectorizedFn = Module.getFunction(VectorizedName);
  if (VectorizedFn) {
    VectorizedFn->setCallingConv(ScalarFn->getCallingConv());
  }
  return VectorizedFn;
}

/// @brief Clone the OpenCL named metadata node with name NodeName
/// @param[in] NodeName The name of the node to clone
///
/// This function works with nodes that follow a specific pattern,
/// specifically nodes that have as their operands other metadata nodes, which
/// in turn have their first operand set to the OpenCL kernel Function. It
/// searches for the node that contains the scalar kernel, and copies all its
/// metadata, which the exception of the Function itself, which is replaced by
/// the vectorized kernel.
void cloneOpenCLNamedMetadataHelper(const VectorizationUnit &VU,
                                    const std::string &NodeName) {
  const Module &M = VU.context().module();

  // Try to get the OpenCL metadata
  NamedMDNode *KernelsMD = M.getNamedMetadata(NodeName);
  if (!KernelsMD) {
    return;
  }

  // Find which metadata node contains the metadata for the scalar function
  MDNode *ScalarKernelMD = nullptr;
  for (auto *KernelMD : KernelsMD->operands()) {
    // The function name is the first operand
    if (KernelMD->getNumOperands() > 0) {
      // Get the Constant containing the function
      ConstantAsMetadata *KernelNameMD =
          dyn_cast_or_null<ConstantAsMetadata>(KernelMD->getOperand(0));
      if (KernelNameMD) {
        // Check if the function in the metadata is the original OpenCL kernel
        if (KernelNameMD->getValue() == VU.scalarFunction()) {
          ScalarKernelMD = KernelMD;
          break;
        }
      }
    }
  }

  // Did we find the correct metadata?
  if (!ScalarKernelMD) {
    return;
  }

  // Replace the kernel name and clone the rest of the metadata
  SmallVector<llvm::Metadata *, 5> KernelMDArgs;
  KernelMDArgs.push_back(
      llvm::ConstantAsMetadata::get(VU.vectorizedFunction()));
  auto MDIt = ScalarKernelMD->op_begin() + 1;
  auto MDEnd = ScalarKernelMD->op_end();
  for (; MDIt != MDEnd; ++MDIt) {
    KernelMDArgs.push_back(*MDIt);
  }

  // Create a new metadata node and add it to the opencl.kernels node
  llvm::MDNode *KernelMDNode =
      llvm::MDNode::get(VU.context().module().getContext(), KernelMDArgs);
  KernelsMD->addOperand(KernelMDNode);
}

/// @brief Create placeholder instructions for arguments that will be
/// vectorized. This is necessary to clone the original function's scalar code
/// into the vectorized function.
///
/// @param[in,out] ValueMap Map to update with the arguments.
SmallVector<Instruction *, 2> createArgumentPlaceholders(
    const VectorizationUnit &VU, Function *VecFunc,
    ValueToValueMapTy &ValueMap) {
  SmallVector<Instruction *, 2> Placeholders;
  const auto &Arguments = VU.arguments();
  unsigned i = 0u;
  for (Argument &DstArg : VecFunc->args()) {
    Argument *SrcArg = Arguments[i++].OldArg;
    DstArg.setName(SrcArg->getName());
    if (DstArg.getType() != SrcArg->getType()) {
      // Map old argument to a temporary placeholder to work around the
      // difference in argument types. This usually happens when vectorizing
      // builtin functions.
      Type *IndexTy = Type::getInt32Ty(VecFunc->getParent()->getContext());
      Constant *Index = Constant::getNullValue(IndexTy);
      auto *const Placeholder = ExtractElementInst::Create(&DstArg, Index);
      ValueMap[SrcArg] = Placeholder;
      Placeholders.push_back(Placeholder);
    } else {
      ValueMap[SrcArg] = &DstArg;
    }
  }
  return Placeholders;
}

}  // namespace

namespace vecz {
std::string getVectorizedFunctionName(StringRef ScalarName, ElementCount VF,
                                      VectorizationChoices Choices,
                                      bool IsBuiltin) {
  const Twine Prefix = Twine(VF.isScalable() ? "nxv" : "v");
  const Twine IsVP = Twine(Choices.vectorPredication() ? "_vp_" : "_");
  return ((IsBuiltin ? VectorizationContext::InternalBuiltinPrefix
                     : Twine("__vecz_")) +
          Prefix + Twine(VF.getKnownMinValue()) + IsVP + ScalarName)
      .str();
}

std::optional<std::tuple<std::string, ElementCount, VectorizationChoices>>
decodeVectorizedFunctionName(StringRef Name) {
  if (!Name.consume_front(VectorizationContext::InternalBuiltinPrefix)) {
    if (!Name.consume_front("__vecz_")) {
      return std::nullopt;
    }
  }

  ElementCount VF;
  bool Scalable = false;
  if (Name.consume_front("nxv")) {
    Scalable = true;
  } else if (!Name.consume_front("v")) {
    return std::nullopt;
  }

  unsigned KnownMin = 0;
  if (Name.consumeInteger(10, KnownMin)) {
    return std::nullopt;
  }

  VF = ElementCount::get(KnownMin, Scalable);

  VectorizationChoices Choices;
  if (Name.consume_front("_vp_")) {
    Choices.enableVectorPredication();
  } else if (!Name.consume_front("_")) {
    return std::nullopt;
  }

  return std::make_tuple(Name.str(), VF, Choices);
}

Function *cloneFunctionToVector(const VectorizationUnit &VU) {
  auto *const VectorizedFn = declareFunction(VU);
  VECZ_ERROR_IF(!VectorizedFn, "declareFunction failed to initialize");

  auto *const ScalarFn = VU.scalarFunction();

  // Map the old arguments to the new ones.
  ValueToValueMapTy ValueMap;
  auto Placeholders = createArgumentPlaceholders(VU, VectorizedFn, ValueMap);

  // Clone the function to preserve instructions that do not need vectorization.
  SmallVector<ReturnInst *, 4> Returns;

  // Setting `moduleChanges` to true allows `llvm::CloneFunctionInto()` to do
  // the work of cloning debug info across translation unit boundaries.
  // However there can be issues with inlined kernels if the inlined kernel
  // still exists in the kernel, and also has a vectorized variant.
  // This value was set to true in this code since LLVM_VERSION_MAJOR > 4 but as
  // of llvm > 12 we need to be a bit more careful with that value as there is
  // more nuance introduced in 22a52dfddc with requisite assertions
  const bool moduleChanges = VectorizedFn->getParent() != ScalarFn->getParent();
  auto cloneMode = moduleChanges ? CloneFunctionChangeType::DifferentModule
                                 : CloneFunctionChangeType::LocalChangesOnly;
  CloneFunctionInto(VectorizedFn, ScalarFn, ValueMap, cloneMode, Returns);

  // Remove unwanted return value attributes.
  if (VectorizedFn->getReturnType()->isVectorTy()) {
    LLVMContext &Ctx = VectorizedFn->getContext();
    AttributeList PAL = VectorizedFn->getAttributes();
    bool RemovedAttribute = false;
    for (const Attribute::AttrKind Kind : {Attribute::ZExt, Attribute::SExt}) {
      if (PAL.hasRetAttr(Kind)) {
        PAL = PAL.removeRetAttribute(Ctx, Kind);
        RemovedAttribute = true;
      }
    }
    if (RemovedAttribute) {
      VectorizedFn->setAttributes(PAL);
    }
  }

  // Override the base function name component for the vectorized function.
  compiler::utils::setBaseFnName(*VectorizedFn, VectorizedFn->getName());

  // Drop any metadata where the scalar kernel already serves as the base or
  // result of vectorization: this vectorized function does not serve as such:
  // not directly in the case of 'derived' metadata, anyway: that relationship
  // will be transitive.
  compiler::utils::dropVeczOrigMetadata(*VectorizedFn);
  compiler::utils::dropVeczDerivedMetadata(*VectorizedFn);

  // Add any 'argument placeholder' instructions to the entry block.
  // Skip over Alloca instructions if there are any.
  BasicBlock &BB = VectorizedFn->getEntryBlock();
  auto InsertPt = BB.getFirstInsertionPt();
  while (isa<AllocaInst>(*InsertPt)) {
    ++InsertPt;
  }

  for (auto *Placeholder : Placeholders) {
    Placeholder->insertBefore(&*InsertPt);
  }

  return VectorizedFn;
}

static DILocation *getDILocation(unsigned Line, unsigned Column, MDNode *Scope,
                                 MDNode *InlinedAt = nullptr) {
  // If no scope is available, this is an unknown location.
  if (!Scope) return DebugLoc();
  return DILocation::get(Scope->getContext(), Line, Column, Scope, InlinedAt,
                         /*ImplicitCode*/ false);
}

void cloneDebugInfo(const VectorizationUnit &VU) {
  DISubprogram *const ScalarDI = VU.scalarFunction()->getSubprogram();
  // We don't have debug info
  if (!ScalarDI) {
    return;
  }

  // Create a DISubprogram entry for the vectorized kernel
  DIBuilder DIB(*VU.scalarFunction()->getParent(), false);
  DICompileUnit *CU =
      DIB.createCompileUnit(dwarf::DW_LANG_OpenCL, ScalarDI->getFile(), "",
                            ScalarDI->isOptimized(), "", 0);
  DISubprogram *const VectorDI = DIB.createFunction(
      CU->getFile(), ScalarDI->getName(),
      StringRef(), /* Don't need a linkage name */
      CU->getFile(), ScalarDI->getLine(), ScalarDI->getType(),
      ScalarDI->getScopeLine(), ScalarDI->getFlags(), ScalarDI->getSPFlags());

  // Point kernel function to a parent compile unit
  VectorDI->replaceUnit(ScalarDI->getUnit());

  VU.vectorizedFunction()->setSubprogram(VectorDI);

  DIB.finalize();

  // Iterate over all the instructions in the kernel looking for
  // intrinsics containing debug info metadata that must be updated.
  // Changing the scope to point to the new vectorized function, rather
  // than the scalar function.

  std::vector<Instruction *> DIIntrinsicsToDelete;
  std::vector<Metadata *> VectorizedLocals;

  for (auto &BBItr : *VU.vectorizedFunction()) {
    for (auto &InstItr : BBItr) {
      // Instruction is a llvm.dbg.value() or llvm.dbg.declare() intrinsic
      // TODO CA-1115 - Support llvm.dbg.addr() intrinsic
      if (DbgInfoIntrinsic *const DII = dyn_cast<DbgInfoIntrinsic>(&InstItr)) {
        // Delete this intrinsic later
        DIIntrinsicsToDelete.push_back(DII);

        // Generate a new DebugLoc pointing to vectorized function
        const DebugLoc &ScalarLoc = DII->getDebugLoc();

        // If location is inlined, we need to change the function it's inlined
        // into to our vectorized kernel, keeping the base location the same.
        DebugLoc VectorLoc;
        const DILocation *InlinedLoc = ScalarLoc.getInlinedAt();
        DISubprogram *OriginalFunc = VectorDI;

        if (InlinedLoc) {
          OriginalFunc = ScalarLoc->getScope()->getSubprogram();
          if (InlinedLoc->getInlinedAt()) {
            // We don't support nested inlined locations currently, abandon
            // creating dbg intrinsic as otherwise it will fail in validation.
            continue;
          }

          const DebugLoc InlinedAtLoc = getDILocation(
              InlinedLoc->getLine(), InlinedLoc->getColumn(), VectorDI);
          VectorLoc = getDILocation(ScalarLoc.getLine(), ScalarLoc.getCol(),
                                    ScalarLoc.getScope(), InlinedAtLoc);
        } else {
          VectorLoc =
              getDILocation(ScalarLoc.getLine(), ScalarLoc.getCol(), VectorDI);
        }

        // New DILocalVariable in the scope of vectorized function
        DILocalVariable *VectorLocal = nullptr;
        if (DbgValueInst *const DVI = dyn_cast<DbgValueInst>(DII)) {
          if (!DVI->getValue()) {
            // Debug value has been optimized out
            continue;
          }

          // Find DILocalVariable the intrinsic references
          const DILocalVariable *const ScalarLocal = DVI->getVariable();

          // Create a copy of DILocalVariable but in vectorized function scope
          if (ScalarLocal->getArg() == 0) {
            VectorLocal = DIB.createAutoVariable(
                OriginalFunc, ScalarLocal->getName(), ScalarLocal->getFile(),
                ScalarLocal->getLine(),
                dyn_cast<DIType>(ScalarLocal->getType()));
          } else {
            VectorLocal = DIB.createParameterVariable(
                OriginalFunc, ScalarLocal->getName(), ScalarLocal->getArg(),
                ScalarLocal->getFile(), ScalarLocal->getLine(),
                dyn_cast<DIType>(ScalarLocal->getType()));
          }

          // New llvm.dbg.value() with correct scope
          DIB.insertDbgValueIntrinsic(DVI->getValue(), VectorLocal,
                                      DVI->getExpression(), VectorLoc, DVI);
        } else if (DbgDeclareInst *const DDI = dyn_cast<DbgDeclareInst>(DII)) {
          // Find DILocalVariable the intrinsic references
          const DILocalVariable *const ScalarLocal = DDI->getVariable();

          // Create a copy of DILocalVariable but in vectorized function scope
          if (ScalarLocal->getArg() == 0) {
            VectorLocal = DIB.createAutoVariable(
                OriginalFunc, ScalarLocal->getName(), ScalarLocal->getFile(),
                ScalarLocal->getLine(),
                dyn_cast<DIType>(ScalarLocal->getType()));
          } else {
            VectorLocal = DIB.createParameterVariable(
                OriginalFunc, ScalarLocal->getName(), ScalarLocal->getArg(),
                ScalarLocal->getFile(), ScalarLocal->getLine(),
                dyn_cast<DIType>(ScalarLocal->getType()));
          }

          // New llvm.dbg.declare() with correct scope
          DIB.insertDeclare(DDI->getAddress(), VectorLocal,
                            DDI->getExpression(), VectorLoc, DDI);
        } else {
          continue;  // No other DbgInfoIntrinsic subclasses
        }

        if (VectorizedLocals.end() == std::find(VectorizedLocals.begin(),
                                                VectorizedLocals.end(),
                                                VectorLocal)) {
          VectorizedLocals.push_back(VectorLocal);
        }
      } else if (InstItr.getDebugLoc()) {
        // Update debug info line numbers to have vectorized kernel scope,
        // taking care to preserve inlined locations.
        const DebugLoc &ScalarLoc = InstItr.getDebugLoc();
        DebugLoc VectorLoc;
        if (DILocation *const InlinedLoc = ScalarLoc.getInlinedAt()) {
          // Don't support nested inlined locations for now
          if (!InlinedLoc->getInlinedAt()) {
            const DebugLoc VectorKernel = getDILocation(
                InlinedLoc->getLine(), InlinedLoc->getColumn(), VectorDI);
            VectorLoc = getDILocation(ScalarLoc.getLine(), ScalarLoc.getCol(),
                                      ScalarLoc.getScope(), VectorKernel);
          }
        } else {
          VectorLoc =
              getDILocation(ScalarLoc.getLine(), ScalarLoc.getCol(), VectorDI);
        }
        InstItr.setDebugLoc(VectorLoc);
      }
    }
  }

  // Delete intrinsics we have replaced
  for (auto Instr : DIIntrinsicsToDelete) {
    Instr->eraseFromParent();
  }

  // Replace temporary MDNode with list of vectorized DILocals we have created
  // In LLVM 7.0 the variables attribute of DISubprogram was changed to
  // retainedNodes
  auto *VectorizedKernelVariables = VectorDI->getRetainedNodes().get();
  assert(VectorizedKernelVariables && "Could not get retained nodes");
  if (VectorizedKernelVariables->isTemporary()) {
    auto NewLocals = MDTuple::getTemporary(
        VectorizedKernelVariables->getContext(), VectorizedLocals);
    VectorizedKernelVariables->replaceAllUsesWith(NewLocals.get());
  }

  return;
}

void cloneOpenCLMetadata(const VectorizationUnit &VU) {
  cloneOpenCLNamedMetadataHelper(VU, "opencl.kernels");
  cloneOpenCLNamedMetadataHelper(VU, "opencl.kernel_wg_size_info");
}

}  // namespace vecz
