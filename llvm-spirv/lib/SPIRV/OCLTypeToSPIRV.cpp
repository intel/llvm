//===- OCLTypeToSPIRV.cpp - Adapt types from OCL for SPIRV ------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements adaptation of OCL types for SPIR-V.
//
// It first maps kernel arguments of OCL opaque types to SPIR-V type, then
// propagates the mapping to the uses of the kernel arguments.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "cltytospv"

#include "OCLTypeToSPIRV.h"
#include "OCLUtil.h"
#include "SPIRVInternal.h"

#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include <iterator>
#include <set>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

char OCLTypeToSPIRVLegacy::ID = 0;

OCLTypeToSPIRVLegacy::OCLTypeToSPIRVLegacy() : ModulePass(ID) {
  initializeOCLTypeToSPIRVLegacyPass(*PassRegistry::getPassRegistry());
}

void OCLTypeToSPIRVLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool OCLTypeToSPIRVLegacy::runOnModule(Module &M) {
  return runOCLTypeToSPIRV(M);
}

OCLTypeToSPIRVBase &OCLTypeToSPIRVPass::run(llvm::Module &M,
                                            llvm::ModuleAnalysisManager &MAM) {
  runOCLTypeToSPIRV(M);
  return *this;
}

OCLTypeToSPIRVBase::OCLTypeToSPIRVBase()
    : BuiltinCallHelper(ManglingRules::None), M(nullptr), Ctx(nullptr) {}

bool OCLTypeToSPIRVBase::runOCLTypeToSPIRV(Module &Module) {
  LLVM_DEBUG(dbgs() << "Enter OCLTypeToSPIRV:\n");
  initialize(Module);
  M = &Module;
  Ctx = &M->getContext();
  AdaptedTy.clear();
  WorkSet.clear();
  auto Src = getSPIRVSource(&Module);
  if (std::get<0>(Src) != spv::SourceLanguageOpenCL_C)
    return false;

  for (auto &F : Module.functions())
    adaptArgumentsByMetadata(&F);

  for (auto &F : Module.functions())
    adaptFunctionArguments(&F);

  adaptArgumentsBySamplerUse(Module);

  while (!WorkSet.empty()) {
    Function *F = *WorkSet.begin();
    WorkSet.erase(WorkSet.begin());

    adaptFunction(F);
  }

  return false;
}

void OCLTypeToSPIRVBase::addAdaptedType(Value *V, Type *Ty) {
  LLVM_DEBUG(dbgs() << "[add adapted type] ";
             V->printAsOperand(dbgs(), true, M);
             dbgs() << " => " << *Ty << '\n');
  AdaptedTy[V] = Ty;
}

void OCLTypeToSPIRVBase::addWork(Function *F) {
  LLVM_DEBUG(dbgs() << "[add work] "; F->printAsOperand(dbgs(), true, M);
             dbgs() << '\n');
  WorkSet.insert(F);
}

/// Create a new function type if \param F has arguments in AdaptedTy, and
/// propagates the adapted arguments to functions called by \param F.
void OCLTypeToSPIRVBase::adaptFunction(Function *F) {
  LLVM_DEBUG(dbgs() << "\n[work on function] ";
             F->printAsOperand(dbgs(), true, M); dbgs() << '\n');
  assert(AdaptedTy.count(F) == 0);

  std::vector<Type *> ArgTys;
  bool Changed = false;
  for (auto &I : F->args()) {
    auto Loc = AdaptedTy.find(&I);
    auto Found = (Loc != AdaptedTy.end());
    Changed |= Found;
    ArgTys.push_back(Found ? Loc->second : I.getType());

    if (Found) {
      Type *Ty = Loc->second;
      for (auto &U : I.uses()) {
        if (auto *CI = dyn_cast<CallInst>(U.getUser())) {
          auto ArgIndex = CI->getArgOperandNo(&U);
          auto *CF = CI->getCalledFunction();
          if (AdaptedTy.count(CF) == 0) {
            addAdaptedType(CF->getArg(ArgIndex), Ty);
            addWork(CF);
          }
        }
      }
    }
  }

  if (!Changed)
    return;

  auto *FT = F->getFunctionType();
  FT = FunctionType::get(FT->getReturnType(), ArgTys, FT->isVarArg());
  addAdaptedType(F, TypedPointerType::get(FT, 0));
}

// Handle functions with sampler arguments that don't get called by
// a kernel function.
void OCLTypeToSPIRVBase::adaptArgumentsBySamplerUse(Module &M) {
  SmallPtrSet<Function *, 5> Processed;

  std::function<void(Function *, unsigned)> TraceArg = [&](Function *F,
                                                           unsigned Idx) {
    // If we have cycles in the call graph in the future, bail out
    // if we've already processed this function.
    if (Processed.insert(F).second == false)
      return;

    for (auto *U : F->users()) {
      auto *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      auto *SamplerArg = CI->getArgOperand(Idx);
      if (!isa<Argument>(SamplerArg) ||
          AdaptedTy.count(SamplerArg) != 0) // Already traced this, move on.
        continue;

      addAdaptedType(SamplerArg, getSPIRVType(OpTypeSampler));
      auto *Caller = cast<Argument>(SamplerArg)->getParent();
      addWork(Caller);
      TraceArg(Caller, cast<Argument>(SamplerArg)->getArgNo());
    }
  };

  for (auto &F : M) {
    if (!F.empty()) // not decl
      continue;
    auto MangledName = F.getName();
    StringRef DemangledName;
    if (!oclIsBuiltin(MangledName, DemangledName, false))
      continue;
    if (DemangledName.find(kSPIRVName::SampledImage) == std::string::npos)
      continue;

    TraceArg(&F, 1);
  }
}

void OCLTypeToSPIRVBase::adaptFunctionArguments(Function *F) {
  auto *TypeMD = F->getMetadata(SPIR_MD_KERNEL_ARG_BASE_TYPE);
  if (TypeMD)
    return;
  bool Changed = false;
  auto *Arg = F->arg_begin();
  SmallVector<Type *, 4> ParamTys;

  // If we couldn't get any information from demangling, there is nothing that
  // can be done.
  if (!getParameterTypes(F, ParamTys))
    return;

  for (unsigned I = 0; I < F->arg_size(); ++I, ++Arg) {
    StructType *NewTy = nullptr;
    if (auto *TPT = dyn_cast<TypedPointerType>(ParamTys[I]))
      NewTy = dyn_cast_or_null<StructType>(TPT->getElementType());
    if (NewTy && NewTy->isOpaque()) {
      auto STName = NewTy->getStructName();
      if (!hasAccessQualifiedName(STName))
        continue;
      if (STName.starts_with(kSPR2TypeName::ImagePrefix)) {
        auto Ty = STName.str();
        auto Acc = getAccessQualifier(Ty);
        auto Desc = getImageDescriptor(ParamTys[I]);
        addAdaptedType(
            &*Arg, getSPIRVType(OpTypeImage, Type::getVoidTy(*Ctx), Desc, Acc));
        Changed = true;
      }
    }
  }
  if (Changed)
    addWork(F);
}

/// Go through all kernel functions, get access qualifier for image and pipe
/// types and use them to map the function arguments to the SPIR-V type.
/// ToDo: Map other OpenCL opaque types to SPIR-V types.
void OCLTypeToSPIRVBase::adaptArgumentsByMetadata(Function *F) {
  auto *TypeMD = F->getMetadata(SPIR_MD_KERNEL_ARG_BASE_TYPE);
  if (!TypeMD)
    return;
  bool Changed = false;
  auto *Arg = F->arg_begin();
  for (unsigned I = 0, E = TypeMD->getNumOperands(); I != E; ++I, ++Arg) {
    auto OCLTyStr = getMDOperandAsString(TypeMD, I);
    if (OCLTyStr == OCL_TYPE_NAME_SAMPLER_T) {
      addAdaptedType(&(*Arg), getSPIRVType(OpTypeSampler));
      Changed = true;
    } else if (OCLTyStr.starts_with("image") && OCLTyStr.ends_with("_t")) {
      auto Ty = (Twine("opencl.") + OCLTyStr).str();
      if (auto *STy = StructType::getTypeByName(F->getContext(), Ty)) {
        auto *ImageTy = TypedPointerType::get(STy, SPIRAS_Global);
        auto Desc = getImageDescriptor(ImageTy);
        auto *AccMD = F->getMetadata(SPIR_MD_KERNEL_ARG_ACCESS_QUAL);
        assert(AccMD && "Invalid access qualifier metadata");
        auto Acc = SPIRSPIRVAccessQualifierMap::map(
            getMDOperandAsString(AccMD, I).str());
        addAdaptedType(
            &*Arg, getSPIRVType(OpTypeImage, Type::getVoidTy(*Ctx), Desc, Acc));
        Changed = true;
      }
    }
  }
  if (Changed)
    addWork(F);
}

// OCL sampler, image and pipe type need to be regularized before converting
// to SPIRV types.
//
// OCL sampler type is represented as i32 in LLVM, however in SPIRV it is
// represented as OpTypeSampler. Also LLVM uses the same pipe type to
// represent pipe types with different underlying data types, however
// in SPIRV they are different types. OCL image and pipe types do not
// encode access qualifier, which is part of SPIRV types for image and pipe.
//
// The function types in LLVM need to be regularized before translating
// to SPIRV function types:
//
// sampler type as i32 -> opencl.sampler_t opaque type
// opencl.pipe_t opaque type with underlying opencl type x and access
//   qualifier y -> opencl.pipe_t.x.y opaque type
// opencl.image_x opaque type with access qualifier y ->
//   opencl.image_x.y opaque type
//
// The converter relies on kernel_arg_base_type to identify the sampler
// type, the underlying data type of pipe type, and access qualifier for
// image and pipe types. The FE is responsible to generate the correct
// kernel_arg_base_type metadata.
//
// Alternatively,the FE may choose to use opencl.sampler_t to represent
// sampler type, use opencl.pipe_t.x.y to represent pipe type with underlying
// opencl data type x and access qualifier y, and use opencl.image_x.y to
// represent image_x type with access qualifier y.
//
Type *OCLTypeToSPIRVBase::getAdaptedArgumentType(Function *F, unsigned ArgNo) {
  Value *Arg = F->getArg(ArgNo);
  auto Loc = AdaptedTy.find(Arg);
  if (Loc == AdaptedTy.end())
    return nullptr;
  return Loc->second;
}

} // namespace SPIRV

AnalysisKey OCLTypeToSPIRVPass::Key;

INITIALIZE_PASS(OCLTypeToSPIRVLegacy, "cltytospv", "Adapt OCL types for SPIR-V",
                false, true)

ModulePass *llvm::createOCLTypeToSPIRVLegacy() {
  return new OCLTypeToSPIRVLegacy();
}
