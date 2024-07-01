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

/// @file
///
/// @brief Defines the RenameStructsPass.

#include <compiler/utils/StructTypeRemapper.h>
#include <compiler/utils/unique_opaque_structs_pass.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <multi_llvm/multi_llvm.h>

using namespace compiler::utils;
using namespace llvm;

/// @brief Indicates whether a function needs to be cloned.
///
/// There are a few ways the undesirable types can exist in a function:
/// * As a return type.
/// * As a parameter type.
/// * As a call to a function returning undesirable type.
/// * The result of an alloca.
/// * Result of a cast of some type.
/// * Reference to a global.
///
/// @param[in] StructTypeRemapper Map from suffixed opaque structs to
/// unsuffixed opaque structs.
/// @param[in] Function function to be checked for cloning.
///
/// @return Whether function should be cloned.
/// @retval true if function should be cloned.
/// @retval false otherwise.
static bool shouldClone(compiler::utils::StructTypeRemapper &StructTypeRemapper,
                        const Function &Func) {
  // First check the return type.
  if (StructTypeRemapper.isRemapped(Func.getReturnType())) {
    return true;
  }

  // Then the arguments.
  for (const Argument &Arg : Func.args()) {
    if (StructTypeRemapper.isRemapped(Arg.getType())) {
      return true;
    }
  }

  // Now look for specific instructions that could introduce the type.
  for (auto &BB : Func) {
    for (auto &I : BB) {
      // We can catch any instruction that produces an undesirable type by
      // just checking its type.
      if (StructTypeRemapper.isRemapped(I.getType())) {
        return true;
      }
    }
  }

  // TODO: Check globals (see CA-3833).

  // If an instruction makes use of a type but
  // isn't of that type e.g. a cast it will necessarily get caught by
  // the above case as it is a use of something which produced that
  // type.

  // If we've got here, we've checked all the cases, so no need to clone.
  return false;
}

/// @brief Constructs a map of suffixed opaque structure types to their
/// unsuffixed versions.
///
/// If a module references opaque structs that have identical names up to a
/// suffix within the context, e.g. opencl.event_t and opencl.event_t this
/// function will return a map mapping the suffixed versions to the unsuffixed
/// versions e.g. map[opencl.event_t.0] = opencl.event_t.
///
/// @param module Module referencing the types in the context.
///
/// @return The map of suffixed structures to the unsuffixed structures.
static compiler::utils::StructMap uniqueOpaqueSuffixedStructs(
    llvm::Module &module) {
  StructMap map;
  for (auto *structTy : module.getIdentifiedStructTypes()) {
    if (!structTy->isOpaque()) {
      continue;
    }

    // Look up each struct in the module by name.
    auto structName = structTy->getName();
    const char *Suffix = ".0123456789";

    // Check whether there is a type in the context with the same name minus a
    // suffix.
    if (auto *ctxStructTy = llvm::StructType::getTypeByName(
            module.getContext(), structName.rtrim(Suffix))) {
      // Make sure it is also opaque.
      if (!ctxStructTy->isOpaque()) {
        continue;
      }

      // If it isn't the same type as the first map the suffixed
      // type to the unsuffixed type.
      if (ctxStructTy != structTy) {
        map[structTy] = ctxStructTy;
      }
    }
  }
  return map;
}

/// @brief Populates list of functions that need to be cloned.
///
/// @param[in] Module module containing the functions to be inspected.
/// @param[in] StructTypeRemapper Map from suffixed opaque structs to
/// unsuffixed opaque structs.
/// @param[out] WorkList vector of functions that need to be processed.
static void populateWorkList(
    Module &Module, compiler::utils::StructTypeRemapper &StructTypeRemapper,
    SmallVectorImpl<Function *> &WorkList) {
  for (auto &Function : Module) {
    // We don't need to touch intrinsics.
    if (Function.isIntrinsic()) {
      continue;
    }

    // Check the function for undesirable types.
    if (shouldClone(StructTypeRemapper, Function)) {
      WorkList.push_back(&Function);
    }
  }
}

static void removeOldFunctions(const SmallVectorImpl<Function *> &OldFuncs) {
  // First we have to delete the bodies of the functions, otherwise we will
  // get issues about uses missing their defs.
  for (auto &OldFunc : OldFuncs) {
    OldFunc->deleteBody();
  }

  // Now we can delete the actual functions.
  for (auto &OldFunc : OldFuncs) {
    OldFunc->eraseFromParent();
  }
}

/// @brief Clones a list of functions updating types within the function.
///
/// Clones a list of functions updating the types of any instances of the
/// undesirable types according to the map that was passed to this pass. A new
/// call graph is constructed and the old functions names are taken by the
/// new functions.
///
/// @param[in] StructTypeRemapper Map from suffixed opaque structs to
/// unsuffixed opaque structs.
/// @param[in] OldFuncs list of functions to clone and update.
static void replaceRemappedTypeRefs(
    compiler::utils::StructTypeRemapper &StructTypeRemapper,
    const SmallVectorImpl<Function *> &OldFuncs) {
  // Maps the old functions to their new versions with updated types.
  // Note: it is important we do this before cloning to catch the case that
  // functions A and B both need updating, but function A calls function B and
  // A is processed before B, otherwise function calls won't be updated during
  // the clone.
  SmallDenseMap<Function *, Function *> FFMap;
  for (auto &OldFunc : OldFuncs) {
    auto *OldFuncTy = OldFunc->getFunctionType();
    // First map the return type.
    auto *RetTy = StructTypeRemapper.remapType(OldFuncTy->getReturnType());

    // Then map the parameter types.
    SmallVector<Type *, 4> ParamTys;
    for (auto ParamTy : OldFuncTy->params()) {
      ParamTys.push_back(StructTypeRemapper.remapType(ParamTy));
    }

    // Create the new function with updated types.
    auto *NewFuncTy = FunctionType::get(RetTy, ParamTys, OldFuncTy->isVarArg());
    auto *NewFunc = Function::Create(NewFuncTy, OldFunc->getLinkage(), "",
                                     OldFunc->getParent());
    NewFunc->setCallingConv(OldFunc->getCallingConv());

    FFMap[OldFunc] = NewFunc;
  }

  // Here we actually do the cloning.
  for (auto &OldFunc : OldFuncs) {
    // We construct a new value map on each iteration to avoid entries in the
    // value map potentially being overwritten during cloning which would then
    // be used be subsequent loop iterations.
    ValueToValueMapTy ValueMap;
    for (auto &pair : FFMap) {
      ValueMap[pair.getFirst()] = pair.getSecond();
    }
    auto *NewFunc = FFMap[OldFunc];
    auto NewArgIterator = NewFunc->arg_begin();
    for (llvm::Argument &Arg : OldFunc->args()) {
      NewArgIterator->setName(Arg.getName());
      ValueMap[&Arg] = &*(NewArgIterator++);
    }
    NewFunc->takeName(OldFunc);

    if (OldFunc->isDeclaration()) {
      // Everything that follows requires a body.
      continue;
    }

    SmallVector<ReturnInst *, 4> Returns;
    CloneFunctionInto(NewFunc, OldFunc, ValueMap,
                      CloneFunctionChangeType::GlobalChanges, Returns, "",
                      /* CodeInfo */ nullptr, &StructTypeRemapper);
    Returns.clear();

    // It's possible we still have references to the old types in our new
    // new function, this can happen via allocas and cast as well as
    // references to global variables.
    for (auto &BB : *NewFunc) {
      for (auto &I : BB) {
        // Anything that defines a undesirable instance will get caught
        // here.
        I.mutateType(StructTypeRemapper.remapType(I.getType()));

        // GEP instructions need to be handled separately.
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          if (StructTypeRemapper.isRemapped(GEP->getSourceElementType())) {
            GEP->setSourceElementType(
                StructTypeRemapper.remapType(GEP->getSourceElementType()));
          }
        }
      }
    }
  }

  // We can now remove any of the misnamed types and any functions that used
  // them.
  removeOldFunctions(OldFuncs);
}

namespace compiler {
namespace utils {
PreservedAnalyses UniqueOpaqueStructsPass::run(Module &Module,
                                               ModuleAnalysisManager &) {
  // Find the opaque types in the module that have suffixes and map them to
  // their unsuffixed versions.
  auto StructMap = uniqueOpaqueSuffixedStructs(Module);
  StructTypeRemapper StructTypeRemapper(StructMap);

  // Build the list of functions we need to process.
  SmallVector<Function *, 8> WorkList;
  populateWorkList(Module, StructTypeRemapper, WorkList);

  // If the set is empty we have no work and can exit early.
  if (WorkList.empty()) {
    return PreservedAnalyses::all();
  }

  // Otherwise, clone the functions, updating the types.
  replaceRemappedTypeRefs(StructTypeRemapper, WorkList);

  // We definitely cloned something by this point, so the module has been
  // modified.
  return PreservedAnalyses::none();
}
}  // namespace utils
}  // namespace compiler
