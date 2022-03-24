//===-------- ModuleSplitter.cpp - split a module into callgraphs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "ModuleSplitter.h"
#include "DeviceGlobals.h"
#include "Support.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <map>
#include <utility>

using namespace llvm;
using namespace module_split;

namespace {

// Identifying name for global scope
constexpr char GLOBAL_SCOPE_NAME[] = "<GLOBAL>";
constexpr char SYCL_SCOPE_NAME[] = "<SYCL>";
constexpr char ESIMD_SCOPE_NAME[] = "<ESIMD>";

constexpr char ATTR_SYCL_MODULE_ID[] = "sycl-module-id";

// Describes scope covered by each entry in the module-entry points map
// populated by the groupEntryPointsByScope function.
enum EntryPointsGroupScope {
  Scope_PerKernel, // one entry per kernel
  Scope_PerModule, // one entry per module
  Scope_Global     // single entry in the map for all kernels
};

bool hasIndirectFunctionsOrCalls(const Module &M) {
  for (const auto &F : M.functions()) {
    // There are functions marked with [[intel::device_indirectly_callable]]
    // attribute, because it instructs us to make this function available to the
    // whole program as it was compiled as a single module.
    if (F.hasFnAttribute("referenced-indirectly"))
      return true;
    if (F.isDeclaration())
      continue;
    // There are indirect calls in the module, which means that we don't know
    // how to group functions so both caller and callee of indirect call are in
    // the same module.
    for (const auto &I : instructions(F)) {
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (!CI->getCalledFunction())
          return true;
    }

    // Function pointer is used somewhere. Follow the same rule as above.
    for (const auto *U : F.users())
      if (!isa<CallInst>(U))
        return true;
  }

  return false;
}

EntryPointsGroupScope selectDeviceCodeGroupScope(const Module &M,
                                                 IRSplitMode Mode,
                                                 bool IROutputOnly) {
  switch (Mode) {
  case SPLIT_PER_TU:
    return Scope_PerModule;

  case SPLIT_PER_KERNEL:
    return Scope_PerKernel;

  case SPLIT_AUTO: {
    if (IROutputOnly) {
      // We allow enabling auto split mode even in presence of -ir-output-only
      // flag, but in this case we are limited by it so we can't do any split
      // at all.
      return Scope_Global;
    }

    if (hasIndirectFunctionsOrCalls(M))
      return Scope_Global;

    // At the moment, we assume that per-source split is the best way of
    // splitting device code and can always be used except for cases handled
    // above.
    return Scope_PerModule;
  }

  case SPLIT_NONE:
    return Scope_Global;
  }

  llvm_unreachable("unsupported split mode");
}

// Return true if the function is a SPIRV or SYCL builtin, e.g.
// _Z28__spirv_GlobalInvocationId_xv
bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.startswith("__spirv_") || FName.startswith("__sycl_");
}

bool isEntryPoint(const Function &F, bool EmitOnlyKernelsAsEntryPoints) {
  // Skip declarations, if any: they should not be included into a vector of
  // entry points groups or otherwise we will end up with incorrectly generated
  // list of symbols.
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  if (CallingConv::SPIR_KERNEL == F.getCallingConv())
    return true;

  if (!EmitOnlyKernelsAsEntryPoints) {
    // If not disabled, SYCL_EXTERNAL functions with sycl-module-id attribute
    // are also considered as entry points (except __spirv_* and __sycl_*
    // functions)
    return F.hasFnAttribute(ATTR_SYCL_MODULE_ID) &&
           !isSpirvSyclBuiltin(F.getName());
  }

  return false;
}

// This function makes one or two groups depending on kernel types (SYCL, ESIMD)
// if SplitEsimd is true. Otherwise, all kernels are collected into one group.
EntryPointGroupVec
groupEntryPointsByKernelType(const Module &M, bool SplitEsimd,
                             bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointGroupVec EntryPointGroups{};
  std::map<StringRef, EntryPointVec> EntryPointMap;

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      continue;

    if (SplitEsimd && F.getMetadata("sycl_explicit_simd"))
      EntryPointMap[ESIMD_SCOPE_NAME].push_back(&F);
    else
      EntryPointMap[SYCL_SCOPE_NAME].push_back(&F);
  }

  if (!EntryPointMap.empty()) {
    for (auto &EPG : EntryPointMap)
      EntryPointGroups.push_back({EPG.first, std::move(EPG.second)});
  } else {
    // No entry points met, record this.
    EntryPointGroups.push_back({SYCL_SCOPE_NAME, {}});
  }

  return EntryPointGroups;
}

// This function decides how entry points of the input module M will be
// distributed ("split") into multiple modules based on the command options and
// IR attributes. The decision is recorded in the output vector EntryPointGroups
// which contains pairs of group id and entry points for that group. Each such
// group along with IR it depends on (globals, functions from its call graph,
// ...) will constitute a separate module.
EntryPointGroupVec groupEntryPointsByScope(const Module &M,
                                           EntryPointsGroupScope EntryScope,
                                           bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointGroupVec EntryPointGroups{};
  std::map<StringRef, EntryPointVec> EntryPointMap;

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      continue;

    switch (EntryScope) {
    case Scope_PerKernel:
      EntryPointMap[F.getName()].push_back(&F);
      break;

    case Scope_PerModule: {
      if (!F.hasFnAttribute(ATTR_SYCL_MODULE_ID))
        // TODO It may make sense to group all entry points w/o the attribute
        // into a separate module rather than issuing an error. Should probably
        // be controlled by an option.
        error("no '" + Twine(ATTR_SYCL_MODULE_ID) +
              "' attribute for entry point '" + F.getName() +
              "', per-module split is not possible");

      Attribute Id = F.getFnAttribute(ATTR_SYCL_MODULE_ID);
      StringRef Val = Id.getValueAsString();
      EntryPointMap[Val].push_back(&F);
      break;
    }

    case Scope_Global:
      // the map key is not significant here
      EntryPointMap[GLOBAL_SCOPE_NAME].push_back(&F);
      break;
    }
  }

  if (!EntryPointMap.empty()) {
    EntryPointGroups.reserve(EntryPointMap.size());
    for (auto &EPG : EntryPointMap)
      EntryPointGroups.push_back({EPG.first, std::move(EPG.second)});
  } else {
    // No entry points met, record this.
    EntryPointGroups.push_back({GLOBAL_SCOPE_NAME, {}});
  }

  return EntryPointGroups;
}

// For device global variables with the 'device_image_scope' property,
// the function checks that there are no usages of a single device global
// variable from kernels grouped to different modules.
void checkImageScopedDeviceGlobals(const Module &M,
                                   const EntryPointGroupVec &Groups) {
  // Early exit if there is only one group
  if (Groups.size() < 2)
    return;

  // Reverse the EntryPointGroupMap to get a map of entry point -> module's name
  unsigned EntryPointNumber = 0;
  for (const auto &Group : Groups)
    EntryPointNumber += static_cast<unsigned>(Group.Functions.size());
  DenseMap<const Function *, StringRef> EntryPointModules(EntryPointNumber);
  for (const auto &Group : Groups)
    for (const auto *F : Group.Functions)
      EntryPointModules.insert({F, Group.GroupId});

  // Processing device global variables with the "device_image_scope" property
  for (auto &GV : M.globals()) {
    if (!isDeviceGlobalVariable(GV) || !hasDeviceImageScopeProperty(GV))
      continue;

    Optional<StringRef> VarEntryPointModule{};
    auto CheckEntryPointModule = [&VarEntryPointModule, &EntryPointModules,
                                  &GV](const auto *F) {
      auto EntryPointModulesIt = EntryPointModules.find(F);
      assert(EntryPointModulesIt != EntryPointModules.end() &&
             "There is no group for an entry point");
      if (!VarEntryPointModule.hasValue()) {
        VarEntryPointModule = EntryPointModulesIt->second;
        return;
      }
      if (EntryPointModulesIt->second != *VarEntryPointModule) {
        error("device_global variable '" + Twine(GV.getName()) +
              "' with property \"device_image_scope\" is used in more "
              "than one device image.");
      }
    };

    SmallSetVector<const User *, 32> Workqueue;
    for (auto *U : GV.users())
      Workqueue.insert(U);

    while (!Workqueue.empty()) {
      const User *U = Workqueue.pop_back_val();
      if (auto *I = dyn_cast<const Instruction>(U)) {
        auto *F = I->getFunction();
        Workqueue.insert(F);
        continue;
      }
      if (auto *F = dyn_cast<const Function>(U)) {
        if (EntryPointModules.count(F))
          CheckEntryPointModule(F);
      }
      for (auto *UU : U->users())
        Workqueue.insert(UU);
    }
  }
}

void collectFunctionsToExtract(SetVector<const GlobalValue *> &GVs,
                               const EntryPointGroup &ModuleEntryPoints) {
  for (const auto *F : ModuleEntryPoints.Functions)
    GVs.insert(F);

  // GVs has SetVector type. This type inserts a value only if it is not yet
  // present there. So, recursion is not expected here.
  decltype(GVs.size()) Idx = 0;
  while (Idx != GVs.size()) {
    const auto *F = cast<Function>(GVs[Idx++]);
    for (const auto &I : instructions(F))
      if (const auto *CB = dyn_cast<CallBase>(&I))
        if (const Function *CF = CB->getCalledFunction())
          if (!CF->isDeclaration())
            GVs.insert(CF);
  }
}

void collectGlobalVarsToExtract(SetVector<const GlobalValue *> &GVs,
                                const Module &M) {
  // It's not easy to trace global variable's uses inside needed functions
  // because global variable can be used inside a combination of operators, so
  // mark all global variables as needed and remove dead ones after cloning.
  // Notice. For device global variables with the 'device_image_scope' property,
  // removing dead ones is a must, the 'checkImageScopedDeviceGlobals' function
  // checks that there are no usages of a single device global variable with the
  // 'device_image_scope' property from multiple modules and the splitter must
  // not add such usages after the check.
  for (const auto &G : M.globals())
    GVs.insert(&G);
}

ModuleDesc extractSubModule(const Module &M,
                            const SetVector<const GlobalValue *> GVs,
                            const EntryPointGroup &ModuleEntryPoints) {
  ModuleDesc Res{nullptr, ModuleEntryPoints};

  // For each group of entry points collect all dependencies.
  ValueToValueMapTy VMap;
  // Clone definitions only for needed globals. Others will be added as
  // declarations and removed later.
  Res.M = CloneModule(M, VMap,
                      [&](const GlobalValue *GV) { return GVs.count(GV); });

  EntryPointVec &NewEPs = Res.EntryPoints.Functions;
  // replace entry points with cloned ones
  std::for_each(NewEPs.begin(), NewEPs.end(),
                [&VMap](const Function *F) { return cast<Function>(VMap[F]); });

  return Res;
}

// TODO: try to move including all passes (cleanup, spec consts, compile time
// properties) in one place and execute MPM.run() only once.
void cleanupSplitModule(Module &SplitM) {
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  ModulePassManager MPM;
  // Do cleanup.
  MPM.addPass(GlobalDCEPass());           // Delete unreachable globals.
  MPM.addPass(StripDeadDebugInfoPass());  // Remove dead debug info.
  MPM.addPass(StripDeadPrototypesPass()); // Remove dead func decls.
  MPM.run(SplitM, MAM);
}

// The function produces a copy of input LLVM IR module M with only those entry
// points that are specified in ModuleEntryPoints vector.
ModuleDesc extractCallGraph(const Module &M,
                            const EntryPointGroup &ModuleEntryPoints) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsToExtract(GVs, ModuleEntryPoints);
  collectGlobalVarsToExtract(GVs, M);

  ModuleDesc SplitM = extractSubModule(M, GVs, ModuleEntryPoints);
  cleanupSplitModule(SplitM.getModule());

  return SplitM;
}

class ModuleCopier : public ModuleSplitterBase {
public:
  using ModuleSplitterBase::ModuleSplitterBase; // to inherit base constructors

  ModuleDesc nextSplit() override {
    return {releaseInputModule(), nextGroup()};
  }
};

class ModuleSplitter : public ModuleSplitterBase {
public:
  using ModuleSplitterBase::ModuleSplitterBase; // to inherit base constructors

  ModuleDesc nextSplit() override {
    return extractCallGraph(getInputModule(), nextGroup());
  }
};

} // namespace

bool module_split::ModuleDesc::isEsimd() {
  return (EntryPoints.GroupId == ESIMD_SCOPE_NAME);
}

std::unique_ptr<ModuleSplitterBase>
module_split::getSplitterByKernelType(std::unique_ptr<Module> M,
                                      bool SplitEsimd,
                                      bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointGroupVec Groups = groupEntryPointsByKernelType(
      *M, SplitEsimd, EmitOnlyKernelsAsEntryPoints);
  bool DoSplit = (Groups.size() > 1);

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(M), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(M), std::move(Groups));
}

std::unique_ptr<ModuleSplitterBase> module_split::getSplitterByMode(
    std::unique_ptr<Module> M, IRSplitMode Mode, bool IROutputOnly,
    bool EmitOnlyKernelsAsEntryPoints, bool DeviceGlobals) {
  EntryPointsGroupScope Scope =
      selectDeviceCodeGroupScope(*M, Mode, IROutputOnly);
  EntryPointGroupVec Groups =
      groupEntryPointsByScope(*M, Scope, EmitOnlyKernelsAsEntryPoints);
  assert(!Groups.empty() && "At least one group is expected");
  if (DeviceGlobals)
    checkImageScopedDeviceGlobals(*M, Groups);
  bool DoSplit = (Mode != SPLIT_NONE &&
                  (Groups.size() > 1 || !Groups.cbegin()->Functions.empty()));

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(M), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(M), std::move(Groups));
}
