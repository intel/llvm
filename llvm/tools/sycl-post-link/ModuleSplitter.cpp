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
using namespace llvm::module_split;

namespace {

// Identifying name for global scope
constexpr char GLOBAL_SCOPE_NAME[] = "<GLOBAL>";
constexpr char SYCL_SCOPE_NAME[] = "<SYCL>";
constexpr char ESIMD_SCOPE_NAME[] = "<ESIMD>";
constexpr char ESIMD_MARKER_MD[] = "sycl_explicit_simd";

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
                                                 IRSplitMode Mode) {
  switch (Mode) {
  case SPLIT_PER_TU:
    return Scope_PerModule;

  case SPLIT_PER_KERNEL:
    return Scope_PerKernel;

  case SPLIT_AUTO: {
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

bool isESIMDFunction(const Function &F) {
  return F.getMetadata(ESIMD_MARKER_MD) != nullptr;
}

// This function makes one or two groups depending on kernel types (SYCL, ESIMD)
EntryPointGroupVec
groupEntryPointsByKernelType(const Module &M, bool EmitOnlyKernelsAsEntryPoints,
                             EntryPointVec *AllowedEntriesVec) {
  SmallPtrSet<const Function *, 32> AllowedEntries;

  if (AllowedEntriesVec) {
    std::copy(AllowedEntriesVec->begin(), AllowedEntriesVec->end(),
              std::inserter(AllowedEntries, AllowedEntries.end()));
  }
  EntryPointGroupVec EntryPointGroups{};
  std::map<StringRef, EntryPointVec> EntryPointMap;

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      continue;
    if (AllowedEntriesVec && (AllowedEntries.find(&F) == AllowedEntries.end()))
      continue;

    if (isESIMDFunction(F))
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
                            EntryPointGroup &&ModuleEntryPoints) {
  // For each group of entry points collect all dependencies.
  ValueToValueMapTy VMap;
  // Clone definitions only for needed globals. Others will be added as
  // declarations and removed later.
  std::unique_ptr<Module> SubM = CloneModule(
      M, VMap, [&](const GlobalValue *GV) { return GVs.count(GV); });
  // Replace entry points with cloned ones.
  EntryPointVec NewEPs;
  const EntryPointVec &EPs = ModuleEntryPoints.Functions;
  NewEPs.reserve(EPs.size());
  std::transform(
      EPs.cbegin(), EPs.cend(), std::inserter(NewEPs, NewEPs.end()),
      [&VMap](const Function *F) { return cast<Function>(VMap[F]); });
  ModuleEntryPoints.Functions = std::move(NewEPs);
  return ModuleDesc{std::move(SubM), std::move(ModuleEntryPoints)};
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
                            EntryPointGroup &&ModuleEntryPoints) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsToExtract(GVs, ModuleEntryPoints);
  collectGlobalVarsToExtract(GVs, M);

  ModuleDesc SplitM = extractSubModule(M, GVs, std::move(ModuleEntryPoints));
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

namespace llvm {
namespace module_split {

std::unique_ptr<ModuleSplitterBase>
getSplitterByKernelType(std::unique_ptr<Module> M,
                        bool EmitOnlyKernelsAsEntryPoints,
                        EntryPointVec *AllowedEntries) {
  EntryPointGroupVec Groups = groupEntryPointsByKernelType(
      *M, EmitOnlyKernelsAsEntryPoints, AllowedEntries);
  bool DoSplit = (Groups.size() > 1);

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(M), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(M), std::move(Groups));
}

std::unique_ptr<ModuleSplitterBase>
getSplitterByMode(std::unique_ptr<Module> M, IRSplitMode Mode,
                  bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointsGroupScope Scope = selectDeviceCodeGroupScope(*M, Mode);
  EntryPointGroupVec Groups =
      groupEntryPointsByScope(*M, Scope, EmitOnlyKernelsAsEntryPoints);
  assert(!Groups.empty() && "At least one group is expected");
  bool DoSplit = (Mode != SPLIT_NONE &&
                  (Groups.size() > 1 || !Groups.cbegin()->Functions.empty()));

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(M), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(M), std::move(Groups));
}

void ModuleSplitterBase::verifyNoCrossModuleDeviceGlobalUsage() {
  const Module &M = *InputModule;
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

#ifndef _NDEBUG

const char *toString(ESIMDStatus S) {
  switch (S) {
  case ESIMDStatus::ESIMD_ONLY:
    return "ESIMD_ONLY";
  case ESIMDStatus::SYCL_ONLY:
    return "SYCL_ONLY";
  case ESIMDStatus::SYCL_AND_ESIMD:
    return "SYCL_AND_ESIMD";
  }
  return "<UNKNOWN_STATUS>";
}

void tab(int N) {
  for (int I = 0; I < N; ++I) {
    llvm::errs() << "  ";
  }
}

void dumpEntryPoints(const EntryPointVec &C, const char *msg, int Tab) {
  tab(Tab);
  llvm::errs() << "ENTRY POINTS"
               << " " << msg << " {\n";
  for (const Function *F : C) {
    tab(Tab);
    llvm::errs() << "  " << F->getName() << "\n";
  }
  tab(Tab);
  llvm::errs() << "}\n";
}

void dumpEntryPoints(const Module &M, bool OnlyKernelsAreEntryPoints,
                     const char *msg, int Tab) {
  tab(Tab);
  llvm::errs() << "ENTRY POINTS (Module)"
               << " " << msg << " {\n";
  for (const auto &F : M) {
    if (isEntryPoint(F, OnlyKernelsAreEntryPoints)) {
      tab(Tab);
      llvm::errs() << "  " << F.getName() << "\n";
    }
  }
  tab(Tab);
  llvm::errs() << "}\n";
}

// Updates Props.HasEsimd to ESIMDStatus::ESIMD_ONLY/SYCL_ONLY if this module
// descriptor is a ESIMD/SYCL part of the ESIMD/SYCL module split. Otherwise
// assumes the module has both SYCL and ESIMD.

void ModuleDesc::assignESIMDProperty() {
  if (EntryPoints.isEsimd()) {
    Props.HasEsimd = ESIMDStatus::ESIMD_ONLY;
  } else if (EntryPoints.isSycl()) {
    Props.HasEsimd = ESIMDStatus::SYCL_ONLY;
  } else {
    Props.HasEsimd = ESIMDStatus::SYCL_AND_ESIMD;
  }
#ifndef _NDEBUG
  verifyESIMDProperty();
#endif // _NDEBUG
}

#ifndef _NDEBUG
void ModuleDesc::verifyESIMDProperty() const {
  if (Props.HasEsimd == ESIMDStatus::SYCL_AND_ESIMD) {
    return; // nothing to verify
  }
  // Verify entry points:
  for (const auto *F : entries()) {
    const bool IsESIMDFunction = isESIMDFunction(*F);

    switch (Props.HasEsimd) {
    case ESIMDStatus::ESIMD_ONLY:
      assert(IsESIMDFunction);
      break;
    case ESIMDStatus::SYCL_ONLY:
      assert(!IsESIMDFunction);
      break;
    default:
      break;
    }
  }
  // No ESIMD functions expected in case of SYCL_ONLY:
  // TODO commented out as this fails with RoundedRangeKernel - when it is
  // created to wrap (call) an ESIMD kernel definition, it is not marked with
  // "sycl_explicit_simd" attribute in the API headers. Thus it leads to extra
  // "SYCL only" module during split. This existed before and needs to be fixed.
  // if (Props.HasEsimd == ESIMDStatus::SYCL_ONLY) {
  //  for (const auto &F : getModule()) {
  //    assert(!isESIMDFunction(F));
  //  }
  //}
}
#endif // _NDEBUG

void ModuleDesc::dump() {
  llvm::errs() << "split_module::ModuleDesc[" << Name << "] {\n";
  llvm::errs() << "  ESIMD:" << toString(Props.HasEsimd)
               << ", SpecConstMet:" << (Props.SpecConstsMet ? "YES" : "NO")
               << "\n";
  dumpEntryPoints(entries(), EntryPoints.getId().str().c_str(), 1);
  llvm::errs() << "}\n";
}
#endif // _NDEBUG

bool EntryPointGroup::isEsimd() const { return GroupId == ESIMD_SCOPE_NAME; }

bool EntryPointGroup::isSycl() const { return GroupId == SYCL_SCOPE_NAME; }

void EntryPointGroup::saveNames(std::vector<std::string> &Dest) const {
  Dest.reserve(Dest.size() + Functions.size());
  std::transform(Functions.cbegin(), Functions.cend(),
                 std::inserter(Dest, Dest.end()),
                 [](const Function *F) { return F->getName().str(); });
}

void EntryPointGroup::rebuildFromNames(const std::vector<std::string> &Names,
                                       const Module &M) {
  Functions.clear();
  Functions.reserve(Names.size());
  auto It0 = Names.cbegin();
  auto It1 = Names.cend();
  std::transform(It0, It1, std::inserter(Functions, Functions.begin()),
                 [&M](const std::string &Name) {
                   const Function *F = M.getFunction(Name);
                   assert(F && "entry point lost");
                   return F;
                 });
}

void EntryPointGroup::rebuild(const Module &M) {
  if (Functions.size() == 0) {
    return;
  }
  EntryPointVec NewFunctions;
  NewFunctions.reserve(Functions.size());
  auto It0 = Functions.cbegin();
  auto It1 = Functions.cend();
  std::transform(It0, It1, std::inserter(NewFunctions, NewFunctions.begin()),
                 [&M](const Function *F) {
                   Function *NewF = M.getFunction(F->getName());
                   assert(NewF && "entry point lost");
                   return NewF;
                 });
  Functions = std::move(NewFunctions);
}

} // namespace module_split
} // namespace llvm
