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
constexpr char ATTR_DOUBLE_GRF[] = "esimd-double-grf";

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
                                                 bool AutoSplitIsGlobalScope) {
  switch (Mode) {
  case SPLIT_PER_TU:
    return Scope_PerModule;

  case SPLIT_PER_KERNEL:
    return Scope_PerKernel;

  case SPLIT_AUTO: {
    if (hasIndirectFunctionsOrCalls(M) || AutoSplitIsGlobalScope)
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
groupEntryPointsByKernelType(const ModuleDesc &MD,
                             bool EmitOnlyKernelsAsEntryPoints) {
  const Module &M = MD.getModule();
  EntryPointGroupVec EntryPointGroups{};
  std::map<StringRef, EntryPointSet> EntryPointMap;

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints) ||
        !MD.isEntryPointCandidate(F))
      continue;

    if (isESIMDFunction(F))
      EntryPointMap[ESIMD_SCOPE_NAME].insert(&F);
    else
      EntryPointMap[SYCL_SCOPE_NAME].insert(&F);
  }

  if (!EntryPointMap.empty()) {
    for (auto &EPG : EntryPointMap) {
      EntryPointGroups.emplace_back(EntryPointGroup{
          EPG.first, std::move(EPG.second), MD.getEntryPointGroup().Props});
      EntryPointGroup &G = EntryPointGroups.back();

      if (G.GroupId == ESIMD_SCOPE_NAME) {
        G.Props.HasESIMD = SyclEsimdSplitStatus::ESIMD_ONLY;
      } else {
        assert(G.GroupId == SYCL_SCOPE_NAME);
        G.Props.HasESIMD = SyclEsimdSplitStatus::SYCL_ONLY;
      }
    }
  } else {
    // No entry points met, record this.
    EntryPointGroups.emplace_back(
        EntryPointGroup{SYCL_SCOPE_NAME, EntryPointSet{}});
    EntryPointGroup &G = EntryPointGroups.back();
    G.Props.HasESIMD = SyclEsimdSplitStatus::SYCL_ONLY;
  }

  return EntryPointGroups;
}

// This function decides how entry points of the input module M will be
// distributed ("split") into multiple modules based on the command options and
// IR attributes. The decision is recorded in the output vector EntryPointGroups
// which contains pairs of group id and entry points for that group. Each such
// group along with IR it depends on (globals, functions from its call graph,
// ...) will constitute a separate module.
EntryPointGroupVec groupEntryPointsByScope(const ModuleDesc &MD,
                                           EntryPointsGroupScope EntryScope,
                                           bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointGroupVec EntryPointGroups{};
  // Use MapVector for deterministic order of traversal (helps tests).
  MapVector<StringRef, EntryPointSet> EntryPointMap;
  const Module &M = MD.getModule();

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints) ||
        !MD.isEntryPointCandidate(F))
      continue;

    switch (EntryScope) {
    case Scope_PerKernel:
      EntryPointMap[F.getName()].insert(&F);
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
      EntryPointMap[Val].insert(&F);
      break;
    }

    case Scope_Global:
      // the map key is not significant here
      EntryPointMap[GLOBAL_SCOPE_NAME].insert(&F);
      break;
    }
  }

  if (!EntryPointMap.empty()) {
    EntryPointGroups.reserve(EntryPointMap.size());
    for (auto &EPG : EntryPointMap) {
      EntryPointGroups.emplace_back(EntryPointGroup{
          EPG.first, std::move(EPG.second), MD.getEntryPointGroup().Props});
      EntryPointGroup &G = EntryPointGroups.back();
      G.Props.Scope = EntryScope;
    }
  } else {
    // No entry points met, record this.
    EntryPointGroups.emplace_back(
        EntryPointGroup{GLOBAL_SCOPE_NAME, EntryPointSet{}});
  }
  return EntryPointGroups;
}

template <class EntryPoinGroupFunc>
EntryPointGroupVec
groupEntryPointsByAttribute(const ModuleDesc &MD, StringRef AttrName,
                            bool EmitOnlyKernelsAsEntryPoints,
                            EntryPoinGroupFunc F) {
  EntryPointGroupVec EntryPointGroups{};
  std::map<StringRef, EntryPointSet> EntryPointMap;
  const Module &M = MD.getModule();

  // Only process module entry points:
  for (const auto &F : M.functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints) ||
        !MD.isEntryPointCandidate(F)) {
      continue;
    }
    if (F.hasFnAttribute(AttrName)) {
      EntryPointMap[AttrName].insert(&F);
    } else {
      EntryPointMap[""].insert(&F);
    }
  }
  if (!EntryPointMap.empty()) {
    EntryPointGroups.reserve(EntryPointMap.size());
    for (auto &EPG : EntryPointMap) {
      EntryPointGroups.emplace_back(EntryPointGroup{
          EPG.first, std::move(EPG.second), MD.getEntryPointGroup().Props});
      F(EntryPointGroups.back());
    }
  } else {
    // No entry points met, record this.
    EntryPointGroups.push_back({GLOBAL_SCOPE_NAME, {}});
    F(EntryPointGroups.back());
  }
  return EntryPointGroups;
}

// Represents a call graph between functions in a module. Nodes are functions,
// edges are "calls" relation.
class CallGraph {
public:
  using FunctionSet = SmallPtrSet<const Function *, 16>;

private:
  std::unordered_map<const Function *, FunctionSet> Graph;
  SmallPtrSet<const Function *, 1> EmptySet;

public:
  CallGraph(const Module &M) {
    for (const auto &F : M) {
      for (const Value *U : F.users()) {
        if (const auto *I = dyn_cast<CallInst>(U)) {
          if (I->getCalledFunction() == &F) {
            const Function *F1 = I->getFunction();
            Graph[F1].insert(&F);
          }
        }
      }
    }
  }

  iterator_range<FunctionSet::const_iterator>
  successors(const Function *F) const {
    auto It = Graph.find(F);
    return (It == Graph.end())
               ? make_range(EmptySet.begin(), EmptySet.end())
               : make_range(It->second.begin(), It->second.end());
  }
};

void collectFunctionsToExtract(SetVector<const GlobalValue *> &GVs,
                               const EntryPointGroup &ModuleEntryPoints,
                               const CallGraph &Deps) {
  for (const auto *F : ModuleEntryPoints.Functions)
    GVs.insert(F);

  // GVs has SetVector type. This type inserts a value only if it is not yet
  // present there. So, recursion is not expected here.
  decltype(GVs.size()) Idx = 0;
  while (Idx < GVs.size()) {
    const auto *F = cast<Function>(GVs[Idx++]);

    for (const Function *F1 : Deps.successors(F)) {
      if (!F1->isDeclaration())
        GVs.insert(F1);
    }
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

ModuleDesc extractSubModule(const ModuleDesc &MD,
                            const SetVector<const GlobalValue *> GVs,
                            EntryPointGroup &&ModuleEntryPoints) {
  const Module &M = MD.getModule();
  // For each group of entry points collect all dependencies.
  ValueToValueMapTy VMap;
  // Clone definitions only for needed globals. Others will be added as
  // declarations and removed later.
  std::unique_ptr<Module> SubM = CloneModule(
      M, VMap, [&](const GlobalValue *GV) { return GVs.count(GV); });
  // Replace entry points with cloned ones.
  EntryPointSet NewEPs;
  const EntryPointSet &EPs = ModuleEntryPoints.Functions;
  std::for_each(EPs.begin(), EPs.end(), [&](const Function *F) {
    NewEPs.insert(cast<Function>(VMap[F]));
  });
  ModuleEntryPoints.Functions = std::move(NewEPs);
  return ModuleDesc{std::move(SubM), std::move(ModuleEntryPoints), MD.Props};
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
ModuleDesc extractCallGraph(const ModuleDesc &MD,
                            EntryPointGroup &&ModuleEntryPoints) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsToExtract(GVs, ModuleEntryPoints, CallGraph{MD.getModule()});
  collectGlobalVarsToExtract(GVs, MD.getModule());

  ModuleDesc SplitM = extractSubModule(MD, GVs, std::move(ModuleEntryPoints));
  cleanupSplitModule(SplitM.getModule());

  return SplitM;
}

class ModuleCopier : public ModuleSplitterBase {
public:
  using ModuleSplitterBase::ModuleSplitterBase; // to inherit base constructors

  ModuleDesc nextSplit() override {
    return ModuleDesc{releaseInputModule(), nextGroup(), Input.Props};
  }
};

class ModuleSplitter : public ModuleSplitterBase {
public:
  ModuleSplitter(ModuleDesc &&MD, EntryPointGroupVec &&GroupVec)
      : ModuleSplitterBase(std::move(MD), std::move(GroupVec)) {}

  ModuleDesc nextSplit() override {
    return extractCallGraph(Input, nextGroup());
  }
};

} // namespace

namespace llvm {
namespace module_split {

std::unique_ptr<ModuleSplitterBase>
getSplitterByKernelType(ModuleDesc &&MD, bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointGroupVec Groups =
      groupEntryPointsByKernelType(MD, EmitOnlyKernelsAsEntryPoints);
  bool DoSplit = (Groups.size() > 1);

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(MD), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(MD), std::move(Groups));
}

std::unique_ptr<ModuleSplitterBase>
getSplitterByMode(ModuleDesc &&MD, IRSplitMode Mode,
                  bool AutoSplitIsGlobalScope,
                  bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointsGroupScope Scope =
      selectDeviceCodeGroupScope(MD.getModule(), Mode, AutoSplitIsGlobalScope);
  EntryPointGroupVec Groups =
      groupEntryPointsByScope(MD, Scope, EmitOnlyKernelsAsEntryPoints);
  assert(!Groups.empty() && "At least one group is expected");
  bool DoSplit = (Mode != SPLIT_NONE &&
                  (Groups.size() > 1 || !Groups.cbegin()->Functions.empty()));

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(MD), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(MD), std::move(Groups));
}

void ModuleSplitterBase::verifyNoCrossModuleDeviceGlobalUsage() {
  const Module &M = getInputModule();
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

#ifndef NDEBUG

const char *toString(SyclEsimdSplitStatus S) {
  switch (S) {
  case SyclEsimdSplitStatus::ESIMD_ONLY:
    return "ESIMD_ONLY";
  case SyclEsimdSplitStatus::SYCL_ONLY:
    return "SYCL_ONLY";
  case SyclEsimdSplitStatus::SYCL_AND_ESIMD:
    return "SYCL_AND_ESIMD";
  }
  return "<UNKNOWN_STATUS>";
}

void tab(int N) {
  for (int I = 0; I < N; ++I) {
    llvm::errs() << "  ";
  }
}

void dumpEntryPoints(const EntryPointSet &C, const char *msg, int Tab) {
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

#endif // NDEBUG

void ModuleDesc::assignMergedProperties(const ModuleDesc &MD1,
                                        const ModuleDesc &MD2) {
  EntryPoints.Props = MD1.EntryPoints.Props.merge(MD2.EntryPoints.Props);
  Props.SpecConstsMet = MD1.Props.SpecConstsMet || MD2.Props.SpecConstsMet;
}

void ModuleDesc::renameDuplicatesOf(const Module &MA, StringRef Suff) {
  Module &MB = getModule();
#ifndef NDEBUG
  DenseSet<StringRef> EntryNamesB;
  const auto It0 = entries().begin();
  const auto It1 = entries().end();
  std::for_each(It0, It1,
                [&](const Function *F) { EntryNamesB.insert(F->getName()); });
#endif // NDEBUG
  for (const GlobalObject &GoA : MA.global_objects()) {
    if (GoA.isDeclaration()) {
      continue;
    }
    StringRef Name = GoA.getName();
    auto *F = dyn_cast<Function>(&GoA);
    GlobalObject *GoB =
        F ? cast_or_null<GlobalObject>(MB.getFunction(Name))
          : cast_or_null<GlobalObject>(MB.getGlobalVariable(Name));

    if (!GoB || GoB->isDeclaration()) {
      // function or variable is not shared or is a declaration in MB
      continue;
    }
#ifndef NDEBUG
    if (F) {
      // this is a shared function, must not be an entry point:
      assert(!EntryNamesB.contains(Name));
    }
#endif // NDEBUG
    // rename the global object in MB:
    GoB->setName(Name + Suff);
  }
}

#ifndef NDEBUG
void ModuleDesc::verifyESIMDProperty() const {
  if (EntryPoints.Props.HasESIMD == SyclEsimdSplitStatus::SYCL_AND_ESIMD) {
    return; // nothing to verify
  }
  // Verify entry points:
  for (const auto *F : entries()) {
    const bool IsESIMDFunction = isESIMDFunction(*F);

    switch (EntryPoints.Props.HasESIMD) {
    case SyclEsimdSplitStatus::ESIMD_ONLY:
      assert(IsESIMDFunction);
      break;
    case SyclEsimdSplitStatus::SYCL_ONLY:
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
  // if (Props.HasEsimd == SyclEsimdSplitStatus::SYCL_ONLY) {
  //  for (const auto &F : getModule()) {
  //    assert(!isESIMDFunction(F));
  //  }
  //}
}

void ModuleDesc::dump() const {
  llvm::errs() << "split_module::ModuleDesc[" << Name << "] {\n";
  llvm::errs() << "  ESIMD:" << toString(EntryPoints.Props.HasESIMD)
               << ", SpecConstMet:" << (Props.SpecConstsMet ? "YES" : "NO")
               << ", DoubleGRF:"
               << (EntryPoints.Props.UsesDoubleGRF ? "YES" : "NO") << "\n";
  dumpEntryPoints(entries(), EntryPoints.GroupId.str().c_str(), 1);
  llvm::errs() << "}\n";
}
#endif // NDEBUG

void EntryPointGroup::saveNames(std::vector<std::string> &Dest) const {
  Dest.reserve(Dest.size() + Functions.size());
  std::transform(Functions.begin(), Functions.end(),
                 std::inserter(Dest, Dest.end()),
                 [](const Function *F) { return F->getName().str(); });
}

void EntryPointGroup::rebuildFromNames(const std::vector<std::string> &Names,
                                       const Module &M) {
  Functions.clear();
  auto It0 = Names.cbegin();
  auto It1 = Names.cend();
  std::for_each(It0, It1, [&](const std::string &Name) {
    const Function *F = M.getFunction(Name);
    assert(F && "entry point lost");
    Functions.insert(F);
  });
}

void EntryPointGroup::rebuild(const Module &M) {
  if (Functions.size() == 0) {
    return;
  }
  EntryPointSet NewFunctions;
  const auto It0 = Functions.begin();
  const auto It1 = Functions.end();
  std::for_each(It0, It1, [&](const Function *F) {
    Function *NewF = M.getFunction(F->getName());
    assert(NewF && "entry point lost");
    NewFunctions.insert(NewF);
  });
  Functions = std::move(NewFunctions);
}

std::unique_ptr<ModuleSplitterBase>
getESIMDDoubleGRFSplitter(ModuleDesc &&MD, bool EmitOnlyKernelsAsEntryPoints) {
  EntryPointGroupVec Groups = groupEntryPointsByAttribute(
      MD, ATTR_DOUBLE_GRF, EmitOnlyKernelsAsEntryPoints,
      [](EntryPointGroup &G) {
        if (G.GroupId == ATTR_DOUBLE_GRF) {
          G.Props.UsesDoubleGRF = true;
        }
      });
  assert(!Groups.empty() && "At least one group is expected");
  assert(Groups.size() <= 2 && "At most 2 groups are expected");

  if (Groups.size() > 1)
    return std::make_unique<ModuleSplitter>(std::move(MD), std::move(Groups));
  else
    return std::make_unique<ModuleCopier>(std::move(MD), std::move(Groups));
}

} // namespace module_split
} // namespace llvm
