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
#include "Support.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"
#include "llvm/SYCLLowerIR/LowerInvokeSimd.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"
#include "llvm/Transforms/IPO/StripSymbols.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <map>
#include <utility>
#include <variant>

using namespace llvm;
using namespace llvm::module_split;

namespace {

// Identifying name for global scope
constexpr char GLOBAL_SCOPE_NAME[] = "<GLOBAL>";
constexpr char SYCL_SCOPE_NAME[] = "<SYCL>";
constexpr char ESIMD_SCOPE_NAME[] = "<ESIMD>";
constexpr char ESIMD_MARKER_MD[] = "sycl_explicit_simd";

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

// Return true if the function is a ESIMD builtin
// The regexp for ESIMD intrinsics:
// /^_Z(\d+)__esimd_\w+/
bool isESIMDBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.startswith("__esimd_");
}

// Return true if the function name starts with "__builtin_"
bool isGenericBuiltin(StringRef FName) {
  return FName.startswith("__builtin_");
}

bool isKernel(const Function &F) {
  return F.getCallingConv() == CallingConv::SPIR_KERNEL;
}

bool isEntryPoint(const Function &F, bool EmitOnlyKernelsAsEntryPoints) {
  // Skip declarations, if any: they should not be included into a vector of
  // entry points groups or otherwise we will end up with incorrectly generated
  // list of symbols.
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  if (isKernel(F))
    return true;

  if (!EmitOnlyKernelsAsEntryPoints) {
    // If not disabled, SYCL_EXTERNAL functions with sycl-module-id attribute
    // are also considered as entry points (except __spirv_* and __sycl_*
    // functions)
    return llvm::sycl::utils::isSYCLExternalFunction(&F) &&
           !isSpirvSyclBuiltin(F.getName()) && !isESIMDBuiltin(F.getName()) &&
           !isGenericBuiltin(F.getName());
  }

  return false;
}

bool isESIMDFunction(const Function &F) {
  return F.getMetadata(ESIMD_MARKER_MD) != nullptr;
}

// This function makes one or two groups depending on kernel types (SYCL, ESIMD)
EntryPointGroupVec
groupEntryPointsByKernelType(ModuleDesc &MD,
                             bool EmitOnlyKernelsAsEntryPoints) {
  Module &M = MD.getModule();
  EntryPointGroupVec EntryPointGroups{};
  std::map<StringRef, EntryPointSet> EntryPointMap;

  // Only process module entry points:
  for (Function &F : M.functions()) {
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
      EntryPointGroups.emplace_back(EPG.first, std::move(EPG.second),
                                    MD.getEntryPointGroup().Props);
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
    EntryPointGroups.emplace_back(SYCL_SCOPE_NAME, EntryPointSet{});
    EntryPointGroup &G = EntryPointGroups.back();
    G.Props.HasESIMD = SyclEsimdSplitStatus::SYCL_ONLY;
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
  FunctionSet AddrTakenFunctions;

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
      if (F.hasAddressTaken()) {
        AddrTakenFunctions.insert(&F);
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

  iterator_range<FunctionSet::const_iterator> addrTakenFunctions() const {
    return make_range(AddrTakenFunctions.begin(), AddrTakenFunctions.end());
  }
};

void collectFunctionsToExtract(SetVector<const GlobalValue *> &GVs,
                               const EntryPointGroup &ModuleEntryPoints,
                               const CallGraph &Deps) {
  for (const auto *F : ModuleEntryPoints.Functions)
    GVs.insert(F);
  // It is conservatively assumed that any address-taken function can be invoked
  // or otherwise used by any function in any module split from the initial one.
  // So such functions along with the call graphs they start are always
  // extracted (and duplicated in each split module). They are not treated as
  // entry points, as SYCL runtime requires that intersection of entry point
  // sets of different device binaries (for the same target) must be empty.
  // TODO: try to determine which split modules really use address-taken
  // functions and only duplicate the functions in such modules. Note that usage
  // may include e.g. function address comparison w/o actual invocation.
  for (const auto *F : Deps.addrTakenFunctions()) {
    if (!isKernel(*F) && (isESIMDFunction(*F) == ModuleEntryPoints.isEsimd()))
      GVs.insert(F);
  }

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

// The function produces a copy of input LLVM IR module M with only those entry
// points that are specified in ModuleEntryPoints vector.
ModuleDesc extractCallGraph(const ModuleDesc &MD,
                            EntryPointGroup &&ModuleEntryPoints,
                            const CallGraph &CG) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsToExtract(GVs, ModuleEntryPoints, CG);
  collectGlobalVarsToExtract(GVs, MD.getModule());

  ModuleDesc SplitM = extractSubModule(MD, GVs, std::move(ModuleEntryPoints));
  SplitM.cleanup();

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
      : ModuleSplitterBase(std::move(MD), std::move(GroupVec)),
        CG(Input.getModule()) {}

  ModuleDesc nextSplit() override {
    return extractCallGraph(Input, nextGroup(), CG);
  }

private:
  CallGraph CG;
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

    std::optional<StringRef> VarEntryPointModule{};
    auto CheckEntryPointModule = [&VarEntryPointModule, &EntryPointModules,
                                  &GV](const auto *F) {
      auto EntryPointModulesIt = EntryPointModules.find(F);
      assert(EntryPointModulesIt != EntryPointModules.end() &&
             "There is no group for an entry point");
      if (!VarEntryPointModule.has_value()) {
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

// Attribute to save current function linkage in before replacement.
// See more comments in ModuleSplitter::fixupLinkageOfDirectInvokeSimdTargets
// declaration.
constexpr char SYCL_ORIG_LINKAGE_ATTR[] = "__sycl_orig_linkage";

void ModuleDesc::fixupLinkageOfDirectInvokeSimdTargets() {
  for (Function &F : *M) {
    if (!F.hasFnAttribute(INVOKE_SIMD_DIRECT_TARGET_ATTR)) {
      continue;
    }
    int L = static_cast<int>(F.getLinkage());
    using LT = GlobalValue::LinkageTypes;

    if (L == static_cast<int>(LT::LinkOnceODRLinkage)) {
      F.addFnAttr(SYCL_ORIG_LINKAGE_ATTR, llvm::utostr(L));
      F.setLinkage(LT::WeakODRLinkage);
    } else if (L == static_cast<int>(LT::LinkOnceAnyLinkage)) {
      F.addFnAttr(SYCL_ORIG_LINKAGE_ATTR, llvm::utostr(L));
      F.setLinkage(LT::WeakAnyLinkage);
    }
  }
}

void ModuleDesc::restoreLinkageOfDirectInvokeSimdTargets() {
  for (Function &F : *M) {
    Attribute A = F.getFnAttribute(SYCL_ORIG_LINKAGE_ATTR);

    if (!A.isValid()) {
      continue;
    }
    F.removeFnAttr(SYCL_ORIG_LINKAGE_ATTR);
    int L = std::stoi(A.getValueAsString().str());
    F.setLinkage(static_cast<GlobalValue::LinkageTypes>(L));
  }
}

// TODO: try to move all passes (cleanup, spec consts, compile time properties)
// in one place and execute MPM.run() only once.
void ModuleDesc::cleanup() {
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  ModulePassManager MPM;
  // Do cleanup.
  MPM.addPass(GlobalDCEPass());           // Delete unreachable globals.
  MPM.addPass(StripDeadDebugInfoPass());  // Remove dead debug info.
  MPM.addPass(StripDeadPrototypesPass()); // Remove dead func decls.
  MPM.run(*M, MAM);
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
               << "\n";
  dumpEntryPoints(entries(), EntryPoints.GroupId.c_str(), 1);
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
    // Sometimes functions considered entry points (those for which isEntryPoint
    // returned true) may be dropped by optimizations, such as AlwaysInliner.
    // For example, if a linkonce_odr function is inlined and there are no other
    // uses, AlwaysInliner drops it. It is responsibility of the user to make an
    // entry point not have internal linkage (such as linkonce_odr) to guarantee
    // its availability in the resulting device binary image.
    if (Function *F = M.getFunction(Name)) {
      Functions.insert(F);
    }
  });
}

namespace {
// This is a helper class, which allows to group/categorize function based on
// provided rules. It is intended to be used in device code split
// implementation.
//
// "Rule" is a simple routine, which returns a string for an llvm::Function
// passed to it. There could be more than one rule and they are applied in order
// of their registration. Results obtained from those rules are concatenated
// together to produce the final result.
//
// There are some predefined rules for the most popular use-cases, like grouping
// functions together based on an attribute value or presence of a metadata.
// However, there is also a possibility to register a custom callback function
// as a rule, to implement custom/more complex logic.
class FunctionsCategorizer {
public:
  FunctionsCategorizer() = default;

  std::string computeCategoryFor(Function *) const;

  // Accepts a callback, which should return a string based on provided
  // function, which will be used as an entry points group identifier.
  void registerRule(const std::function<std::string(Function *)> &Callback) {
    Rules.emplace_back(Rule::RKind::K_Callback, Callback);
  }

  // Creates a simple rule, which adds a value of a string attribute into a
  // resulting identifier.
  void registerSimpleStringAttributeRule(StringRef AttrName) {
    Rules.emplace_back(Rule::RKind::K_SimpleStringAttribute, AttrName);
  }

  // Creates a simple rule, which adds one or another value to a resulting
  // identifier based on the presence of a metadata on a function.
  void registerSimpleFlagAttributeRule(StringRef AttrName,
                                       StringRef IfPresentStr,
                                       StringRef IfAbsentStr = "") {
    Rules.emplace_back(Rule::RKind::K_FlagAttribute,
                       Rule::FlagRuleData{AttrName, IfPresentStr, IfAbsentStr});
  }

  // Creates a simple rule, which adds one or another value to a resulting
  // identifier based on the presence of a metadata on a function.
  void registerSimpleFlagMetadataRule(StringRef MetadataName,
                                      StringRef IfPresentStr,
                                      StringRef IfAbsentStr = "") {
    Rules.emplace_back(
        Rule::RKind::K_FlagMetadata,
        Rule::FlagRuleData{MetadataName, IfPresentStr, IfAbsentStr});
  }

  // Creates a rule, which adds a list of dash-separated integers converted
  // into strings listed in a metadata to a resulting identifier.
  void registerListOfIntegersInMetadataRule(StringRef MetadataName) {
    Rules.emplace_back(Rule::RKind::K_IntegersListMetadata, MetadataName);
  }

  // Creates a rule, which adds a list of sorted dash-separated integers
  // converted into strings listed in a metadata to a resulting identifier.
  void registerListOfIntegersInMetadataSortedRule(StringRef MetadataName) {
    Rules.emplace_back(Rule::RKind::K_SortedIntegersListMetadata, MetadataName);
  }

private:
  struct Rule {
    struct FlagRuleData {
      StringRef Name, IfPresentStr, IfAbsentStr;
    };

  private:
    std::variant<StringRef, FlagRuleData,
                 std::function<std::string(Function *)>>
        Storage;

  public:
    enum class RKind {
      // Custom callback function
      K_Callback,
      // Copy value of the specified attribute, if present
      K_SimpleStringAttribute,
      // Use one or another string based on the specified metadata presence
      K_FlagMetadata,
      // Use one or another string based on the specified attribute presence
      K_FlagAttribute,
      // Concatenate and use list of integers from the specified metadata
      K_IntegersListMetadata,
      // Sort, concatenate and use list of integers from the specified metadata
      K_SortedIntegersListMetadata
    };
    RKind Kind;

    // Returns an index into std::variant<...> Storage defined above, which
    // corresponds to the specified rule Kind.
    constexpr static std::size_t storage_index(RKind K) {
      switch (K) {
      case RKind::K_SimpleStringAttribute:
      case RKind::K_IntegersListMetadata:
      case RKind::K_SortedIntegersListMetadata:
        return 0;
      case RKind::K_Callback:
        return 2;
      case RKind::K_FlagMetadata:
      case RKind::K_FlagAttribute:
        return 1;
      }
      // can't use llvm_unreachable in constexpr context
      return std::variant_npos;
    }

    template <RKind K> auto getStorage() const {
      return std::get<storage_index(K)>(Storage);
    }

    template <typename... Args>
    Rule(RKind K, Args... args) : Storage(args...), Kind(K) {
      assert(storage_index(K) == Storage.index());
    }

    Rule(Rule &&Other) = default;
  };

  std::vector<Rule> Rules;
};

std::string FunctionsCategorizer::computeCategoryFor(Function *F) const {
  SmallString<256> Result;
  for (const auto &R : Rules) {
    switch (R.Kind) {
    case Rule::RKind::K_Callback:
      Result += R.getStorage<Rule::RKind::K_Callback>()(F);
      break;

    case Rule::RKind::K_SimpleStringAttribute: {
      StringRef AttrName = R.getStorage<Rule::RKind::K_SimpleStringAttribute>();
      if (F->hasFnAttribute(AttrName)) {
        Attribute Attr = F->getFnAttribute(AttrName);
        Result += Attr.getValueAsString();
      }
    } break;

    case Rule::RKind::K_FlagMetadata: {
      Rule::FlagRuleData Data = R.getStorage<Rule::RKind::K_FlagMetadata>();
      if (F->hasMetadata(Data.Name))
        Result += Data.IfPresentStr;
      else
        Result += Data.IfAbsentStr;
    } break;

    case Rule::RKind::K_IntegersListMetadata: {
      StringRef MetadataName =
          R.getStorage<Rule::RKind::K_IntegersListMetadata>();
      if (F->hasMetadata(MetadataName)) {
        auto *MDN = F->getMetadata(MetadataName);
        for (const MDOperand &MDOp : MDN->operands())
          Result +=
              "-" + std::to_string(
                        mdconst::extract<ConstantInt>(MDOp)->getZExtValue());
      }
    } break;

    case Rule::RKind::K_SortedIntegersListMetadata: {
      StringRef MetadataName =
          R.getStorage<Rule::RKind::K_IntegersListMetadata>();
      if (F->hasMetadata(MetadataName)) {
        MDNode *MDN = F->getMetadata(MetadataName);

        SmallVector<std::uint64_t, 8> Values;
        for (const MDOperand &MDOp : MDN->operands())
          Values.push_back(mdconst::extract<ConstantInt>(MDOp)->getZExtValue());

        llvm::sort(Values);

        for (std::uint64_t V : Values)
          Result += "-" + std::to_string(V);
      }
    } break;

    case Rule::RKind::K_FlagAttribute: {
      Rule::FlagRuleData Data = R.getStorage<Rule::RKind::K_FlagAttribute>();
      if (F->hasFnAttribute(Data.Name))
        Result += Data.IfPresentStr;
      else
        Result += Data.IfAbsentStr;
    } break;
    }

    Result += "-";
  }

  return (std::string)Result;
}
} // namespace

std::unique_ptr<ModuleSplitterBase>
getDeviceCodeSplitter(ModuleDesc &&MD, IRSplitMode Mode, bool IROutputOnly,
                      bool EmitOnlyKernelsAsEntryPoints) {
  FunctionsCategorizer Categorizer;

  EntryPointsGroupScope Scope =
      selectDeviceCodeGroupScope(MD.getModule(), Mode, IROutputOnly);

  switch (Scope) {
  case Scope_Global:
    // We simply perform entry points filtering, but group all of them together.
    Categorizer.registerRule(
        [](Function *) -> std::string { return GLOBAL_SCOPE_NAME; });
    break;
  case Scope_PerKernel:
    // Per-kernel split is quite simple: every kernel goes into a separate
    // module and that's it, no other rules required.
    Categorizer.registerRule(
        [](Function *F) -> std::string { return F->getName().str(); });
    break;
  case Scope_PerModule:
    // The most complex case, because we should account for many other features
    // like aspects used in a kernel, large-grf mode, reqd-work-group-size, etc.

    // This is core of per-source device code split
    Categorizer.registerSimpleStringAttributeRule(
        sycl::utils::ATTR_SYCL_MODULE_ID);

    // Optional features
    // Note: Add more rules at the end of the list to avoid chaning orders of
    // output files in existing tests.
    Categorizer.registerSimpleStringAttributeRule("sycl-register-alloc-mode");
    Categorizer.registerListOfIntegersInMetadataSortedRule("sycl_used_aspects");
    Categorizer.registerListOfIntegersInMetadataRule("reqd_work_group_size");
    Categorizer.registerSimpleStringAttributeRule(
        sycl::utils::ATTR_SYCL_OPTLEVEL);
    break;
  }

  // std::map is used here to ensure stable ordering of entry point groups,
  // which is based on their contents, this greatly helps LIT tests
  std::map<std::string, EntryPointSet> EntryPointsMap;

  // Only process module entry points:
  for (auto &F : MD.getModule().functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      continue;

    std::string Key = Categorizer.computeCategoryFor(&F);
    EntryPointsMap[std::move(Key)].insert(&F);
  }

  EntryPointGroupVec Groups;

  if (EntryPointsMap.empty()) {
    // No entry points met, record this.
    Groups.emplace_back(GLOBAL_SCOPE_NAME, EntryPointSet{});
  } else {
    Groups.reserve(EntryPointsMap.size());
    // Start with properties of a source module
    EntryPointGroup::Properties MDProps = MD.getEntryPointGroup().Props;
    for (auto &[Key, EntryPoints] : EntryPointsMap)
      Groups.emplace_back(Key, std::move(EntryPoints), MDProps);
  }

  bool DoSplit = (Mode != SPLIT_NONE &&
                  (Groups.size() > 1 || !Groups.cbegin()->Functions.empty()));

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(MD), std::move(Groups));

  return std::make_unique<ModuleCopier>(std::move(MD), std::move(Groups));
}

} // namespace module_split
} // namespace llvm
