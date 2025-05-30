//===-------- ModuleSplitter.cpp - split a module into callgraphs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PassManagerImpl.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/SYCLLowerIR/CleanupSYCLMetadata.h"
#include "llvm/SYCLLowerIR/DeviceGlobals.h"
#include "llvm/SYCLLowerIR/ESIMD/LowerESIMD.h"
#include "llvm/SYCLLowerIR/LowerInvokeSimd.h"
#include "llvm/SYCLLowerIR/SYCLDeviceLibReqMask.h"
#include "llvm/SYCLLowerIR/SYCLJointMatrixTransform.h"
#include "llvm/SYCLLowerIR/SYCLUtils.h"
#include "llvm/SYCLLowerIR/SanitizerKernelMetadata.h"
#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
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

EntryPointsGroupScope selectDeviceCodeGroupScope(const Module &M,
                                                 IRSplitMode Mode,
                                                 bool AutoSplitIsGlobalScope) {
  switch (Mode) {
  case SPLIT_PER_TU:
    return Scope_PerModule;

  case SPLIT_PER_KERNEL:
    return Scope_PerKernel;

  case SPLIT_AUTO: {
    if (AutoSplitIsGlobalScope)
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

  return FName.starts_with("__spirv_") || FName.starts_with("__sycl_");
}

// Return true if the function is a ESIMD builtin
// The regexp for ESIMD intrinsics:
// /^_Z(\d+)__esimd_\w+/
bool isESIMDBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.starts_with("__esimd_");
}

// Return true if the function name starts with "__builtin_"
bool isGenericBuiltin(StringRef FName) {
  return FName.starts_with("__builtin_");
}

bool isKernel(const Function &F) {
  const auto CC = F.getCallingConv();
  return CC == CallingConv::SPIR_KERNEL || CC == CallingConv::AMDGPU_KERNEL ||
         CC == CallingConv::PTX_Kernel;
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

  // Even if we are emitting only kernels as entry points, virtual functions
  // should still be treated as entry points, because they are going to be
  // outlined into separate device images and linked in later.
  return F.hasFnAttribute("indirectly-callable");
}

// Represents "dependency" or "use" graph of global objects (functions and
// global variables) in a module. It is used during device code split to
// understand which global variables and functions (other than entry points)
// should be included into a split module.
//
// Nodes of the graph represent LLVM's GlobalObjects, edges "A" -> "B" represent
// the fact that if "A" is included into a module, then "B" should be included
// as well.
//
// Examples of dependencies which are represented in this graph:
// - Function FA calls function FB
// - Function FA uses global variable GA
// - Global variable GA references (initialized with) function FB
// - Function FA stores address of a function FB somewhere
//
// The following cases are treated as dependencies between global objects:
// 1. Global object A is used within by a global object B in any way (store,
//    bitcast, phi node, call, etc.): "A" -> "B" edge will be added to the
//    graph;
// 2. function A performs an indirect call of a function with signature S and
//    there is a function B with signature S marked with "referenced-indirectly"
//    attribute. "A" -> "B" edge will be added to the graph;
class DependencyGraph {
public:
  using GlobalSet = SmallPtrSet<const GlobalValue *, 16>;

  DependencyGraph(const Module &M, bool AllowDeviceImageDependencies) {
    // Group functions by their signature to handle case (2) described above
    DenseMap<const FunctionType *, DependencyGraph::GlobalSet>
        FuncTypeToFuncsMap;
    for (const auto &F : M.functions()) {
      // Kernels can't be called (either directly or indirectly) in SYCL
      if (isKernel(F))
        continue;

      // Only functions which are marked with "referenced-indireclty" attribute
      // are considered to be indirect callee candidates.
      if (!F.hasFnAttribute("referenced-indirectly"))
        continue;

      FuncTypeToFuncsMap[F.getFunctionType()].insert(&F);
    }

    for (const auto &F : M.functions()) {
      if (canBeImportedFunction(F, AllowDeviceImageDependencies))
        continue;

      // case (1), see comment above the class definition
      for (const Value *U : F.users())
        addUserToGraphRecursively(cast<const User>(U), &F);

      // case (2), see comment above the class definition
      for (const auto &I : instructions(F)) {
        const auto *CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->isIndirectCall()) // Direct calls were handled above
          continue;

        // TODO: consider limiting set of potential callees to functions marked
        // with special attribute (like [[intel::device_indirectly_callable]])
        const FunctionType *Signature = CI->getFunctionType();
        // Note: strictly speaking, virtual functions are allowed to use
        // co-variant return types, i.e. we can actually miss a potential callee
        // here, because it has different signature (different return type).
        // However, this is not a problem for two reasons:
        // - opaque pointers will be enabled at some point and will make
        //   signatures the same in that case
        // - all virtual functions are referenced from vtable and therefore will
        //   anyway be preserved in a module
        const auto &PotentialCallees = FuncTypeToFuncsMap[Signature];
        Graph[&F].insert(PotentialCallees.begin(), PotentialCallees.end());
      }
    }

    // And every global variable (but their handling is a bit simpler)
    for (const auto &GV : M.globals())
      for (const Value *U : GV.users())
        addUserToGraphRecursively(cast<const User>(U), &GV);
  }

  iterator_range<GlobalSet::const_iterator>
  dependencies(const GlobalValue *Val) const {
    auto It = Graph.find(Val);
    return (It == Graph.end())
               ? make_range(EmptySet.begin(), EmptySet.end())
               : make_range(It->second.begin(), It->second.end());
  }

private:
  void addUserToGraphRecursively(const User *Root, const GlobalValue *V) {

    SmallVector<const User *, 8> WorkList;
    WorkList.push_back(Root);

    while (!WorkList.empty()) {
      const User *U = WorkList.pop_back_val();
      if (const auto *I = dyn_cast<const Instruction>(U)) {
        const auto *UFunc = I->getFunction();
        Graph[UFunc].insert(V);
      } else if (isa<const Constant>(U)) {
        if (const auto *GV = dyn_cast<const GlobalVariable>(U))
          Graph[GV].insert(V);
        // This could be a global variable or some constant expression (like
        // bitcast or gep). We trace users of this constant further to reach
        // global objects they are used by and add them to the graph.
        for (const auto *UU : U->users())
          WorkList.push_back(UU);
      } else {
        llvm_unreachable("Unhandled type of function user");
      }
    }
  }

  DenseMap<const GlobalValue *, GlobalSet> Graph;
  SmallPtrSet<const GlobalValue *, 1> EmptySet;
};

void collectFunctionsAndGlobalVariablesToExtract(
    SetVector<const GlobalValue *> &GVs, const Module &M,
    const EntryPointGroup &ModuleEntryPoints, const DependencyGraph &Deps,
    const std::function<bool(const Function *)> &IncludeFunctionPredicate =
        nullptr) {
  // We start with module entry points
  for (const auto *F : ModuleEntryPoints.Functions)
    GVs.insert(F);

  // Non-discardable global variables are also include into the initial set
  for (const auto &GV : M.globals()) {
    if (!GV.isDiscardableIfUnused())
      GVs.insert(&GV);
  }

  // GVs has SetVector type. This type inserts a value only if it is not yet
  // present there. So, recursion is not expected here.
  decltype(GVs.size()) Idx = 0;
  while (Idx < GVs.size()) {
    const auto *Obj = GVs[Idx++];

    for (const GlobalValue *Dep : Deps.dependencies(Obj)) {
      if (const auto *Func = dyn_cast<const Function>(Dep)) {
        if (Func->isDeclaration())
          continue;

        // Functions can be additionally filtered
        if (!IncludeFunctionPredicate || IncludeFunctionPredicate(Func))
          GVs.insert(Func);
      } else {
        // Global variables are added unconditionally
        GVs.insert(Dep);
      }
    }
  }
}

static bool isIntrinsicOrBuiltin(const Function &F) {
  return F.isIntrinsic() || F.getName().starts_with("__") ||
         isSpirvSyclBuiltin(F.getName()) || isESIMDBuiltin(F.getName());
}

// Checks for use of undefined user functions and emits a warning message.
static void
checkForCallsToUndefinedFunctions(const Module &M,
                                  bool AllowDeviceImageDependencies) {
  if (AllowDeviceImageDependencies)
    return;
  for (const Function &F : M) {
    if (!isIntrinsicOrBuiltin(F) && F.isDeclaration() && !F.use_empty())
      WithColor::warning() << "Undefined function " << F.getName()
                           << " found in " << M.getName()
                           << ". This may result in runtime errors.\n";
  }
}

// Check "spirv.ExecutionMode" named metadata in the module and remove nodes
// that reference kernels that have dead prototypes or don't reference any
// kernel at all (nullptr). Dead prototypes are removed as well.
void processSubModuleNamedMetadata(Module *M) {
  auto ExecutionModeMD = M->getNamedMetadata("spirv.ExecutionMode");
  if (!ExecutionModeMD)
    return;

  bool ContainsNodesToRemove = false;
  std::vector<MDNode *> ValueVec;
  for (auto Op : ExecutionModeMD->operands()) {
    assert(Op->getNumOperands() > 0);
    if (!Op->getOperand(0)) {
      ContainsNodesToRemove = true;
      continue;
    }

    // If the first operand is not nullptr then it has to be a kernel
    // function.
    Value *Val = cast<ValueAsMetadata>(Op->getOperand(0))->getValue();
    Function *F = cast<Function>(Val);
    // If kernel function is just a prototype and unused then we can remove it
    // and later remove corresponding spirv.ExecutionMode metadata node.
    if (F->isDeclaration() && F->use_empty()) {
      F->eraseFromParent();
      ContainsNodesToRemove = true;
      continue;
    }

    // Rememver nodes which we need to keep in the module.
    ValueVec.push_back(Op);
  }
  if (!ContainsNodesToRemove)
    return;

  if (ValueVec.empty()) {
    // If all nodes need to be removed then just remove named metadata
    // completely.
    ExecutionModeMD->eraseFromParent();
  } else {
    ExecutionModeMD->clearOperands();
    for (auto MD : ValueVec)
      ExecutionModeMD->addOperand(MD);
  }
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

// The function produces a copy of input LLVM IR module M with only those
// functions and globals that can be called from entry points that are specified
// in ModuleEntryPoints vector, in addition to the entry point functions.
ModuleDesc extractCallGraph(
    const ModuleDesc &MD, EntryPointGroup &&ModuleEntryPoints,
    const DependencyGraph &CG, bool AllowDeviceImageDependencies,
    const std::function<bool(const Function *)> &IncludeFunctionPredicate =
        nullptr) {
  SetVector<const GlobalValue *> GVs;
  collectFunctionsAndGlobalVariablesToExtract(
      GVs, MD.getModule(), ModuleEntryPoints, CG, IncludeFunctionPredicate);

  ModuleDesc SplitM =
      extractSubModule(MD, std::move(GVs), std::move(ModuleEntryPoints));
  // TODO: cleanup pass is now called for each output module at the end of
  // sycl-post-link. This call is redundant. However, we subsequently run
  // GenXSPIRVWriterAdaptor pass that relies on this cleanup. This cleanup call
  // can be removed once that pass no longer depends on this cleanup.
  SplitM.cleanup(AllowDeviceImageDependencies);
  checkForCallsToUndefinedFunctions(SplitM.getModule(),
                                    AllowDeviceImageDependencies);

  return SplitM;
}

// The function is similar to 'extractCallGraph', but it produces a copy of
// input LLVM IR module M with _all_ ESIMD functions and kernels included,
// regardless of whether or not they are listed in ModuleEntryPoints.
ModuleDesc extractESIMDSubModule(
    const ModuleDesc &MD, EntryPointGroup &&ModuleEntryPoints,
    const DependencyGraph &CG, bool AllowDeviceImageDependencies,
    const std::function<bool(const Function *)> &IncludeFunctionPredicate =
        nullptr) {
  SetVector<const GlobalValue *> GVs;
  for (const auto &F : MD.getModule().functions())
    if (isESIMDFunction(F))
      GVs.insert(&F);

  collectFunctionsAndGlobalVariablesToExtract(
      GVs, MD.getModule(), ModuleEntryPoints, CG, IncludeFunctionPredicate);

  ModuleDesc SplitM =
      extractSubModule(MD, std::move(GVs), std::move(ModuleEntryPoints));
  // TODO: cleanup pass is now called for each output module at the end of
  // sycl-post-link. This call is redundant. However, we subsequently run
  // GenXSPIRVWriterAdaptor pass that relies on this cleanup. This cleanup call
  // can be removed once that pass no longer depends on this cleanup.
  SplitM.cleanup(AllowDeviceImageDependencies);

  return SplitM;
}

class ModuleCopier : public ModuleSplitterBase {
public:
  using ModuleSplitterBase::ModuleSplitterBase; // to inherit base constructors

  ModuleDesc nextSplit() override {
    ModuleDesc Desc{releaseInputModule(), nextGroup(), Input.Props};
    // Do some basic optimization like unused symbol removal
    // even if there was no split.
    // TODO: cleanup pass is now called for each output module at the end of
    // sycl-post-link. This call is redundant. However, we subsequently run
    // GenXSPIRVWriterAdaptor pass that relies on this cleanup. This cleanup
    // call can be removed once that pass no longer depends on this cleanup.
    Desc.cleanup(AllowDeviceImageDependencies);
    return Desc;
  }
};

class ModuleSplitter : public ModuleSplitterBase {
public:
  ModuleSplitter(ModuleDesc &&MD, EntryPointGroupVec &&GroupVec,
                 bool AllowDeviceImageDependencies)
      : ModuleSplitterBase(std::move(MD), std::move(GroupVec),
                           AllowDeviceImageDependencies),
        CG(Input.getModule(), AllowDeviceImageDependencies) {}

  ModuleDesc nextSplit() override {
    return extractCallGraph(Input, nextGroup(), CG,
                            AllowDeviceImageDependencies);
  }

private:
  DependencyGraph CG;
};
} // namespace

namespace llvm {
namespace module_split {

std::optional<IRSplitMode> convertStringToSplitMode(StringRef S) {
  static const StringMap<IRSplitMode> Values = {{"kernel", SPLIT_PER_KERNEL},
                                                {"source", SPLIT_PER_TU},
                                                {"auto", SPLIT_AUTO},
                                                {"none", SPLIT_NONE}};

  auto It = Values.find(S);
  if (It == Values.end())
    return std::nullopt;

  return It->second;
}

bool isESIMDFunction(const Function &F) {
  return F.getMetadata(ESIMD_MARKER_MD) != nullptr;
}

Error ModuleSplitterBase::verifyNoCrossModuleDeviceGlobalUsage() {
  const Module &M = getInputModule();
  // Early exit if there is only one group
  if (Groups.size() < 2)
    return Error::success();

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
                                  &GV](const auto *F) -> Error {
      auto EntryPointModulesIt = EntryPointModules.find(F);
      if (EntryPointModulesIt == EntryPointModules.end())
        return createStringError(inconvertibleErrorCode(),
                                 "There is no group for an entry point");

      if (!VarEntryPointModule.has_value()) {
        VarEntryPointModule = EntryPointModulesIt->second;
        return Error::success();
      }

      if (EntryPointModulesIt->second != *VarEntryPointModule)
        return createStringError(
            inconvertibleErrorCode(),
            "device_global variable '" + Twine(GV.getName()) +
                "' with property \"device_image_scope\" is used in more "
                "than one device image.");

      return Error::success();
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
        if (EntryPointModules.count(F)) {
          auto E = CheckEntryPointModule(F);
          if (E)
            return E;
        }
      }
      for (auto *UU : U->users())
        Workqueue.insert(UU);
    }
  }

  return Error::success();
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

// Predicate for Internalize pass. The pass is very aggressive and essentially
// tries to internalize absolutely everything. This function serves as "input
// from a linker" that tells the pass what must be preserved in order to make
// the transformation safe.
static bool mustPreserveGV(const GlobalValue &GV,
                           bool AllowDeviceImageDependencies) {
  if (const Function *F = dyn_cast<Function>(&GV)) {
    // When dynamic linking is supported, we internalize everything (except
    // kernels which are the entry points from host code to device code) that
    // cannot be imported which also means that there is no point of having it
    // visible outside of the current module.
    if (AllowDeviceImageDependencies) {
      const auto CC = F->getCallingConv();
      const bool SpirOrGPU = CC == CallingConv::SPIR_KERNEL ||
                             CC == CallingConv::AMDGPU_KERNEL ||
                             CC == CallingConv::PTX_Kernel;
      return SpirOrGPU ||
             canBeImportedFunction(*F, AllowDeviceImageDependencies);
    }

    // Otherwise, we are being even more aggressive: SYCL modules are expected
    // to be self-contained, meaning that they have no external dependencies.
    // Therefore, we can internalize every function that is not an entry point.
    // One exception here is virtual functions: when they are in use, modules
    // are not self-contained anymore and some device images has to be linked
    // at runtime to resolve all symbols.
    // Functions marked with referenced-indirectly attribute is another
    // exception: that attribute was originally introduced for function pointers
    // and even though its main usage was deprecated and dropped, it is still
    // used in invoke_simd (but that use needs to be revisited).
    return F->hasFnAttribute("sycl-entry-point") ||
           F->hasFnAttribute("indirectly-callable") ||
           F->hasFnAttribute("referenced-indirectly");
  }

  // Otherwise, we don't have enough information about a global and picking a
  // safe side saying that all other globals must be preserved (we should have
  // cleaned up unused globals during dependency graph analysis already).
  return true;
}

void cleanupSYCLRegisteredKernels(Module *M) {
  NamedMDNode *MD = M->getNamedMetadata("sycl_registered_kernels");
  if (!MD)
    return;

  if (MD->getNumOperands() == 0)
    return;

  SmallVector<Metadata *, 8> OperandsToKeep;
  MDNode *RegisterdKernels = MD->getOperand(0);
  for (const MDOperand &Op : RegisterdKernels->operands()) {
    auto RegisteredKernel = cast<MDNode>(Op);
    // Ignore metadata nodes with wrong number of operands.
    if (RegisteredKernel->getNumOperands() != 2)
      continue;

    StringRef MangledName =
        cast<MDString>(RegisteredKernel->getOperand(1))->getString();
    if (M->getFunction(MangledName))
      OperandsToKeep.push_back(RegisteredKernel);
  }
  MD->clearOperands();
  MD->addOperand(MDNode::get(M->getContext(), OperandsToKeep));
}

// TODO: try to move all passes (cleanup, spec consts, compile time properties)
// in one place and execute MPM.run() only once.
void ModuleDesc::cleanup(bool AllowDeviceImageDependencies) {
  // Any definitions of virtual functions should be removed and turned into
  // declarations, they are supposed to be provided by a different module.
  if (!EntryPoints.Props.HasVirtualFunctionDefinitions) {
    for (Function &F : *M)
      if (F.hasFnAttribute("indirectly-callable")) {
        F.deleteBody();
        if (F.hasComdat())
          F.setComdat(nullptr);
      }
  } else {
    // Otherwise externalize them so they are not dropped by GlobalDCE
    for (Function &F : *M)
      if (F.hasFnAttribute("indirectly-callable"))
        F.setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
  }

  // Callback for internalize can't be a lambda with captures, so we propagate
  // necessary information through the module itself.
  if (!AllowDeviceImageDependencies)
    for (Function *F : EntryPoints.Functions)
      F->addFnAttr("sycl-entry-point");

  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  ModulePassManager MPM;
  // Do cleanup.
  MPM.addPass(
      InternalizePass([AllowDeviceImageDependencies](const GlobalValue &GV) {
        return mustPreserveGV(GV, AllowDeviceImageDependencies);
      }));
  MPM.addPass(GlobalDCEPass());           // Delete unreachable globals.
  MPM.addPass(StripDeadDebugInfoPass());  // Remove dead debug info.
  MPM.addPass(StripDeadPrototypesPass()); // Remove dead func decls.
  MPM.run(*M, MAM);

  // Original module may have named metadata (spirv.ExecutionMode) referencing
  // kernels in the module. Some of the Metadata nodes may reference kernels
  // which are not included into the extracted submodule, in such case
  // CloneModule either leaves that metadata nodes as is but they will reference
  // dead prototype of the kernel or operand will be replace with nullptr. So
  // process all nodes in the named metadata and remove nodes which are
  // referencing kernels which are not included into submodule.
  processSubModuleNamedMetadata(M.get());
  cleanupSYCLRegisteredKernels(M.get());
}

bool ModuleDesc::isSpecConstantDefault() const {
  return Props.IsSpecConstantDefault;
}

void ModuleDesc::setSpecConstantDefault(bool Value) {
  Props.IsSpecConstantDefault = Value;
}

ModuleDesc ModuleDesc::clone() const {
  std::unique_ptr<Module> NewModule = CloneModule(getModule());
  ModuleDesc NewMD(std::move(NewModule));
  NewMD.EntryPoints.Props = EntryPoints.Props;
  return NewMD;
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

void ModuleDesc::saveSplitInformationAsMetadata() {
  // Add metadata to the module so we can identify what kind of SYCL/ESIMD split
  // later.
  auto *SplitMD = M->getOrInsertNamedMetadata(SYCL_ESIMD_SPLIT_MD_NAME);
  auto *SplitMDOp = MDNode::get(
      M->getContext(), ConstantAsMetadata::get(ConstantInt::get(
                           Type::getInt8Ty(M->getContext()),
                           static_cast<uint8_t>(EntryPoints.Props.HasESIMD))));
  SplitMD->addOperand(SplitMDOp);

  // Add metadata to the module so we can identify it as the default value spec
  // constants split later.
  if (isSpecConstantDefault())
    M->getOrInsertNamedMetadata(
        SpecConstantsPass::SPEC_CONST_DEFAULT_VAL_MODULE_MD_STRING);
}

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

void EntryPointGroup::rebuild(const Module &M) {
  for (const Function &F : M.functions())
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
      Functions.insert(const_cast<Function *>(&F));
}

std::string ModuleDesc::makeSymbolTable() const {
  std::string ST;
  for (const Function *F : EntryPoints.Functions)
    ST += (Twine(F->getName()) + "\n").str();

  return ST;
}

namespace {

void computeFuncCategoryFromAttribute(const Function &F,
                                      const StringRef AttrName,
                                      SmallString<256> &Result) {
  if (F.hasFnAttribute(AttrName)) {
    const Attribute &Attr = F.getFnAttribute(AttrName);
    Result += Attr.getValueAsString();
  }

  Result += "-";
}

void computeFuncCategoryFromStringMetadata(const Function &F,
                                           const StringRef MetadataName,
                                           SmallString<256> &Result) {
  if (F.hasMetadata(MetadataName)) {
    const auto *MDN = F.getMetadata(MetadataName);
    for (size_t I = 0, E = MDN->getNumOperands(); I < E; ++I) {
      MDString *S = cast<llvm::MDString>(MDN->getOperand(I).get());
      Result += '-';
      Result += S->getString();
    }
  }

  Result += "-";
}

void computeFuncCategoryFromIntegersListMetadata(const Function &F,
                                                 const StringRef MetadataName,
                                                 SmallString<256> &Result) {
  if (F.hasMetadata(MetadataName)) {
    auto *MDN = F.getMetadata(MetadataName);
    for (const MDOperand &MDOp : MDN->operands()) {
      Result += '-';
      Result +=
          std::to_string(mdconst::extract<ConstantInt>(MDOp)->getZExtValue());
    }
  }

  Result += "-";
}

void computeFuncCategoryFromSYCLUsedAspects(const Function &F,
                                            SmallString<256> &Result) {
  if (const MDNode *UsedAspects = F.getMetadata("sycl_used_aspects")) {
    SmallVector<std::uint64_t, 8> Values;
    for (const MDOperand &MDOp : UsedAspects->operands()) {
      if (auto MDN = dyn_cast<MDNode>(MDOp)) {
        assert(MDN->getNumOperands() == 2);
        Values.push_back(
            mdconst::extract<ConstantInt>(MDN->getOperand(1))->getZExtValue());
      } else if (auto C = mdconst::dyn_extract<ConstantInt>(MDOp)) {
        Values.push_back(C->getZExtValue());
      }
    }

    llvm::sort(Values);
    for (std::uint64_t V : Values) {
      Result += '-';
      Result += std::to_string(V);
    }
  }

  Result += "-";
}

/// The function computes a string category for the given \p F.
/// The categories are used to separate functions during the splitting
/// meaning if functions get different categories they shouldn't end up
/// in the same split module.
std::string computeFuncCategoryForSplittingPerSource(const Function &F) {
  SmallString<256> Result;
  computeFuncCategoryFromAttribute(F, sycl::utils::ATTR_SYCL_MODULE_ID, Result);

  // This attribute marks virtual functions and effectively dictates how they
  // should be grouped together. By design we won't split those groups of
  // virtual functions further even if functions from the same group use
  // different optional features and therefore this distinction is put here.
  // TODO: for AOT use case we shouldn't be outlining those and instead should
  // only select those functions which are compatible with the target device.
  computeFuncCategoryFromAttribute(F, "indirectly-callable", Result);

  // Optional features
  // NOTE: Add more categories at the end of the list to avoid changing orders
  // of output files in existing tests.
  computeFuncCategoryFromAttribute(F, "sycl-register-alloc-mode", Result);
  computeFuncCategoryFromAttribute(F, "sycl-grf-size", Result);
  computeFuncCategoryFromSYCLUsedAspects(F, Result);
  computeFuncCategoryFromIntegersListMetadata(F, "reqd_work_group_size",
                                              Result);
  computeFuncCategoryFromIntegersListMetadata(F, "work_group_num_dim", Result);
  computeFuncCategoryFromIntegersListMetadata(F, "intel_reqd_sub_group_size",
                                              Result);
  computeFuncCategoryFromAttribute(F, sycl::utils::ATTR_SYCL_OPTLEVEL, Result);
  computeFuncCategoryFromStringMetadata(F, "sycl_joint_matrix", Result);
  computeFuncCategoryFromStringMetadata(F, "sycl_joint_matrix_mad", Result);
  return std::string(Result);
}

std::string computeFuncCategoryForSplitting(const Function &F,
                                            EntryPointsGroupScope Scope) {
  std::string Category;
  switch (Scope) {
  case Scope_Global:
    // We simply perform entry points filtering, but group all of them together.
    Category = GLOBAL_SCOPE_NAME;
    break;
  case Scope_PerKernel:
    // Per-kernel split is quite simple: every kernel goes into a separate
    // module and that's it, no other rules required.
    Category = F.getName().str();
    break;
  case Scope_PerModule:
    // The most complex case, because we should account for many other features
    // like aspects used in a kernel, large-grf mode, reqd-work-group-size, etc.

    // This is core of per-source device code split
    Category = computeFuncCategoryForSplittingPerSource(F);
    break;
  }

  return Category;
}

} // namespace

std::unique_ptr<ModuleSplitterBase>
getDeviceCodeSplitter(ModuleDesc &&MD, IRSplitMode Mode, bool IROutputOnly,
                      bool EmitOnlyKernelsAsEntryPoints,
                      bool AllowDeviceImageDependencies) {
  EntryPointsGroupScope Scope =
      selectDeviceCodeGroupScope(MD.getModule(), Mode, IROutputOnly);

  // std::map is used here to ensure stable ordering of entry point groups,
  // which is based on their contents, this greatly helps LIT tests
  std::map<std::string, EntryPointSet> EntryPointsMap;

  // Only process module entry points:
  for (auto &F : MD.getModule().functions()) {
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints))
      continue;

    std::string Key = computeFuncCategoryForSplitting(F, Scope);
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
    for (auto &[Key, EntryPoints] : EntryPointsMap) {
      bool HasVirtualFunctions = false;
      for (auto *F : EntryPoints) {
        if (F->hasFnAttribute("indirectly-callable")) {
          HasVirtualFunctions = true;
          break;
        }
      }

      auto PropsCopy = MDProps;
      PropsCopy.HasVirtualFunctionDefinitions = HasVirtualFunctions;
      Groups.emplace_back(Key, std::move(EntryPoints), PropsCopy);
    }
  }

  bool DoSplit = (Mode != SPLIT_NONE &&
                  (Groups.size() > 1 || !Groups.cbegin()->Functions.empty()));

  if (DoSplit)
    return std::make_unique<ModuleSplitter>(std::move(MD), std::move(Groups),
                                            AllowDeviceImageDependencies);

  return std::make_unique<ModuleCopier>(std::move(MD), std::move(Groups),
                                        AllowDeviceImageDependencies);
}

// Splits input module into two:
// - one containing _all_ ESIMD kernels, ESIMD functions and everything they use
// - another one which contains everything else
//
// The most interesting part here is that if a regular SYCL kernel uses a ESIMD
// function (through invoke_simd), it won't be included in non-ESIMD module.
//
// The reason for that is because ESIMD functions should undergo special
// handling and therefore we isolate them all into a separate module completely
// to do so. Due to design choices in passes provided by vc-intrinsics repo, we
// can't handle ESIMD functions _only_ in a mixed module.
//
// Functions, which are used from both ESIMD and non-ESIMD code will be
// duplicated into each module.
//
// If there are dependencies between ESIMD and non-ESIMD code (produced by
// invoke_simd, for example), the modules has to be linked back together to
// avoid undefined behavior at later stages. That is done at higher level,
// outside of this function.
SmallVector<ModuleDesc, 2> splitByESIMD(ModuleDesc &&MD,
                                        bool EmitOnlyKernelsAsEntryPoints,
                                        bool AllowDeviceImageDependencies) {

  SmallVector<module_split::ModuleDesc, 2> Result;
  EntryPointGroupVec EntryPointGroups{};
  EntryPointSet SYCLEntryPoints, ESIMDEntryPoints;
  bool hasESIMDFunctions = false;

  // Only process module entry points:
  for (Function &F : MD.getModule().functions()) {
    if (isESIMDFunction(F))
      hasESIMDFunctions = true;
    if (!isEntryPoint(F, EmitOnlyKernelsAsEntryPoints) ||
        !MD.isEntryPointCandidate(F))
      continue;
    if (isESIMDFunction(F))
      ESIMDEntryPoints.insert(&F);
    else
      SYCLEntryPoints.insert(&F);
  }

  // If there are no ESIMD entry points but there are ESIMD functions,
  // we still need to create an (empty) entry point group so that we
  // can lower the ESIMD functions.
  if (!ESIMDEntryPoints.empty() || hasESIMDFunctions) {
    EntryPointGroups.emplace_back(ESIMD_SCOPE_NAME, std::move(ESIMDEntryPoints),
                                  MD.getEntryPointGroup().Props);
    EntryPointGroup &G = EntryPointGroups.back();
    G.Props.HasESIMD = SyclEsimdSplitStatus::ESIMD_ONLY;
  }

  if (!SYCLEntryPoints.empty() || EntryPointGroups.empty()) {
    EntryPointGroups.emplace_back(SYCL_SCOPE_NAME, std::move(SYCLEntryPoints),
                                  MD.getEntryPointGroup().Props);
    EntryPointGroup &G = EntryPointGroups.back();
    G.Props.HasESIMD = SyclEsimdSplitStatus::SYCL_ONLY;
  }

  if (EntryPointGroups.size() == 1) {
    Result.emplace_back(MD.releaseModulePtr(), std::move(EntryPointGroups[0]),
                        MD.Props);
    return Result;
  }

  DependencyGraph CG(MD.getModule(), AllowDeviceImageDependencies);
  for (auto &Group : EntryPointGroups) {
    if (Group.isEsimd()) {
      // For ESIMD module, we use full call graph of all entry points and all
      // ESIMD functions.
      Result.emplace_back(extractESIMDSubModule(MD, std::move(Group), CG,
                                                AllowDeviceImageDependencies));
    } else {
      // For non-ESIMD module we only use non-ESIMD functions. Additional filter
      // is needed, because there could be uses of ESIMD functions from
      // non-ESIMD functions through invoke_simd. If that is the case, both
      // modules are expected to be linked back together after ESIMD functions
      // were processed and therefore it is fine to return an "incomplete"
      // module here.
      Result.emplace_back(extractCallGraph(
          MD, std::move(Group), CG, AllowDeviceImageDependencies,
          [=](const Function *F) -> bool { return !isESIMDFunction(*F); }));
    }
  }

  return Result;
}

static Error saveModuleIRInFile(Module &M, StringRef FilePath,
                                bool OutputAssembly) {
  int FD = -1;
  if (std::error_code EC = sys::fs::openFileForWrite(FilePath, FD))
    return errorCodeToError(EC);

  raw_fd_ostream OS(FD, true);
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  if (OutputAssembly)
    MPM.addPass(PrintModulePass(OS));
  else
    MPM.addPass(BitcodeWriterPass(OS));

  MPM.run(M, MAM);
  return Error::success();
}

static Expected<SplitModule> saveModuleDesc(ModuleDesc &MD, std::string Prefix,
                                            bool OutputAssembly) {
  SplitModule SM;
  Prefix += OutputAssembly ? ".ll" : ".bc";
  MD.saveSplitInformationAsMetadata();
  Error E = saveModuleIRInFile(MD.getModule(), Prefix, OutputAssembly);
  if (E)
    return E;

  SM.ModuleFilePath = Prefix;
  SM.Symbols = MD.makeSymbolTable();
  return SM;
}

Expected<std::vector<SplitModule>> parseSplitModulesFromFile(StringRef File) {
  auto EntriesMBOrErr = llvm::MemoryBuffer::getFile(File);

  if (!EntriesMBOrErr)
    return createFileError(File, EntriesMBOrErr.getError());

  line_iterator LI(**EntriesMBOrErr);
  if (LI.is_at_eof() || *LI != "[Code|Properties|Symbols]")
    return createStringError(inconvertibleErrorCode(),
                             "invalid SYCL Table file.");

  ++LI;
  std::vector<module_split::SplitModule> Modules;
  while (!LI.is_at_eof()) {
    StringRef Line = *LI;
    if (Line.empty())
      return createStringError(inconvertibleErrorCode(),
                               "invalid SYCL table row.");

    SmallVector<StringRef, 3> Parts;
    Line.split(Parts, "|");
    if (Parts.size() != 3)
      return createStringError(inconvertibleErrorCode(),
                               "invalid SYCL Table row.");

    auto [IRFilePath, PropertyFilePath, SymbolsFilePath] =
        std::tie(Parts[0], Parts[1], Parts[2]);
    if (PropertyFilePath.empty() || SymbolsFilePath.empty())
      return createStringError(inconvertibleErrorCode(),
                               "invalid SYCL Table row.");

    auto MBOrErr = MemoryBuffer::getFile(PropertyFilePath);
    if (!MBOrErr)
      return createFileError(PropertyFilePath, MBOrErr.getError());

    auto &MB = **MBOrErr;
    auto PropSetOrErr = llvm::util::PropertySetRegistry::read(&MB);
    if (!PropSetOrErr)
      return PropSetOrErr.takeError();

    llvm::util::PropertySetRegistry Properties = std::move(**PropSetOrErr);
    MBOrErr = MemoryBuffer::getFile(SymbolsFilePath);
    if (!MBOrErr)
      return createFileError(SymbolsFilePath, MBOrErr.getError());

    auto &MB2 = *MBOrErr;
    std::string Symbols =
        std::string(MB2->getBufferStart(), MB2->getBufferEnd());
    Modules.emplace_back(IRFilePath, std::move(Properties), std::move(Symbols));
    ++LI;
  }

  return Modules;
}

bool runPreSplitProcessingPipeline(Module &M) {
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });

  // Propagate ESIMD attribute to wrapper functions to prevent
  // spurious splits and kernel link errors.
  MPM.addPass(SYCLFixupESIMDKernelWrapperMDPass());

  // After linking device bitcode "llvm.used" holds references to the kernels
  // that are defined in the device image. But after splitting device image into
  // separate kernels we may end up with having references to kernel declaration
  // originating from "llvm.used" in the IR that is passed to llvm-spirv tool,
  // and these declarations cause an assertion in llvm-spirv. To workaround this
  // issue remove "llvm.used" from the input module before performing any other
  // actions.
  MPM.addPass(CleanupSYCLMetadataFromLLVMUsed());

  // There may be device_global variables kept alive in "llvm.compiler.used"
  // to keep the optimizer from wrongfully removing them. llvm.compiler.used
  // symbols are usually removed at backend lowering, but this is handled here
  // for SPIR-V since SYCL compilation uses llvm-spirv, not the SPIR-V backend.
  if (M.getTargetTriple().str().find("spir") != std::string::npos)
    MPM.addPass(RemoveDeviceGlobalFromLLVMCompilerUsed());

  // Sanitizer specific passes.
  if (sycl::isModuleUsingAsan(M) || sycl::isModuleUsingMsan(M) ||
      sycl::isModuleUsingTsan(M))
    MPM.addPass(SanitizerKernelMetadataPass());

  // Transform Joint Matrix builtin calls to align them with SPIR-V friendly
  // LLVM IR specification.
  MPM.addPass(SYCLJointMatrixTransformPass());

  // Do invoke_simd processing before splitting because this:
  // - saves processing time (the pass is run once, even though on larger IR)
  // - doing it before SYCL/ESIMD splitting is required for correctness
  MPM.addPass(SYCLLowerInvokeSimdPass());
  return !MPM.run(M, MAM).areAllPreserved();
}

Expected<std::vector<SplitModule>>
splitSYCLModule(std::unique_ptr<Module> M, ModuleSplitterSettings Settings) {
  ModuleDesc MD = std::move(M); // makeModuleDesc() ?
  // FIXME: false arguments are temporary for now.
  auto Splitter = getDeviceCodeSplitter(std::move(MD), Settings.Mode,
                                        /*IROutputOnly=*/false,
                                        /*EmitOnlyKernelsAsEntryPoints=*/false,
                                        Settings.AllowDeviceImageDependencies);

  size_t ID = 0;
  std::vector<SplitModule> OutputImages;
  while (Splitter->hasMoreSplits()) {
    ModuleDesc MD2 = Splitter->nextSplit();
    MD2.fixupLinkageOfDirectInvokeSimdTargets();

    std::string OutIRFileName = (Settings.OutputPrefix + "_" + Twine(ID)).str();
    auto SplittedImageOrErr =
        saveModuleDesc(MD2, OutIRFileName, Settings.OutputAssembly);
    if (!SplittedImageOrErr)
      return SplittedImageOrErr.takeError();

    OutputImages.emplace_back(std::move(*SplittedImageOrErr));
    ++ID;
  }

  return OutputImages;
}

bool canBeImportedFunction(const Function &F,
                           bool AllowDeviceImageDependencies) {

  // We use sycl dynamic library mechanism to involve bf16 devicelib when
  // necessary, all __devicelib_* functions from native or fallback bf16
  // devicelib will be treated as imported function to user's device image.
  if (llvm::isBF16DeviceLibFuncDecl(F))
    return true;
  // It may be theoretically possible to determine what is importable
  // based solely on function F, but the "SYCL/imported symbols"
  // property list MUST NOT have any imported symbols that are not supplied
  // the exported symbols from another device image.  This will lead to a
  // runtime crash "No device image found for external symbol". Generating
  // precise "SYCL/imported symbols" can be difficult because there exist
  // functions that may look like they can be imported, but are supplied outside
  // of user device code (e.g. _Z38__spirv_JointMatrixWorkItemLength...) In
  // order to be safe and not require perfect name analysis just start with this
  // simple check.
  if (!AllowDeviceImageDependencies)
    return false;

  // SYCL_EXTERNAL property is not recorded for a declaration
  // in a header file.  Thus SYCL IR that is a declaration
  // will be considered as SYCL_EXTERNAL for the purposes of
  // this function.
  if (isIntrinsicOrBuiltin(F) ||
      (!F.isDeclaration() && !llvm::sycl::utils::isSYCLExternalFunction(&F)))
    return false;

  bool ReturnValue = true;
  if (char *NameStr = itaniumDemangle(F.getName())) {
    StringRef DemangledName(NameStr);
    if (DemangledName.starts_with("__"))
      ReturnValue = false;
    free(NameStr);
  }
  return ReturnValue;
}

} // namespace module_split
} // namespace llvm
