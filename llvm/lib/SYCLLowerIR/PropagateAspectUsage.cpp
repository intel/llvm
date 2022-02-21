//===---- PropagateAspectUsage.h - PropagateAspectUsage Pass --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass propagates metadata corresponding to usage of optional device
// features.
//
//===----------------------------------------------------------------------===//
//
#include "llvm/SYCLLowerIR/PropagateAspectUsage.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

namespace {

class SYCLPropagateAspectUsageLegacyPass : public ModulePass {
public:
  static char ID; // Pass identification, replacement for typeid
  SYCLPropagateAspectUsageLegacyPass() : ModulePass(ID) {
    initializeSYCLPropagateAspectUsageLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  PropagateAspectUsagePass Impl;
};

} // namespace

char SYCLPropagateAspectUsageLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLPropagateAspectUsageLegacyPass, "PropagateAspectUsage",
                "Propagate aspect usage", false, false)

ModulePass *llvm::createPropagateAspectUsagePass() {
  return new SYCLPropagateAspectUsageLegacyPass();
}

namespace {

using AspectsSetTy = SmallSet<int, 4>;
using TypeToAspectsMapTy = std::unordered_map<const Type *, AspectsSetTy>;

/// Retrieves from metadata (intel_types_that_use_aspects) types
/// and aspects these types depend on.
TypeToAspectsMapTy getTypesThatUseAspectsFromMetadata(const Module &M) {
  NamedMDNode *Node = M.getNamedMetadata("intel_types_that_use_aspects");
  TypeToAspectsMapTy Result;
  if (!Node)
    return Result;

  LLVMContext &C = M.getContext();
  for (auto OperandIt : Node->operands()) {
    MDNode &N = *OperandIt;
    assert(N.getNumOperands() > 1 && "intel_types_that_use_aspect metadata "
                                     "shouldn't contain empty metadata nodes");

    auto *TypeName = cast<MDString>(N.getOperand(0));
    Type *T = StructType::getTypeByName(C, TypeName->getString());
    assert(T &&
           "invalid type referenced by intel_types_that_use_aspect metadata");

    AspectsSetTy &Aspects = Result[T];
    for (size_t I = 1; I != N.getNumOperands(); ++I) {
      auto *CAM = cast<ConstantAsMetadata>(N.getOperand(I));
      Constant *C = CAM->getValue();
      Aspects.insert(cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

using TypesEdgesTy =
    std::unordered_map<const Type *, std::vector<const Type *>>;

/// Propagates aspects from type @Start to all types which
/// are reachable by edges @Edges by BFS algorithm.
/// Result is recorded in @Aspects.
void propagateAspectsThroughTypes(const TypesEdgesTy &Edges, const Type *Start,
                                  TypeToAspectsMapTy &Aspects) {
  const AspectsSetTy &AspectsToPropagate = Aspects[Start];
  SmallSetVector<const Type *, 16> TypesToPropagate;
  TypesToPropagate.insert(Start);
  for (size_t I = 0; I < TypesToPropagate.size(); ++I) {
    const Type *T = TypesToPropagate[I];
    Aspects[T].insert(AspectsToPropagate.begin(), AspectsToPropagate.end());
    auto It = Edges.find(T);
    if (It != Edges.end())
      TypesToPropagate.insert(It->second.begin(), It->second.end());
  }
}

/// Propagates given aspects to all types in module @M. Function accepts
/// aspects in @TypesWithAspects reference and writes a result in this
/// reference.
/// Type T in the result contains an aspect A if Type T is a composite
/// type (array, struct, vector) which contains elements/fields of
/// another type TT, which in turn uses the aspect A.
/// @TypesWithAspects argument consist of known types with aspects
/// from metadata information.
///
/// The algorithm is the following:
/// 1) Make a list of all structure types from module @M. The list also
///    contains DoubleTy since it is optional as well.
/// 2) Make from list a type graph which consists of nodes corresponding to
///    types and directed edges between nodes. An edge from type A to type B
///    corresponds to the fact that A is contained within B.
///    Examples: B is a pointer to A, B is a struct containing field of type A.
/// 3) For every known type with aspects propagate it's aspects over graph.
///    Every propagation is a separate run of BFS algorithm.
///
/// Time complexity: O((V + E) * T) where T is the number of input types
/// containing aspects.
void propagateAspectsToOtherTypesInModule(
    const Module &M, TypeToAspectsMapTy &TypesWithAspects) {
  std::unordered_set<const Type *> TypesToProcess;
  Type *DoubleTy = Type::getDoubleTy(M.getContext());

  // 6 is taken from sycl/include/CL/sycl/aspects.hpp
  static constexpr int AspectFP64 = 6;
  TypesWithAspects[DoubleTy].insert(AspectFP64);

  TypesToProcess.insert(DoubleTy);
  for (Type *T : M.getIdentifiedStructTypes())
    TypesToProcess.insert(T);

  TypesEdgesTy Edges;
  for (const Type *T : TypesToProcess) {
    for (const Type *TT : T->subtypes()) {
      // If TT = %A*** then we want to get TT = %A
      // The same with arrays and vectors
      while (TT->isPointerTy() || TT->isArrayTy() || TT->isVectorTy()) {
        TT = TT->getContainedType(0);
      }

      // We are not interested in some types. For example, IntTy.
      if (TypesToProcess.count(TT))
        Edges[TT].push_back(T);
    }
  }

  TypeToAspectsMapTy Result;
  for (const Type *T : TypesToProcess)
    propagateAspectsThroughTypes(Edges, T, TypesWithAspects);
}

/// Returns all aspects which might be reached from type @T.
/// It encompases composite structures and pointers.
/// NB! This function inserts new records in @Types map for new discovered
/// types. For the best perfomance pass this map in the next invocations.
const AspectsSetTy &getAspectsFromType(const Type *T,
                                       TypeToAspectsMapTy &Types) {
  auto It = Types.find(T);
  if (It != Types.end())
    return It->second;

  // Empty value is inserted for absent key T.
  // This is essential to no get into infinite recursive loops.
  AspectsSetTy &Result = Types[T];

  for (const Type *TT : T->subtypes()) {
    const AspectsSetTy &Aspects = getAspectsFromType(TT, Types);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  return Result;
}

/// Returns aspects which might be used in instruction @I.
/// Function inspects return type and all operand's types.
/// NB! This function inserts new records in @Types map for new discovered
/// types. For the best perfomance pass this map in the next invocations.
AspectsSetTy getAspectsUsedByInstruction(const Instruction &I,
                                         TypeToAspectsMapTy &Types) {
  Type *ReturnType = I.getType();
  AspectsSetTy Result = getAspectsFromType(ReturnType, Types);
  for (const auto &OperandIt : I.operands()) {
    Type *T = OperandIt->getType();
    const AspectsSetTy &Aspects = getAspectsFromType(T, Types);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  return Result;
}

/// This is a node in a CallGraph. It contains Function link
/// to Callee and corresponding DebugLoc info if it it present.
struct FunctionLinkTy {
  const Function *F = nullptr;
  const DebugLoc *DL = nullptr;
};

using AspectToFunctionLinkMapTy = DenseMap<int, FunctionLinkTy>;
using FunctionToAspectsMapTy = DenseMap<Function *, AspectToFunctionLinkMapTy>;
using CallGraphTy =
    DenseMap<Function *, DenseMap<Function *, const DebugLoc *>>;

std::string constructAspectUsageChain(const Function *F,
                                      const FunctionToAspectsMapTy &Map,
                                      int Aspect) {
  std::string CallChain;
  while (F) {
    auto It = Map.find(F);
    if (It == Map.end())
      break;

    auto &AspectsToFunctionLinkMap = It->second;
    auto AspectIt = AspectsToFunctionLinkMap.find(Aspect);
    assert(AspectIt != AspectsToFunctionLinkMap.end() &&
           "AspectIt is supposed to be determined");
    const DebugLoc *DL = AspectIt->second.DL;
    CallChain += "  ";
    CallChain += demangle(F->getName().str());
    if (DL && *DL) {
      DIScope *DS = cast<DIScope>(DL->getScope());
      CallChain += formatv(" (defined at {0}:{1}:{2})",
                           DS->getDirectory() + sys::path::get_separator() +
                               DS->getFilename(),
                           DL->getLine(), DL->getCol());
    }

    CallChain += "\n";
    F = AspectIt->second.F;
  }

  return CallChain;
}

std::string getAspectStrRepresentation(int Aspect) {
  switch (Aspect) {
  case 0:
    return "host";
  case 1:
    return "cpu";
  case 2:
    return "gpu";
  case 3:
    return "accelerator";
  case 4:
    return "custom";
  case 5:
    return "fp16";
  case 6:
    return "fp64";
  case 7:
    return "int64_base_atomics";
  case 8:
    return "int64_extended_atomics";
  case 9:
    return "image";
  case 10:
    return "online_compiler";
  case 11:
    return "online_linker";
  case 12:
    return "queue_profiling";
  case 13:
    return "usm_device_allocations";
  case 14:
    return "usm_host_allocations";
  case 15:
    return "usm_shared_allocations";
  case 16:
    return "usm_restricted_shared_allocations";
  case 17:
    return "usm_system_allocations";
  case 18:
    return "ext_intel_pci_address";
  case 19:
    return "ext_intel_gpu_eu_count";
  case 20:
    return "ext_intel_gpu_eu_simd_width";
  case 21:
    return "ext_intel_gpu_slices";
  case 22:
    return "ext_intel_gpu_subslices_per_slice";
  case 23:
    return "ext_intel_gpu_eu_count_per_subslice";
  case 24:
    return "ext_intel_max_mem_bandwidth";
  case 25:
    return "ext_intel_mem_channel";
  case 26:
    return "usm_atomic_host_allocations";
  case 27:
    return "usm_atomic_shared_allocations";
  case 28:
    return "atomic64";
  case 29:
    return "ext_intel_device_info_uuid";
  case 30:
    return "ext_oneapi_srgb";
  case 31:
    return "ext_oneapi_native_assert";
  case 32:
    return "host_debuggable";
  case 33:
    return "ext_intel_gpu_hw_threads_per_eu";
  default:
    return std::to_string(Aspect);
  }

  llvm_unreachable("Aspect value isn't recognized");
}

/// Checks that all declared function's aspects correspond to the
/// aspects set @Aspects. If there is a inconsistency then corresponding
/// warning is emitted.
void checkDeclaredAspectsForFunction(Function *F,
                                     const FunctionToAspectsMapTy &Map,
                                     bool isFullDebug) {
  MDNode *MDN = F->getMetadata("intel_declared_aspects");
  if (!MDN)
    return;

  AspectsSetTy DeclaredAspects;
  for (auto &OperandIt : MDN->operands()) {
    const auto *CAM = dyn_cast<const ConstantAsMetadata>(OperandIt);
    assert(CAM &&
           "constant are expected in intel_declared_aspects list's entries");
    DeclaredAspects.insert(
        cast<const ConstantInt>(CAM->getValue())->getSExtValue());
  }

  const AspectToFunctionLinkMapTy &UsedAspects = Map.find(F)->second;
  AspectsSetTy MissedAspects;
  for (const auto &A : UsedAspects) {
    if (DeclaredAspects.count(A.first) == 0)
      MissedAspects.insert(A.first);
  }

  LLVMContext &C = F->getContext();
  for (int Aspect : MissedAspects) {
    auto CallChain = constructAspectUsageChain(F, Map, Aspect);
    C.diagnose(DiagnosticInfoSYCLUnspecAspect(
        F->getName(), getAspectStrRepresentation(Aspect), CallChain,
        isFullDebug));
  }
}

void createUsedAspectsMetadataForFunctions(FunctionToAspectsMapTy &Map) {
  for (auto &It : Map) {
    Function *F = It.first;
    AspectToFunctionLinkMapTy &Aspects = It.second;
    if (Aspects.empty())
      continue;

    LLVMContext &C = F->getContext();
    SmallVector<std::pair<int, FunctionLinkTy>, 16> AspectsVector(
        Aspects.begin(), Aspects.end());
    std::sort(AspectsVector.begin(), AspectsVector.end(),
              [](const auto &Lhs, const auto &Rhs) {
                // sort only by aspects
                return Lhs.first < Rhs.first;
              });

    SmallVector<Metadata *, 16> AspectsMetadata;
    for (const auto &A : AspectsVector)
      AspectsMetadata.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), A.first)));

    MDNode *MDN = MDNode::get(C, AspectsMetadata);
    F->setMetadata("intel_used_aspects", MDN);
  }
}

void checkUsedAndDeclaredAspects(FunctionToAspectsMapTy &Map,
                                 bool isFullDebug) {
  for (auto &It : Map) {
    Function *F = It.first;
    checkDeclaredAspectsForFunction(F, Map, isFullDebug);
  }
}

/// Propagates aspects from leaves up to the top of call graph.
/// NB! Call graph corresponds to call graph of SYCL code which
/// can't contain recursive calls. So there can't be loops in
/// a call graph. But there can be path's intersections.
void propagateAspectsThroughCG(Function *F, CallGraphTy &CG,
                               FunctionToAspectsMapTy &AspectsMap,
                               SmallPtrSet<Function *, 16> &Visited) {
  auto It = CG.find(F);
  if (It == CG.end())
    return;

  AspectToFunctionLinkMapTy LocalAspects;
  for (auto Edge : It->second) {
    Function *Callee = Edge.first;
    if (Visited.insert(Callee).second)
      propagateAspectsThroughCG(Callee, CG, AspectsMap, Visited);

    auto &CalleeAspects = AspectsMap[Callee];
    for (auto AspectIt : CalleeAspects) {
      LocalAspects[AspectIt.first] = FunctionLinkTy{Callee, Edge.second};
    }
  }

  AspectsMap[F].insert(LocalAspects.begin(), LocalAspects.end());
}

/// Processes function's instructions. It analyzes aspect usages with debug
/// information and builds a call graph.
void processFunctionInstructions(Function &F,
                                 FunctionToAspectsMapTy &FunctionToAspects,
                                 TypeToAspectsMapTy &TypesWithAspects,
                                 CallGraphTy &CG) {
  auto &AspectToFunctionLinkMap = FunctionToAspects[&F];
  for (Instruction &I : instructions(F)) {
    AspectsSetTy Aspects = getAspectsUsedByInstruction(I, TypesWithAspects);
    for (int Aspect : Aspects) {
      FunctionToAspects[&F][Aspect] = FunctionLinkTy{nullptr, &I.getDebugLoc()};
    }

    if (auto *DBI = dyn_cast<DbgVariableIntrinsic>(&I)) {
      // Handle a group of llvm.dbg.{addr,declare,value} intrinsics.
      // Example:
      // %tmp = alloca %Struct
      // call void @llvm.dbg.declare(metadata %Struct* %tmp, metadata !1,
      //                             metadata !DIExpression), !dbg !2
      //
      // Here we extract the first intrinsic argument, then extract
      // %Struct type, then analyze known aspects of %Struct and update
      // corresponding aspect's records with debug information !dbg !2.
      Type *T = DBI->getVariableLocationOp(0)->getType();
      Aspects = getAspectsFromType(T, TypesWithAspects);
      for (int Aspect : Aspects) {
        auto It = AspectToFunctionLinkMap.find(Aspect);
        if (It == AspectToFunctionLinkMap.end())
          continue;

        It->second.DL = &I.getDebugLoc();
      }

      continue;
    }

    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (!CI->isIndirectCall() && CI->getCalledFunction())
        CG[&F].try_emplace(CI->getCalledFunction(), &CI->getDebugLoc());
    }
  }
}

/// Returns a map of functions with corresponding used aspects.
FunctionToAspectsMapTy
buildFunctionsToAspectsMap(Module &M, TypeToAspectsMapTy &TypesWithAspects) {
  FunctionToAspectsMapTy FunctionToAspects;
  CallGraphTy CG;
  std::vector<Function *> Kernels;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    if (CC == CallingConv::SPIR_KERNEL)
      Kernels.push_back(&F);

    processFunctionInstructions(F, FunctionToAspects, TypesWithAspects, CG);
  }

  SmallPtrSet<Function *, 16> Visited;
  for (Function *F : Kernels)
    propagateAspectsThroughCG(F, CG, FunctionToAspects, Visited);

  return FunctionToAspects;
}

/// Checks whether module @M contains !DICompileUnit node
/// with emissionKind equal to FullDebug
bool checkModuleHasFullDebugMode(Module &M) {
  for (DICompileUnit *CU : M.debug_compile_units()) {
    if (CU->getEmissionKind() == DICompileUnit::DebugEmissionKind::FullDebug)
      return true;
  }

  return false;
}

} // anonymous namespace

PreservedAnalyses PropagateAspectUsagePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  TypeToAspectsMapTy TypesWithAspects = getTypesThatUseAspectsFromMetadata(M);
  propagateAspectsToOtherTypesInModule(M, TypesWithAspects);

  FunctionToAspectsMapTy FunctionToAspects =
      buildFunctionsToAspectsMap(M, TypesWithAspects);

  createUsedAspectsMetadataForFunctions(FunctionToAspects);
  checkUsedAndDeclaredAspects(FunctionToAspects,
                              checkModuleHasFullDebugMode(M));

  return PreservedAnalyses::all();
}
