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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormatVariadic.h"

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
                "Propagates aspect usage", false, false)

ModulePass *llvm::createPropagateAspectUsagePass() {
  return new SYCLPropagateAspectUsageLegacyPass();
}

namespace {

using AspectsSetTy = SmallSet<int, 4>;
using TypeToAspectsMapTy = std::unordered_map<const Type *, AspectsSetTy>;

/// Retrieves from metadata (intel_types_that_use_aspects) types
/// and aspects these types depend on.
TypeToAspectsMapTy GetTypesThatUseAspectsFromMetadata(const Module &M) {
  NamedMDNode *Node = M.getNamedMetadata("intel_types_that_use_aspects");
  TypeToAspectsMapTy Result;
  if (!Node)
    return Result;

  LLVMContext &C = M.getContext();
  for (auto OperandIt : Node->operands()) {
    MDNode &N = *OperandIt;
    assert(N.getNumOperands() > 1 && "intel_types_that_use_aspect metadata "
                                     "shouldn't contain empty metadata nodes");

    MDString *TypeName = cast<MDString>(N.getOperand(0));
    Type *T = StructType::getTypeByName(C, TypeName->getString());
    assert(T &&
           "invalid type referenced by intel_types_that_use_aspect metadata");

    AspectsSetTy &Aspects = Result[T];
    for (size_t I = 1; I != N.getNumOperands(); ++I) {
      ConstantAsMetadata *CAM = cast<ConstantAsMetadata>(N.getOperand(I));
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
/// Result is recorded in @ResultAspects.
void PropagateAspectsThroughTypes(const TypesEdgesTy &Edges, const Type *Start,
                                  const AspectsSetTy &AspectsToPropagate,
                                  TypeToAspectsMapTy &ResultAspects) {
  SmallPtrSet<const Type *, 16> Visited;
  std::queue<const Type *> queue;
  queue.push(Start);
  while (!queue.empty()) {
    const Type *T = queue.front();
    queue.pop();
    if (Visited.count(T))
      continue;

    Visited.insert(T);
    ResultAspects[T].insert(AspectsToPropagate.begin(),
                            AspectsToPropagate.end());
    if (!Edges.count(T))
      continue;

    for (const Type *TT : Edges.at(T)) {
      if (!Visited.count(TT))
        queue.push(TT);
    }
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
/// 2) Make from list a type graph which consist of nodes corresponding to types
///    and directed edges between nodes. An edge from type A to type B
///    corresponds to the fact that A is contained within B.
///    Examples: B is a pointer to A, B is a struct containing field of type A.
/// 3) For every known type with aspects propagate it's aspects over graph.
///    Every propagation is a separate run of BFS algorithm.
///
/// Time complexity: O((V + E) * T) where T is the number of input types
/// containing aspects.
void PropagateAspectsToOtherTypesInModule(
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
      while (TT->isPointerTy())
        TT = TT->getContainedType(0);

      // We are not interested in some types. For example, IntTy.
      if (!TypesToProcess.count(TT))
        continue;

      Edges[TT].push_back(T);
    }
  }

  TypeToAspectsMapTy Result;
  for (const Type *T : TypesToProcess) {
    AspectsSetTy &Aspects = TypesWithAspects[T];
    PropagateAspectsThroughTypes(Edges, T, Aspects, TypesWithAspects);
  }
}

/// Returns all aspects which might be reached from type @T.
/// It encompases composite structures and pointers.
/// NB! This function inserts new records in @Types map for new discovered
/// types. For the best perfomance pass this map in the next invocations.
AspectsSetTy GetAspectsFromType(const Type *T, TypeToAspectsMapTy &Types) {
  auto It = Types.find(T);
  if (It != Types.end())
    return It->second;

  // This is essential to no get into infinite recursive loops.
  Types[T] = {};
  AspectsSetTy Result;

  for (const Type *TT : T->subtypes()) {
    AspectsSetTy Aspects = GetAspectsFromType(TT, Types);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  Types[T] = Result;
  return Result;
}

/// Returns aspects which might be used in instruction @I.
/// Function inspects return type and all operand's types.
/// NB! This function inserts new records in @Types map for new discovered
/// types. For the best perfomance pass this map in the next invocations.
AspectsSetTy GetAspectsUsedByInstruction(const Instruction &I,
                                         TypeToAspectsMapTy &Types) {
  Type *ReturnType = I.getType();
  AspectsSetTy Result = GetAspectsFromType(ReturnType, Types);
  for (const auto &OperandIt : I.operands()) {
    Type *T = OperandIt->getType();
    AspectsSetTy Aspects = GetAspectsFromType(T, Types);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  return Result;
}

/// Class for emiting warning messages. Unfortunately, LLVMContext
/// doesn't contain tools for that.
class MissedAspectDiagnosticInfo : public DiagnosticInfo {
  Twine Msg;

public:
  MissedAspectDiagnosticInfo(Twine DiagMsg,
                             DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfo(DK_Linker, Severity), Msg(std::move(DiagMsg)) {}

  void print(DiagnosticPrinter &DP) const override { DP << Msg; }
};

void EmitWarning(LLVMContext &C, const StringRef Msg) {
  C.diagnose(MissedAspectDiagnosticInfo(Msg));
}

std::string Join(const AspectsSetTy &C, char sep) {
  std::string S;
  bool FirstOccurence = true;
  for (int Aspect : C) {
    if (!FirstOccurence) {
      S += sep;
      S += ' ';
    }

    FirstOccurence = false;
    S += std::to_string(Aspect);
  }

  return S;
}

/// Checks that all declared function's aspects correspond to the
/// aspects set @Aspects. If there is a inconsistency then corresponding
/// warning is emitted.
template <class Container>
void CheckDeclaredAspectsForFunction(LLVMContext &C, const Function *F,
                                     const Container &UsedAspects) {
  MDNode *MDN = F->getMetadata("intel_declared_aspects");
  if (!MDN)
    return;

  AspectsSetTy DeclaredAspects;
  for (auto &OperandIt : MDN->operands()) {
    const ConstantAsMetadata *CAM =
        dyn_cast<const ConstantAsMetadata>(OperandIt);
    assert(CAM &&
           "constant are expected in intel_declared_aspects list's entries");
    DeclaredAspects.insert(
        cast<const ConstantInt>(CAM->getValue())->getSExtValue());
  }

  AspectsSetTy MissedAspects;
  for (int Aspect : UsedAspects) {
    if (DeclaredAspects.count(Aspect) == 0)
      MissedAspects.insert(Aspect);
  }

  if (!MissedAspects.empty()) {
    std::string AspectsStr = Join(MissedAspects, ',');
    // TODO: demangle function name and aspect's IDs?
    std::string Msg = formatv(
        "for function \"{0}\" there is the list of missed aspects: [{1}]",
        F->getName(), AspectsStr);

    EmitWarning(C, Msg);
  }
}

using FunctionToAspectsMapTy = DenseMap<Function *, SmallSet<int, 4>>;
using CallGraphTy = DenseMap<Function *, SmallPtrSet<Function *, 8>>;

void CreateUsedAspectsMetadataForFunctions(FunctionToAspectsMapTy &Map) {
  for (auto &It : Map) {
    Function *F = It.first;
    AspectsSetTy &Aspects = It.second;
    if (Aspects.empty())
      continue;

    LLVMContext &C = F->getContext();
    SmallVector<Metadata *, 16> AspectsMetadata;
    for (int Aspect : Aspects)
      AspectsMetadata.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), Aspect)));

    MDNode *MDN = MDNode::get(C, AspectsMetadata);
    F->setMetadata("intel_used_aspects", MDN);
  }
}

void CheckUsedAndDeclaredAspects(const FunctionToAspectsMapTy &Map) {
  for (const auto &It : Map) {
    const Function *F = It.first;
    auto &Aspects = It.second;
    if (Aspects.empty())
      continue;

    LLVMContext &C = F->getContext();
    CheckDeclaredAspectsForFunction(C, F, Aspects);
  }
}

/// Propagates aspects from leaves up to the top of call graph.
/// NB! Call graph corresponds to call graph of SYCL code which
/// can't contain recursive calls. So there can't be loops in
/// a call graph. But there can be path's intersections.
void PropagateAspectsThroughCG(Function *F, CallGraphTy &CG,
                               FunctionToAspectsMapTy &AspectsMap,
                               SmallPtrSet<Function *, 16> &Visited) {
  if (CG.count(F) == 0)
    return;

  AspectsSetTy LocalAspects;
  for (Function *Callee : CG[F]) {
    if (!Visited.contains(Callee)) {
      Visited.insert(Callee);
      PropagateAspectsThroughCG(Callee, CG, AspectsMap, Visited);
    }

    auto &CalleeAspects = AspectsMap[Callee];
    LocalAspects.insert(CalleeAspects.begin(), CalleeAspects.end());
  }

  AspectsMap[F].insert(LocalAspects.begin(), LocalAspects.end());
}

/// Returns a map of functions with corresponding used aspects.
FunctionToAspectsMapTy
BuildFunctionsToAspectsMap(Module &M, TypeToAspectsMapTy &TypesWithAspects) {
  FunctionToAspectsMapTy FunctionToAspects;
  CallGraphTy CG;
  std::vector<Function *> Kernels;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    if (CC == CallingConv::SPIR_KERNEL)
      Kernels.push_back(&F);

    for (Instruction &I : instructions(F)) {
      AspectsSetTy Aspects = GetAspectsUsedByInstruction(I, TypesWithAspects);
      FunctionToAspects[&F].insert(Aspects.begin(), Aspects.end());
      if (CallInst *CI = dyn_cast<CallInst>(&I))
        if (!CI->isIndirectCall())
          CG[&F].insert(CI->getCalledFunction());
    }
  }

  SmallPtrSet<Function *, 16> Visited;
  for (Function *F : Kernels)
    PropagateAspectsThroughCG(F, CG, FunctionToAspects, Visited);

  return FunctionToAspects;
}

} // anonymous namespace

PreservedAnalyses PropagateAspectUsagePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  TypeToAspectsMapTy TypesWithAspects = GetTypesThatUseAspectsFromMetadata(M);
  PropagateAspectsToOtherTypesInModule(M, TypesWithAspects);

  FunctionToAspectsMapTy FunctionToAspects =
      BuildFunctionsToAspectsMap(M, TypesWithAspects);

  CreateUsedAspectsMetadataForFunctions(FunctionToAspects);
  CheckUsedAndDeclaredAspects(FunctionToAspects);

  return PreservedAnalyses::all();
}
