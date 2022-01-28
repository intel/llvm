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
using TypeToAspectsMapTy = std::unordered_map<Type *, AspectsSetTy>;

/// Retrieves from metadata (intel_types_that_use_aspects) types
/// and aspects these types depend on.
TypeToAspectsMapTy GetTypesThatUseAspectsFromMetadata(Module &M) {
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
      ConstantAsMetadata *CAM = dyn_cast<ConstantAsMetadata>(N.getOperand(I));
      assert(CAM && "constant metadata is expected in "
                    "intel_types_that_use_aspect list's entry");
      Constant *C = CAM->getValue();
      Aspects.insert(cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

using TypesEdgesTy = std::unordered_map<Type *, std::vector<Type *>>;

/// Propagates aspects from type @Start to all types which
/// are reachable by edges @Edges by BFS algorithm.
/// Result is recorded in @ResultAspects.
void PropagateAspectsThroughTypes(TypesEdgesTy &Edges, Type *Start,
                                  const AspectsSetTy &AspectsToPropagate,
                                  TypeToAspectsMapTy &ResultAspects) {
  SmallPtrSet<Type *, 16> Visited;
  std::queue<Type *> queue;
  queue.push(Start);
  while (!queue.empty()) {
    Type *T = queue.front();
    queue.pop();
    if (Visited.count(T))
      continue;

    Visited.insert(T);
    ResultAspects[T].insert(AspectsToPropagate.begin(),
                            AspectsToPropagate.end());
    for (Type *TT : Edges[T]) {
      if (Visited.count(TT))
        continue;

      queue.push(TT);
    }
  }
}

/// Returns all types with corresponding aspects from module @M. Type T in the
/// result contains an aspect A if there is a way with instance of T to access
/// by internal fields or pointers another type TT which uses corresponding
/// aspect A.
/// @TypesWithAspects argument consist of known types with aspects
/// from metadata information.
///
/// The algorithm is the following:
/// 1) Make a list of all structure types from module @M. List also consist of
///    DoubleTy since it is optional as well.
/// 2) Make from list a type graph which consist of nodes corresponding to types
///    and directed edges between nodes. An edge from type A to type B
///    corresponds to the fact that A is accessible from an instance of type B
///    (by direct field or pointer in B).
/// 3) For every known type with aspects propagate it's aspects over graph.
///    Every propagation is a separate run of BFS algorithm.
///
/// Time complexity: O((V + E) * T) where T is the number of input types
/// containing aspects.
TypeToAspectsMapTy
GetTypesWithAspectsFromModule(Module &M, TypeToAspectsMapTy &TypesWithAspects) {
  std::unordered_set<Type *> Types;
  Type *DoubleTy = Type::getDoubleTy(M.getContext());
  static constexpr int AspectFP64 = 6; // TODO: add the link to spec
  TypesWithAspects[DoubleTy].insert(AspectFP64);

  Types.insert(DoubleTy);
  for (Type *T : M.getIdentifiedStructTypes()) {
    Types.insert(T);
  }

  std::unordered_map<Type *, std::vector<Type *>> Edges;
  for (Type *T : Types) {
    for (size_t i = 0; i != T->getNumContainedTypes(); ++i) {
      Type *TT = T->getContainedType(i);

      // If TT = %A*** then we want to get TT = %A
      while (TT->isPointerTy()) {
        TT = TT->getContainedType(0);
      }

      if (!Types.count(TT)) {
        continue; // We are not interested in some types. For example, IntTy.
      }

      Edges[TT].push_back(T);
    }
  }

  TypeToAspectsMapTy Result;
  for (auto it : TypesWithAspects) {
    Type *T = it.first;
    AspectsSetTy Aspects;
    Aspects.insert(it.second.begin(), it.second.end());
    PropagateAspectsThroughTypes(Edges, T, Aspects, Result);
  }

  return Result;
}

/// GetAspectsFromType() function finds all aspects which might be
/// reached from type @T. It encompases composite structures and
/// pointers.
/// It is important to maintain and pass @Cache structure so that procedure
/// doesn't get into infite loop.
AspectsSetTy GetAspectsFromType(Type *T, TypeToAspectsMapTy &Types,
                                TypeToAspectsMapTy &Cache) {
  auto it = Types.find(T);
  if (it != Types.end()) {
    Cache[T] = it->second;
    return it->second;
  }

  it = Cache.find(T);
  if (it != Cache.end()) {
    return it->second;
  }

  Cache[T] = {};
  AspectsSetTy Result;

  for (size_t i = 0; i != T->getNumContainedTypes(); ++i) {
    Type *TT = T->getContainedType(i);
    AspectsSetTy Aspects = GetAspectsFromType(TT, Types, Cache);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  Cache[T] = Result;
  return Result;
}

AspectsSetTy GetAspectsUsedByInstruction(Instruction &I,
                                         TypeToAspectsMapTy &Types,
                                         TypeToAspectsMapTy &Cache) {
  Type *ReturnType = I.getType();
  SmallSet<int, 4> Aspects = GetAspectsFromType(ReturnType, Types, Cache);
  SmallSet<int, 4> Result = Aspects;

  int i = 0;
  for (auto OpIt = I.op_begin(); OpIt != I.op_end(); ++OpIt, ++i) {
    Type *T = (*OpIt)->getType();
    Aspects = GetAspectsFromType(T, Types, Cache);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  return Result;
}

class MissedAspectDiagnosticInfo : public DiagnosticInfo {
  Twine Msg;

public:
  MissedAspectDiagnosticInfo(Twine DiagMsg,
                             DiagnosticSeverity Severity = DS_Warning)
      : DiagnosticInfo(DK_Linker, Severity), Msg(std::move(DiagMsg)) {}

  void print(DiagnosticPrinter &DP) const override { DP << Msg; }
};

void emitWarning(LLVMContext &C, const StringRef Msg) {
  C.diagnose(MissedAspectDiagnosticInfo(Msg));
}

template <class Container> std::string Join(const Container &C, char sep) {
  std::string S;
  auto it = C.begin();
  for (size_t i = 0; i != C.size(); ++i, ++it) {
    if (i > 0)
      S += ", ";

    S += std::to_string(*it);
  }

  return S;
}

template <class Container>
void CheckDeclaredAspects(LLVMContext &C, const Function *F,
                          const Container &Aspects) {
  MDNode *MDN = F->getMetadata("intel_declared_aspects");
  if (!MDN)
    return;

  AspectsSetTy DeclaredAspects;
  for (unsigned i = 0; i != MDN->getNumOperands(); ++i) {
    const ConstantAsMetadata *CM =
        dyn_cast<const ConstantAsMetadata>(MDN->getOperand(i));
    DeclaredAspects.insert(
        dyn_cast<const ConstantInt>(CM->getValue())->getSExtValue());
  }

  SmallVector<int> MissedAspects;
  for (int aspect : Aspects) {
    if (DeclaredAspects.count(aspect) == 0) {
      MissedAspects.push_back(aspect);
    }
  }

  if (!MissedAspects.empty()) {
    std::string AspectsStr = Join(MissedAspects, ',');
    // TODO: demangle function name and aspect's IDs?
    std::string Msg = formatv(
        "for function \"{0}\" there is the list of missed aspects: [{1}]",
        F->getName(), AspectsStr);

    emitWarning(C, Msg);
  }
}

using FunctionToAspectsMapTy = DenseMap<Function *, SmallSet<int, 4>>;
using CallGraphTy = DenseMap<Function *, SmallPtrSet<Function *, 8>>;

void CreateAspectMetadataForFunctions(FunctionToAspectsMapTy &Map) {
  for (auto &it : Map) {
    Function *F = it.first;
    auto &Aspects = it.second;
    if (Aspects.empty())
      continue;

    LLVMContext &C = F->getContext();
    std::vector<Metadata *> AspectsMetadata;
    AspectsMetadata.reserve(Aspects.size());
    for (int aspect : Aspects) {
      AspectsMetadata.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), aspect)));
    }

    MDNode *MDN = MDNode::get(C, AspectsMetadata);
    F->setMetadata("intel_used_aspects", MDN);
  }
}

void CheckAllDeclaredAspects(FunctionToAspectsMapTy &Map) {
  for (auto &it : Map) {
    Function *F = it.first;
    auto &Aspects = it.second;
    if (Aspects.empty())
      continue;

    LLVMContext &C = F->getContext();
    CheckDeclaredAspects(C, F, Aspects);
  }
}

void PropagateAspects(Function *F, CallGraphTy &CG,
                      FunctionToAspectsMapTy &AspectsMap,
                      SmallPtrSet<Function *, 16> &Visited) {
  SmallSet<int, 8> LocalAspects;
  for (Function *Callee : CG[F]) {
    if (Visited.contains(Callee)) {
      Visited.insert(Callee);
      PropagateAspects(Callee, CG, AspectsMap, Visited);
    }

    auto &CalleeAspects = AspectsMap[Callee];
    LocalAspects.insert(CalleeAspects.begin(), CalleeAspects.end());
  }

  AspectsMap[F].insert(LocalAspects.begin(), LocalAspects.end());
}

FunctionToAspectsMapTy
GetFunctionsAspects(Module &M, TypeToAspectsMapTy &TypesWithAspects) {
  FunctionToAspectsMapTy FunctionToAspects;
  CallGraphTy CG;
  std::vector<Function *> Kernels;
  TypeToAspectsMapTy Cache;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      Kernels.push_back(&F);
    }

    for (auto &I : instructions(F)) {
      SmallSet<int, 4> Aspects =
          GetAspectsUsedByInstruction(I, TypesWithAspects, Cache);
      FunctionToAspects[&F].insert(Aspects.begin(), Aspects.end());
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (!CI->isIndirectCall())
          CG[&F].insert(CI->getCalledFunction());
      }
    }
  }

  SmallPtrSet<Function *, 16> Visited;
  for (Function *F : Kernels) {
    PropagateAspects(F, CG, FunctionToAspects, Visited);
  }

  return FunctionToAspects;
}

} // anonymous namespace

PreservedAnalyses PropagateAspectUsagePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  TypeToAspectsMapTy TypesWithAspects = GetTypesThatUseAspectsFromMetadata(M);
  TypesWithAspects = GetTypesWithAspectsFromModule(M, TypesWithAspects);

  FunctionToAspectsMapTy FunctionToAspects =
      GetFunctionsAspects(M, TypesWithAspects);

  CreateAspectMetadataForFunctions(FunctionToAspects);
  CheckAllDeclaredAspects(FunctionToAspects);

  return PreservedAnalyses::all();
}
