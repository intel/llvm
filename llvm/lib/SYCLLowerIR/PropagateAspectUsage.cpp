#include "llvm/SYCLLowerIR/PropagateAspectUsage.h"

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
#include "llvm/Support/raw_ostream.h"

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
    return !PA.areAllPreserved(); // TODO: figure out
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

using AspectsTy = SmallSet<int, 4>;
using TypesWithAspectsTy = std::unordered_map<Type *, AspectsTy>;

TypesWithAspectsTy GetTypesWithAspectsFromMetadata(Module &M) {
  NamedMDNode *Node = M.getNamedMetadata("intel_types_that_use_aspects");
  TypesWithAspectsTy Result;
  if (!Node)
    return Result;

  LLVMContext &C = M.getContext();
  for (size_t i = 0; i != Node->getNumOperands(); ++i) {
    MDNode *N = Node->getOperand(i);
    assert(N->getNumOperands() > 1 && "intel_types_that_use_aspect metadata "
                                      "shouldn't contain empty metadata nodes");

    MDString *TypeName = cast<MDString>(N->getOperand(0));
    Type *T = StructType::getTypeByName(C, TypeName->getString());
    if (!T) {
      // TODO: warn?
      continue;
    }

    AspectsTy &Aspects = Result[T];
    for (size_t j = 1; j != N->getNumOperands(); ++j) {
      Constant *C = cast<ConstantAsMetadata>(N->getOperand(j))->getValue();
      Aspects.insert(cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

using TypesEdgesTy = std::unordered_map<Type *, std::vector<Type *>>;

void PropagateAspectsThroughTypes(TypesEdgesTy &Edges, Type *Start,
                                  AspectsTy &Aspects,
                                  TypesWithAspectsTy &ResultAspects) {
  std::unordered_set<Type *> Visited;
  std::queue<Type *> queue;
  queue.push(Start);
  while (!queue.empty()) {
    Type *T = queue.front();
    queue.pop();
    dbgs() << "Pop: Type*: " << T << ", Type: " << *T << "\n";
    if (Visited.count(T))
      continue;

    Visited.insert(T);
    ResultAspects[T].insert(Aspects.begin(), Aspects.end());
    for (Type *TT : Edges[T]) {
      if (Visited.count(TT))
        continue;

      dbgs() << "Push: Type*: " << TT << ", Type: " << *TT << "\n";
      queue.push(TT);
    }
  }
}

TypesWithAspectsTy
GetAllTypesWithAspects(std::unordered_set<Type *> &Types,
                       TypesWithAspectsTy &TypesWithAspects) {
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

  TypesWithAspectsTy Result;
  for (auto it : TypesWithAspects) {
    Type *T = it.first;
    AspectsTy Aspects;
    Aspects.insert(it.second.begin(), it.second.end());
    PropagateAspectsThroughTypes(Edges, T, Aspects, Result);
  }

  return Result;
}

AspectsTy GetAspectsFromType(Type *T, TypesWithAspectsTy &Types,
                             TypesWithAspectsTy &Cache) {
  dbgs() << "CheckType: " << *T << ", ptr: " << T << "\n";
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
  AspectsTy Result;

  dbgs() << "Check contained types. num: " << T->getNumContainedTypes() << "\n";
  for (size_t i = 0; i != T->getNumContainedTypes(); ++i) {
    Type *TT = T->getContainedType(i);
    dbgs() << "Check contained Type, index: " << i << " Type: " << *TT
           << ", ptr: " << TT << "\n";
    AspectsTy Aspects = GetAspectsFromType(TT, Types, Cache);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  Cache[T] = Result;
  return Result;
}

AspectsTy GetAspectsUsedByInstruction(Instruction &I, TypesWithAspectsTy &Types,
                                      TypesWithAspectsTy &Cache) {
  dbgs() << "GetAspectsUsedByInstruction: " << I << "\n";

  Type *ReturnType = I.getType();
  SmallSet<int, 4> Aspects = GetAspectsFromType(ReturnType, Types, Cache);
  SmallSet<int, 4> Result = Aspects;

  dbgs() << "Check Operands. NumOperands: " << I.getNumOperands() << "\n";
  int i = 0;
  for (auto OpIt = I.op_begin(); OpIt != I.op_end(); ++OpIt, ++i) {
    dbgs() << "Check Operand: " << i << ", : " << **OpIt << "\n";
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
  if (MDN == nullptr)
    return;

  SmallSet<int, 10> DeclaredAspects;
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

    C.diagnose(MissedAspectDiagnosticInfo(Msg));
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

void PrintTypes(std::unordered_map<Type *, SmallSet<int, 4>> &M) {
  dbgs() << "PrintTypes:\n";
  for (auto it : M) {
    dbgs() << "Type*: " << it.first << ", Type: " << *it.first
           << ", Aspects: " << Join(it.second, ',') << "\n";
  }
}

PreservedAnalyses PropagateAspectUsagePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  TypesWithAspectsTy TypesWithAspects = GetTypesWithAspectsFromMetadata(M);
  Type *DoubleTy = Type::getDoubleTy(M.getContext());
  TypesWithAspects[DoubleTy].insert(
      6); // TODO: use aspects::fp64 instead of constant?

  std::unordered_set<Type *> Types;
  Types.insert(DoubleTy);
  for (Type *T : M.getIdentifiedStructTypes()) {
    Types.insert(T);
  }

  TypesWithAspectsTy TypesWithAspects2 =
      GetAllTypesWithAspects(Types, TypesWithAspects);
  PrintTypes(TypesWithAspects2);

  FunctionToAspectsMapTy FunctionToAspectsMap;
  CallGraphTy CG;
  std::vector<Function *> Kernels;
  TypesWithAspectsTy Cache;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      Kernels.push_back(&F);
    }

    for (auto &I : instructions(F)) {
      SmallSet<int, 4> Aspects =
          GetAspectsUsedByInstruction(I, TypesWithAspects2, Cache);
      FunctionToAspectsMap[&F].insert(Aspects.begin(), Aspects.end());
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        if (!CI->isIndirectCall())
          CG[&F].insert(CI->getCalledFunction());
      }
    }
  }

  SmallPtrSet<Function *, 16> Visited;
  for (Function *F : Kernels) {
    PropagateAspects(F, CG, FunctionToAspectsMap, Visited);
  }

  CreateAspectMetadataForFunctions(FunctionToAspectsMap);
  CheckAllDeclaredAspects(FunctionToAspectsMap);

  return PreservedAnalyses::all();
}
