#include "llvm/SYCLLowerIR/PropagateAspectUsage.h"
#include "../IR/LLVMContextImpl.h"
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

using TypesMap = std::unordered_map<Type *, SmallVector<int>>;

TypesMap GetTypesWithAspectsFromMetadata(Module &M) {
  NamedMDNode *Node = M.getNamedMetadata("intel_types_that_use_aspects");
  TypesMap Result;
  if (!Node)
    return Result;

  LLVMContextImpl &CI = *M.getContext().pImpl;
  for (unsigned i = 0; i != Node->getNumOperands(); ++i) {
    MDNode *N = Node->getOperand(i);
    assert(N->getNumOperands() > 1 && "intel_types_that_use_aspect metadata shouldn't contain empty metadata nodes");

    MDString *TypeNameMD = cast<MDString>(N->getOperand(0));
    StringRef Name = TypeNameMD->getString();
    auto it2 = CI.NamedStructTypes.find(Name);
    if (it2 == CI.NamedStructTypes.end()) {
      continue; // Skip it because type is unknown in module M.
    }

    Type *T = it2->second;
    auto it = Result.emplace(T, SmallVector<int>()).first;
    for (unsigned j = 1; j != N->getNumOperands(); ++j) {
      Constant *C = cast<ConstantAsMetadata>(N->getOperand(j))->getValue();
      it->second.push_back(cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

TypesMap::const_iterator
CheckTypeContainsOptionalType(const Type *T, const TypesMap &Types) {
  dbgs() << "CheckType: " << *T << ", ptr: " << T << "\n";
  auto it = Types.find(const_cast<Type *>(T));
  if (it != Types.end())
    return it;

  dbgs() << "Check contained types. num: " << T->getNumContainedTypes() << "\n";
  for (size_t i = 0; i != T->getNumContainedTypes(); ++i) {
    Type *TT = T->getContainedType(i);
    it = Types.find(const_cast<Type *>(TT));
    dbgs() << "Check contained Type, index: " << i << " Type: " << *TT
           << ", ptr: " << TT << "\n";
    if (it != Types.end()) {
      return it;
    }
  }

  return Types.end();
}

TypesMap::const_iterator
CheckInstUsesOptionalDeviceFeature(const Instruction &I,
                                   const TypesMap &Types) {
  dbgs() << "CheckInstUsesOptionalDeviceFeature: " << I << "\n";
  const Type *ReturnType = I.getType();
  dbgs() << "Check ReturnType: " << *I.getType() << "\n";
  auto it = CheckTypeContainsOptionalType(ReturnType, Types);
  if (it != Types.end()) {
    return it;
  }

  dbgs() << "Return Type doesn't contain searched type\n";
  dbgs() << "Check Operands. NumOperands: " << I.getNumOperands() << "\n";
  std::string type;
  int i = 0;
  for (auto OpIt = I.op_begin(); OpIt != I.op_end(); ++OpIt, ++i) {
    dbgs() << "Check Operand: " << i << ", : " << **OpIt << "\n";
    it = CheckTypeContainsOptionalType((*OpIt)->getType(), Types);
    if (it != Types.end()) {
      dbgs() << "Searched type found: " << it->first << "\n";
      return it;
    }
  }

  dbgs() << "No type is found\n";
  return Types.end();
}

class Graph {
public:
  struct Node {
    Function *F = nullptr;
    std::vector<int> Edges;
    std::vector<int> ReversedEdges;
    SmallSet<int, 10> Aspects;

    Node(Function *F) : F(F) {}
  };

  void addEdge(Function *From, Function *To) {
    if (FunctionIndex.count(From) == 0 || FunctionIndex.count(To) == 0) {
      assert(false && "couldn't find entries for functions");
    }

    int IndexFrom = FunctionIndex[From];
    int IndexTo = FunctionIndex[To];
    nodes[IndexFrom].Edges.push_back(IndexTo);
    nodes[IndexTo].ReversedEdges.push_back(IndexFrom);
  }

  void createNodeForFunction(Function *F) {
    if (FunctionIndex.count(F)) {
      return;
    }

    FunctionIndex[F] = static_cast<int>(nodes.size());
    nodes.push_back(Graph::Node(F));
  }

  size_t getNumNodes() const {
    return nodes.size();
  }

  std::vector<Node> nodes;
  std::unordered_map<Function *, int> FunctionIndex;
};

void DFS(const Graph &G, int NodeIndex,
         std::vector<int> &Topsort) {
  for (int edge : G.nodes[NodeIndex].ReversedEdges) {
    DFS(G, edge, Topsort);
  }

  Topsort.push_back(NodeIndex);
}

std::vector<int> BuildTopsort(const Graph &G, const std::vector<Function *> &Kernels) {
  std::vector<int> Topsort;
  for (Function *F : Kernels) {
    int Index = G.FunctionIndex.at(F);
    DFS(G, Index, Topsort);
  }

  return Topsort;
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
    std::string Msg = formatv("for function \"{0}\" there is the list of missed aspects: [{1}]",
                              F->getName(), AspectsStr);

    C.diagnose(MissedAspectDiagnosticInfo(Msg));
  }
}

Graph BuildGraph(Module &M, const TypesMap &TypesWithAspects) {
  Graph G;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    G.createNodeForFunction(&F);
  }

  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    for (User *U : F.users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      G.addEdge(&F, CI->getFunction());
    }

    int NodeIndex = G.FunctionIndex[&F];
    for (auto &I : instructions(F)) {
      auto it = CheckInstUsesOptionalDeviceFeature(I, TypesWithAspects);
      if (it == TypesWithAspects.end()) {
        continue;
      }

      const auto &Aspects = it->second;
      G.nodes[NodeIndex].Aspects.insert(Aspects.begin(), Aspects.end());
      break;
    }
  }

  return G;
}

void PropagateAspectsInCallGraph(Graph &G, const std::vector<int> &Order) {
  for (int Index : Order) {
    Graph::Node &From = G.nodes[Index];
    for (int Edge : From.Edges) {
      Graph::Node &To = G.nodes[Edge];
      for (int aspect : From.Aspects) {
        To.Aspects.insert(aspect);
      }
    }
  }
}

void CreateAspectMetadataForFunctions(Graph &G) {
  for (Graph::Node &n : G.nodes) {
    if (n.Aspects.empty())
      continue;

    LLVMContext &C = n.F->getContext();
    std::vector<Metadata *> Aspects;
    Aspects.reserve(n.Aspects.size());
    for (int aspect : n.Aspects) {
      Aspects.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), aspect)));
    }

    MDNode *MDN = MDNode::get(C, Aspects);
    n.F->setMetadata("intel_used_aspects", MDN);

    CheckDeclaredAspects(C, n.F, n.Aspects);
  }
}

void CheckAllDeclaredAspects(Graph &G) {
  for (Graph::Node &n : G.nodes) {
    if (n.Aspects.empty())
      continue;

    LLVMContext &C = n.F->getContext();
    std::vector<Metadata *> Aspects;
    Aspects.reserve(n.Aspects.size());
    for (int aspect : n.Aspects) {
      Aspects.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), aspect)));
    }

    MDNode *MDN = MDNode::get(C, Aspects);
    n.F->setMetadata("intel_used_aspects", MDN);

    CheckDeclaredAspects(C, n.F, n.Aspects);
  }
}

PreservedAnalyses PropagateAspectUsagePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  TypesMap TypesWithAspects = GetTypesWithAspectsFromMetadata(M);
  LLVMContext &C = M.getContext();
  TypesWithAspects.insert({&C.pImpl->DoubleTy, SmallVector<int>{6}}); // TODO: use aspects::fp64 instead of constant?

  std::vector<Function *> SPIRKernels;
  for (Function &F : M.functions()) {
    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      SPIRKernels.push_back(&F);
    }
  }

  Graph G = BuildGraph(M, TypesWithAspects);
  auto Topsort = BuildTopsort(G, SPIRKernels);
  PropagateAspectsInCallGraph(G, Topsort);
  CreateAspectMetadataForFunctions(G);
  CheckAllDeclaredAspects(G);

  return PreservedAnalyses::all();
}
