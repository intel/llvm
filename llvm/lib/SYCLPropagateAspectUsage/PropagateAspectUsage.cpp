#include "llvm/SYCLPropagateAspectUsage/PropagateAspectUsage.h"
#include "../IR/LLVMContextImpl.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
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

using TypesMap = std::unordered_map<std::string, SmallVector<int>>;
using TypesMap2 = std::unordered_map<Type *, SmallVector<int>>;

TypesMap2 GetTypesWithAspectsFromMetadata(Module &M) {
  NamedMDNode *node = M.getNamedMetadata("intel_types_that_use_aspects");
  TypesMap2 Result;
  if (!node)
    return Result;

  LLVMContextImpl &CI = *M.getContext().pImpl;
  for (unsigned i = 0; i != node->getNumOperands(); ++i) {
    dbgs() << "index: " << i << "\n";
    MDNode *n = node->getOperand(i);
    unsigned numOperands = n->getNumOperands();
    assert(numOperands > 1);
    Metadata *mname = n->getOperand(0);
    MDString *sname = dyn_cast<MDString>(mname);
    if (sname == nullptr) {
      assert(false);
    }

    StringRef name = sname->getString();
    auto it2 = CI.NamedStructTypes.find(name);
    if (it2 == CI.NamedStructTypes.end()) {
      continue;
    }

    Type *t = it2->second;
    auto it = Result.insert({t, SmallVector<int>()}).first;
    for (unsigned j = 1; j != numOperands; ++j) {
      Constant *C = dyn_cast<ConstantAsMetadata>(n->getOperand(j))->getValue();
      if (!C) {
        continue;
      }

      it->second.push_back(dyn_cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

TypesMap2::const_iterator
CheckTypeContainsOptionalType(const Type *T, const TypesMap2 &Types) {
  dbgs() << "CheckType: " << *T << ", ptr: " << T << "\n";
  auto it = Types.find(const_cast<Type *>(T));
  if (it != Types.end())
    return it;

  dbgs() << "Check contained types. num: " << T->getNumContainedTypes() << "\n";
  for (int i = 0; i != T->getNumContainedTypes(); ++i) {
    Type *TT = T->getContainedType(i);
    it = Types.find(const_cast<Type *>(TT));
    dbgs() << "Check contained Type, index: " << i << " Type: " << *TT
           << ", ptr: " << TT << "\n";
    if (it != Types.end()) {
      dbgs() << "Success\n";
      return it;
    }
    dbgs() << "Fail\n";
  }

  return Types.end();
}

TypesMap2::const_iterator
CheckInstUsesOptionalDeviceFeature(const Instruction &I,
                                   const TypesMap2 &Types) {
  dbgs() << "CheckInstUsesOptionalDeviceFeature: " << I << "\n";
  const Type *ReturnType = I.getType();
  dbgs() << "Check ReturnType: " << *I.getType() << "\n";
  auto it = CheckTypeContainsOptionalType(ReturnType, Types);
  if (it != Types.end()) {
    dbgs() << "Exit\n";
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
    uint32_t count = 0;
    std::vector<int> edges;
    std::vector<int> reversedEdges;
    SmallSet<int, 10> aspects;

    Node(Function &F) : F(&F) {}
  };

  void addEdge(Function *from, Function *to) {
    if (functionIndex.count(from) == 0 || functionIndex.count(to) == 0) {
      assert(false && "couldn't find entries for functions");
    }

    int indexFrom = functionIndex[from];
    int indexTo = functionIndex[to];
    nodes[indexFrom].edges.push_back(indexTo);
    nodes[indexTo].reversedEdges.push_back(indexFrom);
  }

  std::vector<Node> nodes;
  std::unordered_map<Function *, int> functionIndex;
};

void dfs2(const Graph &G, int NodeIndex, std::vector<int8_t> &Visited,
          std::vector<int> &Topsort) {
  if (Visited[NodeIndex])
    return;

  Visited[NodeIndex] = 1;
  for (int edge : G.nodes[NodeIndex].reversedEdges) {
    dfs2(G, edge, Visited, Topsort);
  }

  Topsort.push_back(NodeIndex);
}

std::vector<int> dfs(const Graph &G, const std::vector<Function *> Kernels) {
  std::vector<int> topsort;
  std::vector<int8_t> visited(G.nodes.size(), 0);
  for (Function *F : Kernels) {
    int index = G.functionIndex.at(F);
    dfs2(G, index, visited, topsort);
  }

  return topsort;
}

template <typename T> void PrintMap(const T &m) {
  dbgs() << "Map: size: " << m.size() << "\n";
  for (const auto it : m) {
    dbgs() << "key: " << it.first << "\n";
  }
}

PreservedAnalyses PropagateAspectUsagePass::run(Module &M,
                                                ModuleAnalysisManager &MAM) {
  auto TypesWithAspects = GetTypesWithAspectsFromMetadata(M);
  LLVMContext &C = M.getContext();
  TypesWithAspects.insert({&C.pImpl->DoubleTy, SmallVector<int>{6}});
  PrintMap(TypesWithAspects);

  Graph g;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    g.nodes.push_back(Graph::Node(F));
    g.functionIndex[&F] = g.nodes.size() - 1;
  }

  std::vector<Function *> SPIRKernels;
  for (Function &F : M.functions()) {
    auto CC = F.getCallingConv();
    if (CC != CallingConv::SPIR_FUNC && CC != CallingConv::SPIR_KERNEL)
      continue;

    if (F.getCallingConv() == CallingConv::SPIR_KERNEL) {
      SPIRKernels.push_back(&F);
    }

    int nodeIndex = g.functionIndex[&F];
    for (User *U : F.users()) {
      CallInst *CI = dyn_cast<CallInst>(U);
      if (!CI)
        continue;

      g.addEdge(&F, CI->getFunction());
    }

    for (const auto &I : instructions(F)) {
      auto it = CheckInstUsesOptionalDeviceFeature(I, TypesWithAspects);
      if (it == TypesWithAspects.end()) {
        continue;
      }

      const auto &aspects = it->second;
      g.nodes[nodeIndex].aspects.insert(aspects.begin(), aspects.end());
      break;
    }
  }

  auto topsort = dfs(g, SPIRKernels);
  for (int index : topsort) {
    auto &node = g.nodes[index];
    for (int edge : node.edges) {
      Graph::Node &n2 = g.nodes[edge];
      for (int aspect : node.aspects) {
        n2.aspects.insert(aspect);
      }
    }
  }

  for (Graph::Node &n : g.nodes) {
    if (n.aspects.empty())
      continue;

    LLVMContext &C = n.F->getContext();
    std::vector<Metadata *> aspects;
    aspects.reserve(n.aspects.size());
    for (int aspect : n.aspects) {
      aspects.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), aspect)));
    }

    MDNode *MDN = MDNode::get(C, aspects);
    n.F->setMetadata("intel_used_aspects", MDN);
  }

  return PreservedAnalyses::all();
}
