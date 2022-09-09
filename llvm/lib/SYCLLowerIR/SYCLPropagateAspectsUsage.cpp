//===---- SYCLPropagateAspectsUsage.cpp - SYCLPropagateAspectsUsage Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pass propagates optional kernel features metadata through a module call graph
//
// The pass consists of four main steps:
//
// I. It builds Type -> set of aspects mapping for usage in step II
// II. It builds Function -> set of aspects mapping to use in further steps
// III. FIXME: this step is not yet implemented
//      Analyzes aspects usage and emit warnings if necessary
// IV. Generates metadata with information about aspects used by each function
//
// Note: step I is not strictly necessary, because we can simply check if a
// function actually uses one or another type to say whether or not it uses any
// aspects. However, from customers point of view it could be more transparent
// that if a function is declared accepting an optional type, then it means that
// it uses an associated aspect, regardless of whether or not compiler was able
// to optimize out that variable.
//
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SYCLPropagateAspectsUsage.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Path.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

namespace {

using AspectsSetTy = SmallSet<int, 4>;
using TypeToAspectsMapTy = std::unordered_map<const Type *, AspectsSetTy>;

/// Retrieves from metadata (intel_types_that_use_aspects) types
/// and aspects these types depend on.
TypeToAspectsMapTy getTypesThatUseAspectsFromMetadata(const Module &M) {
  const NamedMDNode *Node = M.getNamedMetadata("intel_types_that_use_aspects");
  TypeToAspectsMapTy Result;
  if (!Node)
    return Result;

  LLVMContext &C = M.getContext();
  for (const auto OperandIt : Node->operands()) {
    const MDNode &N = *OperandIt;
    assert(N.getNumOperands() > 1 && "intel_types_that_use_aspect metadata "
                                     "shouldn't contain empty metadata nodes");

    const auto *TypeName = cast<MDString>(N.getOperand(0));
    const Type *T = StructType::getTypeByName(C, TypeName->getString());
    assert(T &&
           "invalid type referenced by intel_types_that_use_aspect metadata");

    AspectsSetTy &Aspects = Result[T];
    for (size_t I = 1; I != N.getNumOperands(); ++I) {
      const auto *CAM = cast<ConstantAsMetadata>(N.getOperand(I));
      const Constant *C = CAM->getValue();
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
    const auto It = Edges.find(T);
    if (It != Edges.end())
      TypesToPropagate.insert(It->second.begin(), It->second.end());
  }
}

/// Propagates given aspects to all types in module @M. Function accepts
/// aspects in @TypesWithAspects reference and writes a result in this
/// reference.
/// Type T in the result uses an aspect A if Type T is a composite
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
  const Type *DoubleTy = Type::getDoubleTy(M.getContext());

  // 6 is taken from sycl/include/CL/sycl/aspects.hpp
  // Note: that magic number must strictly correspond to the one assigned to
  // 'fp64' value of 'aspect' enum.
  // FIXME: we should develop some kind of mechanism which will allow us to
  // avoid hardcoding this number here and having a build dependency between
  // the compiler and the runtime. See intel/llvm#5892
  static constexpr int AspectFP64 = 6;
  TypesWithAspects[DoubleTy].insert(AspectFP64);

  TypesToProcess.insert(DoubleTy);
  for (const Type *T : M.getIdentifiedStructTypes())
    TypesToProcess.insert(T);

  TypesEdgesTy Edges;
  for (const Type *T : TypesToProcess) {
    for (const Type *TT : T->subtypes()) {
      if (TT->isPointerTy())
        // We don't know the pointee type in opaque pointers world
        continue;

      // If TT = [4 x [4 x [4 x %A]]] then we want to get TT = %A
      // The same with vectors
      while (TT->isArrayTy() || TT->isVectorTy()) {
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
  const auto It = Types.find(T);
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
  const Type *ReturnType = I.getType();
  AspectsSetTy Result = getAspectsFromType(ReturnType, Types);
  for (const auto &OperandIt : I.operands()) {
    const AspectsSetTy &Aspects =
        getAspectsFromType(OperandIt->getType(), Types);
    Result.insert(Aspects.begin(), Aspects.end());
  }

  return Result;
}

using FunctionToAspectsMapTy = DenseMap<Function *, AspectsSetTy>;
using CallGraphTy = DenseMap<Function *, SmallPtrSet<Function *, 8>>;

void createUsedAspectsMetadataForFunctions(FunctionToAspectsMapTy &Map) {
  for (auto &It : Map) {
    AspectsSetTy &Aspects = It.second;
    if (Aspects.empty())
      continue;

    Function *F = It.first;
    LLVMContext &C = F->getContext();

    SmallVector<Metadata *, 16> AspectsMetadata;
    for (const auto &A : Aspects)
      AspectsMetadata.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), A)));

    MDNode *MDN = MDNode::get(C, AspectsMetadata);
    F->setMetadata("intel_used_aspects", MDN);
  }
}

/// Propagates aspects from leaves up to the top of call graph.
/// NB! Call graph corresponds to call graph of SYCL code which
/// can't contain recursive calls. So there can't be loops in
/// a call graph. But there can be path's intersections.
void propagateAspectsThroughCG(Function *F, CallGraphTy &CG,
                               FunctionToAspectsMapTy &AspectsMap,
                               SmallPtrSet<const Function *, 16> &Visited) {
  const auto It = CG.find(F);
  if (It == CG.end())
    return;

  AspectsSetTy LocalAspects;
  for (Function *Callee : It->second) {
    if (Visited.insert(Callee).second)
      propagateAspectsThroughCG(Callee, CG, AspectsMap, Visited);

    const auto &CalleeAspects = AspectsMap[Callee];
    LocalAspects.insert(CalleeAspects.begin(), CalleeAspects.end());
  }

  AspectsMap[F].insert(LocalAspects.begin(), LocalAspects.end());
}

/// Processes a function:
///  - checks if return and argument types are using any aspects
///  - checks if instructions are using any aspects
///  - updates call graph information
///
void processFunction(Function &F, FunctionToAspectsMapTy &FunctionToAspects,
                     TypeToAspectsMapTy &TypesWithAspects, CallGraphTy &CG) {
  const AspectsSetTy RetTyAspects =
      getAspectsFromType(F.getReturnType(), TypesWithAspects);
  FunctionToAspects[&F].insert(RetTyAspects.begin(), RetTyAspects.end());
  for (Argument &Arg : F.args()) {
    const AspectsSetTy ArgAspects =
        getAspectsFromType(Arg.getType(), TypesWithAspects);
    FunctionToAspects[&F].insert(ArgAspects.begin(), ArgAspects.end());
  }

  for (Instruction &I : instructions(F)) {
    const AspectsSetTy Aspects =
        getAspectsUsedByInstruction(I, TypesWithAspects);
    FunctionToAspects[&F].insert(Aspects.begin(), Aspects.end());

    if (const auto *CI = dyn_cast<CallInst>(&I)) {
      if (!CI->isIndirectCall() && CI->getCalledFunction())
        CG[&F].insert(CI->getCalledFunction());
    }
  }
}

// Return true if the function is a SPIRV or SYCL builtin, e.g.
// _Z28__spirv_GlobalInvocationId_xv
// Note: this function was copied from sycl-post-link/ModuleSplitter.cpp and the
// definition of entry point (i.e. implementation of the function) should be in
// sync between those two.
bool isSpirvSyclBuiltin(StringRef FName) {
  if (!FName.consume_front("_Z"))
    return false;
  // now skip the digits
  FName = FName.drop_while([](char C) { return std::isdigit(C); });

  return FName.startswith("__spirv_") || FName.startswith("__sycl_");
}

bool isEntryPoint(const Function &F) {
  // Skip declarations, we can't analyze them
  if (F.isDeclaration())
    return false;

  // Kernels are always considered to be entry points
  if (CallingConv::SPIR_KERNEL == F.getCallingConv())
    return true;

  // FIXME: sycl-post-link allows to disable treating SYCL_EXTERNAL's as entry
  // points - do we need similar flag here?
  // SYCL_EXTERNAL functions with sycl-module-id attribute
  // are also considered as entry points (except __spirv_* and __sycl_*
  // functions)
  return F.hasFnAttribute("sycl-module-id") && !isSpirvSyclBuiltin(F.getName());
}

/// Returns a map of functions with corresponding used aspects.
FunctionToAspectsMapTy
buildFunctionsToAspectsMap(Module &M, TypeToAspectsMapTy &TypesWithAspects) {
  FunctionToAspectsMapTy FunctionToAspects;
  CallGraphTy CG;
  std::vector<Function *> EntryPoints;
  for (Function &F : M.functions()) {
    if (F.isDeclaration())
      continue;

    if (isEntryPoint(F))
      EntryPoints.push_back(&F);

    processFunction(F, FunctionToAspects, TypesWithAspects, CG);
  }

  SmallPtrSet<const Function *, 16> Visited;
  for (Function *F : EntryPoints)
    propagateAspectsThroughCG(F, CG, FunctionToAspects, Visited);

  return FunctionToAspects;
}

} // anonymous namespace

PreservedAnalyses
SYCLPropagateAspectsUsagePass::run(Module &M, ModuleAnalysisManager &MAM) {
  TypeToAspectsMapTy TypesWithAspects = getTypesThatUseAspectsFromMetadata(M);
  propagateAspectsToOtherTypesInModule(M, TypesWithAspects);

  FunctionToAspectsMapTy FunctionToAspects =
      buildFunctionsToAspectsMap(M, TypesWithAspects);

  createUsedAspectsMetadataForFunctions(FunctionToAspects);
  // FIXME: check and diagnose if a function uses an aspect which was not
  // declared through [[sycl::device_has()]] attribute

  return PreservedAnalyses::all();
}
