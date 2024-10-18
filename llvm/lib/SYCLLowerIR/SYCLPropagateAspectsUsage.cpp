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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

static cl::opt<std::string> ClSyclFixedTargets(
    "sycl-propagate-aspects-usage-fixed-targets",
    cl::desc("Specify target device(s) all device code in the translation unit "
             "is expected to be runnable on"),
    cl::Hidden, cl::init(""));

static cl::opt<std::string> ClSyclExcludeAspects(
    "sycl-propagate-aspects-usage-exclude-aspects",
    cl::desc("Specify aspects to exclude when propagating aspect usage"),
    cl::Hidden, cl::init(""));

namespace {

using AspectsSetTy = SmallSet<int, 4>;
using TypeToAspectsMapTy = std::unordered_map<const Type *, AspectsSetTy>;

/// Retrieves from metadata (sycl_types_that_use_aspects) types
/// and aspects these types depend on.
TypeToAspectsMapTy getTypesThatUseAspectsFromMetadata(const Module &M) {
  const NamedMDNode *Node = M.getNamedMetadata("sycl_types_that_use_aspects");
  TypeToAspectsMapTy Result;
  if (!Node)
    return Result;

  LLVMContext &C = M.getContext();
  for (const MDNode *N : Node->operands()) {
    assert(N->getNumOperands() > 1 && "intel_types_that_use_aspect metadata "
                                      "shouldn't contain empty metadata nodes");

    const auto *TypeName = cast<MDString>(N->getOperand(0));
    const Type *T = StructType::getTypeByName(C, TypeName->getString());
    assert(T &&
           "invalid type referenced by intel_types_that_use_aspect metadata");

    AspectsSetTy &Aspects = Result[T];
    for (const MDOperand &Op : drop_begin(N->operands())) {
      const Constant *C = cast<ConstantAsMetadata>(Op)->getValue();
      Aspects.insert(cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

using AspectValueToNameMapTy = SmallMapVector<StringRef, int, 32>;

/// Retrieves from metadata (sycl_aspects) the mapping between SYCL aspect names
/// and their integral values.
AspectValueToNameMapTy getAspectsFromMetadata(const Module &M) {
  const NamedMDNode *Node = M.getNamedMetadata("sycl_aspects");
  AspectValueToNameMapTy Result;
  if (!Node)
    return Result;

  for (const MDNode *N : Node->operands()) {
    assert(N->getNumOperands() == 2 &&
           "Each operand of sycl_aspects must be a pair.");

    // The aspect's name is the first operand.
    const auto *AspectName = cast<MDString>(N->getOperand(0));

    // The aspect's integral value is the second operand.
    const auto *AspectCAM = cast<ConstantAsMetadata>(N->getOperand(1));
    const Constant *AspectC = AspectCAM->getValue();

    Result[AspectName->getString()] =
        cast<ConstantInt>(AspectC)->getSExtValue();
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
  // The TypesToPropagate is being updated inside the loop, so no range-for.
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
/// @AspectValues argument consist of known aspect values and their names.
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
    const Module &M, TypeToAspectsMapTy &TypesWithAspects,
    AspectValueToNameMapTy &AspectValues) {
  std::unordered_set<const Type *> TypesToProcess;
  const Type *DoubleTy = Type::getDoubleTy(M.getContext());

  // Find the value of the fp64 aspect from the aspect values map and register
  // it as a special-case type with aspect for double.
  auto FP64AspectIt = AspectValues.find("fp64");
  assert(FP64AspectIt != AspectValues.end() &&
         "fp64 aspect was not found in the aspect values.");
  TypesWithAspects[DoubleTy].insert(FP64AspectIt->second);

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

/// Utility function to identify FP64 conversion instruction
bool isFP64ConversionInstruction(const Instruction &I) {
  switch (I.getOpcode()) {
  default:
    return false;
  case Instruction::Alloca:
  case Instruction::GetElementPtr:
  case Instruction::Load:
  case Instruction::Store:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
  case Instruction::SIToFP:
  case Instruction::UIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::FCmp:
  case Instruction::PHI:
  case Instruction::ExtractValue:
  case Instruction::InsertValue:
  case Instruction::Ret:
    return true;
  // In case of call instructions, we check if the definition of the called
  // function is present inside the current module. If present, we conclude
  // that the call instruction does not require FP64 computations and return
  // 'true'. Otherwise, we return 'false'.
  // In case of memory intrinsics, FP64 computations are not required and we
  // return 'true'.
  // TODO: Identify other cases similar to memory intrinsics.
  // TODO: Try to handle FP64 usage in call instructions in sycl-post-link.
  case Instruction::Call:
    if (cast<CallInst>(&I)->getCalledFunction()->isDeclaration())
      return isa<MemIntrinsic>(&I);
    else
      return true;
  }
}

/// Returns 'true' if Type has double type
bool hasDoubleType(const Type *T) {
  if (T->isDoubleTy())
    return true;
  for (const Type *TT : T->subtypes())
    if (hasDoubleType(TT))
      return true;
  return false;
}

/// Returns 'true' if Instruction has double type
bool hasDoubleType(const Instruction &I) {
  const Type *ReturnType = I.getType();
  if (auto *AI = dyn_cast<const AllocaInst>(&I)) {
    // Return type of an alloca is a pointer and in opaque pointers world we
    // don't know which type it points to. Therefore, explicitly checking the
    // allocated type instead
    ReturnType = AI->getAllocatedType();
  }
  if (hasDoubleType(ReturnType))
    return true;
  for (const auto &OperandIt : I.operands()) {
    if (const auto *GV =
            dyn_cast<const GlobalValue>(OperandIt->stripPointerCasts())) {
      if (hasDoubleType(GV->getValueType()))
        return true;
    } else {
      if (hasDoubleType(OperandIt->getType()))
        return true;
    }
  }

  // Opaque pointer arguments may hide types of pointer arguments until elements
  // inside the types are accessed through a GEP instruction. However, this will
  // not be caught by the operands check above, so we must extract the
  // information directly from the GEP.
  if (auto *GEPI = dyn_cast<const GetElementPtrInst>(&I))
    if (hasDoubleType(GEPI->getSourceElementType()))
      return true;

  return false;
}

/// Returns aspects which might be used in instruction @I.
/// Function inspects return type and all operand's types.
/// NB! This function inserts new records in @Types map for new discovered
/// types. For the best perfomance pass this map in the next invocations.
AspectsSetTy getAspectsUsedByInstruction(const Instruction &I,
                                         TypeToAspectsMapTy &Types) {
  const Type *ReturnType = I.getType();
  if (auto *AI = dyn_cast<const AllocaInst>(&I)) {
    // Return type of an alloca is a pointer and in opaque pointers world we
    // don't know which type it points to. Therefore, explicitly checking the
    // allocated type instead
    ReturnType = AI->getAllocatedType();
  }
  AspectsSetTy Result = getAspectsFromType(ReturnType, Types);
  auto AddAspectsFromType = [&](Type *Ty) {
    const AspectsSetTy &Aspects = getAspectsFromType(Ty, Types);
    Result.insert(Aspects.begin(), Aspects.end());
  };
  for (const auto &OperandIt : I.operands()) {
    if (const auto *GV =
            dyn_cast<const GlobalValue>(OperandIt->stripPointerCasts()))
      AddAspectsFromType(GV->getValueType());
    else
      AddAspectsFromType(OperandIt->getType());
  }

  // Opaque pointer arguments may hide types of pointer arguments until elements
  // inside the types are accessed through a GEP instruction. However, this will
  // not be caught by the operands check above, so we must extract the
  // information directly from the GEP.
  if (auto *GEPI = dyn_cast<const GetElementPtrInst>(&I))
    AddAspectsFromType(GEPI->getSourceElementType());

  if (const MDNode *InstApsects = I.getMetadata("sycl_used_aspects")) {
    for (const MDOperand &MDOp : InstApsects->operands()) {
      const Constant *C = cast<ConstantAsMetadata>(MDOp)->getValue();
      Result.insert(cast<ConstantInt>(C)->getSExtValue());
    }
  }

  return Result;
}

using FunctionToAspectsMapTy = DenseMap<Function *, AspectsSetTy>;
using ConditionsSetTy = SmallSet<int, 4>;
using FunctionToConditionExpressionsMapTy = DenseMap<Function *, ConditionsSetTy>;
using ConditionExpressionsAndConditionalAspectsTy = SmallVector<std::pair<ConditionsSetTy, AspectsSetTy>>;
using FunctionToConditionExpressionsAndConditionalAspectsTy = DenseMap<Function *, ConditionExpressionsAndConditionalAspectsTy>;
using FunctionSetTy = SmallPtrSet<Function *, 8>;
using CallGraphTy = DenseMap<Function *, FunctionSetTy>;

// Finds the first function in a list that uses a given aspect. Returns nullptr
// if none of the functions satisfy the criteria.
Function *findFirstAspectUsageCallee(
    const SmallPtrSetImpl<Function *> &Callees,
    const FunctionToAspectsMapTy &AspectsMap, int Aspect,
    SmallPtrSetImpl<const Function *> *Visited = nullptr) {
  for (Function *Callee : Callees) {
    if (Visited && !Visited->insert(Callee).second)
      continue;

    auto AspectIt = AspectsMap.find(Callee);
    if (AspectIt != AspectsMap.end() && AspectIt->second.contains(Aspect))
      return Callee;
  }
  return nullptr;
}

// Constructs an aspect usage chain for a given aspect from the function to the
// last callee in the first found chain.
void constructAspectUsageChain(const Function *F,
                               const FunctionToAspectsMapTy &AspectsMap,
                               const CallGraphTy &CG, int Aspect,
                               SmallVectorImpl<Function *> &CallChain,
                               SmallPtrSetImpl<const Function *> &Visited) {
  const auto EdgeIt = CG.find(F);
  if (EdgeIt == CG.end())
    return;

  if (Function *AspectUsingCallee = findFirstAspectUsageCallee(
          EdgeIt->second, AspectsMap, Aspect, &Visited)) {
    CallChain.push_back(AspectUsingCallee);
    constructAspectUsageChain(AspectUsingCallee, AspectsMap, CG, Aspect,
                              CallChain, Visited);
  }
}

// Simplified function for getting the call chain of a given function. See
// constructAspectUsageChain.
SmallVector<Function *, 8>
getAspectUsageChain(const Function *F, const FunctionToAspectsMapTy &AspectsMap,
                    const CallGraphTy &CG, int Aspect) {
  SmallVector<Function *, 8> CallChain;
  SmallPtrSet<const Function *, 16> Visited;
  constructAspectUsageChain(F, AspectsMap, CG, Aspect, CallChain, Visited);
  return CallChain;
}

void createUsedAspectsMetadataForFunctions(
    FunctionToAspectsMapTy &FunctionToUsedAspects,
    FunctionToAspectsMapTy &FunctionToDeclaredAspects,
    const AspectsSetTy &ExcludeAspectVals) {
  for (auto &[F, Aspects] : FunctionToUsedAspects) {
    LLVMContext &C = F->getContext();

    // Create a set of unique aspects. First we add the ones from the found
    // aspects that have not been excluded.
    AspectsSetTy UniqueAspects;
    for (const int &A : Aspects)
      if (!ExcludeAspectVals.contains(A))
        UniqueAspects.insert(A);

    // The aspects that were propagated via declared aspects are always
    // added to the metadata.
    for (const int &A : FunctionToDeclaredAspects[F])
      UniqueAspects.insert(A);

    // If there are no new aspects, we can just keep the old metadata.
    if (UniqueAspects.empty())
      continue;

    // If there is new metadata, merge it with the old aspects. We preserve
    // the excluded ones.
    if (const MDNode *ExistingAspects = F->getMetadata("sycl_used_aspects")) {
      for (const MDOperand &MDOp : ExistingAspects->operands()) {
        const Constant *C = cast<ConstantAsMetadata>(MDOp)->getValue();
        UniqueAspects.insert(cast<ConstantInt>(C)->getSExtValue());
      }
    }

    // Create new metadata.
    SmallVector<Metadata *, 16> AspectsMetadata;
    for (const int &A : UniqueAspects)
      AspectsMetadata.push_back(ConstantAsMetadata::get(
          ConstantInt::getSigned(Type::getInt32Ty(C), A)));

    MDNode *MDN = MDNode::get(C, AspectsMetadata);
    F->setMetadata("sycl_used_aspects", MDN);
  }
}

void createConditionallyUsedAspectsMetadataForFunctions(
    FunctionToConditionExpressionsAndConditionalAspectsTy
        &FunctionToConditionExpressionsAndConditionalAspects,
    const AspectsSetTy &ExcludeAspectVals) {
  for (auto &[F, ConditionsAndAspectsVec] :
       FunctionToConditionExpressionsAndConditionalAspects) {
    LLVMContext &C = F->getContext();

    // Create a set of unique conditions and aspects. First we add the ones from
    // the found aspects that have not been excluded.
    ConditionExpressionsAndConditionalAspectsTy UniqueConditionsAndAspects;
    for (const auto &ConditionsAndAspectsPair : ConditionsAndAspectsVec) {
      AspectsSetTy FilteredAspects;
      for (const auto &A : ConditionsAndAspectsPair.second)
        if (!ExcludeAspectVals.contains(A)) {
          FilteredAspects.insert(A);
        }

      if (FilteredAspects.empty())
        continue;

      UniqueConditionsAndAspects.push_back(
          std::make_pair(ConditionsAndAspectsPair.first, FilteredAspects));
    }

    // If there is new metadata, merge it with the old aspects. We preserve
    // the excluded ones.

    if (const MDNode *ExistingAspects =
            F->getMetadata("sycl_conditionally_used_aspects")) {
      for (const MDOperand &MDOp : ExistingAspects->operands()) {
        if (MDNode *PairNode = dyn_cast<MDNode>(MDOp)) {
          if (PairNode->getNumOperands() != 2)
            continue;

          MDNode *ConditionNode = dyn_cast<MDNode>(PairNode->getOperand(0));
          ConditionsSetTy ConditionsSet;
          if (ConditionNode) {
            for (const MDOperand &CondOp : ConditionNode->operands()) {
              const Constant *C = cast<ConstantAsMetadata>(CondOp)->getValue();
              ConditionsSet.insert(cast<ConstantInt>(C)->getSExtValue());
            }
          }

          MDNode *AspectsNode = dyn_cast<MDNode>(PairNode->getOperand(1));
          AspectsSetTy AspectsSet;
          if (AspectsNode) {
            for (const MDOperand &AspOp : AspectsNode->operands()) {
              const Constant *C = cast<ConstantAsMetadata>(AspOp)->getValue();
              AspectsSet.insert(cast<ConstantInt>(C)->getSExtValue());
            }
          }
          UniqueConditionsAndAspects.push_back(
              std::make_pair(ConditionsSet, AspectsSet));
        }
      }
    }

    // Create new metadata.
    SmallVector<Metadata *, 16> AspectsMetadata;

    for (const auto &Pair : ConditionsAndAspectsVec) {
      // Metadata node for conditions
      SmallVector<Metadata *, 4> ConditionsMD;
      for (int Condition : Pair.first) {
        ConditionsMD.push_back(ConstantAsMetadata::get(
            ConstantInt::getSigned(Type::getInt32Ty(C), Condition)));
      }
      MDNode *ConditionNode = MDNode::get(C, ConditionsMD);

      // Metadata node for aspects
      SmallVector<Metadata *, 4> AspectsMD;
      for (int Aspect : Pair.second) {
        AspectsMD.push_back(ConstantAsMetadata::get(
            ConstantInt::getSigned(Type::getInt32Ty(C), Aspect)));
      }
      MDNode *AspectsNode = MDNode::get(C, AspectsMD);

      // Create the pair metadata node that holds the condition and aspects
      // nodes
      AspectsMetadata.push_back(MDNode::get(C, {ConditionNode, AspectsNode}));
    }

    MDNode *MDN = MDNode::get(C, AspectsMetadata);
    F->setMetadata("sycl_conditionally_used_aspects", MDN);
  }
}

/// Checks that all aspects determined to be used by a given function are in
/// that function's sycl_declared_aspects metadata if present. A warning
/// diagnostic is produced for each aspect this check fails for.
void validateUsedAspectsForFunctions(const FunctionToAspectsMapTy &Map,
                                     const AspectValueToNameMapTy &AspectValues,
                                     const std::vector<Function *> &EntryPoints,
                                     const CallGraphTy &CG) {
  for (auto &It : Map) {
    const AspectsSetTy &Aspects = It.second;
    if (Aspects.empty())
      continue;

    Function *F = It.first;
    AspectsSetTy DeviceHasAspectSet;
    bool OriginatedFromAttribute = true;
    if (const MDNode *DeviceHasMD = F->getMetadata("sycl_declared_aspects")) {
      // Entry points will have their declared aspects from their kernel call.
      // To avoid double warnings, we skip them.
      if (is_contained(EntryPoints, F))
        continue;
      for (const MDOperand &DeviceHasMDOp : DeviceHasMD->operands()) {
        const auto *CAM = cast<ConstantAsMetadata>(DeviceHasMDOp);
        const Constant *C = CAM->getValue();
        DeviceHasAspectSet.insert(cast<ConstantInt>(C)->getSExtValue());
      }
      OriginatedFromAttribute = true;
    } else if (F->hasFnAttribute("sycl-device-has")) {
      Attribute DeviceHasAttr = F->getFnAttribute("sycl-device-has");
      SmallVector<StringRef, 4> AspectValStrs;
      DeviceHasAttr.getValueAsString().split(
          AspectValStrs, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (StringRef AspectValStr : AspectValStrs) {
        int AspectVal = -1;
        bool AspectValStrConverted = !AspectValStr.getAsInteger(10, AspectVal);
        // Avoid unused warning when asserts are disabled.
        std::ignore = AspectValStrConverted;
        assert(AspectValStrConverted &&
               "Aspect value in sycl-device-has is not an integer.");
        DeviceHasAspectSet.insert(AspectVal);
      }
      OriginatedFromAttribute = false;
    } else {
      continue;
    }

    for (int Aspect : Aspects) {
      if (!DeviceHasAspectSet.contains(Aspect)) {
        auto AspectNameIt = std::find_if(
            AspectValues.begin(), AspectValues.end(),
            [=](auto AspectIt) { return Aspect == AspectIt.second; });
        assert(AspectNameIt != AspectValues.end() &&
               "Used aspect is not part of the existing aspects");
        // We may encounter an entry point when using the device_has property.
        // In this case we act like the usage came from the first callee to
        // avoid repeat warnings on the same line.
        Function *AdjustedOriginF =
            is_contained(EntryPoints, F)
                ? findFirstAspectUsageCallee(CG.find(F)->second, Map, Aspect)
                : F;
        assert(AdjustedOriginF &&
               "Adjusted function pointer for aspect usage is null");
        SmallVector<Function *, 8> CallChain =
            getAspectUsageChain(AdjustedOriginF, Map, CG, Aspect);
        diagnoseAspectsMismatch(AdjustedOriginF, CallChain, AspectNameIt->first,
                                OriginatedFromAttribute);
      }
    }
  }
}

struct FunctionsClassifiedByTheirFunctionCall {
  FunctionSetTy CalledUnconditionally;
  FunctionSetTy CalledConditionally;
};

void addAllCalledFunctionsRecursivelyToFunctionsClassifiedByTheirFunctionCallStruct(
    Function *F, const CallGraphTy &CG,
    FunctionsClassifiedByTheirFunctionCall &Funcs,
    SmallPtrSet<const Function *, 16> &Visited) {
  const auto It = CG.find(F);
  if (It == CG.end())
    return;
  for (Function *Callee : It->second) {
    if (!Callee->hasFnAttribute("sycl-call-if-on-device-conditionally") &&
        Funcs.CalledUnconditionally.contains(F)) {
      Funcs.CalledUnconditionally.insert(Callee);
    }
    if (Callee->hasFnAttribute("sycl-call-if-on-device-conditionally") ||
        Funcs.CalledConditionally.contains(F)) {
      Funcs.CalledConditionally.insert(Callee);
    }
    if (Visited.insert(Callee).second)
      addAllCalledFunctionsRecursivelyToFunctionsClassifiedByTheirFunctionCallStruct(
          Callee, CG, Funcs, Visited);
  }
}

/// Propagates aspects from leaves up to the top of call graph.
/// NB! Call graph corresponds to call graph of SYCL code which
/// can't contain recursive calls. So there can't be loops in
/// a call graph. But there can be path's intersections.
void propagateAspectsThroughCG(Function *F, const CallGraphTy &CG,
                               FunctionToAspectsMapTy &AspectsMap,
                               FunctionSetTy FunctionsCalledUnconditionally,
                               SmallPtrSet<const Function *, 16> &Visited) {
  const auto It = CG.find(F);
  if (It == CG.end())
    return;

  AspectsSetTy LocalAspects;
  for (Function *Callee : It->second) {
    if (Visited.insert(Callee).second)
      propagateAspectsThroughCG(Callee, CG, AspectsMap,
                                FunctionsCalledUnconditionally, Visited);

    const auto &CalleeAspects = AspectsMap[Callee];
    LocalAspects.insert(CalleeAspects.begin(), CalleeAspects.end());
  }
  if (FunctionsCalledUnconditionally.contains(F))
    AspectsMap[F].insert(LocalAspects.begin(), LocalAspects.end());
}

using PathsContainingConditionalCallersTy =
    std::vector<std::vector<Function *>>;

void identifyPathsContainingConditionalCallers(
    Function *F, const CallGraphTy &CG, std::vector<Function *> &CurrentPath,
    PathsContainingConditionalCallersTy &Paths) {
  const auto It = CG.find(F);
  if (It == CG.end()) {
    CurrentPath.push_back(F);
    for (auto Func : CurrentPath)
      if (Func->hasFnAttribute("sycl-call-if-on-device-conditionally")) {
        Paths.push_back(CurrentPath);
        break;
      }
    CurrentPath.pop_back();
    return;
  }

  CurrentPath.push_back(F);

  for (Function *Callee : It->second) {
    identifyPathsContainingConditionalCallers(Callee, CG, CurrentPath, Paths);
  }

  CurrentPath.pop_back();
}

void propagateConditionExpressionsAndConditionalAspectsThroughCG(
    const PathsContainingConditionalCallersTy &Paths,
    FunctionToConditionExpressionsAndConditionalAspectsTy
        &ConditionsAndAspectsMap,
    FunctionToConditionExpressionsMapTy &ConditionsMap,
    FunctionToAspectsMapTy &AspectsMap) {
  auto Contains =
      [](ConditionExpressionsAndConditionalAspectsTy ConditionsAndAspectsVec,
         std::pair<ConditionsSetTy, AspectsSetTy> Search) {
        for (const auto &ConditionAndAspectsPair : ConditionsAndAspectsVec)
          if ((ConditionAndAspectsPair.first == Search.first) &&
              (ConditionAndAspectsPair.second == Search.second))
            return true;
        return false;
      };

  // TODO: need to optimize finding duplicates and combining pairs with the same
  // Condition Expressions or Aspects
  for (const auto &Path : Paths)
    for (int I = Path.size() - 1; I >= 0; --I) {
      if (I != Path.size() - 1) {
        const auto &CalleeAspects = AspectsMap[Path[I + 1]];
        AspectsMap[Path[I]].insert(CalleeAspects.begin(), CalleeAspects.end());
        if (ConditionsMap[Path[I]].empty())
          ConditionsMap[Path[I]] = ConditionsMap[Path[I + 1]];
      }
      auto NewPair =
          std::make_pair(ConditionsMap[Path[I]], AspectsMap[Path[I]]);
      if ((!NewPair.first.empty()) && (!NewPair.second.empty())) {
        if (ConditionsAndAspectsMap[Path[I]].empty())
          ConditionsAndAspectsMap[Path[I]].push_back(NewPair);
        else
          if (!Contains(ConditionsAndAspectsMap[Path[I]], NewPair))
              ConditionsAndAspectsMap[Path[I]].push_back(NewPair);
      }
      if (I != Path.size() - 1)
        for (const auto &ConditionAndAspectsPair : ConditionsAndAspectsMap[Path[I + 1]])
          if (!Contains(ConditionsAndAspectsMap[Path[I]], ConditionAndAspectsPair))
              ConditionsAndAspectsMap[Path[I]].push_back(ConditionAndAspectsPair);
      }
}

void propagateConditionExpressionsThroughCG(
    Function *F, const CallGraphTy &CG,
    FunctionToConditionExpressionsMapTy &ConditionsMap) {
  // TODO: re-write using Paths from identifyPathsContainingConditionalCallers func
  const auto It = CG.find(F);
  if (It == CG.end())
    return;

  FunctionToConditionExpressionsMapTy LocalConditions;
  for (Instruction &I : instructions(F)) {
    if (const auto *CI = dyn_cast<CallInst>(&I)) {
      if (Function *CalledFunction = CI->getCalledFunction();
          !CI->isIndirectCall() && CalledFunction) {
        if (CalledFunction->hasFnAttribute(
                "sycl-call-if-on-device-conditionally")) {
          // Start the loop with the 2nd argument (counting from 0) as 0 arg by
          // design doc is Conditional Action, 1st arg is "this" pointer for the
          // application's callable object, and all Condition Expressions start
          // with 2nd argument
          for (unsigned J = 2; J < CI->arg_size(); ++J) {
            Value *Arg = CI->getArgOperand(J);
            if (auto *ConstInt = dyn_cast<ConstantInt>(Arg)) {
              // Get the integer value and add it to the set
              LocalConditions[CalledFunction].insert(
                  static_cast<int>(ConstInt->getSExtValue()));
            }
          }
        }
      }
    }
  }

  for (Function *Callee : It->second) {
    auto NewConditions = ConditionsMap[F];
    if (Callee->hasFnAttribute("sycl-call-if-on-device-conditionally")) {
      NewConditions.insert(LocalConditions[Callee].begin(),
                           LocalConditions[Callee].end());
    }
    ConditionsMap[Callee] = NewConditions;
    propagateConditionExpressionsThroughCG(Callee, CG, ConditionsMap);
  }
}

/// Processes a function:
///  - checks if return and argument types are using any aspects
///  - checks if instructions are using any aspects
///  - updates call graph information
///  - checks if function has "!sycl_used_aspects",
///    "!sycl_conditionally_used_aspects" and "!sycl_declared_aspects"
///    metadata and if so collects aspects from this metadata
void processFunction(Function &F, FunctionToAspectsMapTy &FunctionToUsedAspects,
                     FunctionToAspectsMapTy &FunctionToConditionallyUsedAspects,
                     FunctionToAspectsMapTy &FunctionToDeclaredAspects,
                     const FunctionsClassifiedByTheirFunctionCall &Funcs,
                     TypeToAspectsMapTy &TypesWithAspects,
                     const AspectValueToNameMapTy &AspectValues,
                     bool FP64ConvEmu) {
  bool FunctionIsUnconditional = Funcs.CalledUnconditionally.contains(&F);
  bool FunctionIsConditional = Funcs.CalledConditionally.contains(&F);
  auto FP64AspectIt = AspectValues.find("fp64");
  assert(FP64AspectIt != AspectValues.end() &&
         "fp64 aspect was not found in the aspect values.");
  auto FP64Aspect = FP64AspectIt->second;
  const AspectsSetTy RetTyAspects =
      getAspectsFromType(F.getReturnType(), TypesWithAspects);
  for (const auto &Aspect : RetTyAspects)
    if (!FP64ConvEmu || (Aspect != FP64Aspect) ||
        !hasDoubleType(F.getReturnType())) {
      if (FunctionIsUnconditional)
        FunctionToUsedAspects[&F].insert(Aspect);
      if (FunctionIsConditional)
        FunctionToConditionallyUsedAspects[&F].insert(Aspect);
    }
  for (Argument &Arg : F.args()) {
    const AspectsSetTy ArgAspects =
        getAspectsFromType(Arg.getType(), TypesWithAspects);
    for (const auto &Aspect : ArgAspects)
      if (!FP64ConvEmu || (Aspect != FP64Aspect) ||
          !hasDoubleType(Arg.getType())) {
        if (FunctionIsUnconditional)
          FunctionToUsedAspects[&F].insert(Aspect);
        if (FunctionIsConditional)
          FunctionToConditionallyUsedAspects[&F].insert(Aspect);
      }
  }

  for (Instruction &I : instructions(F)) {
    const AspectsSetTy Aspects =
        getAspectsUsedByInstruction(I, TypesWithAspects);
    for (const auto &Aspect : Aspects)
      if (!FP64ConvEmu || (Aspect != FP64Aspect) || !hasDoubleType(I) ||
          !isFP64ConversionInstruction(I)) {
        if (FunctionIsUnconditional)
          FunctionToUsedAspects[&F].insert(Aspect);
        if (FunctionIsConditional)
          FunctionToConditionallyUsedAspects[&F].insert(Aspect);
      }
  }

  auto CollectAspectsFromMD = [&F](const char* MDName, FunctionToAspectsMapTy &Map) {
    if (const MDNode *MD = F.getMetadata(MDName)) {
      AspectsSetTy Aspects;
      for (const MDOperand &Op : MD->operands()) {
        Constant *C = cast<ConstantAsMetadata>(Op.get())->getValue();
        Aspects.insert(cast<ConstantInt>(C)->getSExtValue());
      }
      Map[&F].insert(Aspects.begin(), Aspects.end());
    }
  };
  CollectAspectsFromMD("sycl_used_aspects", FunctionToUsedAspects);
  CollectAspectsFromMD("sycl_declared_aspects", FunctionToDeclaredAspects);
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

  return FName.starts_with("__spirv_") || FName.starts_with("__sycl_");
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

void setSyclFixedTargetsMD(const std::vector<Function *> &EntryPoints,
                           const SmallVector<StringRef, 8> &Targets,
                           AspectValueToNameMapTy &AspectValues) {
  if (EntryPoints.empty())
    return;

  SmallVector<Metadata *, 8> TargetsMD;
  LLVMContext &C = EntryPoints[0]->getContext();

  for (const auto &Target : Targets) {
    if (!Target.empty()) {
      auto AspectIt = AspectValues.find(Target);
      if (AspectIt != AspectValues.end()) {
        auto ConstIntTarget =
            ConstantInt::getSigned(Type::getInt32Ty(C), AspectIt->second);
        TargetsMD.push_back(ConstantAsMetadata::get(ConstIntTarget));
      }
    }
  }

  MDNode *MDN = MDNode::get(C, TargetsMD);
  for (Function *F : EntryPoints)
    F->setMetadata("sycl_fixed_targets", MDN);
}

CallGraphTy getCallGraph(Module &M) {
  CallGraphTy CG;
  for (Function &F : M.functions()) {
    for (Instruction &I : instructions(F)) {
      if (const auto *CI = dyn_cast<CallInst>(&I)) {
        if (Function *CalledFunction = CI->getCalledFunction();
            !CI->isIndirectCall() && CalledFunction) {
          if (CalledFunction->hasFnAttribute(
                  "sycl-call-if-on-device-conditionally")) {
            if (Function *CallableObj =
                    dyn_cast<Function>(CI->getArgOperand(0))) {
              CG[&F].insert(CalledFunction);
              CG[CalledFunction].insert(CallableObj);
            }
          } else
            CG[&F].insert(CalledFunction);
        }
      }
    }
  }
  return CG;
}

/// Returns a map of functions with corresponding used aspects.
std::tuple<FunctionToAspectsMapTy,
           FunctionToConditionExpressionsAndConditionalAspectsTy,
           FunctionToAspectsMapTy>
buildFunctionsToAspectsMap(Module &M, TypeToAspectsMapTy &TypesWithAspects,
                           const AspectValueToNameMapTy &AspectValues,
                           const std::vector<Function *> &EntryPoints,
                           bool ValidateAspects, bool FP64ConvEmu) {
  FunctionToAspectsMapTy FunctionToUsedAspects;
  FunctionToAspectsMapTy FunctionToConditionallyUsedAspects;
  FunctionToAspectsMapTy FunctionToDeclaredAspects;

  // Create call graph which includes conditional actions
  CallGraphTy CG = getCallGraph(M);

  // Separate functions which called conditionally and unconditionally
  FunctionsClassifiedByTheirFunctionCall FuncsClassifiedByTheirFunctionCall;
  auto addFunctionToFunctionsClassifiedByTheirFunctionCallStruct =
      [&FuncsClassifiedByTheirFunctionCall](Function *F) {
        if (!F->hasFnAttribute("sycl-call-if-on-device-conditionally"))
          FuncsClassifiedByTheirFunctionCall.CalledUnconditionally.insert(F);
        else
          FuncsClassifiedByTheirFunctionCall.CalledConditionally.insert(F);
      };

  SmallPtrSet<const Function *, 16> Visited;
  for (Function *F : EntryPoints) {
    addFunctionToFunctionsClassifiedByTheirFunctionCallStruct(F);
    addAllCalledFunctionsRecursivelyToFunctionsClassifiedByTheirFunctionCallStruct(
        F, CG, FuncsClassifiedByTheirFunctionCall, Visited);
  }
  for (Function &F : M.functions())
    if (!Visited.contains(&F))
      addFunctionToFunctionsClassifiedByTheirFunctionCallStruct(&F);

  for (Function &F : M.functions()) {
    processFunction(
        F, FunctionToUsedAspects, FunctionToConditionallyUsedAspects,
        FunctionToDeclaredAspects, FuncsClassifiedByTheirFunctionCall,
        TypesWithAspects, AspectValues, FP64ConvEmu);
  }

  Visited.clear();
  for (Function *F : EntryPoints)
    propagateAspectsThroughCG(
        F, CG, FunctionToUsedAspects,
        FuncsClassifiedByTheirFunctionCall.CalledUnconditionally, Visited);

  Visited.clear();

  FunctionToConditionExpressionsMapTy FunctionToConditionExpressions;
  std::vector<std::vector<Function *>> Paths;
  for (Function *F : EntryPoints) {
    propagateConditionExpressionsThroughCG(F, CG, FunctionToConditionExpressions);
    std::vector<Function *> CurrentPath;
    identifyPathsContainingConditionalCallers(F, CG, CurrentPath, Paths);
  }

  FunctionToConditionExpressionsAndConditionalAspectsTy
      FunctionToCondExpsAndCondAspects;
  propagateConditionExpressionsAndConditionalAspectsThroughCG(
      Paths, FunctionToCondExpsAndCondAspects, FunctionToConditionExpressions,
      FunctionToConditionallyUsedAspects);
  if (ValidateAspects)
    validateUsedAspectsForFunctions(FunctionToUsedAspects, AspectValues,
                                    EntryPoints, CG);

  // The set of aspects from FunctionToDeclaredAspects should be merged to the
  // set of FunctionToUsedAspects after validateUsedAspectsForFunctions call to
  // avoid errors during validation.
  Visited.clear();
  for (Function *F : EntryPoints)
    propagateAspectsThroughCG(
        F, CG, FunctionToDeclaredAspects,
        FuncsClassifiedByTheirFunctionCall.CalledUnconditionally, Visited);

  return {std::move(FunctionToUsedAspects),
          std::move(FunctionToCondExpsAndCondAspects),
          std::move(FunctionToDeclaredAspects)};
}

} // anonymous namespace

PreservedAnalyses
SYCLPropagateAspectsUsagePass::run(Module &M, ModuleAnalysisManager &MAM) {
  TypeToAspectsMapTy TypesWithAspects = getTypesThatUseAspectsFromMetadata(M);
  AspectValueToNameMapTy AspectValues = getAspectsFromMetadata(M);

  // If there is no metadata for aspect values the source code must not have
  // included the SYCL headers. In that case there should also not be any types
  // that use aspects, so we can skip this pass.
  if (AspectValues.empty()) {
    assert(TypesWithAspects.empty() &&
           "sycl_aspects metadata is missing but "
           "sycl_types_that_use_aspects is present.");
    return PreservedAnalyses::all();
  }

  if (ClSyclFixedTargets.getNumOccurrences() > 0)
    StringRef(ClSyclFixedTargets)
        .split(TargetFixedAspects, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  if (ClSyclExcludeAspects.getNumOccurrences() > 0) {
    SmallVector<StringRef, 4> ExcludedAspectsVec;
    StringRef(ClSyclExcludeAspects)
        .split(ExcludedAspectsVec, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    ExcludedAspects.insert(ExcludedAspectsVec.begin(),
                           ExcludedAspectsVec.end());
  }

  std::vector<Function *> EntryPoints;
  for (Function &F : M.functions())
    if (isEntryPoint(F))
      EntryPoints.push_back(&F);

  propagateAspectsToOtherTypesInModule(M, TypesWithAspects, AspectValues);

  auto [FunctionToUsedAspects,
        FunctionToConditionExpressionsAndConditionalAspects,
        FunctionToDeclaredAspects] =
      buildFunctionsToAspectsMap(M, TypesWithAspects, AspectValues, EntryPoints,
                                 ValidateAspectUsage, FP64ConvEmu);

  // Create a set of excluded aspect values.
  AspectsSetTy ExcludedAspectVals;
  for (const StringRef &AspectName : ExcludedAspects) {
    const auto AspectValIter = AspectValues.find(AspectName);
    assert(AspectValIter != AspectValues.end() &&
           "Excluded aspect does not have a corresponding value.");
    ExcludedAspectVals.insert(AspectValIter->second);
  }

  createUsedAspectsMetadataForFunctions(
      FunctionToUsedAspects, FunctionToDeclaredAspects, ExcludedAspectVals);
  createConditionallyUsedAspectsMetadataForFunctions(
      FunctionToConditionExpressionsAndConditionalAspects, ExcludedAspectVals);

  setSyclFixedTargetsMD(EntryPoints, TargetFixedAspects, AspectValues);
  return PreservedAnalyses::all();
}
