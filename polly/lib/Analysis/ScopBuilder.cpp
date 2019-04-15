//===- ScopBuilder.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Create a polyhedral description for a static control flow region.
//
// The pass creates a polyhedral description of the Scops detected by the SCoP
// detection derived from their LLVM-IR code.
//
//===----------------------------------------------------------------------===//

#include "polly/ScopBuilder.h"
#include "polly/Options.h"
#include "polly/ScopDetection.h"
#include "polly/ScopDetectionDiagnostic.h"
#include "polly/ScopInfo.h"
#include "polly/Support/GICHelper.h"
#include "polly/Support/SCEVValidator.h"
#include "polly/Support/ScopHelper.h"
#include "polly/Support/VirtualInstruction.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;
using namespace polly;

#define DEBUG_TYPE "polly-scops"

STATISTIC(ScopFound, "Number of valid Scops");
STATISTIC(RichScopFound, "Number of Scops containing a loop");
STATISTIC(InfeasibleScops,
          "Number of SCoPs with statically infeasible context.");

bool polly::ModelReadOnlyScalars;

static cl::opt<bool, true> XModelReadOnlyScalars(
    "polly-analyze-read-only-scalars",
    cl::desc("Model read-only scalar values in the scop description"),
    cl::location(ModelReadOnlyScalars), cl::Hidden, cl::ZeroOrMore,
    cl::init(true), cl::cat(PollyCategory));

static cl::opt<bool> UnprofitableScalarAccs(
    "polly-unprofitable-scalar-accs",
    cl::desc("Count statements with scalar accesses as not optimizable"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> DetectFortranArrays(
    "polly-detect-fortran-arrays",
    cl::desc("Detect Fortran arrays and use this for code generation"),
    cl::Hidden, cl::init(false), cl::cat(PollyCategory));

static cl::opt<bool> DetectReductions("polly-detect-reductions",
                                      cl::desc("Detect and exploit reductions"),
                                      cl::Hidden, cl::ZeroOrMore,
                                      cl::init(true), cl::cat(PollyCategory));

// Multiplicative reductions can be disabled separately as these kind of
// operations can overflow easily. Additive reductions and bit operations
// are in contrast pretty stable.
static cl::opt<bool> DisableMultiplicativeReductions(
    "polly-disable-multiplicative-reductions",
    cl::desc("Disable multiplicative reductions"), cl::Hidden, cl::ZeroOrMore,
    cl::init(false), cl::cat(PollyCategory));

enum class GranularityChoice { BasicBlocks, ScalarIndependence, Stores };

static cl::opt<GranularityChoice> StmtGranularity(
    "polly-stmt-granularity",
    cl::desc(
        "Algorithm to use for splitting basic blocks into multiple statements"),
    cl::values(clEnumValN(GranularityChoice::BasicBlocks, "bb",
                          "One statement per basic block"),
               clEnumValN(GranularityChoice::ScalarIndependence, "scalar-indep",
                          "Scalar independence heuristic"),
               clEnumValN(GranularityChoice::Stores, "store",
                          "Store-level granularity")),
    cl::init(GranularityChoice::ScalarIndependence), cl::cat(PollyCategory));

void ScopBuilder::buildPHIAccesses(ScopStmt *PHIStmt, PHINode *PHI,
                                   Region *NonAffineSubRegion,
                                   bool IsExitBlock) {
  // PHI nodes that are in the exit block of the region, hence if IsExitBlock is
  // true, are not modeled as ordinary PHI nodes as they are not part of the
  // region. However, we model the operands in the predecessor blocks that are
  // part of the region as regular scalar accesses.

  // If we can synthesize a PHI we can skip it, however only if it is in
  // the region. If it is not it can only be in the exit block of the region.
  // In this case we model the operands but not the PHI itself.
  auto *Scope = LI.getLoopFor(PHI->getParent());
  if (!IsExitBlock && canSynthesize(PHI, *scop, &SE, Scope))
    return;

  // PHI nodes are modeled as if they had been demoted prior to the SCoP
  // detection. Hence, the PHI is a load of a new memory location in which the
  // incoming value was written at the end of the incoming basic block.
  bool OnlyNonAffineSubRegionOperands = true;
  for (unsigned u = 0; u < PHI->getNumIncomingValues(); u++) {
    Value *Op = PHI->getIncomingValue(u);
    BasicBlock *OpBB = PHI->getIncomingBlock(u);
    ScopStmt *OpStmt = scop->getIncomingStmtFor(PHI->getOperandUse(u));

    // Do not build PHI dependences inside a non-affine subregion, but make
    // sure that the necessary scalar values are still made available.
    if (NonAffineSubRegion && NonAffineSubRegion->contains(OpBB)) {
      auto *OpInst = dyn_cast<Instruction>(Op);
      if (!OpInst || !NonAffineSubRegion->contains(OpInst))
        ensureValueRead(Op, OpStmt);
      continue;
    }

    OnlyNonAffineSubRegionOperands = false;
    ensurePHIWrite(PHI, OpStmt, OpBB, Op, IsExitBlock);
  }

  if (!OnlyNonAffineSubRegionOperands && !IsExitBlock) {
    addPHIReadAccess(PHIStmt, PHI);
  }
}

void ScopBuilder::buildScalarDependences(ScopStmt *UserStmt,
                                         Instruction *Inst) {
  assert(!isa<PHINode>(Inst));

  // Pull-in required operands.
  for (Use &Op : Inst->operands())
    ensureValueRead(Op.get(), UserStmt);
}

void ScopBuilder::buildEscapingDependences(Instruction *Inst) {
  // Check for uses of this instruction outside the scop. Because we do not
  // iterate over such instructions and therefore did not "ensure" the existence
  // of a write, we must determine such use here.
  if (scop->isEscaping(Inst))
    ensureValueWrite(Inst);
}

/// Check that a value is a Fortran Array descriptor.
///
/// We check if V has the following structure:
/// %"struct.array1_real(kind=8)" = type { i8*, i<zz>, i<zz>,
///                                   [<num> x %struct.descriptor_dimension] }
///
///
/// %struct.descriptor_dimension = type { i<zz>, i<zz>, i<zz> }
///
/// 1. V's type name starts with "struct.array"
/// 2. V's type has layout as shown.
/// 3. Final member of V's type has name "struct.descriptor_dimension",
/// 4. "struct.descriptor_dimension" has layout as shown.
/// 5. Consistent use of i<zz> where <zz> is some fixed integer number.
///
/// We are interested in such types since this is the code that dragonegg
/// generates for Fortran array descriptors.
///
/// @param V the Value to be checked.
///
/// @returns True if V is a Fortran array descriptor, False otherwise.
bool isFortranArrayDescriptor(Value *V) {
  PointerType *PTy = dyn_cast<PointerType>(V->getType());

  if (!PTy)
    return false;

  Type *Ty = PTy->getElementType();
  assert(Ty && "Ty expected to be initialized");
  auto *StructArrTy = dyn_cast<StructType>(Ty);

  if (!(StructArrTy && StructArrTy->hasName()))
    return false;

  if (!StructArrTy->getName().startswith("struct.array"))
    return false;

  if (StructArrTy->getNumElements() != 4)
    return false;

  const ArrayRef<Type *> ArrMemberTys = StructArrTy->elements();

  // i8* match
  if (ArrMemberTys[0] != Type::getInt8PtrTy(V->getContext()))
    return false;

  // Get a reference to the int type and check that all the members
  // share the same int type
  Type *IntTy = ArrMemberTys[1];
  if (ArrMemberTys[2] != IntTy)
    return false;

  // type: [<num> x %struct.descriptor_dimension]
  ArrayType *DescriptorDimArrayTy = dyn_cast<ArrayType>(ArrMemberTys[3]);
  if (!DescriptorDimArrayTy)
    return false;

  // type: %struct.descriptor_dimension := type { ixx, ixx, ixx }
  StructType *DescriptorDimTy =
      dyn_cast<StructType>(DescriptorDimArrayTy->getElementType());

  if (!(DescriptorDimTy && DescriptorDimTy->hasName()))
    return false;

  if (DescriptorDimTy->getName() != "struct.descriptor_dimension")
    return false;

  if (DescriptorDimTy->getNumElements() != 3)
    return false;

  for (auto MemberTy : DescriptorDimTy->elements()) {
    if (MemberTy != IntTy)
      return false;
  }

  return true;
}

Value *ScopBuilder::findFADAllocationVisible(MemAccInst Inst) {
  // match: 4.1 & 4.2 store/load
  if (!isa<LoadInst>(Inst) && !isa<StoreInst>(Inst))
    return nullptr;

  // match: 4
  if (Inst.getAlignment() != 8)
    return nullptr;

  Value *Address = Inst.getPointerOperand();

  const BitCastInst *Bitcast = nullptr;
  // [match: 3]
  if (auto *Slot = dyn_cast<GetElementPtrInst>(Address)) {
    Value *TypedMem = Slot->getPointerOperand();
    // match: 2
    Bitcast = dyn_cast<BitCastInst>(TypedMem);
  } else {
    // match: 2
    Bitcast = dyn_cast<BitCastInst>(Address);
  }

  if (!Bitcast)
    return nullptr;

  auto *MallocMem = Bitcast->getOperand(0);

  // match: 1
  auto *MallocCall = dyn_cast<CallInst>(MallocMem);
  if (!MallocCall)
    return nullptr;

  Function *MallocFn = MallocCall->getCalledFunction();
  if (!(MallocFn && MallocFn->hasName() && MallocFn->getName() == "malloc"))
    return nullptr;

  // Find all uses the malloc'd memory.
  // We are looking for a "store" into a struct with the type being the Fortran
  // descriptor type
  for (auto user : MallocMem->users()) {
    /// match: 5
    auto *MallocStore = dyn_cast<StoreInst>(user);
    if (!MallocStore)
      continue;

    auto *DescriptorGEP =
        dyn_cast<GEPOperator>(MallocStore->getPointerOperand());
    if (!DescriptorGEP)
      continue;

    // match: 5
    auto DescriptorType =
        dyn_cast<StructType>(DescriptorGEP->getSourceElementType());
    if (!(DescriptorType && DescriptorType->hasName()))
      continue;

    Value *Descriptor = dyn_cast<Value>(DescriptorGEP->getPointerOperand());

    if (!Descriptor)
      continue;

    if (!isFortranArrayDescriptor(Descriptor))
      continue;

    return Descriptor;
  }

  return nullptr;
}

Value *ScopBuilder::findFADAllocationInvisible(MemAccInst Inst) {
  // match: 3
  if (!isa<LoadInst>(Inst) && !isa<StoreInst>(Inst))
    return nullptr;

  Value *Slot = Inst.getPointerOperand();

  LoadInst *MemLoad = nullptr;
  // [match: 2]
  if (auto *SlotGEP = dyn_cast<GetElementPtrInst>(Slot)) {
    // match: 1
    MemLoad = dyn_cast<LoadInst>(SlotGEP->getPointerOperand());
  } else {
    // match: 1
    MemLoad = dyn_cast<LoadInst>(Slot);
  }

  if (!MemLoad)
    return nullptr;

  auto *BitcastOperator =
      dyn_cast<BitCastOperator>(MemLoad->getPointerOperand());
  if (!BitcastOperator)
    return nullptr;

  Value *Descriptor = dyn_cast<Value>(BitcastOperator->getOperand(0));
  if (!Descriptor)
    return nullptr;

  if (!isFortranArrayDescriptor(Descriptor))
    return nullptr;

  return Descriptor;
}

bool ScopBuilder::buildAccessMultiDimFixed(MemAccInst Inst, ScopStmt *Stmt) {
  Value *Val = Inst.getValueOperand();
  Type *ElementType = Val->getType();
  Value *Address = Inst.getPointerOperand();
  const SCEV *AccessFunction =
      SE.getSCEVAtScope(Address, LI.getLoopFor(Inst->getParent()));
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFunction));
  enum MemoryAccess::AccessType AccType =
      isa<LoadInst>(Inst) ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  if (auto *BitCast = dyn_cast<BitCastInst>(Address)) {
    auto *Src = BitCast->getOperand(0);
    auto *SrcTy = Src->getType();
    auto *DstTy = BitCast->getType();
    // Do not try to delinearize non-sized (opaque) pointers.
    if ((SrcTy->isPointerTy() && !SrcTy->getPointerElementType()->isSized()) ||
        (DstTy->isPointerTy() && !DstTy->getPointerElementType()->isSized())) {
      return false;
    }
    if (SrcTy->isPointerTy() && DstTy->isPointerTy() &&
        DL.getTypeAllocSize(SrcTy->getPointerElementType()) ==
            DL.getTypeAllocSize(DstTy->getPointerElementType()))
      Address = Src;
  }

  auto *GEP = dyn_cast<GetElementPtrInst>(Address);
  if (!GEP)
    return false;

  std::vector<const SCEV *> Subscripts;
  std::vector<int> Sizes;
  std::tie(Subscripts, Sizes) = getIndexExpressionsFromGEP(GEP, SE);
  auto *BasePtr = GEP->getOperand(0);

  if (auto *BasePtrCast = dyn_cast<BitCastInst>(BasePtr))
    BasePtr = BasePtrCast->getOperand(0);

  // Check for identical base pointers to ensure that we do not miss index
  // offsets that have been added before this GEP is applied.
  if (BasePtr != BasePointer->getValue())
    return false;

  std::vector<const SCEV *> SizesSCEV;

  const InvariantLoadsSetTy &ScopRIL = scop->getRequiredInvariantLoads();

  Loop *SurroundingLoop = Stmt->getSurroundingLoop();
  for (auto *Subscript : Subscripts) {
    InvariantLoadsSetTy AccessILS;
    if (!isAffineExpr(&scop->getRegion(), SurroundingLoop, Subscript, SE,
                      &AccessILS))
      return false;

    for (LoadInst *LInst : AccessILS)
      if (!ScopRIL.count(LInst))
        return false;
  }

  if (Sizes.empty())
    return false;

  SizesSCEV.push_back(nullptr);

  for (auto V : Sizes)
    SizesSCEV.push_back(SE.getSCEV(
        ConstantInt::get(IntegerType::getInt64Ty(BasePtr->getContext()), V)));

  addArrayAccess(Stmt, Inst, AccType, BasePointer->getValue(), ElementType,
                 true, Subscripts, SizesSCEV, Val);
  return true;
}

bool ScopBuilder::buildAccessMultiDimParam(MemAccInst Inst, ScopStmt *Stmt) {
  if (!PollyDelinearize)
    return false;

  Value *Address = Inst.getPointerOperand();
  Value *Val = Inst.getValueOperand();
  Type *ElementType = Val->getType();
  unsigned ElementSize = DL.getTypeAllocSize(ElementType);
  enum MemoryAccess::AccessType AccType =
      isa<LoadInst>(Inst) ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  const SCEV *AccessFunction =
      SE.getSCEVAtScope(Address, LI.getLoopFor(Inst->getParent()));
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFunction));

  assert(BasePointer && "Could not find base pointer");

  auto &InsnToMemAcc = scop->getInsnToMemAccMap();
  auto AccItr = InsnToMemAcc.find(Inst);
  if (AccItr == InsnToMemAcc.end())
    return false;

  std::vector<const SCEV *> Sizes = {nullptr};

  Sizes.insert(Sizes.end(), AccItr->second.Shape->DelinearizedSizes.begin(),
               AccItr->second.Shape->DelinearizedSizes.end());

  // In case only the element size is contained in the 'Sizes' array, the
  // access does not access a real multi-dimensional array. Hence, we allow
  // the normal single-dimensional access construction to handle this.
  if (Sizes.size() == 1)
    return false;

  // Remove the element size. This information is already provided by the
  // ElementSize parameter. In case the element size of this access and the
  // element size used for delinearization differs the delinearization is
  // incorrect. Hence, we invalidate the scop.
  //
  // TODO: Handle delinearization with differing element sizes.
  auto DelinearizedSize =
      cast<SCEVConstant>(Sizes.back())->getAPInt().getSExtValue();
  Sizes.pop_back();
  if (ElementSize != DelinearizedSize)
    scop->invalidate(DELINEARIZATION, Inst->getDebugLoc(), Inst->getParent());

  addArrayAccess(Stmt, Inst, AccType, BasePointer->getValue(), ElementType,
                 true, AccItr->second.DelinearizedSubscripts, Sizes, Val);
  return true;
}

bool ScopBuilder::buildAccessMemIntrinsic(MemAccInst Inst, ScopStmt *Stmt) {
  auto *MemIntr = dyn_cast_or_null<MemIntrinsic>(Inst);

  if (MemIntr == nullptr)
    return false;

  auto *L = LI.getLoopFor(Inst->getParent());
  auto *LengthVal = SE.getSCEVAtScope(MemIntr->getLength(), L);
  assert(LengthVal);

  // Check if the length val is actually affine or if we overapproximate it
  InvariantLoadsSetTy AccessILS;
  const InvariantLoadsSetTy &ScopRIL = scop->getRequiredInvariantLoads();

  Loop *SurroundingLoop = Stmt->getSurroundingLoop();
  bool LengthIsAffine = isAffineExpr(&scop->getRegion(), SurroundingLoop,
                                     LengthVal, SE, &AccessILS);
  for (LoadInst *LInst : AccessILS)
    if (!ScopRIL.count(LInst))
      LengthIsAffine = false;
  if (!LengthIsAffine)
    LengthVal = nullptr;

  auto *DestPtrVal = MemIntr->getDest();
  assert(DestPtrVal);

  auto *DestAccFunc = SE.getSCEVAtScope(DestPtrVal, L);
  assert(DestAccFunc);
  // Ignore accesses to "NULL".
  // TODO: We could use this to optimize the region further, e.g., intersect
  //       the context with
  //          isl_set_complement(isl_set_params(getDomain()))
  //       as we know it would be undefined to execute this instruction anyway.
  if (DestAccFunc->isZero())
    return true;

  auto *DestPtrSCEV = dyn_cast<SCEVUnknown>(SE.getPointerBase(DestAccFunc));
  assert(DestPtrSCEV);
  DestAccFunc = SE.getMinusSCEV(DestAccFunc, DestPtrSCEV);
  addArrayAccess(Stmt, Inst, MemoryAccess::MUST_WRITE, DestPtrSCEV->getValue(),
                 IntegerType::getInt8Ty(DestPtrVal->getContext()),
                 LengthIsAffine, {DestAccFunc, LengthVal}, {nullptr},
                 Inst.getValueOperand());

  auto *MemTrans = dyn_cast<MemTransferInst>(MemIntr);
  if (!MemTrans)
    return true;

  auto *SrcPtrVal = MemTrans->getSource();
  assert(SrcPtrVal);

  auto *SrcAccFunc = SE.getSCEVAtScope(SrcPtrVal, L);
  assert(SrcAccFunc);
  // Ignore accesses to "NULL".
  // TODO: See above TODO
  if (SrcAccFunc->isZero())
    return true;

  auto *SrcPtrSCEV = dyn_cast<SCEVUnknown>(SE.getPointerBase(SrcAccFunc));
  assert(SrcPtrSCEV);
  SrcAccFunc = SE.getMinusSCEV(SrcAccFunc, SrcPtrSCEV);
  addArrayAccess(Stmt, Inst, MemoryAccess::READ, SrcPtrSCEV->getValue(),
                 IntegerType::getInt8Ty(SrcPtrVal->getContext()),
                 LengthIsAffine, {SrcAccFunc, LengthVal}, {nullptr},
                 Inst.getValueOperand());

  return true;
}

bool ScopBuilder::buildAccessCallInst(MemAccInst Inst, ScopStmt *Stmt) {
  auto *CI = dyn_cast_or_null<CallInst>(Inst);

  if (CI == nullptr)
    return false;

  if (CI->doesNotAccessMemory() || isIgnoredIntrinsic(CI) || isDebugCall(CI))
    return true;

  bool ReadOnly = false;
  auto *AF = SE.getConstant(IntegerType::getInt64Ty(CI->getContext()), 0);
  auto *CalledFunction = CI->getCalledFunction();
  switch (AA.getModRefBehavior(CalledFunction)) {
  case FMRB_UnknownModRefBehavior:
    llvm_unreachable("Unknown mod ref behaviour cannot be represented.");
  case FMRB_DoesNotAccessMemory:
    return true;
  case FMRB_DoesNotReadMemory:
  case FMRB_OnlyAccessesInaccessibleMem:
  case FMRB_OnlyAccessesInaccessibleOrArgMem:
    return false;
  case FMRB_OnlyReadsMemory:
    GlobalReads.emplace_back(Stmt, CI);
    return true;
  case FMRB_OnlyReadsArgumentPointees:
    ReadOnly = true;
    LLVM_FALLTHROUGH;
  case FMRB_OnlyAccessesArgumentPointees: {
    auto AccType = ReadOnly ? MemoryAccess::READ : MemoryAccess::MAY_WRITE;
    Loop *L = LI.getLoopFor(Inst->getParent());
    for (const auto &Arg : CI->arg_operands()) {
      if (!Arg->getType()->isPointerTy())
        continue;

      auto *ArgSCEV = SE.getSCEVAtScope(Arg, L);
      if (ArgSCEV->isZero())
        continue;

      auto *ArgBasePtr = cast<SCEVUnknown>(SE.getPointerBase(ArgSCEV));
      addArrayAccess(Stmt, Inst, AccType, ArgBasePtr->getValue(),
                     ArgBasePtr->getType(), false, {AF}, {nullptr}, CI);
    }
    return true;
  }
  }

  return true;
}

void ScopBuilder::buildAccessSingleDim(MemAccInst Inst, ScopStmt *Stmt) {
  Value *Address = Inst.getPointerOperand();
  Value *Val = Inst.getValueOperand();
  Type *ElementType = Val->getType();
  enum MemoryAccess::AccessType AccType =
      isa<LoadInst>(Inst) ? MemoryAccess::READ : MemoryAccess::MUST_WRITE;

  const SCEV *AccessFunction =
      SE.getSCEVAtScope(Address, LI.getLoopFor(Inst->getParent()));
  const SCEVUnknown *BasePointer =
      dyn_cast<SCEVUnknown>(SE.getPointerBase(AccessFunction));

  assert(BasePointer && "Could not find base pointer");
  AccessFunction = SE.getMinusSCEV(AccessFunction, BasePointer);

  // Check if the access depends on a loop contained in a non-affine subregion.
  bool isVariantInNonAffineLoop = false;
  SetVector<const Loop *> Loops;
  findLoops(AccessFunction, Loops);
  for (const Loop *L : Loops)
    if (Stmt->contains(L)) {
      isVariantInNonAffineLoop = true;
      break;
    }

  InvariantLoadsSetTy AccessILS;

  Loop *SurroundingLoop = Stmt->getSurroundingLoop();
  bool IsAffine = !isVariantInNonAffineLoop &&
                  isAffineExpr(&scop->getRegion(), SurroundingLoop,
                               AccessFunction, SE, &AccessILS);

  const InvariantLoadsSetTy &ScopRIL = scop->getRequiredInvariantLoads();
  for (LoadInst *LInst : AccessILS)
    if (!ScopRIL.count(LInst))
      IsAffine = false;

  if (!IsAffine && AccType == MemoryAccess::MUST_WRITE)
    AccType = MemoryAccess::MAY_WRITE;

  addArrayAccess(Stmt, Inst, AccType, BasePointer->getValue(), ElementType,
                 IsAffine, {AccessFunction}, {nullptr}, Val);
}

void ScopBuilder::buildMemoryAccess(MemAccInst Inst, ScopStmt *Stmt) {
  if (buildAccessMemIntrinsic(Inst, Stmt))
    return;

  if (buildAccessCallInst(Inst, Stmt))
    return;

  if (buildAccessMultiDimFixed(Inst, Stmt))
    return;

  if (buildAccessMultiDimParam(Inst, Stmt))
    return;

  buildAccessSingleDim(Inst, Stmt);
}

void ScopBuilder::buildAccessFunctions() {
  for (auto &Stmt : *scop) {
    if (Stmt.isBlockStmt()) {
      buildAccessFunctions(&Stmt, *Stmt.getBasicBlock());
      continue;
    }

    Region *R = Stmt.getRegion();
    for (BasicBlock *BB : R->blocks())
      buildAccessFunctions(&Stmt, *BB, R);
  }

  // Build write accesses for values that are used after the SCoP.
  // The instructions defining them might be synthesizable and therefore not
  // contained in any statement, hence we iterate over the original instructions
  // to identify all escaping values.
  for (BasicBlock *BB : scop->getRegion().blocks()) {
    for (Instruction &Inst : *BB)
      buildEscapingDependences(&Inst);
  }
}

bool ScopBuilder::shouldModelInst(Instruction *Inst, Loop *L) {
  return !Inst->isTerminator() && !isIgnoredIntrinsic(Inst) &&
         !canSynthesize(Inst, *scop, &SE, L);
}

/// Generate a name for a statement.
///
/// @param BB     The basic block the statement will represent.
/// @param BBIdx  The index of the @p BB relative to other BBs/regions.
/// @param Count  The index of the created statement in @p BB.
/// @param IsMain Whether this is the main of all statement for @p BB. If true,
///               no suffix will be added.
/// @param IsLast Uses a special indicator for the last statement of a BB.
static std::string makeStmtName(BasicBlock *BB, long BBIdx, int Count,
                                bool IsMain, bool IsLast = false) {
  std::string Suffix;
  if (!IsMain) {
    if (UseInstructionNames)
      Suffix = '_';
    if (IsLast)
      Suffix += "last";
    else if (Count < 26)
      Suffix += 'a' + Count;
    else
      Suffix += std::to_string(Count);
  }
  return getIslCompatibleName("Stmt", BB, BBIdx, Suffix, UseInstructionNames);
}

/// Generate a name for a statement that represents a non-affine subregion.
///
/// @param R    The region the statement will represent.
/// @param RIdx The index of the @p R relative to other BBs/regions.
static std::string makeStmtName(Region *R, long RIdx) {
  return getIslCompatibleName("Stmt", R->getNameStr(), RIdx, "",
                              UseInstructionNames);
}

void ScopBuilder::buildSequentialBlockStmts(BasicBlock *BB, bool SplitOnStore) {
  Loop *SurroundingLoop = LI.getLoopFor(BB);

  int Count = 0;
  long BBIdx = scop->getNextStmtIdx();
  std::vector<Instruction *> Instructions;
  for (Instruction &Inst : *BB) {
    if (shouldModelInst(&Inst, SurroundingLoop))
      Instructions.push_back(&Inst);
    if (Inst.getMetadata("polly_split_after") ||
        (SplitOnStore && isa<StoreInst>(Inst))) {
      std::string Name = makeStmtName(BB, BBIdx, Count, Count == 0);
      scop->addScopStmt(BB, Name, SurroundingLoop, Instructions);
      Count++;
      Instructions.clear();
    }
  }

  std::string Name = makeStmtName(BB, BBIdx, Count, Count == 0);
  scop->addScopStmt(BB, Name, SurroundingLoop, Instructions);
}

/// Is @p Inst an ordered instruction?
///
/// An unordered instruction is an instruction, such that a sequence of
/// unordered instructions can be permuted without changing semantics. Any
/// instruction for which this is not always the case is ordered.
static bool isOrderedInstruction(Instruction *Inst) {
  return Inst->mayHaveSideEffects() || Inst->mayReadOrWriteMemory();
}

/// Join instructions to the same statement if one uses the scalar result of the
/// other.
static void joinOperandTree(EquivalenceClasses<Instruction *> &UnionFind,
                            ArrayRef<Instruction *> ModeledInsts) {
  for (Instruction *Inst : ModeledInsts) {
    if (isa<PHINode>(Inst))
      continue;

    for (Use &Op : Inst->operands()) {
      Instruction *OpInst = dyn_cast<Instruction>(Op.get());
      if (!OpInst)
        continue;

      // Check if OpInst is in the BB and is a modeled instruction.
      auto OpVal = UnionFind.findValue(OpInst);
      if (OpVal == UnionFind.end())
        continue;

      UnionFind.unionSets(Inst, OpInst);
    }
  }
}

/// Ensure that the order of ordered instructions does not change.
///
/// If we encounter an ordered instruction enclosed in instructions belonging to
/// a different statement (which might as well contain ordered instructions, but
/// this is not tested here), join them.
static void
joinOrderedInstructions(EquivalenceClasses<Instruction *> &UnionFind,
                        ArrayRef<Instruction *> ModeledInsts) {
  SetVector<Instruction *> SeenLeaders;
  for (Instruction *Inst : ModeledInsts) {
    if (!isOrderedInstruction(Inst))
      continue;

    Instruction *Leader = UnionFind.getLeaderValue(Inst);
    bool Inserted = SeenLeaders.insert(Leader);
    if (Inserted)
      continue;

    // Merge statements to close holes. Say, we have already seen statements A
    // and B, in this order. Then we see an instruction of A again and we would
    // see the pattern "A B A". This function joins all statements until the
    // only seen occurrence of A.
    for (Instruction *Prev : reverse(SeenLeaders)) {
      // Items added to 'SeenLeaders' are leaders, but may have lost their
      // leadership status when merged into another statement.
      Instruction *PrevLeader = UnionFind.getLeaderValue(SeenLeaders.back());
      if (PrevLeader == Leader)
        break;
      UnionFind.unionSets(Prev, Leader);
    }
  }
}

/// If the BasicBlock has an edge from itself, ensure that the PHI WRITEs for
/// the incoming values from this block are executed after the PHI READ.
///
/// Otherwise it could overwrite the incoming value from before the BB with the
/// value for the next execution. This can happen if the PHI WRITE is added to
/// the statement with the instruction that defines the incoming value (instead
/// of the last statement of the same BB). To ensure that the PHI READ and WRITE
/// are in order, we put both into the statement. PHI WRITEs are always executed
/// after PHI READs when they are in the same statement.
///
/// TODO: This is an overpessimization. We only have to ensure that the PHI
/// WRITE is not put into a statement containing the PHI itself. That could also
/// be done by
/// - having all (strongly connected) PHIs in a single statement,
/// - unite only the PHIs in the operand tree of the PHI WRITE (because it only
///   has a chance of being lifted before a PHI by being in a statement with a
///   PHI that comes before in the basic block), or
/// - when uniting statements, ensure that no (relevant) PHIs are overtaken.
static void joinOrderedPHIs(EquivalenceClasses<Instruction *> &UnionFind,
                            ArrayRef<Instruction *> ModeledInsts) {
  for (Instruction *Inst : ModeledInsts) {
    PHINode *PHI = dyn_cast<PHINode>(Inst);
    if (!PHI)
      continue;

    int Idx = PHI->getBasicBlockIndex(PHI->getParent());
    if (Idx < 0)
      continue;

    Instruction *IncomingVal =
        dyn_cast<Instruction>(PHI->getIncomingValue(Idx));
    if (!IncomingVal)
      continue;

    UnionFind.unionSets(PHI, IncomingVal);
  }
}

void ScopBuilder::buildEqivClassBlockStmts(BasicBlock *BB) {
  Loop *L = LI.getLoopFor(BB);

  // Extracting out modeled instructions saves us from checking
  // shouldModelInst() repeatedly.
  SmallVector<Instruction *, 32> ModeledInsts;
  EquivalenceClasses<Instruction *> UnionFind;
  Instruction *MainInst = nullptr;
  for (Instruction &Inst : *BB) {
    if (!shouldModelInst(&Inst, L))
      continue;
    ModeledInsts.push_back(&Inst);
    UnionFind.insert(&Inst);

    // When a BB is split into multiple statements, the main statement is the
    // one containing the 'main' instruction. We select the first instruction
    // that is unlikely to be removed (because it has side-effects) as the main
    // one. It is used to ensure that at least one statement from the bb has the
    // same name as with -polly-stmt-granularity=bb.
    if (!MainInst && (isa<StoreInst>(Inst) ||
                      (isa<CallInst>(Inst) && !isa<IntrinsicInst>(Inst))))
      MainInst = &Inst;
  }

  joinOperandTree(UnionFind, ModeledInsts);
  joinOrderedInstructions(UnionFind, ModeledInsts);
  joinOrderedPHIs(UnionFind, ModeledInsts);

  // The list of instructions for statement (statement represented by the leader
  // instruction). The order of statements instructions is reversed such that
  // the epilogue is first. This makes it easier to ensure that the epilogue is
  // the last statement.
  MapVector<Instruction *, std::vector<Instruction *>> LeaderToInstList;

  // Collect the instructions of all leaders. UnionFind's member iterator
  // unfortunately are not in any specific order.
  for (Instruction &Inst : reverse(*BB)) {
    auto LeaderIt = UnionFind.findLeader(&Inst);
    if (LeaderIt == UnionFind.member_end())
      continue;

    std::vector<Instruction *> &InstList = LeaderToInstList[*LeaderIt];
    InstList.push_back(&Inst);
  }

  // Finally build the statements.
  int Count = 0;
  long BBIdx = scop->getNextStmtIdx();
  bool MainFound = false;
  for (auto &Instructions : reverse(LeaderToInstList)) {
    std::vector<Instruction *> &InstList = Instructions.second;

    // If there is no main instruction, make the first statement the main.
    bool IsMain;
    if (MainInst)
      IsMain = std::find(InstList.begin(), InstList.end(), MainInst) !=
               InstList.end();
    else
      IsMain = (Count == 0);
    if (IsMain)
      MainFound = true;

    std::reverse(InstList.begin(), InstList.end());
    std::string Name = makeStmtName(BB, BBIdx, Count, IsMain);
    scop->addScopStmt(BB, Name, L, std::move(InstList));
    Count += 1;
  }

  // Unconditionally add an epilogue (last statement). It contains no
  // instructions, but holds the PHI write accesses for successor basic blocks,
  // if the incoming value is not defined in another statement if the same BB.
  // The epilogue will be removed if no PHIWrite is added to it.
  std::string EpilogueName = makeStmtName(BB, BBIdx, Count, !MainFound, true);
  scop->addScopStmt(BB, EpilogueName, L, {});
}

void ScopBuilder::buildStmts(Region &SR) {
  if (scop->isNonAffineSubRegion(&SR)) {
    std::vector<Instruction *> Instructions;
    Loop *SurroundingLoop =
        getFirstNonBoxedLoopFor(SR.getEntry(), LI, scop->getBoxedLoops());
    for (Instruction &Inst : *SR.getEntry())
      if (shouldModelInst(&Inst, SurroundingLoop))
        Instructions.push_back(&Inst);
    long RIdx = scop->getNextStmtIdx();
    std::string Name = makeStmtName(&SR, RIdx);
    scop->addScopStmt(&SR, Name, SurroundingLoop, Instructions);
    return;
  }

  for (auto I = SR.element_begin(), E = SR.element_end(); I != E; ++I)
    if (I->isSubRegion())
      buildStmts(*I->getNodeAs<Region>());
    else {
      BasicBlock *BB = I->getNodeAs<BasicBlock>();
      switch (StmtGranularity) {
      case GranularityChoice::BasicBlocks:
        buildSequentialBlockStmts(BB);
        break;
      case GranularityChoice::ScalarIndependence:
        buildEqivClassBlockStmts(BB);
        break;
      case GranularityChoice::Stores:
        buildSequentialBlockStmts(BB, true);
        break;
      }
    }
}

void ScopBuilder::buildAccessFunctions(ScopStmt *Stmt, BasicBlock &BB,
                                       Region *NonAffineSubRegion) {
  assert(
      Stmt &&
      "The exit BB is the only one that cannot be represented by a statement");
  assert(Stmt->represents(&BB));

  // We do not build access functions for error blocks, as they may contain
  // instructions we can not model.
  if (isErrorBlock(BB, scop->getRegion(), LI, DT))
    return;

  auto BuildAccessesForInst = [this, Stmt,
                               NonAffineSubRegion](Instruction *Inst) {
    PHINode *PHI = dyn_cast<PHINode>(Inst);
    if (PHI)
      buildPHIAccesses(Stmt, PHI, NonAffineSubRegion, false);

    if (auto MemInst = MemAccInst::dyn_cast(*Inst)) {
      assert(Stmt && "Cannot build access function in non-existing statement");
      buildMemoryAccess(MemInst, Stmt);
    }

    // PHI nodes have already been modeled above and terminators that are
    // not part of a non-affine subregion are fully modeled and regenerated
    // from the polyhedral domains. Hence, they do not need to be modeled as
    // explicit data dependences.
    if (!PHI)
      buildScalarDependences(Stmt, Inst);
  };

  const InvariantLoadsSetTy &RIL = scop->getRequiredInvariantLoads();
  bool IsEntryBlock = (Stmt->getEntryBlock() == &BB);
  if (IsEntryBlock) {
    for (Instruction *Inst : Stmt->getInstructions())
      BuildAccessesForInst(Inst);
    if (Stmt->isRegionStmt())
      BuildAccessesForInst(BB.getTerminator());
  } else {
    for (Instruction &Inst : BB) {
      if (isIgnoredIntrinsic(&Inst))
        continue;

      // Invariant loads already have been processed.
      if (isa<LoadInst>(Inst) && RIL.count(cast<LoadInst>(&Inst)))
        continue;

      BuildAccessesForInst(&Inst);
    }
  }
}

MemoryAccess *ScopBuilder::addMemoryAccess(
    ScopStmt *Stmt, Instruction *Inst, MemoryAccess::AccessType AccType,
    Value *BaseAddress, Type *ElementType, bool Affine, Value *AccessValue,
    ArrayRef<const SCEV *> Subscripts, ArrayRef<const SCEV *> Sizes,
    MemoryKind Kind) {
  bool isKnownMustAccess = false;

  // Accesses in single-basic block statements are always executed.
  if (Stmt->isBlockStmt())
    isKnownMustAccess = true;

  if (Stmt->isRegionStmt()) {
    // Accesses that dominate the exit block of a non-affine region are always
    // executed. In non-affine regions there may exist MemoryKind::Values that
    // do not dominate the exit. MemoryKind::Values will always dominate the
    // exit and MemoryKind::PHIs only if there is at most one PHI_WRITE in the
    // non-affine region.
    if (Inst && DT.dominates(Inst->getParent(), Stmt->getRegion()->getExit()))
      isKnownMustAccess = true;
  }

  // Non-affine PHI writes do not "happen" at a particular instruction, but
  // after exiting the statement. Therefore they are guaranteed to execute and
  // overwrite the old value.
  if (Kind == MemoryKind::PHI || Kind == MemoryKind::ExitPHI)
    isKnownMustAccess = true;

  if (!isKnownMustAccess && AccType == MemoryAccess::MUST_WRITE)
    AccType = MemoryAccess::MAY_WRITE;

  auto *Access = new MemoryAccess(Stmt, Inst, AccType, BaseAddress, ElementType,
                                  Affine, Subscripts, Sizes, AccessValue, Kind);

  scop->addAccessFunction(Access);
  Stmt->addAccess(Access);
  return Access;
}

void ScopBuilder::addArrayAccess(ScopStmt *Stmt, MemAccInst MemAccInst,
                                 MemoryAccess::AccessType AccType,
                                 Value *BaseAddress, Type *ElementType,
                                 bool IsAffine,
                                 ArrayRef<const SCEV *> Subscripts,
                                 ArrayRef<const SCEV *> Sizes,
                                 Value *AccessValue) {
  ArrayBasePointers.insert(BaseAddress);
  auto *MemAccess = addMemoryAccess(Stmt, MemAccInst, AccType, BaseAddress,
                                    ElementType, IsAffine, AccessValue,
                                    Subscripts, Sizes, MemoryKind::Array);

  if (!DetectFortranArrays)
    return;

  if (Value *FAD = findFADAllocationInvisible(MemAccInst))
    MemAccess->setFortranArrayDescriptor(FAD);
  else if (Value *FAD = findFADAllocationVisible(MemAccInst))
    MemAccess->setFortranArrayDescriptor(FAD);
}

void ScopBuilder::ensureValueWrite(Instruction *Inst) {
  // Find the statement that defines the value of Inst. That statement has to
  // write the value to make it available to those statements that read it.
  ScopStmt *Stmt = scop->getStmtFor(Inst);

  // It is possible that the value is synthesizable within a loop (such that it
  // is not part of any statement), but not after the loop (where you need the
  // number of loop round-trips to synthesize it). In LCSSA-form a PHI node will
  // avoid this. In case the IR has no such PHI, use the last statement (where
  // the value is synthesizable) to write the value.
  if (!Stmt)
    Stmt = scop->getLastStmtFor(Inst->getParent());

  // Inst not defined within this SCoP.
  if (!Stmt)
    return;

  // Do not process further if the instruction is already written.
  if (Stmt->lookupValueWriteOf(Inst))
    return;

  addMemoryAccess(Stmt, Inst, MemoryAccess::MUST_WRITE, Inst, Inst->getType(),
                  true, Inst, ArrayRef<const SCEV *>(),
                  ArrayRef<const SCEV *>(), MemoryKind::Value);
}

void ScopBuilder::ensureValueRead(Value *V, ScopStmt *UserStmt) {
  // TODO: Make ScopStmt::ensureValueRead(Value*) offer the same functionality
  // to be able to replace this one. Currently, there is a split responsibility.
  // In a first step, the MemoryAccess is created, but without the
  // AccessRelation. In the second step by ScopStmt::buildAccessRelations(), the
  // AccessRelation is created. At least for scalar accesses, there is no new
  // information available at ScopStmt::buildAccessRelations(), so we could
  // create the AccessRelation right away. This is what
  // ScopStmt::ensureValueRead(Value*) does.

  auto *Scope = UserStmt->getSurroundingLoop();
  auto VUse = VirtualUse::create(scop.get(), UserStmt, Scope, V, false);
  switch (VUse.getKind()) {
  case VirtualUse::Constant:
  case VirtualUse::Block:
  case VirtualUse::Synthesizable:
  case VirtualUse::Hoisted:
  case VirtualUse::Intra:
    // Uses of these kinds do not need a MemoryAccess.
    break;

  case VirtualUse::ReadOnly:
    // Add MemoryAccess for invariant values only if requested.
    if (!ModelReadOnlyScalars)
      break;

    LLVM_FALLTHROUGH;
  case VirtualUse::Inter:

    // Do not create another MemoryAccess for reloading the value if one already
    // exists.
    if (UserStmt->lookupValueReadOf(V))
      break;

    addMemoryAccess(UserStmt, nullptr, MemoryAccess::READ, V, V->getType(),
                    true, V, ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
                    MemoryKind::Value);

    // Inter-statement uses need to write the value in their defining statement.
    if (VUse.isInter())
      ensureValueWrite(cast<Instruction>(V));
    break;
  }
}

void ScopBuilder::ensurePHIWrite(PHINode *PHI, ScopStmt *IncomingStmt,
                                 BasicBlock *IncomingBlock,
                                 Value *IncomingValue, bool IsExitBlock) {
  // As the incoming block might turn out to be an error statement ensure we
  // will create an exit PHI SAI object. It is needed during code generation
  // and would be created later anyway.
  if (IsExitBlock)
    scop->getOrCreateScopArrayInfo(PHI, PHI->getType(), {},
                                   MemoryKind::ExitPHI);

  // This is possible if PHI is in the SCoP's entry block. The incoming blocks
  // from outside the SCoP's region have no statement representation.
  if (!IncomingStmt)
    return;

  // Take care for the incoming value being available in the incoming block.
  // This must be done before the check for multiple PHI writes because multiple
  // exiting edges from subregion each can be the effective written value of the
  // subregion. As such, all of them must be made available in the subregion
  // statement.
  ensureValueRead(IncomingValue, IncomingStmt);

  // Do not add more than one MemoryAccess per PHINode and ScopStmt.
  if (MemoryAccess *Acc = IncomingStmt->lookupPHIWriteOf(PHI)) {
    assert(Acc->getAccessInstruction() == PHI);
    Acc->addIncoming(IncomingBlock, IncomingValue);
    return;
  }

  MemoryAccess *Acc = addMemoryAccess(
      IncomingStmt, PHI, MemoryAccess::MUST_WRITE, PHI, PHI->getType(), true,
      PHI, ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
      IsExitBlock ? MemoryKind::ExitPHI : MemoryKind::PHI);
  assert(Acc);
  Acc->addIncoming(IncomingBlock, IncomingValue);
}

void ScopBuilder::addPHIReadAccess(ScopStmt *PHIStmt, PHINode *PHI) {
  addMemoryAccess(PHIStmt, PHI, MemoryAccess::READ, PHI, PHI->getType(), true,
                  PHI, ArrayRef<const SCEV *>(), ArrayRef<const SCEV *>(),
                  MemoryKind::PHI);
}

void ScopBuilder::buildDomain(ScopStmt &Stmt) {
  isl::id Id = isl::id::alloc(scop->getIslCtx(), Stmt.getBaseName(), &Stmt);

  Stmt.Domain = scop->getDomainConditions(&Stmt);
  Stmt.Domain = Stmt.Domain.set_tuple_id(Id);
}

void ScopBuilder::collectSurroundingLoops(ScopStmt &Stmt) {
  isl::set Domain = Stmt.getDomain();
  BasicBlock *BB = Stmt.getEntryBlock();

  Loop *L = LI.getLoopFor(BB);

  while (L && Stmt.isRegionStmt() && Stmt.getRegion()->contains(L))
    L = L->getParentLoop();

  SmallVector<llvm::Loop *, 8> Loops;

  while (L && Stmt.getParent()->getRegion().contains(L)) {
    Loops.push_back(L);
    L = L->getParentLoop();
  }

  Stmt.NestLoops.insert(Stmt.NestLoops.begin(), Loops.rbegin(), Loops.rend());
}

/// Return the reduction type for a given binary operator.
static MemoryAccess::ReductionType getReductionType(const BinaryOperator *BinOp,
                                                    const Instruction *Load) {
  if (!BinOp)
    return MemoryAccess::RT_NONE;
  switch (BinOp->getOpcode()) {
  case Instruction::FAdd:
    if (!BinOp->isFast())
      return MemoryAccess::RT_NONE;
    LLVM_FALLTHROUGH;
  case Instruction::Add:
    return MemoryAccess::RT_ADD;
  case Instruction::Or:
    return MemoryAccess::RT_BOR;
  case Instruction::Xor:
    return MemoryAccess::RT_BXOR;
  case Instruction::And:
    return MemoryAccess::RT_BAND;
  case Instruction::FMul:
    if (!BinOp->isFast())
      return MemoryAccess::RT_NONE;
    LLVM_FALLTHROUGH;
  case Instruction::Mul:
    if (DisableMultiplicativeReductions)
      return MemoryAccess::RT_NONE;
    return MemoryAccess::RT_MUL;
  default:
    return MemoryAccess::RT_NONE;
  }
}

void ScopBuilder::checkForReductions(ScopStmt &Stmt) {
  SmallVector<MemoryAccess *, 2> Loads;
  SmallVector<std::pair<MemoryAccess *, MemoryAccess *>, 4> Candidates;

  // First collect candidate load-store reduction chains by iterating over all
  // stores and collecting possible reduction loads.
  for (MemoryAccess *StoreMA : Stmt) {
    if (StoreMA->isRead())
      continue;

    Loads.clear();
    collectCandidateReductionLoads(StoreMA, Loads);
    for (MemoryAccess *LoadMA : Loads)
      Candidates.push_back(std::make_pair(LoadMA, StoreMA));
  }

  // Then check each possible candidate pair.
  for (const auto &CandidatePair : Candidates) {
    bool Valid = true;
    isl::map LoadAccs = CandidatePair.first->getAccessRelation();
    isl::map StoreAccs = CandidatePair.second->getAccessRelation();

    // Skip those with obviously unequal base addresses.
    if (!LoadAccs.has_equal_space(StoreAccs)) {
      continue;
    }

    // And check if the remaining for overlap with other memory accesses.
    isl::map AllAccsRel = LoadAccs.unite(StoreAccs);
    AllAccsRel = AllAccsRel.intersect_domain(Stmt.getDomain());
    isl::set AllAccs = AllAccsRel.range();

    for (MemoryAccess *MA : Stmt) {
      if (MA == CandidatePair.first || MA == CandidatePair.second)
        continue;

      isl::map AccRel =
          MA->getAccessRelation().intersect_domain(Stmt.getDomain());
      isl::set Accs = AccRel.range();

      if (AllAccs.has_equal_space(Accs)) {
        isl::set OverlapAccs = Accs.intersect(AllAccs);
        Valid = Valid && OverlapAccs.is_empty();
      }
    }

    if (!Valid)
      continue;

    const LoadInst *Load =
        dyn_cast<const LoadInst>(CandidatePair.first->getAccessInstruction());
    MemoryAccess::ReductionType RT =
        getReductionType(dyn_cast<BinaryOperator>(Load->user_back()), Load);

    // If no overlapping access was found we mark the load and store as
    // reduction like.
    CandidatePair.first->markAsReductionLike(RT);
    CandidatePair.second->markAsReductionLike(RT);
  }
}

void ScopBuilder::collectCandidateReductionLoads(
    MemoryAccess *StoreMA, SmallVectorImpl<MemoryAccess *> &Loads) {
  ScopStmt *Stmt = StoreMA->getStatement();

  auto *Store = dyn_cast<StoreInst>(StoreMA->getAccessInstruction());
  if (!Store)
    return;

  // Skip if there is not one binary operator between the load and the store
  auto *BinOp = dyn_cast<BinaryOperator>(Store->getValueOperand());
  if (!BinOp)
    return;

  // Skip if the binary operators has multiple uses
  if (BinOp->getNumUses() != 1)
    return;

  // Skip if the opcode of the binary operator is not commutative/associative
  if (!BinOp->isCommutative() || !BinOp->isAssociative())
    return;

  // Skip if the binary operator is outside the current SCoP
  if (BinOp->getParent() != Store->getParent())
    return;

  // Skip if it is a multiplicative reduction and we disabled them
  if (DisableMultiplicativeReductions &&
      (BinOp->getOpcode() == Instruction::Mul ||
       BinOp->getOpcode() == Instruction::FMul))
    return;

  // Check the binary operator operands for a candidate load
  auto *PossibleLoad0 = dyn_cast<LoadInst>(BinOp->getOperand(0));
  auto *PossibleLoad1 = dyn_cast<LoadInst>(BinOp->getOperand(1));
  if (!PossibleLoad0 && !PossibleLoad1)
    return;

  // A load is only a candidate if it cannot escape (thus has only this use)
  if (PossibleLoad0 && PossibleLoad0->getNumUses() == 1)
    if (PossibleLoad0->getParent() == Store->getParent())
      Loads.push_back(&Stmt->getArrayAccessFor(PossibleLoad0));
  if (PossibleLoad1 && PossibleLoad1->getNumUses() == 1)
    if (PossibleLoad1->getParent() == Store->getParent())
      Loads.push_back(&Stmt->getArrayAccessFor(PossibleLoad1));
}

void ScopBuilder::buildAccessRelations(ScopStmt &Stmt) {
  for (MemoryAccess *Access : Stmt.MemAccs) {
    Type *ElementType = Access->getElementType();

    MemoryKind Ty;
    if (Access->isPHIKind())
      Ty = MemoryKind::PHI;
    else if (Access->isExitPHIKind())
      Ty = MemoryKind::ExitPHI;
    else if (Access->isValueKind())
      Ty = MemoryKind::Value;
    else
      Ty = MemoryKind::Array;

    auto *SAI = scop->getOrCreateScopArrayInfo(Access->getOriginalBaseAddr(),
                                               ElementType, Access->Sizes, Ty);
    Access->buildAccessRelation(SAI);
    scop->addAccessData(Access);
  }
}

#ifndef NDEBUG
static void verifyUse(Scop *S, Use &Op, LoopInfo &LI) {
  auto PhysUse = VirtualUse::create(S, Op, &LI, false);
  auto VirtUse = VirtualUse::create(S, Op, &LI, true);
  assert(PhysUse.getKind() == VirtUse.getKind());
}

/// Check the consistency of every statement's MemoryAccesses.
///
/// The check is carried out by expecting the "physical" kind of use (derived
/// from the BasicBlocks instructions resides in) to be same as the "virtual"
/// kind of use (derived from a statement's MemoryAccess).
///
/// The "physical" uses are taken by ensureValueRead to determine whether to
/// create MemoryAccesses. When done, the kind of scalar access should be the
/// same no matter which way it was derived.
///
/// The MemoryAccesses might be changed by later SCoP-modifying passes and hence
/// can intentionally influence on the kind of uses (not corresponding to the
/// "physical" anymore, hence called "virtual"). The CodeGenerator therefore has
/// to pick up the virtual uses. But here in the code generator, this has not
/// happened yet, such that virtual and physical uses are equivalent.
static void verifyUses(Scop *S, LoopInfo &LI, DominatorTree &DT) {
  for (auto *BB : S->getRegion().blocks()) {
    for (auto &Inst : *BB) {
      auto *Stmt = S->getStmtFor(&Inst);
      if (!Stmt)
        continue;

      if (isIgnoredIntrinsic(&Inst))
        continue;

      // Branch conditions are encoded in the statement domains.
      if (Inst.isTerminator() && Stmt->isBlockStmt())
        continue;

      // Verify all uses.
      for (auto &Op : Inst.operands())
        verifyUse(S, Op, LI);

      // Stores do not produce values used by other statements.
      if (isa<StoreInst>(Inst))
        continue;

      // For every value defined in the block, also check that a use of that
      // value in the same statement would not be an inter-statement use. It can
      // still be synthesizable or load-hoisted, but these kind of instructions
      // are not directly copied in code-generation.
      auto VirtDef =
          VirtualUse::create(S, Stmt, Stmt->getSurroundingLoop(), &Inst, true);
      assert(VirtDef.getKind() == VirtualUse::Synthesizable ||
             VirtDef.getKind() == VirtualUse::Intra ||
             VirtDef.getKind() == VirtualUse::Hoisted);
    }
  }

  if (S->hasSingleExitEdge())
    return;

  // PHINodes in the SCoP region's exit block are also uses to be checked.
  if (!S->getRegion().isTopLevelRegion()) {
    for (auto &Inst : *S->getRegion().getExit()) {
      if (!isa<PHINode>(Inst))
        break;

      for (auto &Op : Inst.operands())
        verifyUse(S, Op, LI);
    }
  }
}
#endif

/// Return the block that is the representing block for @p RN.
static inline BasicBlock *getRegionNodeBasicBlock(RegionNode *RN) {
  return RN->isSubRegion() ? RN->getNodeAs<Region>()->getEntry()
                           : RN->getNodeAs<BasicBlock>();
}

void ScopBuilder::buildScop(Region &R, AssumptionCache &AC,
                            OptimizationRemarkEmitter &ORE) {
  scop.reset(new Scop(R, SE, LI, DT, *SD.getDetectionContext(&R), ORE));

  buildStmts(R);

  // Create all invariant load instructions first. These are categorized as
  // 'synthesizable', therefore are not part of any ScopStmt but need to be
  // created somewhere.
  const InvariantLoadsSetTy &RIL = scop->getRequiredInvariantLoads();
  for (BasicBlock *BB : scop->getRegion().blocks()) {
    if (isErrorBlock(*BB, scop->getRegion(), LI, DT))
      continue;

    for (Instruction &Inst : *BB) {
      LoadInst *Load = dyn_cast<LoadInst>(&Inst);
      if (!Load)
        continue;

      if (!RIL.count(Load))
        continue;

      // Invariant loads require a MemoryAccess to be created in some statement.
      // It is not important to which statement the MemoryAccess is added
      // because it will later be removed from the ScopStmt again. We chose the
      // first statement of the basic block the LoadInst is in.
      ArrayRef<ScopStmt *> List = scop->getStmtListFor(BB);
      assert(!List.empty());
      ScopStmt *RILStmt = List.front();
      buildMemoryAccess(Load, RILStmt);
    }
  }
  buildAccessFunctions();

  // In case the region does not have an exiting block we will later (during
  // code generation) split the exit block. This will move potential PHI nodes
  // from the current exit block into the new region exiting block. Hence, PHI
  // nodes that are at this point not part of the region will be.
  // To handle these PHI nodes later we will now model their operands as scalar
  // accesses. Note that we do not model anything in the exit block if we have
  // an exiting block in the region, as there will not be any splitting later.
  if (!R.isTopLevelRegion() && !scop->hasSingleExitEdge()) {
    for (Instruction &Inst : *R.getExit()) {
      PHINode *PHI = dyn_cast<PHINode>(&Inst);
      if (!PHI)
        break;

      buildPHIAccesses(nullptr, PHI, nullptr, true);
    }
  }

  // Create memory accesses for global reads since all arrays are now known.
  auto *AF = SE.getConstant(IntegerType::getInt64Ty(SE.getContext()), 0);
  for (auto GlobalReadPair : GlobalReads) {
    ScopStmt *GlobalReadStmt = GlobalReadPair.first;
    Instruction *GlobalRead = GlobalReadPair.second;
    for (auto *BP : ArrayBasePointers)
      addArrayAccess(GlobalReadStmt, MemAccInst(GlobalRead), MemoryAccess::READ,
                     BP, BP->getType(), false, {AF}, {nullptr}, GlobalRead);
  }

  scop->buildInvariantEquivalenceClasses();

  /// A map from basic blocks to their invalid domains.
  DenseMap<BasicBlock *, isl::set> InvalidDomainMap;

  if (!scop->buildDomains(&R, DT, LI, InvalidDomainMap)) {
    LLVM_DEBUG(
        dbgs() << "Bailing-out because buildDomains encountered problems\n");
    return;
  }

  scop->addUserAssumptions(AC, DT, LI, InvalidDomainMap);

  // Initialize the invalid domain.
  for (ScopStmt &Stmt : scop->Stmts)
    if (Stmt.isBlockStmt())
      Stmt.setInvalidDomain(InvalidDomainMap[Stmt.getEntryBlock()]);
    else
      Stmt.setInvalidDomain(InvalidDomainMap[getRegionNodeBasicBlock(
          Stmt.getRegion()->getNode())]);

  // Remove empty statements.
  // Exit early in case there are no executable statements left in this scop.
  scop->removeStmtNotInDomainMap();
  scop->simplifySCoP(false);
  if (scop->isEmpty()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because SCoP is empty\n");
    return;
  }

  // The ScopStmts now have enough information to initialize themselves.
  for (ScopStmt &Stmt : *scop) {
    collectSurroundingLoops(Stmt);

    buildDomain(Stmt);
    buildAccessRelations(Stmt);

    if (DetectReductions)
      checkForReductions(Stmt);
  }

  // Check early for a feasible runtime context.
  if (!scop->hasFeasibleRuntimeContext()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because of unfeasible context (early)\n");
    return;
  }

  // Check early for profitability. Afterwards it cannot change anymore,
  // only the runtime context could become infeasible.
  if (!scop->isProfitable(UnprofitableScalarAccs)) {
    scop->invalidate(PROFITABLE, DebugLoc());
    LLVM_DEBUG(
        dbgs() << "Bailing-out because SCoP is not considered profitable\n");
    return;
  }

  scop->buildSchedule(LI);

  scop->finalizeAccesses();

  scop->realignParams();
  scop->addUserContext();

  // After the context was fully constructed, thus all our knowledge about
  // the parameters is in there, we add all recorded assumptions to the
  // assumed/invalid context.
  scop->addRecordedAssumptions();

  scop->simplifyContexts();
  if (!scop->buildAliasChecks(AA)) {
    LLVM_DEBUG(dbgs() << "Bailing-out because could not build alias checks\n");
    return;
  }

  scop->hoistInvariantLoads();
  scop->canonicalizeDynamicBasePtrs();
  scop->verifyInvariantLoads();
  scop->simplifySCoP(true);

  // Check late for a feasible runtime context because profitability did not
  // change.
  if (!scop->hasFeasibleRuntimeContext()) {
    LLVM_DEBUG(dbgs() << "Bailing-out because of unfeasible context (late)\n");
    return;
  }

#ifndef NDEBUG
  verifyUses(scop.get(), LI, DT);
#endif
}

ScopBuilder::ScopBuilder(Region *R, AssumptionCache &AC, AliasAnalysis &AA,
                         const DataLayout &DL, DominatorTree &DT, LoopInfo &LI,
                         ScopDetection &SD, ScalarEvolution &SE,
                         OptimizationRemarkEmitter &ORE)
    : AA(AA), DL(DL), DT(DT), LI(LI), SD(SD), SE(SE) {
  DebugLoc Beg, End;
  auto P = getBBPairForRegion(R);
  getDebugLocations(P, Beg, End);

  std::string Msg = "SCoP begins here.";
  ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ScopEntry", Beg, P.first)
           << Msg);

  buildScop(*R, AC, ORE);

  LLVM_DEBUG(dbgs() << *scop);

  if (!scop->hasFeasibleRuntimeContext()) {
    InfeasibleScops++;
    Msg = "SCoP ends here but was dismissed.";
    LLVM_DEBUG(dbgs() << "SCoP detected but dismissed\n");
    scop.reset();
  } else {
    Msg = "SCoP ends here.";
    ++ScopFound;
    if (scop->getMaxLoopDepth() > 0)
      ++RichScopFound;
  }

  if (R->isTopLevelRegion())
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ScopEnd", End, P.first)
             << Msg);
  else
    ORE.emit(OptimizationRemarkAnalysis(DEBUG_TYPE, "ScopEnd", End, P.second)
             << Msg);
}
