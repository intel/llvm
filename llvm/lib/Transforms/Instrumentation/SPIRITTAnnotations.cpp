//===---- SPIRITTAnnotations.cpp - SPIR Instrumental Annotations Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which adds instrumental calls to annotate SPIR
// synchronization instructions. This can be used for kernel profiling.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/SPIRITTAnnotations.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

/** Following functions are used for ITT instrumentation:
 * * * * * * * * * * *
 * Notify tools work-item execution has started
 *
 * /param[in] group_id Pointer to array of 3 integers that uniquely identify
 *  group withing a kernel
 * /param[in] wi_id Globally unique work-item id
 * /param[in] wg_size Number of work-items in given group
 *
 * void __itt_offload_wi_start(size_t* group_id, size_t wi_id,
 *                             uint32_t wg_size);
 * * * * * * * * * * *
 * Notify tools work-item execution resumed (e.g. after barrier)
 *
 * /param[in] group_id Pointer to array of 3 integers that uniquely identify
 *  group withing a kernel.
 * /param[in] wi_id Globally unique work-item id.
 *
 * void __itt_offload_wi_resume(size_t* group_id, size_t wi_id);
 * * * * * * * * * * *
 * Notify tools work-item execution has finished
 *
 * /param[in] group_id Pointer to array of 3 integers that uniquely identify
 *  group withing a kernel.
 * /param[in] wi_id Globally unique work-item id.
 *
 * void __itt_offload_wi_finish(size_t* group_id, size_t wi_id);
 * * * * * * * * * * *
 * Notify tools work-item has reached a barrier
 *
 * /param[in] barrier_id Unique barrier id. If multi-barriers are not supported.
 * Pass 0 for barrier_id. Notify tools work-item has reached a barier.
 *
 * void __itt_offload_wg_barrier(uintptr_t barrier_id);
 * * * * * * * * * * *
 * Purpose of this pass is to add wrapper calls to these instructions.
 * Also this pass adds annotations to atomic instructions:
 * * * * * * * * * * *
 * Atomic operation markup
 *
 * /param[in] object Memory location which is used in atomic operation
 * /param[in] op_type Operation type
 * /param[in] mem_order Memory ordering semantic
 *
 * void __itt_offload_atomic_op_start(void* object,
 *                                    __itt_atomic_mem_op_t op_type,
 *                                    __itt_atomic_mem_order_t mem_order);
 * * * * * * * * * * *
 * Atomic operation markup
 *
 * /param[in] object Memory location which is used in atomic operation
 * /param[in] op_type Operation type
 * /param[in] mem_order Memory ordering semantic
 *
 * void __itt_offload_atomic_op_finish(void* object,
 *                                     __itt_atomic_mem_op_t op_type,
 *                                     __itt_atomic_mem_order_t mem_order);
 **/

using namespace llvm;

namespace {
constexpr char SPIRV_PREFIX[] = "__spirv_";
constexpr char SPIRV_CONTROL_BARRIER[] = "ControlBarrier";
constexpr char SPIRV_GROUP_ALL[] = "GroupAll";
constexpr char SPIRV_GROUP_ANY[] = "GroupAny";
constexpr char SPIRV_GROUP_BROADCAST[] = "GroupBroadcast";
constexpr char SPIRV_GROUP_IADD[] = "GroupIAdd";
constexpr char SPIRV_GROUP_FADD[] = "GroupFAdd";
constexpr char SPIRV_GROUP_FMIN[] = "GroupFMin";
constexpr char SPIRV_GROUP_UMIN[] = "GroupUMin";
constexpr char SPIRV_GROUP_SMIN[] = "GroupSMin";
constexpr char SPIRV_GROUP_FMAX[] = "GroupFMax";
constexpr char SPIRV_GROUP_UMAX[] = "GroupUMax";
constexpr char SPIRV_GROUP_SMAX[] = "GroupSMax";
constexpr char SPIRV_ATOMIC_INST[] = "Atomic";
constexpr char SPIRV_ATOMIC_LOAD[] = "AtomicLoad";
constexpr char SPIRV_ATOMIC_STORE[] = "AtomicStore";
constexpr char ITT_ANNOTATION_WI_START[] = "__itt_offload_wi_start_wrapper";
constexpr char ITT_ANNOTATION_WI_RESUME[] = "__itt_offload_wi_resume_wrapper";
constexpr char ITT_ANNOTATION_WI_FINISH[] = "__itt_offload_wi_finish_wrapper";
constexpr char ITT_ANNOTATION_WG_BARRIER[] = "__itt_offload_wg_barrier_wrapper";
constexpr char ITT_ANNOTATION_ATOMIC_START[] = "__itt_offload_atomic_op_start";
constexpr char ITT_ANNOTATION_ATOMIC_FINISH[] =
    "__itt_offload_atomic_op_finish";

// Wrapper for the pass to make it working with the old pass manager
class SPIRITTAnnotationsLegacyPass : public ModulePass {
public:
  static char ID;
  SPIRITTAnnotationsLegacyPass() : ModulePass(ID) {
    initializeSPIRITTAnnotationsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  // run the SPIRITTAnnotations pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  SPIRITTAnnotationsPass Impl;
};

} // namespace

char SPIRITTAnnotationsLegacyPass::ID = 0;
INITIALIZE_PASS(SPIRITTAnnotationsLegacyPass, "SPIRITTAnnotations",
                "Insert ITT annotations in SPIR code", false, false)

// Public interface to the SPIRITTAnnotationsPass.
ModulePass *llvm::createSPIRITTAnnotationsLegacyPass() {
  return new SPIRITTAnnotationsLegacyPass();
}

namespace {

// Check for calling convention of a function. Return true if it's SPIR kernel.
inline bool isSPIRKernel(Function &F) {
  return F.getCallingConv() == CallingConv::SPIR_KERNEL;
}

Instruction *emitCall(Module &M, Type *RetTy, StringRef FunctionName,
                      ArrayRef<Value *> Args, Instruction *InsertBefore) {
  SmallVector<Type *, 8> ArgTys(Args.size());
  for (unsigned I = 0; I < Args.size(); ++I)
    ArgTys[I] = Args[I]->getType();
  auto *FT = FunctionType::get(RetTy, ArgTys, false /*isVarArg*/);
  FunctionCallee FC = M.getOrInsertFunction(FunctionName, FT);
  assert(FC.getCallee() && "Instruction creation failed");
  Function *F = cast<Function>(FC.getCallee());
  F->setCallingConv(CallingConv::SPIR_FUNC);
  CallInst *NewCall =
      CallInst::Create(FT, FC.getCallee(), Args, "", InsertBefore);
  NewCall->setCallingConv(F->getCallingConv());
  return NewCall;
}

// Insert instrumental annotation calls, that has no arguments (for example
// work items start/finish/resume and barrier annotation.
void insertSimpleInstrumentationCall(Module &M, StringRef Name,
                                     Instruction *Position,
                                     const DebugLoc &DL) {
  Type *VoidTy = Type::getVoidTy(M.getContext());
  ArrayRef<Value *> Args;
  Instruction *InstrumentationCall = emitCall(M, VoidTy, Name, Args, Position);
  assert(InstrumentationCall && "Instrumentation call creation failed");
  InstrumentationCall->setDebugLoc(DL);
}

// Insert instrumental annotation calls for SPIR-V atomics.
bool insertAtomicInstrumentationCall(Module &M, StringRef Name,
                                     CallInst *AtomicFun, Instruction *Position,
                                     StringRef AtomicName) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  IntegerType *Int32Ty = IntegerType::get(Ctx, 32);
  // __spirv_Atomic... instructions have following arguments:
  // Pointer, Memory Scope, Memory Semantics and others. To construct Atomic
  // annotation instructions we need Pointer and Memory Semantic arguments
  // taken from the original Atomic instruction.
  Value *Ptr = dyn_cast<Value>(AtomicFun->getArgOperand(0));
  assert(Ptr && "Failed to get a pointer argument of atomic instruction");
  // Second parameter of Atomic Start/Finish annotation is an Op code of
  // the instruction, encoded into a value of enum, defined like this on user's/
  // profiler's side:
  // enum __itt_atomic_mem_op_t
  // {
  //   __itt_mem_load = 0,
  //   __itt_mem_store = 1,
  //   __itt_mem_update = 2
  // }
  ConstantInt *AtomicOp =
      StringSwitch<ConstantInt *>(AtomicName)
          .StartsWith(SPIRV_ATOMIC_LOAD, ConstantInt::get(Int32Ty, 0))
          .StartsWith(SPIRV_ATOMIC_STORE, ConstantInt::get(Int32Ty, 1))
          .Default(ConstantInt::get(Int32Ty, 2));
  // Third parameter of Atomic Start/Finish annotation is an ordering
  // semantic of the instruction, encoded into a value of enum, defined like
  // this on user's/profiler's side:
  // enum __itt_atomic_mem_order_t
  // {
  //   __itt_mem_order_relaxed = 0,        // SPIR-V 0x0
  //   __itt_mem_order_acquire = 1,        // SPIR-V 0x2
  //   __itt_mem_order_release = 2,        // SPIR-V 0x4
  //   __itt_mem_order_acquire_release = 3 // SPIR-V 0x8
  // }
  // which isn't 1:1 mapped on SPIR-V memory ordering mask (aside of a
  // differencies in values between SYCL mem order and SPIR-V mem order, SYCL RT
  // also applies Memory Semantic mask, like WorkgroupMemory (0x100)), need to
  // align it.
  auto *MemFlag = dyn_cast<ConstantInt>(AtomicFun->getArgOperand(2));
  // TODO: add non-constant memory order processing
  if (!MemFlag)
    return false;
  uint64_t IntMemFlag = MemFlag->getValue().getZExtValue();
  uint64_t Order;
  if (IntMemFlag & 0x2)
    Order = 1;
  else if (IntMemFlag & 0x4)
    Order = 2;
  else if (IntMemFlag & 0x8)
    Order = 3;
  else
    Order = 0;
  PointerType *Int8PtrAS4Ty = PointerType::get(IntegerType::get(Ctx, 8), 4);
  Ptr = CastInst::CreatePointerBitCastOrAddrSpaceCast(Ptr, Int8PtrAS4Ty, "",
                                                      Position);
  Value *MemOrder = ConstantInt::get(Int32Ty, Order);
  Value *Args[] = {Ptr, AtomicOp, MemOrder};
  Instruction *InstrumentationCall = emitCall(M, VoidTy, Name, Args, Position);
  assert(InstrumentationCall && "Instrumentation call creation failed");
  InstrumentationCall->setDebugLoc(AtomicFun->getDebugLoc());
  return true;
}

} // namespace

PreservedAnalyses SPIRITTAnnotationsPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  assert(StringRef(M.getTargetTriple()).startswith("spir"));
  bool IRModified = false;
  std::vector<StringRef> SPIRVCrossWGInstuctions = {
      SPIRV_CONTROL_BARRIER, SPIRV_GROUP_ALL,  SPIRV_GROUP_ANY,
      SPIRV_GROUP_BROADCAST, SPIRV_GROUP_IADD, SPIRV_GROUP_FADD,
      SPIRV_GROUP_FMIN,      SPIRV_GROUP_UMIN, SPIRV_GROUP_SMIN,
      SPIRV_GROUP_FMAX,      SPIRV_GROUP_UMAX, SPIRV_GROUP_SMAX};

  for (Function &F : M) {
    // Do not annotate:
    // - declarations
    // - ESIMD functions (TODO: consider enabling instrumentation)
    if (F.isDeclaration() || F.getMetadata("sycl_explicit_simd"))
      continue;

    // Work item start/finish annotations are only for SPIR kernels
    bool IsSPIRKernel = isSPIRKernel(F);

    // At the beggining of a kernel insert work item start annotation
    // instruction.
    if (IsSPIRKernel) {
      Instruction *InsertPt = &*inst_begin(F);
      if (InsertPt->isDebugOrPseudoInst())
        InsertPt = InsertPt->getNextNonDebugInstruction();
      assert(InsertPt && "Function does not have any real instructions.");
      insertSimpleInstrumentationCall(M, ITT_ANNOTATION_WI_START, InsertPt,
                                      InsertPt->getDebugLoc());
      IRModified = true;
    }

    for (BasicBlock &BB : F) {
      // Insert Finish instruction before return instruction
      if (IsSPIRKernel)
        if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
          insertSimpleInstrumentationCall(M, ITT_ANNOTATION_WI_FINISH, RI,
                                          RI->getDebugLoc());
          IRModified = true;
        }
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee)
          continue;
        StringRef CalleeName = Callee->getName();
        // Process only calls to functions which names starts with __spirv_
        size_t PrefixPosFound = CalleeName.find(SPIRV_PREFIX);
        if (PrefixPosFound == StringRef::npos)
          continue;
        CalleeName =
            CalleeName.drop_front(PrefixPosFound + /*len of SPIR-V prefix*/ 8);
        // Annotate barrier and other cross WG calls
        if (std::any_of(SPIRVCrossWGInstuctions.begin(),
                        SPIRVCrossWGInstuctions.end(),
                        [&CalleeName](StringRef Name) {
                          return CalleeName.startswith(Name);
                        })) {
          Instruction *InstAfterBarrier = CI->getNextNode();
          const DebugLoc &DL = CI->getDebugLoc();
          insertSimpleInstrumentationCall(M, ITT_ANNOTATION_WG_BARRIER, CI, DL);
          insertSimpleInstrumentationCall(M, ITT_ANNOTATION_WI_RESUME,
                                          InstAfterBarrier, DL);
          IRModified = true;
        } else if (CalleeName.startswith(SPIRV_ATOMIC_INST)) {
          Instruction *InstAfterAtomic = CI->getNextNode();
          IRModified |= insertAtomicInstrumentationCall(
              M, ITT_ANNOTATION_ATOMIC_START, CI, CI, CalleeName);
          IRModified |= insertAtomicInstrumentationCall(
              M, ITT_ANNOTATION_ATOMIC_FINISH, CI, InstAfterAtomic, CalleeName);
        }
      }
    }
  }

  return IRModified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
