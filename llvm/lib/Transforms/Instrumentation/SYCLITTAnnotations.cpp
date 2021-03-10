//===---- SYCLITTAnnotations.cpp - SYCL Instrumental Annotations Pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A transformation pass which adds instrumental calls to annotate SYCL
// synchronization instructions. This can be used for kernel profiling.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/SYCLITTAnnotations.h"

#include "llvm/InitializePasses.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Function.h"

/** Following instrumentations will be linked from libdevice:
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
 * Notify tools work-item has reached a barier
 *
 * /param[in] barrier_id Unique barrier id. If multi-barriers are not supported.
 * Pass 0 for barrier_id. Notify tools work-item has reached a barier.
 *
 * void __itt_offload_wg_barrier(uintptr_t barrier_id);
 * * * * * * * * * * *
 * Purpose of this pass is to add wrapper calls to these instructions.
 */

using namespace llvm;

namespace {
constexpr char SPIRV_CONTROL_BARRIER[] = "__spirv_ControlBarrier";
constexpr char SPIRV_GROUP_ALL[] = "__spirv_GroupAll";
constexpr char SPIRV_GROUP_ANY[] = "__spirv_GroupAny";
constexpr char SPIRV_GROUP_BROADCAST[] = "__spirv_GroupBroadcast";
constexpr char SPIRV_GROUP_IADD[] = "__spirv_GroupIAdd";
constexpr char SPIRV_GROUP_FADD[] = "__spirv_GroupFAdd";
constexpr char SPIRV_GROUP_FMIN[] = "__spirv_GroupFMin";
constexpr char SPIRV_GROUP_UMIN[] = "__spirv_GroupUMin";
constexpr char SPIRV_GROUP_SMIN[] = "__spirv_GroupSMin";
constexpr char SPIRV_GROUP_FMAX[] = "__spirv_GroupFMax";
constexpr char SPIRV_GROUP_UMAX[] = "__spirv_GroupUMax";
constexpr char SPIRV_GROUP_SMAX[] = "__spirv_GroupSMax";
constexpr char SPIRV_ATOMIC_INST[] = "__spirv_Atomic";
constexpr char SPIRV_ATOMIC_LOAD[] = "__spirv_AtomicLoad";
constexpr char SPIRV_ATOMIC_STORE[] = "__spirv_AtomicSTORE";
constexpr char ITT_ANNOTATION_WI_START[] = "__itt_offload_wi_start_wrapper";
constexpr char ITT_ANNOTATION_WI_RESUME[] = "__itt_offload_wi_resume_wrapper";
constexpr char ITT_ANNOTATION_WI_FINISH[] = "__itt_offload_wi_finish_wrapper";
constexpr char ITT_ANNOTATION_WG_BARRIER[] = "__itt_offload_wg_barrier_wrapper";
constexpr char ITT_ANNOTATION_ATOMIC_START[] = "__itt_offload_atomic_op_start";
constexpr char ITT_ANNOTATION_ATOMIC_FINISH[] =
    "__itt_offload_atomic_op_finish";

// Wrapper for the pass to make it working with the old pass manager
class SYCLITTAnnotationsLegacyPass : public ModulePass {
public:
  static char ID;
  SYCLITTAnnotationsLegacyPass() : ModulePass(ID) {
    initializeSYCLITTAnnotationsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  // run the SYCLITTAnnotations pass on the specified module
  bool runOnModule(Module &M) override {
    ModuleAnalysisManager MAM;
    auto PA = Impl.run(M, MAM);
    return !PA.areAllPreserved();
  }

private:
  SYCLITTAnnotationsPass Impl;
};

} // namespace

char SYCLITTAnnotationsLegacyPass::ID = 0;
INITIALIZE_PASS(SYCLITTAnnotationsLegacyPass, "SYCLITTAnnotations",
                "Insert ITT annotations in SYCL code", false, false)

// Public interface to the SYCLITTAnnotationsPass.
ModulePass *llvm::createSYCLITTAnnotationsPass() {
  return new SYCLITTAnnotationsLegacyPass();
}

namespace {

// Check for calling convention of a function. If it's spir_kernel - consider
// the function to be a SYCL kernel.
bool isSyclKernel(Function &F) {
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
  auto *Call = CallInst::Create(FT, FC.getCallee(), Args, "", InsertBefore);
  return Call;
}

// Insert instrumental annotation calls, that has no arguments (for example
// work items start/finish/resume and barrier annotation.
bool insertSimpleInstrumentationCall(Module &M, StringRef Name,
                                     Instruction *Position) {
  Type *VoidTy = Type::getVoidTy(M.getContext());
  ArrayRef<Value *> Args;
  Instruction *InstrumentationCall =
      emitCall(M, VoidTy, Name, Args, Position);
  assert(InstrumentationCall && "Instrumentation call creation failed");
  return true;
}

// Insert instrumental annotation calls for SPIR-V atomics.
bool insertAtomicInstrumentationCall(Module &M, StringRef Name,
                                     CallInst *AtomicFun,
                                     Instruction *Position) {
  LLVMContext &Ctx = M.getContext();
  Type *VoidTy = Type::getVoidTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  // __spirv_Atomic... instructions have following arguments:
  // Pointer, Memory Scope, Memory Semantics and others. To construct Atomic
  // annotation instructions we need Pointer and Memory Semantic arguments
  // taken from the original Atomic instruction.
  Value *Ptr = dyn_cast<Value>(AtomicFun->getArgOperand(0));
  StringRef AtomicName = AtomicFun->getName();
  Value *AtomicOp;
  // Second parameter of Atomic Start/Finish annotation is an Op code of
  // the instruction, encoded into a value of enum, defined like this on user's/
  // profiler's side:
  // enum __itt_atomic_mem_op_t
  // {
  //   __itt_mem_load = 0,
  //   __itt_mem_store = 1,
  //   __itt_mem_update = 2
  // }
  if (AtomicName.contains(SPIRV_ATOMIC_LOAD))
    AtomicOp = ConstantInt::get(Int32Ty, 0);
  else if (AtomicName.contains(SPIRV_ATOMIC_STORE))
    AtomicOp = ConstantInt::get(Int32Ty, 1);
  else
    AtomicOp = ConstantInt::get(Int32Ty, 2);
  // TODO: Third parameter of Atomic Start/Finish annotation is an ordering
  // semantic of the instruction, encoded into a value of enum, defined like
  // this on user's/profiler's side:
  // enum __itt_atomic_mem_order_t
  // {
  //   __itt_mem_order_relaxed = 0,
  //   __itt_mem_order_acquire = 1,
  //   __itt_mem_order_release = 2
  // }
  // which isn't 1:1 mapped on SPIR-V memory ordering mask, need to align it.
  ConstantInt *MemSemantic = dyn_cast<ConstantInt>(AtomicFun->getArgOperand(2));
  Value *Args[] = {Ptr, AtomicOp, MemSemantic};
  Instruction *InstrumentationCall =
      emitCall(M, VoidTy, Name, Args, Position);
  assert(InstrumentationCall && "Instrumentation call creation failed");
  return true;
}

} // namespace

PreservedAnalyses SYCLITTAnnotationsPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  bool IRModified = false;
  std::vector<StringRef> SPIRVCrossWGInstuctions = {
      SPIRV_CONTROL_BARRIER, SPIRV_GROUP_ALL, SPIRV_GROUP_ANY,
      SPIRV_GROUP_BROADCAST, SPIRV_GROUP_IADD, SPIRV_GROUP_FADD,
      SPIRV_GROUP_FMIN, SPIRV_GROUP_UMIN, SPIRV_GROUP_SMIN, SPIRV_GROUP_FMAX,
      SPIRV_GROUP_UMAX, SPIRV_GROUP_SMAX };

  for (Function &F : M) {
    // Annotate only SYCL kernels
    if (F.isDeclaration() || !isSyclKernel(F))
      continue;

    // At the beggining of a kernel insert work item start annotation
    // instruction.
    IRModified |= insertSimpleInstrumentationCall(M, ITT_ANNOTATION_WI_START,
                                                  &*inst_begin(F));

    for (BasicBlock &BB : F) {
      // Insert Finish instruction before return instruction
      if (ReturnInst *RI = dyn_cast<ReturnInst>(BB.getTerminator()))
        IRModified |=
            insertSimpleInstrumentationCall(M, ITT_ANNOTATION_WI_FINISH, RI);
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI)
          continue;
        Function *Callee = CI->getCalledFunction();
        if (!Callee)
          continue;
        StringRef CalleeName = Callee->getName();
        // Annotate barrier and other cross WG calls
        if (std::any_of(SPIRVCrossWGInstuctions.begin(),
                        SPIRVCrossWGInstuctions.end(),
                        [&CalleeName](StringRef Name) {
                          return CalleeName.contains(Name);
                        })) {
          Instruction *InstAfterBarrier = CI->getNextNode();
          IRModified |= insertSimpleInstrumentationCall(
              M, ITT_ANNOTATION_WG_BARRIER, CI);
          IRModified |= insertSimpleInstrumentationCall(
              M, ITT_ANNOTATION_WI_RESUME, InstAfterBarrier);
        } else if (CalleeName.contains(SPIRV_ATOMIC_INST)) {
          Instruction *InstAfterAtomic = CI->getNextNode();
          IRModified |= insertAtomicInstrumentationCall(
              M, ITT_ANNOTATION_ATOMIC_START, CI, CI);
          IRModified |= insertAtomicInstrumentationCall(
              M, ITT_ANNOTATION_ATOMIC_FINISH, CI, InstAfterAtomic);
        }
      }
    }
  }

  return IRModified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
