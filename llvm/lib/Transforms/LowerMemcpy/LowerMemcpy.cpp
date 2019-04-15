//===- LowerMemcpy.cpp - ------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Lower aggregate copies, memset, memcpy, memmov intrinsics into loops when
// the size is large or is not a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "LowerMemcpy.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#define DEBUG_TYPE "nvptx"

using namespace llvm;

namespace {

// actual analysis class, which is a functionpass
struct LowerMemcpy : public InstVisitor<LowerMemcpy>,
                     public FunctionPass {

  void visit(Value& V) {
    if (Instruction *I = dyn_cast<Instruction>(&V)) {
      visit(*I);
    }
  }

  void visit(Instruction& I) {
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(&I)) {
      visitBitCastInst(*BCI);
    } else if (MemCpyInst *MCI = dyn_cast<MemCpyInst>(&I)) {
      visitMemCpyInst(*MCI);
    }
  }

  void visitBitCastInst(BitCastInst& I) {
    data_type = I.getSrcTy()->getPointerElementType();
    data_addr = I.getOperand(0);
  }

  void visitMemCpyInst(MemCpyInst& I) {
    Value *ori_src_addr = I.getRawSource();
    Value *ori_dst_addr = I.getRawDest();
    visit(*ori_src_addr);
    Type *src_type = data_type;
    Value *src_addr = data_addr;
    visit(*ori_dst_addr);
    Type *dst_type = data_type;
    Value *dst_addr = data_addr;

    ConstantInt *CI = dyn_cast<ConstantInt>(I.getLength());
    if (CI != nullptr) {
      createMemCpyLoopKnownSize(I, 
                                src_addr, src_type, 
                                dst_addr, dst_type,
                                CI);
    } else {
      createMemCpyLoopUnknownSize(I,
                                  src_addr, src_type,
                                  dst_addr, dst_type);
    }
  }

  void createMemCpyLoopKnownSize(MemCpyInst &I,
                                 Value *src_addr,
                                 Type *src_type,
                                 Value *dst_addr,
                                 Type *dst_type,
                                 ConstantInt *copy_len) {
    if (src_addr == nullptr || src_type == nullptr || 
        dst_addr == nullptr || dst_type == nullptr) 
      return;

    if (src_type != dst_type) 
      return;

    if (!src_type->isStructTy())
      return; 

    if (copy_len->isZero())
      return;

    BasicBlock *pre_loop_BB = I.getParent();
    BasicBlock *post_loop_BB = nullptr;
    Function *parent_fn = pre_loop_BB->getParent();
    LLVMContext &ctx = pre_loop_BB->getContext();
    auto dl = parent_fn->getParent()->getDataLayout();

    Type *iter_type = copy_len->getType();
    StructType *copy_type = dyn_cast<StructType>(src_type);
    auto *sdl = dl.getStructLayout(copy_type);
    uint64_t struct_size = sdl->getSizeInBytes();
    uint64_t loop_count = copy_len->getZExtValue()/struct_size;

 
    if (loop_count == 1) {
      IRBuilder<> InstBuilder(&I);
      Value *load = InstBuilder.CreateLoad(copy_type, src_addr);
      InstBuilder.CreateStore(load, dst_addr);
    } else if (loop_count > 0){
      //split
      post_loop_BB = pre_loop_BB->splitBasicBlock(&I, "memcpy_split");
      BasicBlock *loop_BB =
          BasicBlock::Create(ctx, "load-store-loop", parent_fn, post_loop_BB);
      pre_loop_BB->getTerminator()->setSuccessor(0, loop_BB);

      // fill-in loop BB
      // iter
      IRBuilder<> LoopBuilder(loop_BB);
      PHINode *loop_index = LoopBuilder.CreatePHI(iter_type, 2, "loop-index");
      loop_index->addIncoming(ConstantInt::get(iter_type, 0U), pre_loop_BB);
      // assignment
      Value *src_GEP =
          LoopBuilder.CreateInBoundsGEP(copy_type, src_addr, loop_index);
      Value *load = LoopBuilder.CreateLoad(copy_type, src_GEP);
      Value *dst_GEP =
          LoopBuilder.CreateInBoundsGEP(copy_type, dst_addr, loop_index);
      LoopBuilder.CreateStore(load, dst_GEP);
      // inc
      Value *new_index = 
          LoopBuilder.CreateAdd(loop_index, ConstantInt::get(iter_type, 1U));
      loop_index->addIncoming(new_index, loop_BB);
      // br
      Constant *loop_ci = ConstantInt::get(iter_type, loop_count);
      LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(new_index, loop_ci),
                               loop_BB, post_loop_BB);
      //metadata
      MDNode *loop_id = MDNode::get(ctx, MDString::get(ctx, "llvm.loop"));
      loop_BB->getTerminator()->setMetadata(LLVMContext::MD_loop, loop_id);
    }
  }

  void createMemCpyLoopUnknownSize(MemCpyInst &I,
                                 Value *src_addr,
                                 Type *src_type,
                                 Value *dst_addr,
                                 Type *dst_type) {
    if (src_addr == nullptr || src_type == nullptr || 
        dst_addr == nullptr || dst_type == nullptr) 
      return;

    if (src_type != dst_type) 
      return;

    if (!src_type->isStructTy())
      return; 

    Value *copy_len = I.getLength();

    BasicBlock *pre_loop_BB = I.getParent();
    BasicBlock *post_loop_BB = 
        pre_loop_BB->splitBasicBlock(&I, "post-loop-memcpy-expansion");
    Function *parent_fn = pre_loop_BB->getParent();
    LLVMContext &ctx = pre_loop_BB->getContext();
    auto &dl = parent_fn->getParent()->getDataLayout();

    IRBuilder<> PLBuilder(pre_loop_BB->getTerminator());

    Type *iter_type = copy_len->getType();
    IntegerType *i_len_ty = dyn_cast<IntegerType>(iter_type);
    StructType *copy_type = dyn_cast<StructType>(src_type);
    auto *sdl = dl.getStructLayout(copy_type);
    uint64_t struct_size = sdl->getSizeInBytes();
    ConstantInt *ci_struct_size = ConstantInt::get(i_len_ty, struct_size);
    Value *run_time_loop_cnt = PLBuilder.CreateUDiv(copy_len, ci_struct_size); 

    BasicBlock *loop_BB =
        BasicBlock::Create(ctx, "load-memcpy-expansion", parent_fn, post_loop_BB);
    pre_loop_BB->getTerminator()->setSuccessor(0, loop_BB);

    IRBuilder<> LoopBuilder(loop_BB);

    // fill-in loop BB
    // iter
    PHINode *loop_index = LoopBuilder.CreatePHI(iter_type, 2, "loop-index");
    loop_index->addIncoming(ConstantInt::get(iter_type, 0U), pre_loop_BB);
    // assignment
    Value *src_GEP =
        LoopBuilder.CreateInBoundsGEP(copy_type, src_addr, loop_index);
    Value *load = LoopBuilder.CreateLoad(copy_type, src_GEP);
    Value *dst_GEP =
        LoopBuilder.CreateInBoundsGEP(copy_type, dst_addr, loop_index);
    LoopBuilder.CreateStore(load, dst_GEP);
    // inc
    Value *new_index = 
        LoopBuilder.CreateAdd(loop_index, ConstantInt::get(iter_type, 1U));
    loop_index->addIncoming(new_index, loop_BB);
    // br
    LoopBuilder.CreateCondBr(LoopBuilder.CreateICmpULT(new_index, run_time_loop_cnt),
                             loop_BB, post_loop_BB);
    //metadata
    MDNode *loop_id = MDNode::get(ctx, MDString::get(ctx, "llvm.loop"));
    loop_BB->getTerminator()->setMetadata(LLVMContext::MD_loop, loop_id);
  }

  Type *data_type;
  Value *data_addr;

  static char ID;

  LowerMemcpy() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<StackProtector>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "Lower struct memcpys into loops";
  }
};

char LowerMemcpy::ID = 0;
static RegisterPass<LowerMemcpy> X("lower-memcpy", "lower memcpy to for loops");

bool LowerMemcpy::runOnFunction(Function &F) {
  SmallVector<MemCpyInst *, 4> memcpy_calls;

  // const DataLayout &DL = F.getParent()->getDataLayout();
  // LLVMContext &Context = F.getParent()->getContext();
  // const TargetTransformInfo &TTI =
  //    getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);

  // Collect all aggregate loads and mem* calls.
  for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
    for (BasicBlock::iterator II = BI->begin(), IE = BI->end(); II != IE;
         ++II) {
      if (MemCpyInst *memcpy_call = dyn_cast<MemCpyInst>(II)) {
          memcpy_calls.push_back(memcpy_call); 
      }
    }
  }

  if (memcpy_calls.size() == 0) {
    return false;
  }

  // Transform memcpy calls.
  for (MemCpyInst *memcpy_call : memcpy_calls) {
    visitMemCpyInst(*memcpy_call);
    memcpy_call->eraseFromParent();
  }

  return true;
}

} // namespace

namespace llvm {
void initializeLowerMemcpyPass(PassRegistry &);
}

// use hello.cpp
INITIALIZE_PASS(LowerMemcpy, "lowememcpy",
                "Lower struct llvm.mem* intrinsics into loops",
                false, false)

//FunctionPass *llvm::createMemcpy() {
////  return new LowerMemcpy();
//    return nullptr;
//}
