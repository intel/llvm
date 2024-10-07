// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <compiler/utils/attributes.h>
#include <compiler/utils/barrier_regions.h>
#include <compiler/utils/builtin_info.h>
#include <compiler/utils/group_collective_helpers.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/pass_functions.h>
#include <compiler/utils/sub_group_analysis.h>
#include <compiler/utils/vectorization_factor.h>
#include <compiler/utils/work_item_loops_pass.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Local.h>
#include <multi_llvm/multi_llvm.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>

using namespace llvm;

#define NDEBUG_WI_LOOPS
#define DEBUG_TYPE "work-item-loops"

namespace compiler {
namespace utils {

/// @brief A subclass of the generic Barrier which is used by the
/// WorkItemLoopsPass.
///
/// It adds additional fields used when creating wrapper kernels.
class BarrierWithLiveVars : public Barrier {
 public:
  BarrierWithLiveVars(llvm::Module &m, llvm::Function &f,
                      VectorizationInfo vf_info, bool IsDebug)
      : Barrier(m, f, IsDebug), vf_info(vf_info) {}

  VectorizationInfo getVFInfo() const { return vf_info; }

  AllocaInst *getMemSpace() const { return mem_space; }
  void setMemSpace(AllocaInst *ai) { mem_space = ai; }

  void setSize0(Value *v) { size0 = v; }
  Value *getSize0() const { return size0; }

  void setTotalSize(Value *v) { totalSize = v; }
  Value *getTotalSize() const { return totalSize; }

  Value *getStructSize() const { return structSize; }
  void setStructSize(Value *v) { structSize = v; }

  AllocaInst *getDebugAddr() const { return debug_addr; }
  void setDebugAddr(AllocaInst *ai) { debug_addr = ai; }

 private:
  VectorizationInfo vf_info;

  // Alloca representing the memory for the live variables for a given kernel,
  // with enough space for each individual work-item in a work-group to have
  // its own view.
  //
  // This is typically used to hold Z*Y*(X/vec_width) individual instances of
  // the live-variables structure.
  AllocaInst *mem_space = nullptr;

  // Alloca holding the address of the live vars struct for the
  // currently executing work item.
  AllocaInst *debug_addr = nullptr;

  // The number of items along the primary dimension
  Value *size0 = nullptr;

  // The total number of items
  Value *totalSize = nullptr;

  /// @brief The size of the struct in bytes, if the barrier contains
  /// scalables
  Value *structSize = nullptr;
};

}  // namespace utils
}  // namespace compiler

namespace {
#ifndef NDEBUG_WI_LOOPS
/// @brief Generate IR level printf function call Debug function only.
///
/// @param[in] format Format string string.
/// @param[in] module Current module.
/// @param[in] v Value for printing.
/// @param[in] bb Basic block insertion point for @p v.
///
/// @return Return instruction to be checked.
Instruction *IRPrintf(const std::string format, Module &module, Value *v,
                      BasicBlock *bb) {
  LLVMContext &context = module.getContext();
  PointerType *ptr_type = PointerType::getUnqual(IntegerType::get(context, 8));

  SmallVector<Type *, 16> args;
  args.push_back(ptr_type);
  FunctionType *printf_type =
      FunctionType::get(IntegerType::get(context, 32), args, true);

  bool isDeclared = true;
  Function *func_printf = module.getFunction("printf");
  if (!func_printf) {
    func_printf = Function::Create(printf_type, GlobalValue::ExternalLinkage,
                                   "printf", &module);
    isDeclared = false;
  }

  ArrayType *array_type =
      ArrayType::get(IntegerType::get(context, 8), format.size() + 1);
  GlobalVariable *str;
  if (isDeclared) {
    str = new GlobalVariable(
        module, array_type, true, GlobalValue::PrivateLinkage, 0, ".str",
        nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 2, false);
  } else {
    str = new GlobalVariable(
        module, array_type, true, GlobalValue::PrivateLinkage, 0, ".str",
        nullptr, GlobalValue::ThreadLocalMode::NotThreadLocal, 0, false);
  }
  str->setAlignment(MaybeAlign(1));

  Constant *const_array = ConstantDataArray::getString(context, format, true);
  SmallVector<Constant *, 16> indices;
  ConstantInt *cst_8 = ConstantInt::get(context, APInt(64, StringRef("0"), 10));
  indices.push_back(cst_8);
  indices.push_back(cst_8);
  Constant *cst_ptr = ConstantExpr::getGetElementPtr(nullptr, str, indices);

  str->setInitializer(const_array);

  SmallVector<Value *, 8> call_params;
  call_params.push_back(cst_ptr);
  call_params.push_back(v);

  CallInst *call = CallInst::Create(func_printf, call_params, "", bb);

  return call;
}
#endif  // NDEBUG_WI_LOOPS

Value *materializeVF(IRBuilder<> &builder,
                     compiler::utils::VectorizationFactor vf) {
  auto &m = *builder.GetInsertBlock()->getModule();
  Constant *multiple =
      ConstantInt::get(compiler::utils::getSizeType(m), vf.getKnownMin());
  return !vf.isScalable() ? multiple : builder.CreateVScale(multiple);
}

struct ScheduleGenerator {
  ScheduleGenerator(Module &m,
                    const compiler::utils::BarrierWithLiveVars &barrierMain,
                    const compiler::utils::BarrierWithLiveVars *barrierTail,
                    compiler::utils::BuiltinInfo &BI)
      : module(m),
        context(m.getContext()),
        barrierMain(barrierMain),
        barrierTail(barrierTail),
        BI(BI),
        i32Ty(Type::getInt32Ty(context)) {
    set_local_id =
        BI.getOrDeclareMuxBuiltin(compiler::utils::eMuxBuiltinSetLocalId, m);
    set_subgroup_id =
        BI.getOrDeclareMuxBuiltin(compiler::utils::eMuxBuiltinSetSubGroupId, m);
    assert(set_local_id && set_subgroup_id && "Missing mux builtins");
  }
  Module &module;
  LLVMContext &context;
  const compiler::utils::BarrierWithLiveVars &barrierMain;
  const compiler::utils::BarrierWithLiveVars *barrierTail;
  compiler::utils::BuiltinInfo &BI;

  SmallVector<Value *, 8> args;
  Function *set_local_id = nullptr;
  Function *set_subgroup_id = nullptr;
  Type *const i32Ty;

  uint32_t workItemDim0 = 0;
  uint32_t workItemDim1 = 1;
  uint32_t workItemDim2 = 2;
  Value *localSizeDim[3];

  AllocaInst *nextID = nullptr;
  Value *mainLoopLimit = nullptr;
  Value *peel = nullptr;
  bool emitTail = true;
  bool isVectorPredicated = false;
  bool wrapperHasMain = false;
  bool wrapperHasTail = false;

  DILocation *wrapperDbgLoc = nullptr;

  Value *createLinearLiveVarsPtr(
      const compiler::utils::BarrierWithLiveVars &barrier, IRBuilder<> &ir,
      Value *index) {
    Value *const mem_space = barrier.getMemSpace();
    if (!mem_space) {
      return nullptr;
    }

    // Calculate the offset for where the live variables of the current
    // work item (within the nested loops) are stored.
    // Loop i,j,k  -->  ((i * dim1) + j) * size0 + k
    // memory access pattern should not depend on the vectorization
    // dimension

    Value *live_var_ptr;
    if (!barrier.getStructSize()) {
      Value *const live_var_mem_idxs[] = {index};
      live_var_ptr = ir.CreateInBoundsGEP(barrier.getLiveVarsType(), mem_space,
                                          live_var_mem_idxs);
    } else {
      // index into the byte buffer
      auto *const byteOffset = ir.CreateMul(index, barrier.getStructSize());
      Value *const live_var_mem_idxs[] = {byteOffset};
      live_var_ptr =
          ir.CreateInBoundsGEP(ir.getInt8Ty(), mem_space, live_var_mem_idxs);

      // cast to the live mem type
      live_var_ptr = ir.CreatePointerCast(
          live_var_ptr,
          PointerType::get(
              barrier.getLiveVarsType(),
              cast<PointerType>(live_var_ptr->getType())->getAddressSpace()));
    }

    return live_var_ptr;
  }

  Value *createLiveVarsPtr(const compiler::utils::BarrierWithLiveVars &barrier,
                           IRBuilder<> &ir, Value *dim_0, Value *dim_1,
                           Value *dim_2, Value *VF = nullptr) {
    Value *const mem_space = barrier.getMemSpace();
    if (!mem_space) {
      return nullptr;
    }

    // Calculate the offset for where the live variables of the current
    // work item (within the nested loops) are stored.
    // Loop i,j,k  -->  ((i * dim1) + j) * size0 + k
    // memory access pattern should not depend on the vectorization
    // dimension
    auto *const i_offset = ir.CreateMul(dim_2, localSizeDim[workItemDim1]);
    auto *const j_offset =
        ir.CreateMul(ir.CreateAdd(i_offset, dim_1), barrier.getSize0());
    auto *const k_offset = VF ? ir.CreateUDiv(dim_0, VF) : dim_0;
    auto *const offset = ir.CreateAdd(j_offset, k_offset);

    return createLinearLiveVarsPtr(barrier, ir, offset);
  }

  void recreateDebugIntrinsics(
      const compiler::utils::BarrierWithLiveVars &barrier, BasicBlock *block,
      StoreInst *SI) {
    DIBuilder DIB(module, /*AllowUnresolved*/ false);
    auto RecreateDebugIntrinsic = [&](DILocalVariable *const old_var,
                                      const unsigned live_var_offset) {
      const uint64_t dwPlusOp = dwarf::DW_OP_plus_uconst;
      // Use a DWARF expression to point to byte offset in struct where
      // the variable lives. This involves dereferencing the pointer
      // stored in `live_vars_debug_addr` to get the start of the live
      // vars struct, then using a byte offset into the struct for the
      // particular variable.
      auto expr = DIB.createExpression(
          ArrayRef<uint64_t>{dwarf::DW_OP_deref, dwPlusOp, live_var_offset});
      // Remap this debug variable to its new scope.
      auto *new_var = DIB.createAutoVariable(
          block->getParent()->getSubprogram(), old_var->getName(),
          old_var->getFile(), old_var->getLine(), old_var->getType(),
          /*AlwaysPreserve=*/false, DINode::FlagZero,
          old_var->getAlignInBits());
      // Create intrinsic
#if LLVM_VERSION_GREATER_EQUAL(19, 0)
      if (!module.IsNewDbgInfoFormat) {
        auto *const DII = DIB.insertDeclare(barrier.getDebugAddr(), new_var,
                                            expr, wrapperDbgLoc, block)
                              .get<Instruction *>();

        // Bit of a HACK to produce the same debug output as the Mem2Reg
        // pass used to do.
        auto *const DVIntrinsic = cast<DbgVariableIntrinsic>(DII);
        ConvertDebugDeclareToDebugValue(DVIntrinsic, SI, DIB);
      } else {
        auto *const DVR = static_cast<DbgVariableRecord *>(
            DIB.insertDeclare(barrier.getDebugAddr(), new_var, expr,
                              wrapperDbgLoc, block)
                .get<DbgRecord *>());

        // This is nasty, but LLVM errors out on trailing debug info, we need a
        // subsequent instruction even if we delete it immediately afterwards.
        auto *DummyInst = new UnreachableInst(module.getContext(), block);

        // Bit of a HACK to produce the same debug output as the Mem2Reg
        // pass used to do.
        ConvertDebugDeclareToDebugValue(DVR, SI, DIB);

        DummyInst->eraseFromParent();
      }
#else
      auto *const DII = DIB.insertDeclare(barrier.getDebugAddr(), new_var, expr,
                                          wrapperDbgLoc, block);

      // Bit of a HACK to produce the same debug output as the Mem2Reg
      // pass used to do.
      auto *const DVIntrinsic = cast<DbgVariableIntrinsic>(DII);
      ConvertDebugDeclareToDebugValue(DVIntrinsic, SI, DIB);
#endif
    };
    for (auto debug_pair : barrier.getDebugIntrinsics()) {
      RecreateDebugIntrinsic(debug_pair.first->getVariable(),
                             debug_pair.second);
    }
#if LLVM_VERSION_GREATER_EQUAL(19, 0)
    for (auto debug_pair : barrier.getDebugDbgVariableRecords()) {
      RecreateDebugIntrinsic(debug_pair.first->getVariable(),
                             debug_pair.second);
    }
#endif
  }

  void createWorkItemLoopBody(
      const compiler::utils::BarrierWithLiveVars &barrier, IRBuilder<> &ir,
      BasicBlock *block, unsigned i, Value *dim_0, Value *dim_1, Value *dim_2,
      Value *accumulator = nullptr, Value *VF = nullptr,
      Value *offset = nullptr) {
    auto new_kernel_args = args;
    if (accumulator) {
      new_kernel_args.push_back(accumulator);
    }

    // If the work item ID is a nullptr we take it to mean this barrier region
    // doesn't need to use the barrier struct.
    if (dim_0) {
      assert(dim_1 && dim_2 && "unexpected null Work item IDs");

      // set our local id
      auto *const local_id = offset ? ir.CreateAdd(offset, dim_0) : dim_0;
      ir.CreateCall(set_local_id,
                    {ConstantInt::get(i32Ty, workItemDim0), local_id})
          ->setCallingConv(set_local_id->getCallingConv());

      auto *const live_var_ptr =
          createLiveVarsPtr(barrier, ir, dim_0, dim_1, dim_2, VF);
      if (live_var_ptr) {
        new_kernel_args.push_back(live_var_ptr);

        if (auto *debug_addr = barrier.getDebugAddr()) {
          // Update the alloca holding the address of the live vars struct for
          // currently executing work item.
          auto *const live_var_ptr_cast =
              ir.CreatePointerBitCastOrAddrSpaceCast(
                  live_var_ptr, debug_addr->getAllocatedType());
          auto *const SI = ir.CreateStore(live_var_ptr_cast, debug_addr);

          // Recreate all the debug intrinsics pointing at location in live
          // variables struct. We only need to do this once before the first
          // barrier.
          if (i == compiler::utils::kBarrier_FirstID) {
            recreateDebugIntrinsics(barrier, block, SI);
          }
        }
      }
    }

    auto &subkernel = *barrier.getSubkernel(i);

    // call the original function now we've setup all the info!
    CallInst *ci = ir.CreateCall(&subkernel, new_kernel_args);
    // add a debug location for this call so that later inlining correctly
    // updates the debug metadata of all inlined instructions.
    if (wrapperDbgLoc) {
      ci->setDebugLoc(wrapperDbgLoc);
    }
    ci->setCallingConv(subkernel.getCallingConv());
    ci->setAttributes(compiler::utils::getCopiedFunctionAttrs(subkernel));

#ifndef NDEBUG_WI_LOOPS
    IRPrintf(std::string("return.kernel.body=%d\x0A"), module, ci, block);
#endif  // NDEBUG_WI_LOOPS

    // And update the location of where we need to go to next (if we need to)
    const auto &successors = barrier.getSuccessorIds(i);
    if (successors.size() > 1) {
      ir.CreateStore(ci, nextID);
    }
  }

  // Create a 1D loop to execute all the work items in a 'barrier', reducing
  // across an accumulator.
  std::pair<BasicBlock *, Value *> makeReductionLoop(
      const compiler::utils::BarrierWithLiveVars &barrier,
      const compiler::utils::GroupCollective &WGC, BasicBlock *block, Value *op,
      Value *accumulator) {
    auto *const accTy = accumulator->getType();
    Function *const func = block->getParent();

    // Induction variables
    auto *const totalSize = barrier.getTotalSize();

    compiler::utils::CreateLoopOpts inner_opts;
    inner_opts.IVs = {accumulator};
    inner_opts.disableVectorize = true;

    BasicBlock *preheader = block;
    BasicBlock *exitBlock = nullptr;
    PHINode *resultPhi = nullptr;

    auto *const zero =
        Constant::getNullValue(compiler::utils::getSizeType(module));

    if (auto *const loopLimitConst = dyn_cast<Constant>(totalSize)) {
      if (loopLimitConst->isZeroValue()) {
        // No iterations at all!
        return {block, accumulator};
      }
      preheader = block;
    } else {
      preheader =
          BasicBlock::Create(context, "ca_work_group_reduce_preheader", func);

      exitBlock =
          BasicBlock::Create(context, "ca_work_group_reduce_exit", func);
      preheader->moveAfter(block);
      exitBlock->moveAfter(preheader);

      auto *const needLoop = CmpInst::Create(
          Instruction::ICmp, CmpInst::ICMP_NE, zero, totalSize, "", block);

      BranchInst::Create(preheader, exitBlock, needLoop, block);

      resultPhi = PHINode::Create(accTy, 2, "WGC_reduce", exitBlock);
      resultPhi->addIncoming(accumulator, block);
    }

    BasicBlock *latchBlock = nullptr;

    // linearly looping through the work items
    exitBlock = compiler::utils::createLoop(
        preheader, exitBlock, zero, totalSize, inner_opts,
        [&](BasicBlock *block, Value *index, ArrayRef<Value *> ivs,
            MutableArrayRef<Value *> ivsNext) -> BasicBlock * {
          IRBuilder<> ir(block);
          auto *const liveVars = createLinearLiveVarsPtr(barrier, ir, index);
          compiler::utils::Barrier::LiveValuesHelper live_values(barrier, block,
                                                                 liveVars);

          IRBuilder<> ir_load(block);
          auto *const itemOp =
              live_values.getReload(op, ir_load, "_load", /*reuse*/ true);

          // Do the reduction here..
          accumulator = compiler::utils::createBinOpForRecurKind(
              ir, ivs[0], itemOp, WGC.Recurrence);
          ivsNext[0] = accumulator;
          latchBlock = block;

          return block;
        });

    if (!resultPhi) {
      assert(exitBlock != latchBlock && "createLoop didn't create a loop");
      resultPhi = PHINode::Create(accTy, 1, "WGC_reduce", exitBlock);
    }
    resultPhi->addIncoming(accumulator, latchBlock);
    return {exitBlock, resultPhi};
  }

  void getUniformValues(BasicBlock *block,
                        const compiler::utils::BarrierWithLiveVars &barrier,
                        MutableArrayRef<Value *> values) {
    auto *const zero =
        Constant::getNullValue(compiler::utils::getSizeType(module));
    IRBuilder<> ir(block);
    auto *const barrier0 = ir.CreateInBoundsGEP(barrier.getLiveVarsType(),
                                                barrier.getMemSpace(), {zero});
    compiler::utils::Barrier::LiveValuesHelper live_values(barrier, block,
                                                           barrier0);
    for (auto &value : values) {
      value = live_values.getReload(value, ir, "_load", true);
    }
  }

  std::optional<compiler::utils::GroupCollective> getBarrierGroupCollective(
      const compiler::utils::BarrierWithLiveVars &Barrier, unsigned BarrierID) {
    auto *const BarrierCall = Barrier.getBarrierCall(BarrierID);
    if (!BarrierCall) {
      return std::nullopt;
    }

    auto Builtin = BI.analyzeBuiltin(*BarrierCall->getCalledFunction());
    return BI.isMuxGroupCollective(Builtin.ID);
  }

  std::tuple<BasicBlock *, Value *,
             std::optional<compiler::utils::GroupCollective>>
  makeWorkGroupCollectiveLoops(BasicBlock *block, unsigned barrierID) {
    auto *const groupCall = barrierMain.getBarrierCall(barrierID);
    if (!groupCall) {
      return {block, nullptr, std::nullopt};
    }

    auto Info = getBarrierGroupCollective(barrierMain, barrierID);
    if (!Info || !Info->isWorkGroupScope()) {
      return {block, nullptr, std::nullopt};
    }

    switch (Info->Op) {
      case compiler::utils::GroupCollective::OpKind::Reduction:
      case compiler::utils::GroupCollective::OpKind::All:
      case compiler::utils::GroupCollective::OpKind::Any: {
        auto *const ty = groupCall->getType();
        auto *const accumulator =
            compiler::utils::getNeutralVal(Info->Recurrence, ty);
        auto [loop_exit_block, accum] = makeReductionLoop(
            barrierMain, *Info, block, groupCall->getOperand(1), accumulator);
        if (barrierTail) {
          auto *const groupTailInst = barrierTail->getBarrierCall(barrierID);
          std::tie(loop_exit_block, accum) =
              makeReductionLoop(*barrierTail, *Info, loop_exit_block,
                                groupTailInst->getOperand(1), accum);
        }
        if (groupCall->hasName()) {
          accum->takeName(groupCall);
        }
        return std::make_tuple(loop_exit_block, accum, Info);
      }
      case compiler::utils::GroupCollective::OpKind::ScanInclusive:
      case compiler::utils::GroupCollective::OpKind::ScanExclusive: {
        auto *const ty = groupCall->getType();
        auto *const accumulator =
            compiler::utils::getIdentityVal(Info->Recurrence, ty);
        return {block, accumulator, Info};
      }
      case compiler::utils::GroupCollective::OpKind::Broadcast: {
        // First we need to get the item ID values from the barrier struct.
        // These should be uniform but they may still be variables. It should
        // be safe to get them from the barrier struct at index zero.
        auto *const zero =
            Constant::getNullValue(compiler::utils::getSizeType(module));

        Function *const func = block->getParent();
        BasicBlock *mainUniformBlock = block;
        BasicBlock *tailUniformBlock = nullptr;

        auto *const totalSize = barrierMain.getTotalSize();
        if (auto *const loopLimitConst = dyn_cast<Constant>(totalSize)) {
          // If we know for a fact that the main struct has at least one item,
          // we can just use that. Otherwise, we need to use the tail struct.
          if (loopLimitConst->isZeroValue()) {
            mainUniformBlock = nullptr;
            if (barrierTail) {
              tailUniformBlock = block;
            }
          }
        } else if (barrierTail) {
          // If we have a variable number of main items, it could be zero at
          // runtime, so we need an alternative way to get the values.
          mainUniformBlock =
              BasicBlock::Create(context, "ca_main_uniform_load", func);
          tailUniformBlock =
              BasicBlock::Create(context, "ca_tail_uniform_load", func);

          auto *const needTail = CmpInst::Create(
              Instruction::ICmp, CmpInst::ICMP_EQ, totalSize, zero, "", block);
          BranchInst::Create(tailUniformBlock, mainUniformBlock, needTail,
                             block);
        }

        if (!mainUniformBlock && !tailUniformBlock) {
          return {block, nullptr, std::nullopt};
        }

        Value *idsMain[] = {zero, zero, zero};
        Value *idsTail[] = {zero, zero, zero};
        if (mainUniformBlock) {
          idsMain[0] = groupCall->getOperand(2);
          idsMain[1] = groupCall->getOperand(3);
          idsMain[2] = groupCall->getOperand(4);
          getUniformValues(mainUniformBlock, barrierMain, idsMain);
        }

        if (tailUniformBlock) {
          auto *const tailGroupCall = barrierTail->getBarrierCall(barrierID);
          assert(tailGroupCall &&
                 "No corresponding work group broadcast in tail kernel");
          idsTail[0] = tailGroupCall->getOperand(2);
          idsTail[1] = tailGroupCall->getOperand(3);
          idsTail[2] = tailGroupCall->getOperand(4);
          getUniformValues(tailUniformBlock, *barrierTail, idsTail);
        }

        // If both barrier structs had to be used, we need to merge the result.
        if (mainUniformBlock && tailUniformBlock) {
          block = BasicBlock::Create(context, "ca_merge_uniform_load", func);
          BranchInst::Create(block, tailUniformBlock);
          BranchInst::Create(block, mainUniformBlock);

          for (size_t i = 0; i != 3; ++i) {
            auto *mergePhi = PHINode::Create(idsMain[i]->getType(), 2,
                                             "uniform_merge", block);
            mergePhi->addIncoming(idsMain[i], mainUniformBlock);
            mergePhi->addIncoming(idsTail[i], tailUniformBlock);
            idsMain[i] = mergePhi;
          }
        }

        IRBuilder<> ir(block);
        auto *const op = groupCall->getOperand(1);

        // Compute the address of the value in the main barrier struct
        auto *const VF = materializeVF(ir, barrierMain.getVFInfo().vf);
        auto *const liveVars = createLiveVarsPtr(barrierMain, ir, idsMain[0],
                                                 idsMain[1], idsMain[2], VF);
        compiler::utils::Barrier::LiveValuesHelper live_values(barrierMain,
                                                               block, liveVars);
        auto *const GEPmain = live_values.getGEP(op);
        assert(GEPmain && "Could not get broadcasted value");

        if (barrierTail) {
          const bool VP = barrierTail->getVFInfo().IsVectorPredicated;

          // Compute the address of the value in the tail barrier struct
          auto *const offsetDim0 = ir.CreateSub(idsMain[0], mainLoopLimit);
          auto *const liveVarsTail =
              createLiveVarsPtr(*barrierTail, ir, offsetDim0, idsMain[1],
                                idsMain[2], VP ? VF : nullptr);
          compiler::utils::Barrier::LiveValuesHelper live_values(
              *barrierTail, block, liveVarsTail);

          auto *const opTail =
              barrierTail->getBarrierCall(barrierID)->getOperand(1);
          auto *const GEPtail = live_values.getGEP(opTail);
          assert(GEPtail && "Could not get tail-broadcasted value");

          // Select the main GEP or the tail GEP to load from
          auto *const cond = ir.CreateICmpUGE(idsMain[0], mainLoopLimit);

          auto *const select = ir.CreateSelect(cond, GEPtail, GEPmain);

          auto *const result = ir.CreateLoad(op->getType(), select);
          result->takeName(groupCall);

          return {block, result, Info};
        } else {
          auto *const result = ir.CreateLoad(op->getType(), GEPmain);
          result->takeName(groupCall);
          return {block, result, Info};
        }
      }
      default:
        break;
    }
    return {block, nullptr, std::nullopt};
  }

  // Create loops to execute all the main work items, and then all the
  // left-over tail work items at the end.
  BasicBlock *makeWorkItemLoops(BasicBlock *block, unsigned barrierID) {
    Value *accum = nullptr;
    std::optional<compiler::utils::GroupCollective> collective;
    std::tie(block, accum, collective) =
        makeWorkGroupCollectiveLoops(block, barrierID);

    // Work-group scans should be using linear work-item loops.
    assert((!collective || !collective->isScan()) && "No support for scans");

    auto *const zero =
        Constant::getNullValue(compiler::utils::getSizeType(module));
    auto *const i32Zero = Constant::getNullValue(i32Ty);
    auto *const func = block->getParent();

    // The subgroup induction variable, set to the value of the subgroup ID at
    // the end of the last loop (i.e. beginning of the next loop)
    Value *nextSubgroupIV = i32Zero;

    // looping through num groups in the first (innermost)
    // dimension
    BasicBlock *mainPreheaderBB = block;
    BasicBlock *mainExitBB = nullptr;

    // We need to ensure any subgroup IV is defined on the path in which
    // the vector loop is skipped.
    PHINode *subgroupMergePhi = nullptr;

    // If we are emitting a tail, we might need to bypass the vector loop (if
    // the local size is less than the vector width).
    if (emitTail) {
      if (auto *const loopLimitConst = dyn_cast<Constant>(mainLoopLimit)) {
        if (loopLimitConst->isZeroValue()) {
          // No vector iterations at all!
          mainPreheaderBB = nullptr;
          mainExitBB = block;
        }
      } else {
        mainPreheaderBB = BasicBlock::Create(
            context, "ca_work_item_x_vector_preheader", func);

        mainExitBB =
            BasicBlock::Create(context, "ca_work_item_x_vector_exit", func);
        mainPreheaderBB->moveAfter(block);
        mainExitBB->moveAfter(mainPreheaderBB);

        subgroupMergePhi = PHINode::Create(i32Ty, 2, "", mainExitBB);
        subgroupMergePhi->addIncoming(i32Zero, block);

        auto *const needMain =
            CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE, zero,
                            mainLoopLimit, "", block);

        BranchInst::Create(mainPreheaderBB, mainExitBB, needMain, block);
      }
    }

    assert((mainPreheaderBB || !wrapperHasMain) &&
           "Vector loops in one barrier block but not another?");

    if (mainPreheaderBB) {
      wrapperHasMain = true;
      // Subgroup induction variables
      compiler::utils::CreateLoopOpts outer_opts;
      outer_opts.IVs = {i32Zero};

      // looping through num groups in the third (outermost) dimension
      mainExitBB = compiler::utils::createLoop(
          mainPreheaderBB, mainExitBB, zero, localSizeDim[workItemDim2],
          outer_opts,
          [&](BasicBlock *block, Value *dim_2, ArrayRef<Value *> ivs2,
              MutableArrayRef<Value *> ivsNext2) -> BasicBlock * {
            // if we need to set the local id, do so here.
            IRBuilder<> ir(block);
            ir.CreateCall(set_local_id,
                          {ConstantInt::get(i32Ty, workItemDim2), dim_2})
                ->setCallingConv(set_local_id->getCallingConv());

            compiler::utils::CreateLoopOpts middle_opts;
            middle_opts.IVs = ivs2.vec();

            // looping through num groups in the second dimension
            BasicBlock *exit1 = compiler::utils::createLoop(
                block, nullptr, zero, localSizeDim[workItemDim1], middle_opts,
                [&](BasicBlock *block, Value *dim_1, ArrayRef<Value *> ivs1,
                    MutableArrayRef<Value *> ivsNext1) -> BasicBlock * {
                  IRBuilder<> ir(block);
                  ir.CreateCall(set_local_id,
                                {ConstantInt::get(i32Ty, workItemDim1), dim_1})
                      ->setCallingConv(set_local_id->getCallingConv());

                  // Materialize the scale factor at the beginning of the
                  // preheader
                  IRBuilder<> irph(mainPreheaderBB,
                                   mainPreheaderBB->getFirstInsertionPt());
                  auto *VF = materializeVF(irph, barrierMain.getVFInfo().vf);

                  compiler::utils::CreateLoopOpts inner_opts;
                  inner_opts.indexInc = VF;
                  inner_opts.IVs = ivs1.vec();

                  BasicBlock *exit0 = compiler::utils::createLoop(
                      block, nullptr, zero, mainLoopLimit, inner_opts,
                      [&](BasicBlock *block, Value *dim_0,
                          ArrayRef<Value *> ivs0,
                          MutableArrayRef<Value *> ivsNext0) -> BasicBlock * {
                        IRBuilder<> ir(block);

                        // set our subgroup id
                        ir.CreateCall(set_subgroup_id, {ivs0[0]})
                            ->setCallingConv(set_subgroup_id->getCallingConv());

                        createWorkItemLoopBody(barrierMain, ir, block,
                                               barrierID, dim_0, dim_1, dim_2,
                                               accum, VF);

                        nextSubgroupIV =
                            ir.CreateAdd(ivs0[0], ConstantInt::get(i32Ty, 1));
                        ivsNext0[0] = nextSubgroupIV;

                        return block;
                      });

                  // Don't forget to update the subgroup IV phi.
                  ivsNext1[0] = nextSubgroupIV;

                  return exit0;
                });

            // Don't forget to update the subgroup IV phi.
            ivsNext2[0] = nextSubgroupIV;

            if (subgroupMergePhi) {
              subgroupMergePhi->addIncoming(nextSubgroupIV, exit1);
            }

            return exit1;
          });
    }

    // looping through num groups in the first
    // (innermost) dimension
    BasicBlock *tailPreheaderBB = mainExitBB;
    BasicBlock *tailExitBB = nullptr;

    if (emitTail && peel) {
      // We might need to bypass the tail loop.
      if (auto *const peelConst = dyn_cast<Constant>(peel)) {
        if (peelConst->isZeroValue()) {
          // No tail iterations at all!
          tailPreheaderBB = nullptr;
          tailExitBB = mainExitBB;
        }
      } else {
        tailPreheaderBB = BasicBlock::Create(
            context, "ca_work_item_x_scalar_preheader", func);

        tailExitBB =
            BasicBlock::Create(context, "ca_work_item_x_scalar_exit", func);
        tailPreheaderBB->moveAfter(mainExitBB);
        tailExitBB->moveAfter(tailPreheaderBB);

        auto *const needPeeling = CmpInst::Create(
            Instruction::ICmp, CmpInst::ICMP_NE, zero, peel, "", mainExitBB);

        BranchInst::Create(tailPreheaderBB, tailExitBB, needPeeling,
                           mainExitBB);
      }
    } else {
      tailPreheaderBB = nullptr;
      tailExitBB = mainExitBB;
    }

    assert((tailPreheaderBB || !wrapperHasTail) &&
           "Tail loops in one barrier block but not another?");

    if (tailPreheaderBB) {
      assert(barrierTail);
      wrapperHasTail = true;
      // Subgroup induction variables
      compiler::utils::CreateLoopOpts outer_opts;
      outer_opts.IVs = {subgroupMergePhi ? subgroupMergePhi : nextSubgroupIV};

      // looping through num groups in the third (outermost) dimension
      tailExitBB = compiler::utils::createLoop(
          tailPreheaderBB, tailExitBB, zero, localSizeDim[workItemDim2],
          outer_opts,
          [&](BasicBlock *block, Value *dim_2, ArrayRef<Value *> ivs2,
              MutableArrayRef<Value *> ivsNext2) -> BasicBlock * {
            // set the local id
            IRBuilder<> ir(block);
            ir.CreateCall(set_local_id,
                          {ConstantInt::get(i32Ty, workItemDim2), dim_2})
                ->setCallingConv(set_local_id->getCallingConv());

            compiler::utils::CreateLoopOpts middle_opts;
            middle_opts.IVs = ivs2.vec();

            // looping through num groups in the second dimension
            BasicBlock *exit1 = compiler::utils::createLoop(
                block, nullptr, zero, localSizeDim[workItemDim1], middle_opts,
                [&](BasicBlock *block, Value *dim_1, ArrayRef<Value *> ivs1,
                    MutableArrayRef<Value *> ivsNext1) -> BasicBlock * {
                  IRBuilder<> ir(block);
                  ir.CreateCall(set_local_id,
                                {ConstantInt::get(i32Ty, workItemDim1), dim_1})
                      ->setCallingConv(set_local_id->getCallingConv());

                  compiler::utils::CreateLoopOpts inner_opts;
                  inner_opts.IVs = ivs1.vec();
                  inner_opts.disableVectorize = true;

                  BasicBlock *exit0 = compiler::utils::createLoop(
                      block, nullptr, zero, peel, inner_opts,
                      [&](BasicBlock *block, Value *dim_0,
                          ArrayRef<Value *> ivs0,
                          MutableArrayRef<Value *> ivsNext0) -> BasicBlock * {
                        IRBuilder<> ir(block);

                        if (set_subgroup_id) {
                          // set our subgroup id
                          ir.CreateCall(set_subgroup_id, {ivs0[0]})
                              ->setCallingConv(
                                  set_subgroup_id->getCallingConv());
                        }

                        createWorkItemLoopBody(
                            *barrierTail, ir, block, barrierID, dim_0, dim_1,
                            dim_2, accum, /*VF*/ nullptr, mainLoopLimit);

                        nextSubgroupIV =
                            ir.CreateAdd(ivs0[0], ConstantInt::get(i32Ty, 1));
                        ivsNext0[0] = nextSubgroupIV;

                        return block;
                      });

                  // Don't forget to update the subgroup IV phi.
                  ivsNext1[0] = nextSubgroupIV;

                  return exit0;
                });

            // Don't forget to update the subgroup IV phi.
            ivsNext2[0] = nextSubgroupIV;

            return exit1;
          });
    }
    return tailExitBB;
  }

  // Create loops to execute all work items in local linear ID order.
  BasicBlock *makeLinearWorkItemLoops(BasicBlock *block, unsigned barrierID) {
    Value *accum = nullptr;
    std::optional<compiler::utils::GroupCollective> collective;
    std::tie(block, accum, collective) =
        makeWorkGroupCollectiveLoops(block, barrierID);

    bool isScan = collective && collective->isScan();
    bool isExclusiveScan =
        isScan && collective->Op ==
                      compiler::utils::GroupCollective::OpKind::ScanExclusive;
    // The scan types can differ between 'main' and 'tail' kernels.
    bool isTailExclusiveScan = false;
    if (isScan && barrierTail) {
      const auto tailInfo = getBarrierGroupCollective(*barrierTail, barrierID);
      assert(tailInfo && "No corresponding work group scan in tail kernel");
      isTailExclusiveScan =
          tailInfo->Op ==
          compiler::utils::GroupCollective::OpKind::ScanExclusive;
    }

    auto *const zero =
        Constant::getNullValue(compiler::utils::getSizeType(module));
    auto *const i32Zero = Constant::getNullValue(i32Ty);
    auto *const func = block->getParent();

    // The subgroup induction variable, set to the value of the subgroup ID at
    // the end of the last loop (i.e. beginning of the next loop)
    Value *nextSubgroupIV = i32Zero;

    // The work-group scan induction variable, set to the current scan value at
    // the end of the last loop (i.e. beginning of the next loop)
    Value *nextScanIV = accum;

    // We need to ensure any subgroup IV is defined on the path in which
    // the vector loop is skipped.
    PHINode *subgroupMergePhi = nullptr;
    // Same with the scan IV
    PHINode *scanMergePhi = nullptr;

    compiler::utils::CreateLoopOpts outer_opts;
    outer_opts.IVs.push_back(i32Zero);
    outer_opts.loopIVNames.push_back("sg.z");
    if (isScan) {
      outer_opts.IVs.push_back(nextScanIV);
      outer_opts.loopIVNames.push_back("scan.z");
    }

    // looping through num groups in the third (outermost) dimension
    return compiler::utils::createLoop(
        block, nullptr, zero, localSizeDim[workItemDim2], outer_opts,
        [&](BasicBlock *block, Value *dim_2, ArrayRef<Value *> ivs2,
            MutableArrayRef<Value *> ivsNext2) -> BasicBlock * {
          // set the local id
          IRBuilder<> ir(block);
          ir.CreateCall(set_local_id,
                        {ConstantInt::get(i32Ty, workItemDim2), dim_2})
              ->setCallingConv(set_local_id->getCallingConv());

          compiler::utils::CreateLoopOpts middle_opts;
          middle_opts.IVs = ivs2.vec();
          middle_opts.loopIVNames.push_back("sg.y");
          if (isScan) {
            middle_opts.loopIVNames.push_back("scan.y");
          }

          // looping through num groups in the second dimension
          BasicBlock *exit1 = compiler::utils::createLoop(
              block, nullptr, zero, localSizeDim[workItemDim1], middle_opts,
              [&](BasicBlock *block, Value *dim_1, ArrayRef<Value *> ivs1,
                  MutableArrayRef<Value *> ivsNext1) -> BasicBlock * {
                IRBuilder<> ir(block);
                ir.CreateCall(set_local_id,
                              {ConstantInt::get(i32Ty, workItemDim1), dim_1})
                    ->setCallingConv(set_local_id->getCallingConv());

                // looping through num groups in the first (innermost)
                // dimension
                BasicBlock *mainPreheaderBB = block;
                BasicBlock *mainExitBB = nullptr;

                // If we are emitting a tail, we might need to bypass the
                // main loop (if the local size is less than the main loop
                // width).
                if (emitTail) {
                  if (auto *const loopLimitConst =
                          dyn_cast<Constant>(mainLoopLimit)) {
                    if (loopLimitConst->isZeroValue()) {
                      // No main iterations at all!
                      mainPreheaderBB = nullptr;
                      mainExitBB = block;
                    }
                  } else {
                    mainPreheaderBB = BasicBlock::Create(
                        context, "ca_work_item_x_main_preheader", func);

                    mainExitBB = BasicBlock::Create(
                        context, "ca_work_item_x_main_exit", func);
                    mainPreheaderBB->moveAfter(block);
                    mainExitBB->moveAfter(mainPreheaderBB);

                    subgroupMergePhi =
                        PHINode::Create(i32Ty, 2, "sg.merge", mainExitBB);
                    subgroupMergePhi->addIncoming(ivs1[0], block);

                    if (isScan) {
                      scanMergePhi = PHINode::Create(accum->getType(), 2,
                                                     "scan.merge", mainExitBB);
                      scanMergePhi->addIncoming(ivs1[1], block);
                    }

                    auto *const needMain =
                        CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE,
                                        zero, mainLoopLimit, "", block);

                    BranchInst::Create(mainPreheaderBB, mainExitBB, needMain,
                                       block);
                  }
                }

                assert((mainPreheaderBB || !wrapperHasMain) &&
                       "Main loops in one barrier block but not another?");

                if (mainPreheaderBB) {
                  wrapperHasMain = true;
                  BasicBlock *mainLoopBB = nullptr;

                  // Materialize the scale factor at the beginning of the
                  // preheader
                  IRBuilder<> irph(mainPreheaderBB,
                                   mainPreheaderBB->getFirstInsertionPt());
                  auto *VF = materializeVF(irph, barrierMain.getVFInfo().vf);

                  compiler::utils::CreateLoopOpts inner_vf_opts;
                  inner_vf_opts.indexInc = VF;
                  inner_vf_opts.IVs = ivs1.vec();
                  inner_vf_opts.loopIVNames.push_back("sg.x.main");
                  if (isScan) {
                    inner_vf_opts.loopIVNames.push_back("scan.y.main");
                  }

                  mainExitBB = compiler::utils::createLoop(
                      mainPreheaderBB, mainExitBB, zero, mainLoopLimit,
                      inner_vf_opts,
                      [&](BasicBlock *block, Value *dim_0,
                          ArrayRef<Value *> ivs0,
                          MutableArrayRef<Value *> ivsNext0) -> BasicBlock * {
                        IRBuilder<> ir(block);

                        if (set_subgroup_id) {
                          // set our subgroup id
                          ir.CreateCall(set_subgroup_id, {ivs0[0]})
                              ->setCallingConv(
                                  set_subgroup_id->getCallingConv());
                        }

                        if (isScan) {
                          auto *const barrierCall =
                              barrierMain.getBarrierCall(barrierID);
                          auto *const liveVars = createLiveVarsPtr(
                              barrierMain, ir, dim_0, dim_1, dim_2, VF);
                          compiler::utils::Barrier::LiveValuesHelper
                              live_values(barrierMain, block, liveVars);
                          auto *const itemOp = live_values.getReload(
                              barrierCall->getOperand(1), ir, "_load",
                              /*reuse*/ true);
                          nextScanIV = compiler::utils::createBinOpForRecurKind(
                              ir, ivs0[1], itemOp, collective->Recurrence);
                          accum = isExclusiveScan ? ivs0[1] : nextScanIV;
                          ivsNext0[1] = nextScanIV;
                        }

                        createWorkItemLoopBody(barrierMain, ir, block,
                                               barrierID, dim_0, dim_1, dim_2,
                                               accum, VF);

                        nextSubgroupIV =
                            ir.CreateAdd(ivs0[0], ConstantInt::get(i32Ty, 1),
                                         "sg.x.main.inc");
                        ivsNext0[0] = nextSubgroupIV;

                        // Move the exit after the loop block, as it reads more
                        // logically.
                        mainLoopBB = block;
                        if (mainExitBB) {
                          mainExitBB->moveAfter(mainLoopBB);
                        }

                        return block;
                      });

                  if (subgroupMergePhi) {
                    subgroupMergePhi->addIncoming(nextSubgroupIV, mainLoopBB);
                  }

                  if (scanMergePhi) {
                    scanMergePhi->addIncoming(nextScanIV, mainLoopBB);
                  }
                }
                assert(mainExitBB && "didn't create a loop exit block!");

                // looping through num groups in the first
                // (innermost) dimension
                BasicBlock *tailPreheaderBB = mainExitBB;
                BasicBlock *tailExitBB = nullptr;

                if (emitTail && peel) {
                  // We might need to bypass the tail loop.
                  if (auto *const peelConst = dyn_cast<Constant>(peel)) {
                    if (peelConst->isZeroValue()) {
                      // No tail iterations at all!
                      tailPreheaderBB = nullptr;
                      tailExitBB = mainExitBB;
                    }
                  } else {
                    tailPreheaderBB = BasicBlock::Create(
                        context, "ca_work_item_x_tail_preheader", func);

                    tailExitBB = BasicBlock::Create(
                        context, "ca_work_item_x_tail_exit", func);
                    tailPreheaderBB->moveAfter(mainExitBB);
                    tailExitBB->moveAfter(tailPreheaderBB);

                    auto *const needPeeling =
                        CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_NE,
                                        zero, peel, "", mainExitBB);

                    BranchInst::Create(tailPreheaderBB, tailExitBB, needPeeling,
                                       mainExitBB);
                  }
                } else {
                  tailPreheaderBB = nullptr;
                  tailExitBB = mainExitBB;
                }

                assert((tailPreheaderBB || !wrapperHasTail) &&
                       "Tail loops in one barrier block but not another?");

                if (tailPreheaderBB) {
                  assert(barrierTail);
                  wrapperHasTail = true;
                  // Subgroup induction variables
                  SmallVector<Value *, 2> subgroupIVs0 = {
                      subgroupMergePhi ? subgroupMergePhi : nextSubgroupIV};
                  if (isScan) {
                    subgroupIVs0.push_back(scanMergePhi ? scanMergePhi
                                                        : nextScanIV);
                  }

                  BasicBlock *tailLoopBB = nullptr;
                  if (barrierTail->getVFInfo().IsVectorPredicated) {
                    IRBuilder<> ir(tailPreheaderBB);
                    if (set_subgroup_id) {
                      // set our subgroup id
                      ir.CreateCall(set_subgroup_id, {subgroupIVs0[0]})
                          ->setCallingConv(set_subgroup_id->getCallingConv());
                    }

                    if (isScan) {
                      assert(barrierTail);
                      auto *const barrierCall =
                          barrierTail->getBarrierCall(barrierID);
                      auto *const liveVars = createLiveVarsPtr(
                          *barrierTail, ir, zero, dim_1, dim_2, nullptr);
                      compiler::utils::Barrier::LiveValuesHelper live_values(
                          *barrierTail, tailPreheaderBB, liveVars);
                      auto *const itemOp = live_values.getReload(
                          barrierCall->getOperand(1), ir, "_load",
                          /*reuse*/ true);
                      nextScanIV = compiler::utils::createBinOpForRecurKind(
                          ir, subgroupIVs0[1], itemOp, collective->Recurrence);
                      accum =
                          isTailExclusiveScan ? subgroupIVs0[1] : nextScanIV;
                    }

                    createWorkItemLoopBody(*barrierTail, ir, tailPreheaderBB,
                                           barrierID, zero, dim_1, dim_2, accum,
                                           /*VF*/ nullptr, mainLoopLimit);

                    nextSubgroupIV = ir.CreateAdd(subgroupIVs0[0],
                                                  ConstantInt::get(i32Ty, 1),
                                                  "sg.x.tail.inc");
                    assert(tailExitBB);
                    ir.CreateBr(tailExitBB);
                    tailLoopBB = tailPreheaderBB;
                  } else {
                    compiler::utils::CreateLoopOpts inner_scalar_opts;
                    inner_scalar_opts.disableVectorize = true;
                    inner_scalar_opts.IVs.assign(subgroupIVs0.begin(),
                                                 subgroupIVs0.end());
                    inner_scalar_opts.loopIVNames.push_back("sg.x.tail");
                    if (isScan) {
                      inner_scalar_opts.loopIVNames.push_back("scan.x.tail");
                    }

                    tailExitBB = compiler::utils::createLoop(
                        tailPreheaderBB, tailExitBB, zero, peel,
                        inner_scalar_opts,
                        [&](BasicBlock *block, Value *dim_0,
                            ArrayRef<Value *> ivs0,
                            MutableArrayRef<Value *> ivsNext0) -> BasicBlock * {
                          IRBuilder<> ir(block);

                          if (set_subgroup_id) {
                            // set our subgroup id
                            ir.CreateCall(set_subgroup_id, {ivs0[0]})
                                ->setCallingConv(
                                    set_subgroup_id->getCallingConv());
                          }

                          if (isScan) {
                            assert(barrierTail);
                            auto *const barrierCall =
                                barrierTail->getBarrierCall(barrierID);
                            auto *const liveVars = createLiveVarsPtr(
                                *barrierTail, ir, dim_0, dim_1, dim_2, nullptr);
                            compiler::utils::Barrier::LiveValuesHelper
                                live_values(*barrierTail, block, liveVars);
                            auto *const itemOp = live_values.getReload(
                                barrierCall->getOperand(1), ir, "_load",
                                /*reuse*/ true);
                            nextScanIV =
                                compiler::utils::createBinOpForRecurKind(
                                    ir, ivs0[1], itemOp,
                                    collective->Recurrence);
                            accum = isTailExclusiveScan ? ivs0[1] : nextScanIV;
                            ivsNext0[1] = nextScanIV;
                          }

                          createWorkItemLoopBody(
                              *barrierTail, ir, block, barrierID, dim_0, dim_1,
                              dim_2, accum, /*VF*/ nullptr, mainLoopLimit);

                          nextSubgroupIV =
                              ir.CreateAdd(ivs0[0], ConstantInt::get(i32Ty, 1),
                                           "sg.x.tail.inc");
                          ivsNext0[0] = nextSubgroupIV;

                          tailLoopBB = block;
                          // Move the exit after the loop block, as it reads
                          // more logically.
                          if (tailExitBB) {
                            tailExitBB->moveAfter(tailLoopBB);
                          }

                          return block;
                        });
                  }

                  // Merge the main and tail subgroup IVs together in the
                  // tail exit, since we may have skipped either main or
                  // tail loops.
                  if (subgroupMergePhi) {
                    auto *scalarSubgroupIV = nextSubgroupIV;
                    nextSubgroupIV = PHINode::Create(
                        i32Ty, 2, "sg.main.tail.merge", tailExitBB);
                    cast<PHINode>(nextSubgroupIV)
                        ->addIncoming(scalarSubgroupIV, tailLoopBB);
                    cast<PHINode>(nextSubgroupIV)
                        ->addIncoming(subgroupMergePhi, mainExitBB);
                  }

                  if (scanMergePhi) {
                    auto *scalarScanIV = nextScanIV;
                    nextScanIV =
                        PHINode::Create(accum->getType(), 2,
                                        "scan.main.tail.merge", tailExitBB);
                    cast<PHINode>(nextScanIV)
                        ->addIncoming(scalarScanIV, tailLoopBB);
                    cast<PHINode>(nextScanIV)
                        ->addIncoming(scanMergePhi, mainExitBB);
                  }
                }
                // Don't forget to update the subgroup IV phi.
                ivsNext1[0] = nextSubgroupIV;
                if (isScan) {
                  // ... or the scan IV phi.
                  ivsNext1[1] = nextScanIV;
                }
                return tailExitBB;
              });

          // Don't forget to update the subgroup IV phi.
          ivsNext2[0] = nextSubgroupIV;
          if (isScan) {
            // ... or the scan IV phi.
            ivsNext2[1] = nextScanIV;
          }
          return exit1;
        });
  }

  // It executes only the first work item in the work group
  BasicBlock *makeRunOneWorkItem(BasicBlock *block, unsigned barrierID) {
    // "Once" scheduled barriers shouldn't need the local id set.
    IRBuilder<> ir(block);
    createWorkItemLoopBody(barrierTail ? *barrierTail : barrierMain, ir, block,
                           barrierID, nullptr, nullptr, nullptr, nullptr);
    return block;
  }
};

// Emits code to set up the storage allocated to a live-vars structure.
//
// Allocates enough space for sizeZ * sizeY * sizeX work-items. Note that Z/Y/X
// here corresponds to the current outermost to innermost vectorized
// dimensions, rather than in their absolutist sense.
void setUpLiveVarsAlloca(compiler::utils::BarrierWithLiveVars &barrier,
                         IRBuilder<> &B, Value *const sizeZ, Value *const sizeY,
                         Value *const sizeX, StringRef name, bool isDebug) {
  barrier.setSize0(sizeX);
  Value *const live_var_size = B.CreateMul(sizeX, B.CreateMul(sizeY, sizeZ));
  barrier.setTotalSize(live_var_size);
  AllocaInst *live_var_mem_space;
  auto &m = *B.GetInsertBlock()->getModule();
  auto *const size_ty = compiler::utils::getSizeType(m);
  const auto scalablesSize = barrier.getLiveVarMemSizeScalable();
  if (scalablesSize == 0) {
    live_var_mem_space =
        B.CreateAlloca(barrier.getLiveVarsType(), live_var_size, name);
    live_var_mem_space->setAlignment(
        MaybeAlign(barrier.getLiveVarMaxAlignment()).valueOrOne());
    barrier.setMemSpace(live_var_mem_space);
  } else {
    const auto fixedSize = barrier.getLiveVarMemSizeFixed();
    // We ensure that the VFs are the same between the main and tail.
    auto *const vscale =
        B.CreateVScale(ConstantInt::get(size_ty, scalablesSize));
    auto *const structSize =
        B.CreateAdd(vscale, ConstantInt::get(size_ty, fixedSize));
    auto *const buffer_size = B.CreateMul(structSize, live_var_size);

    live_var_mem_space = B.CreateAlloca(B.getInt8Ty(), buffer_size, name);
    live_var_mem_space->setAlignment(
        MaybeAlign(barrier.getLiveVarMaxAlignment()).valueOrOne());
    barrier.setMemSpace(live_var_mem_space);
    barrier.setStructSize(structSize);
  }

  if (isDebug) {
    barrier.setDebugAddr(B.CreateAlloca(live_var_mem_space->getType(), nullptr,
                                        "live_vars_peel_dbg"));
  }
}

}  // namespace

Function *compiler::utils::WorkItemLoopsPass::makeWrapperFunction(
    BarrierWithLiveVars &barrierMain, BarrierWithLiveVars *barrierTail,
    StringRef baseName, Module &M, compiler::utils::BuiltinInfo &BI) {
  Function &mainF = barrierMain.getFunc();

  // The reference function is that which we expect to hold the reference
  // version of various pieces of data, such as metadata. It's the tail
  // function if one exists, else it's the main function.
  Function &refF = barrierTail ? barrierTail->getFunc() : barrierMain.getFunc();

  const bool emitTail = barrierTail != nullptr;

  auto mainInfo = barrierMain.getVFInfo();
  auto tailInfo =
      emitTail ? barrierTail->getVFInfo() : std::optional<VectorizationInfo>();

  const auto workItemDim0 = 0;
  const auto workItemDim1 = 1;
  const auto workItemDim2 = 2;

  LLVMContext &context = M.getContext();

  Function *new_wrapper =
      createKernelWrapperFunction(mainF, ".mux-barrier-wrapper");

  new_wrapper->setName(baseName + ".mux-barrier-wrapper");
  // Ensure the base name is recorded
  setBaseFnName(*new_wrapper, baseName);

  // An inlinable function call in a function with debug info *must* be given
  // a debug location.
  DILocation *wrapperDbgLoc = nullptr;
  if (new_wrapper->getSubprogram()) {
    wrapperDbgLoc = DILocation::get(context, /*line*/ 0, /*col*/ 0,
                                    new_wrapper->getSubprogram());
  }

  IRBuilder<> entryIR(BasicBlock::Create(context, "entry", new_wrapper));

  auto *const i32Ty = Type::getInt32Ty(context);

  auto sizeTyBytes = getSizeTypeBytes(M);

  auto *VF = materializeVF(entryIR, barrierMain.getVFInfo().vf);
  Value *localSizeDim[3];

  if (auto wgs = parseRequiredWGSMetadata(refF)) {
    localSizeDim[0] = entryIR.getIntN(8 * sizeTyBytes, (*wgs)[0]);
    localSizeDim[1] = entryIR.getIntN(8 * sizeTyBytes, (*wgs)[1]);
    localSizeDim[2] = entryIR.getIntN(8 * sizeTyBytes, (*wgs)[2]);
  } else {
    const uint32_t max_work_dim = parseMaxWorkDimMetadata(refF).value_or(3);

    // Fill out a default local size of 1x1x1.
    std::fill(std::begin(localSizeDim), std::end(localSizeDim),
              entryIR.getIntN(8 * sizeTyBytes, 1));

    auto *const get_local_size =
        BI.getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalSize, M);
    assert(get_local_size && "Missing __mux_get_local_size");

    auto ci0 =
        entryIR.CreateCall(get_local_size, entryIR.getInt32(0), "local_size.x");
    ci0->setCallingConv(get_local_size->getCallingConv());
    localSizeDim[0] = ci0;

    if (max_work_dim > 1) {
      auto ci1 = entryIR.CreateCall(get_local_size, entryIR.getInt32(1),
                                    "local_size.y");
      ci1->setCallingConv(get_local_size->getCallingConv());
      localSizeDim[1] = ci1;
    }

    if (max_work_dim > 2) {
      auto ci2 = entryIR.CreateCall(get_local_size, entryIR.getInt32(2),
                                    "local_size.z");
      ci2->setCallingConv(get_local_size->getCallingConv());
      localSizeDim[2] = ci2;
    }
  }

  // Assume that local sizes are never zero. This prevents LLVM "saving" our
  // loops by inserting llvm.umax (or its equivalent) to stop the loops we're
  // about to create from causing headaches:
  //   %iv.next = add i64 nuw %iv, 1
  //   %exit = icmp eq i64 %iv.next, %localsizeY
  //   br i1 %exit, label %exit.the.loop, %continue.the.loop
  // If LLVM doesn't know that %localsizey is never zero, it rightly determines
  // that a zero size would cause problems, since we'd have to overflow our i64
  // to exit the loop, but we've marked the increment as 'nuw'. So it inserts
  // an llvm.umax to ensure the size is at least 1. Since we know our local
  // sizes are never zero, an llvm.assume intrinsic prevents this from
  // happening.
  // We want to insert a call to __mux__set_max_sub_group_size after these
  // assumptions, to keep track of the last one we've inserted.
  Instruction *setMaxSubgroupSizeInsertPt = nullptr;
  for (auto i = 0; i < 3; i++) {
    auto *const nonZero = entryIR.CreateICmpNE(
        localSizeDim[i], ConstantInt::get(localSizeDim[i]->getType(), 0));
    setMaxSubgroupSizeInsertPt = entryIR.CreateAssumption(nonZero);
  }

  const bool isVectorPredicated = barrierMain.getVFInfo().IsVectorPredicated;

  Value *mainLoopLimit = localSizeDim[workItemDim0];
  Value *peel = nullptr;
  if (emitTail) {
    peel = entryIR.CreateSRem(mainLoopLimit, VF, "peel");
    mainLoopLimit = entryIR.CreateSub(mainLoopLimit, peel, "mainLoopLimit");
  }

  // Set the number of subgroups in this kernel
  {
    auto setNumSubgroupsFn =
        BI.getOrDeclareMuxBuiltin(eMuxBuiltinSetNumSubGroups, M);
    assert(setNumSubgroupsFn && "Missing __mux_set_num_sub_groups");
    // First, compute Z * Y
    auto *const numSubgroupsZY = entryIR.CreateMul(
        localSizeDim[workItemDim2], localSizeDim[workItemDim1], "sg.zy");
    // Now multiply by the number of subgroups in the X dimension.
    auto *numSubgroupsX = entryIR.CreateUDiv(mainLoopLimit, VF, "sg.main.x");
    // Add on any tail iterations here.
    if (peel) {
      numSubgroupsX = entryIR.CreateAdd(numSubgroupsX, peel, "sg.x");
    } else if (isVectorPredicated) {
      // Vector predication will use an extra subgroup to mop up any remainder.
      auto *const leftover = entryIR.CreateSRem(mainLoopLimit, VF, "peel");
      auto *hasLeftover = entryIR.CreateICmp(
          CmpInst::ICMP_NE, leftover, ConstantInt::get(leftover->getType(), 0),
          "sg.has.vp");
      hasLeftover = entryIR.CreateZExt(hasLeftover, numSubgroupsX->getType());
      numSubgroupsX = entryIR.CreateAdd(numSubgroupsX, hasLeftover, "sg.x");
    }
    auto *numSubgroups =
        entryIR.CreateMul(numSubgroupsZY, numSubgroupsX, "sg.zyx");
    if (numSubgroups->getType() != i32Ty) {
      numSubgroups = entryIR.CreateTrunc(numSubgroups, i32Ty);
    }
    entryIR.CreateCall(setNumSubgroupsFn, {numSubgroups});
  }

  if (barrierMain.hasLiveVars()) {
    // The size in the first dimension is divided by the vectorization factor.
    // When vector-predicated, this result is rounded up: (LIM + VF - 1) / VF.
    // This catches cases where we need two loop iterations, e.g., VF=4 and
    // size=7, where rounding down would give one.
    Value *numerator = mainLoopLimit;
    if (isVectorPredicated) {
      Value *const vf_minus_1 =
          entryIR.CreateSub(VF, ConstantInt::get(VF->getType(), 1));
      numerator = entryIR.CreateAdd(mainLoopLimit, vf_minus_1);
    }
    Value *const size0 = entryIR.CreateUDiv(numerator, VF);

    setUpLiveVarsAlloca(barrierMain, entryIR, localSizeDim[workItemDim2],
                        localSizeDim[workItemDim1], size0, "live_variables",
                        IsDebug);
  }

  // Amazingly, it's possible for the tail kernel to have live vars in its
  // barriers, even when the main kernel does not.
  if (emitTail && barrierTail->hasLiveVars()) {
    Value *size0 = peel;
    if (barrierTail->getVFInfo().IsVectorPredicated) {
      // If the tail is predicated, it will only have a single (vectorized) item
      // along the X axis, or none.
      auto *const hasLeftover = entryIR.CreateICmp(
          CmpInst::ICMP_NE, peel, ConstantInt::get(peel->getType(), 0),
          "tail.has.vp");
      size0 = entryIR.CreateZExt(hasLeftover, peel->getType());
    }
    setUpLiveVarsAlloca(*barrierTail, entryIR, localSizeDim[workItemDim2],
                        localSizeDim[workItemDim1], size0,
                        "live_variables_peel", IsDebug);
  }

  // next means next barrier id. This variable is uninitialized to begin with,
  // and is set by the first pass below
  IntegerType *index_type = i32Ty;
  AllocaInst *nextID =
      entryIR.CreateAlloca(index_type, nullptr, "next_barrier_id");

  SmallVector<BasicBlock *, 8> bbs;
  const unsigned num_blocks = barrierMain.getNumSubkernels();
  assert(!emitTail || barrierTail->getNumSubkernels() == num_blocks);

  for (unsigned i = kBarrier_EndID; i <= num_blocks; i++) {
    BasicBlock *bb = BasicBlock::Create(context, "sw.bb", new_wrapper);
    bbs.push_back(bb);
  }

  ScheduleGenerator schedule(M, barrierMain, barrierTail, BI);
  schedule.workItemDim0 = workItemDim0;
  schedule.workItemDim1 = workItemDim1;
  schedule.workItemDim2 = workItemDim2;
  schedule.localSizeDim[0] = localSizeDim[0];
  schedule.localSizeDim[1] = localSizeDim[1];
  schedule.localSizeDim[2] = localSizeDim[2];
  schedule.wrapperDbgLoc = wrapperDbgLoc;
  schedule.nextID = nextID;
  schedule.mainLoopLimit = mainLoopLimit;
  schedule.emitTail = emitTail;
  schedule.isVectorPredicated = isVectorPredicated;
  schedule.peel = peel;

  // Make call instruction for first new kernel. It follows wrapper function's
  // parameters.
  for (auto &arg : new_wrapper->args()) {
    schedule.args.push_back(&arg);
  }

  // Branch directly into the first basic block.
  entryIR.CreateBr(bbs[kBarrier_FirstID]);

  for (unsigned i = kBarrier_EndID; i <= num_blocks; i++) {
    // Keep it linear
    BasicBlock *const block = bbs[i];
    block->moveAfter(&new_wrapper->back());

    if (i == kBarrier_EndID) {
      // This basic block breaks us out of our function, thus we return!
      ReturnInst::Create(context, block);
    } else {
      // Re-issue the barrier's memory fence before the work-item loops
      if (auto *const CI = barrierMain.getBarrierCall(i)) {
        auto *const callee = CI->getCalledFunction();
        const auto builtin = BI.analyzeBuiltin(*callee);
        if (builtin.ID == compiler::utils::eMuxBuiltinWorkGroupBarrier) {
          IRBuilder<> B(block);
          auto *MemBarrier =
              BI.getOrDeclareMuxBuiltin(eMuxBuiltinMemBarrier, M);
          assert(MemBarrier);
          Value *Ops[2] = {CI->getOperand(1), CI->getOperand(2)};

          auto *const Call = B.CreateCall(MemBarrier, Ops);

          // Patch up any operands that were non-constants by fetching them from
          // the barrier struct. We do this after creating the call because we
          // need an instruction to function as an insert point.
          if (!isa<Constant>(Ops[0]) || !isa<Constant>(Ops[1])) {
            // We expect these values to be uniform so it should be safe to get
            // from the barrier struct at index zero. Barriers are convergent,
            // so there should be no chance that the value does not exist.
            auto *const zero =
                Constant::getNullValue(compiler::utils::getSizeType(M));
            IRBuilder<> ir(Call);
            auto *const barrier0 =
                ir.CreateInBoundsGEP(barrierMain.getLiveVarsType(),
                                     barrierMain.getMemSpace(), {zero});

            Barrier::LiveValuesHelper live_values(barrierMain, Call, barrier0);

            size_t op_index = 0;
            for (auto *const op : Ops) {
              if (!isa<Constant>(op)) {
                auto *const new_op =
                    live_values.getReload(op, ir, "_load", /*reuse*/ true);
                Call->setArgOperand(op_index, new_op);
              }
              ++op_index;
            }
          }
          Call->setDebugLoc(wrapperDbgLoc);
        }
      }

      auto *const exitBlock = [&]() {
        switch (barrierMain.getSchedule(i)) {
          default:
            assert(!"Unexpected barrier schedule enum");
            LLVM_FALLTHROUGH;
          case BarrierSchedule::Unordered:
          case BarrierSchedule::ScalarTail:
            if (tailInfo && tailInfo->IsVectorPredicated) {
              return schedule.makeLinearWorkItemLoops(block, i);
            }
            return schedule.makeWorkItemLoops(block, i);

          case BarrierSchedule::Once:
            return schedule.makeRunOneWorkItem(block, i);

          case BarrierSchedule::Linear:
            return schedule.makeLinearWorkItemLoops(block, i);
        }
      }();

      // the last basic block in our function!
      IRBuilder<> exitIR(exitBlock);

      const auto &successors = barrierMain.getSuccessorIds(i);
      const auto num_succ = successors.size();

      if (num_succ == 1) {
        // If there is only one successor, we can branch directly to it
        exitIR.CreateBr(bbs[successors.front()]);
      } else if (num_succ == 2) {
        // If there are exactly two successors, we can use a conditional branch
        auto *const bb_id = ConstantInt::get(index_type, successors[0]);
        auto *const br_block =
            BasicBlock::Create(context, "barrier.branch", new_wrapper);
        auto *const ld_next_id = new LoadInst(index_type, nextID, "", br_block);
        auto *const cmp_id =
            CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_EQ, ld_next_id,
                            bb_id, "", br_block);
        BranchInst::Create(bbs[successors[0]], bbs[successors[1]], cmp_id,
                           br_block);

        exitIR.CreateBr(br_block);
      } else if (num_succ == 0) {
        // If a barrier region has no successor, we just emit a call to
        // llvm.trap and unreachable. A barrier region can have zero successors
        // if all its terminators end in unreachable. Since there are no
        // successors, it is not possible to continue and therefore we emit an
        // unreachable here.

        // TODO: we should be flagging up unreachables sooner, so that we avoid
        // wrapping barrier regions with no successors with work item loops,
        // and we should also make sure that the barrier region has no
        // successors because of all its terminators ending in unreachable.
        // If it's not the case we may want to handle that differently.
        auto trap =
            M.getOrInsertFunction("llvm.trap", Type::getVoidTy(context));
        exitIR.CreateCall(trap);
        exitIR.CreateUnreachable();
      } else {
        // Make a basic block with a switch to jump to the next subkernel
        auto *const switch_body =
            BasicBlock::Create(context, "barrier.switch", new_wrapper);
        LoadInst *const ld_next_id =
            new LoadInst(index_type, nextID, "", switch_body);
        SwitchInst *const sw = SwitchInst::Create(
            ld_next_id, bbs[successors[0]], num_succ, switch_body);
        for (const auto i : successors) {
          sw->addCase(ConstantInt::get(index_type, i), bbs[i]);
        }
        exitIR.CreateBr(switch_body);
      }
    }
  }

  bbs[kBarrier_EndID]->moveAfter(&new_wrapper->back());
  bbs[kBarrier_EndID]->setName("kernel.exit");

  // Set the subgroup maximum size in this kernel wrapper.
  // There are three cases:
  //
  // 1. With no vectorization:
  //    get_max_sub_group_size() = mux sub-group size
  //
  // 2. With predicated vectorization:
  //    get_max_sub_group_size() = min(vector_width,
  //    local_size_in_vectorization_dimension)
  //
  // 3. Without predicated vectorization:
  //    get_max_sub_group_size() = local_size_in_vectorization_dimension
  //    < vector_width ? mux sub-group size : vector_width
  {
    // Reset the insertion point back to the wrapper entry block, after VF was
    // materialized.
    entryIR.SetInsertPoint(setMaxSubgroupSizeInsertPt);
    auto setMaxSubgroupSizeFn =
        BI.getOrDeclareMuxBuiltin(eMuxBuiltinSetMaxSubGroupSize, M);
    assert(setMaxSubgroupSizeFn && "Missing __mux_set_max_sub_group_size");
    // Assume no vectorization to begin with i.e. get_max_sub_group_size() = mux
    // sub-group size.
    Value *maxSubgroupSize = entryIR.getInt32(getMuxSubgroupSize(refF));
    if (schedule.wrapperHasMain) {
      auto *localSizeInVecDim = localSizeDim[workItemDim0];
      auto *cmp = entryIR.CreateICmpULT(localSizeInVecDim, VF);
      if (isVectorPredicated) {
        maxSubgroupSize = entryIR.CreateSelect(cmp, localSizeInVecDim, VF);
      } else {
        maxSubgroupSize = entryIR.CreateSelect(
            cmp, ConstantInt::get(VF->getType(), getMuxSubgroupSize(refF)), VF);
      }
      if (maxSubgroupSize->getType() != i32Ty) {
        maxSubgroupSize = entryIR.CreateTrunc(maxSubgroupSize, i32Ty);
      }
    }
    entryIR.CreateCall(setMaxSubgroupSizeFn, {maxSubgroupSize});
  }

  // Remap any constant expression which take a reference to the old function
  // FIXME: What about the main function?
  for (auto *user : make_early_inc_range(refF.users())) {
    if (ConstantExpr *constant = dyn_cast<ConstantExpr>(user)) {
      remapConstantExpr(constant, &refF, new_wrapper);
    } else if (ConstantArray *ca = dyn_cast<ConstantArray>(user)) {
      remapConstantArray(ca, &refF, new_wrapper);
    } else if (!isa<CallInst>(user)) {
      llvm_unreachable(
          "Cannot handle user of function being anything other than a "
          "ConstantExpr, ConstantArray or CallInst");
    }
  }
  // We output the number of uses here to lit test that the number of uses was
  // not increased by the remap functions.
  LLVM_DEBUG(dbgs() << "Uses of " << refF.getName() << ": " << refF.getNumUses()
                    << "\n");

  // Forcibly disable the tail info if we know we've omitted it.
  if (!schedule.wrapperHasMain || !schedule.wrapperHasTail) {
    // If we're missing a main loop then the tail loop becomes the main from
    // the perspective of the metadata: have that steal the tail loop info. We
    // should always have a main loop with an optional tail.
    if (!schedule.wrapperHasMain) {
      if (schedule.wrapperHasTail && tailInfo) {
        mainInfo = *tailInfo;
      } else {
        // If we have neither a main nor a tail (which may happen at kernel
        // compile time but we should never actually execute such a kernel -
        // we already assume the local sizes are never zero, see elsewhere in
        // this pass) then encode a token info metadata of 1.
        mainInfo =
            VectorizationInfo{VectorizationFactor::getScalar(), workItemDim0,
                              /*isVectorPredicated*/ false};
      }
    }
    tailInfo = std::nullopt;
  }

  encodeWrapperFnMetadata(*new_wrapper, mainInfo, tailInfo);

  // The subkernels can be marked as internal since its external uses have been
  // superceded by this wrapper. This will help it get DCE'd once inlined. Any
  // existing calls to this subkernel (e.g., another kernel calling this
  // kernel) will prevent it from being removed unnecessarily.
  barrierMain.getFunc().setLinkage(Function::InternalLinkage);
  if (barrierTail) {
    barrierTail->getFunc().setLinkage(Function::InternalLinkage);
  }

  return new_wrapper;
}

struct BarrierWrapperInfo {
  StringRef BaseName;
  // Information about the 'main' kernel
  Function *MainF;
  compiler::utils::VectorizationInfo MainInfo;
  // Optional information about the 'tail' kernel
  Function *TailF = nullptr;
  std::optional<compiler::utils::VectorizationInfo> TailInfo = std::nullopt;
  // A 'tail' kernel which was explicitly omitted.
  Function *SkippedTailF = nullptr;
};

PreservedAnalyses compiler::utils::WorkItemLoopsPass::run(
    Module &M, ModuleAnalysisManager &MAM) {
  // Cache the functions we're interested in as this pass introduces new ones
  // which we don't want to run over.
  SmallVector<BarrierWrapperInfo, 4> MainTailPairs;
  const auto &GSGI = MAM.getResult<compiler::utils::SubgroupAnalysis>(M);

  for (auto &F : M.functions()) {
    if (!isKernelEntryPt(F)) {
      continue;
    }

    const auto BaseName = getBaseFnNameOrFnName(F);
    auto VeczToOrigFnData = parseVeczToOrigFnLinkMetadata(F);

    const auto WorkItemDim0 = 0;

    const VectorizationInfo scalarTailInfo{VectorizationFactor::getScalar(),
                                           WorkItemDim0,
                                           /*IsVectorPredicated*/ false};

    if (!VeczToOrigFnData) {
      // If there was no vectorization metadata, it's a scalar kernel.
      MainTailPairs.push_back({BaseName, &F, scalarTailInfo});
      continue;
    }

    // If we got a vectorized kernel, wrap it using the vectorization factor.
    const auto MainInfo = VeczToOrigFnData->second;

    // Start out assuming scalar tail, which is the default behaviour...
    auto TailInfo = scalarTailInfo;
    auto *TailFunc = VeczToOrigFnData->first;
    // ... and search for a linked vector-predicated tail, which we prefer.
    if (!MainInfo.IsVectorPredicated && TailFunc) {
      SmallVector<LinkMetadataResult, 4> LinkedFns;
      parseOrigToVeczFnLinkMetadata(*TailFunc, LinkedFns);
      for (const auto &Link : LinkedFns) {
        // Restrict our option to strict VF==VF matches.
        if (Link.first != &F && Link.second.vf == MainInfo.vf &&
            Link.second.IsVectorPredicated) {
          TailFunc = Link.first;
          TailInfo = Link.second;
          break;
        }
      }
    }

    std::optional<size_t> LocalSizeInVecDim;
    if (auto WGS = parseRequiredWGSMetadata(F)) {
      LocalSizeInVecDim = (*WGS)[WorkItemDim0];
    }

    // We can skip the tail in the following circumstances:
    // * If we have no tail function (trusting that this is okay)
    // * Vector-predicated kernels handle their own tails
    // * The user has explicitly forced us to omit tails
    // * We can prove that the vectorization factor fits the required/known
    //   local work-group size
    if (!TailFunc || MainInfo.IsVectorPredicated || ForceNoTail ||
        (LocalSizeInVecDim && !MainInfo.vf.isScalable() &&
         *LocalSizeInVecDim % MainInfo.vf.getKnownMin() == 0)) {
      MainTailPairs.push_back({BaseName, &F, MainInfo, /*TailF*/ nullptr,
                               /*TailInfo*/ std::nullopt,
                               /*SkippedTailF*/ TailFunc});
    } else {
      // Else, emit a tail using the tail function.
      MainTailPairs.push_back({BaseName, &F, MainInfo, TailFunc, TailInfo});
    }
  }

  if (MainTailPairs.empty()) {
    return PreservedAnalyses::all();
  }

  // Prune redundant wrappers we don't want to create for the sake of compile
  // time.
  SmallPtrSet<const Function *, 4> RedundantMains;
  for (const auto &P : MainTailPairs) {
    // If we're creating a wrapper with a skipped 'tail' or a scalar 'tail', we
    // don't want to create another wrapper where the scalar tail is the
    // 'main', unless that tail is useful as a fallback sub-group kernel. A
    // fallback sub-group kernel is one for which:
    // * The 'main' is not a degenerate sub-group kernel. These are always safe
    // to run so the fallback is unnecessary.
    // * The 'main' has a required sub-group size that isn't the scalar size.
    // * The 'main' and 'tail' kernels both make use of sub-group builtins. If
    // neither do, there's no need for the fallback.
    // * The 'main' kernel uses sub-groups but the 'main' vectorization factor
    // cleanly divides the known local work-group size.
    if (P.SkippedTailF || (P.TailInfo && P.TailInfo->vf.isScalar())) {
      const auto *TailF = P.SkippedTailF ? P.SkippedTailF : P.TailF;
      if (hasDegenerateSubgroups(*P.MainF) ||
          getReqdSubgroupSize(*P.MainF).value_or(1) != 1 ||
          (!GSGI.usesSubgroups(*P.MainF) && !GSGI.usesSubgroups(*TailF))) {
        RedundantMains.insert(TailF);
      } else if (auto wgs = parseRequiredWGSMetadata(*P.MainF)) {
        const uint64_t local_size_x = wgs.value()[0];
        if (!P.MainInfo.IsVectorPredicated &&
            !(local_size_x % P.MainInfo.vf.getKnownMin())) {
          RedundantMains.insert(TailF);
        }
      }
    }
    // If we're creating a wrapper with a VP 'tail', we don't want to create
    // another wrapper where the VP is the 'main'
    if (!P.MainInfo.IsVectorPredicated && P.TailInfo &&
        P.TailInfo->IsVectorPredicated) {
      RedundantMains.insert(P.TailF);
    }
  }

  MainTailPairs.erase(
      std::remove_if(MainTailPairs.begin(), MainTailPairs.end(),
                     [&RedundantMains](const BarrierWrapperInfo &I) {
                       return RedundantMains.contains(I.MainF);
                     }),
      MainTailPairs.end());

  SmallPtrSet<Function *, 4> Wrappers;
  auto &BI = MAM.getResult<BuiltinInfoAnalysis>(M);

  for (const auto &P : MainTailPairs) {
    assert(P.MainF && "Missing main function");
    // Construct the main barrier
    BarrierWithLiveVars MainBarrier(M, *P.MainF, P.MainInfo, IsDebug);
    MainBarrier.Run(MAM);

    // Tail kernels are optional
    if (!P.TailF) {
      Wrappers.insert(
          makeWrapperFunction(MainBarrier, nullptr, P.BaseName, M, BI));
    } else {
      // Construct the tail barrier
      assert(P.TailInfo && "Missing tail info");
      BarrierWithLiveVars TailBarrier(M, *P.TailF, *P.TailInfo, IsDebug);
      TailBarrier.Run(MAM);

      Wrappers.insert(
          makeWrapperFunction(MainBarrier, &TailBarrier, P.BaseName, M, BI));
    }
  }

  // At this point we mandate that any kernels that haven't been wrapped with
  // work-item loops can't be kernels, nor entry points.
  for (auto &F : M) {
    if (isKernelEntryPt(F) && !Wrappers.contains(&F)) {
      dropIsKernel(F);
      // FIXME: Also mark them as internal in case they contain symbols we
      // haven't resolved as part of the work-item loop wrapping process. We
      // rely on GlobalOptPass to remove such functions; this is the same root
      // issue as CA-4126.
      F.setLinkage(GlobalValue::InternalLinkage);
    }
  }

  return PreservedAnalyses::none();
}
