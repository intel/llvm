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
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TinyPtrVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/LCSSA.h>
#include <llvm/Transforms/Utils/Local.h>
#include <multi_llvm/multi_llvm.h>

#include <optional>

using namespace llvm;

#define NDEBUG_BARRIER
#define DEBUG_TYPE "barrier-regions"

namespace {
using AlignIntTy = uint64_t;

/// @brief it returns true if and only if the instruction is a work group
/// collective call, and returns false otherwise.
std::optional<compiler::utils::GroupCollective> getWorkGroupCollectiveCall(
    Instruction *inst, compiler::utils::BuiltinInfo &bi) {
  auto *const ci = dyn_cast_or_null<CallInst>(inst);
  if (!ci) {
    return std::nullopt;
  }

  Function *callee = ci->getCalledFunction();
  assert(callee && "could not get called function");
  auto info = bi.isMuxGroupCollective(bi.analyzeBuiltin(*callee).ID);
  if (info && info->isWorkGroupScope()) {
    return info;
  }
  return std::nullopt;
}

/// @brief Builds a stub function containing only a void return instruction.
///
/// @note This is useful for client debuggers that want to break on a
/// particular barrier and work item. Customer specific passes can fill the
/// contents since it may involve inline assembly for breakpoint traps. The
/// stub function takes a single i32 argument which is an id identifying the
/// barrier which invoked the stub. A client debugger should be able to read
/// this argument using the arch calling convention even without debug info
/// since it's always the first argument, although customer passes may
/// rearrange parameters later.
///
/// @param[in] name What to name the stub function.
/// @param[in] module Current module.
/// @param[in] cc Calling convention for function.
///
/// @return Return function created.
Function *MakeStubFunction(StringRef name, Module &module, CallingConv::ID cc) {
  // If we've already created a stub return the existing function
  if (Function *existing = module.getFunction(name)) {
    return existing;
  }

  auto &context = module.getContext();
  // 32-bit integer parameter
  IntegerType *int32_type = IntegerType::get(context, 32);
  // Function returns void
  FunctionType *func_type =
      FunctionType::get(Type::getVoidTy(context), {int32_type}, false);

  // Create function in module
  Function *stub_func =
      Function::Create(func_type, Function::ExternalLinkage, name, &module);

  // Don't inline the function since we want the debugger to be able to hook it
  stub_func->addFnAttr(Attribute::NoInline);

  // we don't use exceptions
  stub_func->addFnAttr(Attribute::NoUnwind);
  stub_func->setCallingConv(cc);

  // No stub or cloned function should have SPIR_KERNEL calling convention.
  // Please consider using SPIR_FUNC instead of SPIR_KERNEL. In case the
  // original code has a different calling convention, we should preserve that
  // one.
  assert(cc != CallingConv::SPIR_KERNEL && "calling convention mismatch");

  // Single basic block containing only a return void instruction
  IRBuilder<> IRBuilder(BasicBlock::Create(context, "entry", stub_func));
  IRBuilder.CreateRetVoid();

  // Build debug info for function if compiled with -g
  DIBuilder DIB(module, /*AllowUnresolved*/ false);

  // Find module compile unit
  auto *cu = DIB.createCompileUnit(
      dwarf::DW_LANG_OpenCL, DIB.createFile("debug", "/"), "", false, "", 0);

  // Create DISubprogram metadata for function
  auto type_array =
      DIB.getOrCreateTypeArray({DIB.createUnspecifiedParameter()});
  auto subprogram_type = DIB.createSubroutineType(type_array);
  auto DISubprogram = DIB.createFunction(
      cu->getFile(), name, name, cu->getFile(), 0, subprogram_type, 0,
      DINode::FlagZero, DISubprogram::SPFlagDefinition);

  // Set function compile unit
  DISubprogram->replaceUnit(cu);

  // Assigned debug info to function
  stub_func->setSubprogram(DISubprogram);

  DIB.finalize();

  return stub_func;
}

/// @brief Check whether this value is valid as def.
///
/// @param[in] v Value for checking.
///
/// @return True = valid for definition, False = not valid.
inline bool CheckValidDef(Value *v) {
  return !(isa<BranchInst>(v) || isa<ReturnInst>(v));
}

/// @brief Check whether this value is valid as use.
///
/// @param[in] v - value for checking.
///
/// @return True = valid for use, False = not valid.
inline bool CheckValidUse(Value *v) {
  return !(isa<Constant>(v) || isa<BasicBlock>(v) || isa<MetadataAsValue>(v));
}

bool IsRematerializableBuiltinCall(Value *v, compiler::utils::BuiltinInfo &bi) {
  if (auto *call = dyn_cast<CallInst>(v)) {
    assert(call->getCalledFunction() && "Could not get called function");
    const auto B = bi.analyzeBuiltin(*call->getCalledFunction());
    if (B.properties & compiler::utils::eBuiltinPropertyRematerializable) {
      for (auto &op : call->operands()) {
        if (isa<Instruction>(op.get())) {
          return false;
        }
      }
      return true;
    }
  }
  return false;
}

// It traces through instructions with a single Instruction operand, looking
// for work item functions or function arguments.
bool IsTrivialValue(Value *v, unsigned depth,
                    compiler::utils::BuiltinInfo &bi) {
  while (depth--) {
    auto *const I = dyn_cast<Instruction>(v);
    if (!I || IsRematerializableBuiltinCall(v, bi)) {
      return true;
    }

    // Pass through a vector splat to the splatted value
    if (auto *const shuffle = dyn_cast<ShuffleVectorInst>(I)) {
      if (shuffle->isZeroEltSplat()) {
        if (auto *const ins =
                dyn_cast<InsertElementInst>(shuffle->getOperand(0))) {
          if (auto *const src = dyn_cast<Instruction>(ins->getOperand(1))) {
            v = src;
            continue;
          } else {
            // Splat of a non-Instruction (i.e. an Argument)
            return true;
          }
        }
      }
      return false;
    }

    // Consider only certain trivial operations
    if (!I->isBinaryOp() && !I->isCast() && !I->isUnaryOp()) {
      return false;
    }

    Value *chain = nullptr;
    for (auto *op : I->operand_values()) {
      if (auto *const opI = dyn_cast<Instruction>(op)) {
        if (!chain) {
          chain = opI;
        } else if (chain != op) {
          // It's non-trivial if it has more than one Instruction operand.
          return false;
        }
      }
    }

    // It's trivial if it didn't have any operands that were instructions.
    if (!chain) {
      return true;
    }

    v = chain;
  }
  return false;
}

// GEPs typically have a low cost, allow up to 1 non-trivial operand
// (including the pointer operand as well as the indices).
bool IsTrivialGEP(Value *v, SmallVectorImpl<Value *> &operands) {
  auto *const GEP = dyn_cast<GetElementPtrInst>(v);
  if (!GEP) {
    return false;
  }

  unsigned inst_ops = 0;
  for (auto *op : GEP->operand_values()) {
    if (isa<Instruction>(op) && ++inst_ops > 1) {
      return false;
    }
  }

  for (auto *op : GEP->operand_values()) {
    if (isa<Instruction>(op)) {
      operands.push_back(op);
    }
  }
  return true;
}

/// @brief Update all basic block edges for PHINodes, and drop edges from
/// basic blocks that are not in the the new function (which only consists of
/// the subset of blocks that make up one region).
///
/// @param[in] BB Basic block to process.
/// @param[in] vmap Map for value for cloning.
void UpdateAndTrimPHINodeEdges(BasicBlock *BB, ValueToValueMapTy &vmap) {
  for (auto &phi : BB->phis()) {
    for (unsigned i = 0; i < phi.getNumIncomingValues(); i++) {
      const BasicBlock *incoming_bb = phi.getIncomingBlock(i);

      // If the incoming basic block was processed during cloning then
      // update the edge, if it wasn't then it is not in the region so
      // remove it.
      if (vmap.count(incoming_bb)) {
        Value *updated_bb = vmap[incoming_bb];
        phi.setIncomingBlock(i, cast<BasicBlock>(updated_bb));
      } else {
        // Note: Updating the loop iterator to reflect the updated
        // post-deletion indices.
        phi.removeIncomingValue(i--);
      }
    }
  }
}

/// @brief Returns true if the type is a struct type containing any scalable
/// vectors in its list of elements
bool isStructWithScalables(Type *ty) {
  if (auto *const struct_ty = dyn_cast<StructType>(ty)) {
    return any_of(struct_ty->elements(),
                  [](Type *ty) { return isa<ScalableVectorType>(ty); });
  }
  return false;
}

}  // namespace

Value *compiler::utils::Barrier::LiveValuesHelper::getExtractValueGEP(
    const Value *live) {
  if (auto *const extract = dyn_cast<ExtractValueInst>(live)) {
    // We can't handle extracts with multiple indices
    if (extract->getIndices().size() == 1) {
      return getGEP(extract->getAggregateOperand(), extract->getIndices()[0]);
    }
  }
  return nullptr;
}

Value *compiler::utils::Barrier::LiveValuesHelper::getGEP(const Value *live,
                                                          unsigned member_idx) {
  auto key = std::make_pair(live, member_idx);
  if (auto gep_it = live_GEPs.find(key); gep_it != live_GEPs.end()) {
    return gep_it->second;
  }

  Value *gep;
  Type *data_ty = live->getType();
  if (auto *AI = dyn_cast<AllocaInst>(live)) {
    data_ty = AI->getAllocatedType();
  }

  if (auto field_it = barrier.live_variable_index_map_.find(key);
      field_it != barrier.live_variable_index_map_.end()) {
    LLVMContext &context = barrier.module_.getContext();
    const unsigned field_index = field_it->second;
    Value *live_variable_info_idxs[2] = {
        ConstantInt::get(Type::getInt32Ty(context), 0),
        ConstantInt::get(Type::getInt32Ty(context), field_index)};

    gep = gepBuilder.CreateInBoundsGEP(barrier.live_var_mem_ty_, barrier_struct,
                                       live_variable_info_idxs,
                                       Twine("live_gep_") + live->getName());
  } else if (auto field_it = barrier.live_variable_scalables_map_.find(key);
             field_it != barrier.live_variable_scalables_map_.end()) {
    const unsigned field_offset = field_it->second;
    Value *scaled_offset = nullptr;

    LLVMContext &context = barrier.module_.getContext();
    if (field_offset != 0) {
      if (!vscale) {
        Type *size_type = gepBuilder.getIntNTy(barrier.size_t_bytes * 8);
        vscale = gepBuilder.CreateIntrinsic(Intrinsic::vscale, size_type, {});
      }
      scaled_offset = gepBuilder.CreateMul(
          vscale, gepBuilder.getIntN(barrier.size_t_bytes * 8, field_offset));
    } else {
      scaled_offset = ConstantInt::get(Type::getInt32Ty(context), 0);
    }

    Value *live_variable_info_idxs[3] = {
        ConstantInt::get(Type::getInt32Ty(context), 0),
        ConstantInt::get(Type::getInt32Ty(context),
                         barrier.live_var_mem_scalables_index),
        scaled_offset,
    };

    // Gep into the raw byte buffer
    gep = gepBuilder.CreateInBoundsGEP(
        barrier.live_var_mem_ty_, barrier_struct, live_variable_info_idxs,
        Twine("live_gep_scalable_") + live->getName());

    // Cast the pointer to the scalable vector type
    gep = gepBuilder.CreatePointerCast(
        gep,
        PointerType::get(
            data_ty,
            cast<PointerType>(barrier_struct->getType())->getAddressSpace()));
  } else {
    // Fall back and see if this live variable is actually a decomposed
    // structure type.
    return getExtractValueGEP(live);
  }

  // Cache this GEP for later
  live_GEPs[key] = gep;

  return gep;
}

Value *compiler::utils::Barrier::LiveValuesHelper::getReload(Value *live,
                                                             IRBuilderBase &ir,
                                                             const char *name,
                                                             bool reuse) {
  auto &mapped = reloads[live];
  if (reuse && mapped) {
    return mapped;
  }

  if (Value *v = getGEP(live)) {
    if (!isa<AllocaInst>(live)) {
      // If live variable is not allocainst, insert load.
      if (!isStructWithScalables(live->getType())) {
        v = ir.CreateLoad(live->getType(), v, Twine(live->getName(), name));
      } else {
        auto *const struct_ty = cast<StructType>(live->getType());
        // Start off with a poison value, and build the struct up member by
        // member, reloading each member at a time from their respective
        // offsets.
        v = PoisonValue::get(struct_ty);
        for (auto [idx, ty] : enumerate(struct_ty->elements())) {
          auto *const elt_addr = getGEP(live, idx);
          assert(elt_addr && "Could not get address of struct element");
          auto *const reload =
              ir.CreateLoad(ty, elt_addr, Twine(live->getName(), name));
          v = ir.CreateInsertValue(v, reload, idx);
        }
      }
    }
    mapped = v;
    return v;
  }

  if (auto *I = dyn_cast<Instruction>(live)) {
    // Save these
    auto insPoint = ir.GetInsertPoint();
    auto *const insBB = ir.GetInsertBlock();

    if (!reuse || !mapped) {
      auto *clone = I->clone();
      clone->setName(I->getName());
      clone->setDebugLoc(DebugLoc());
      ir.Insert(clone);
      if (gepBuilder.GetInsertPoint() == ir.GetInsertPoint()) {
        gepBuilder.SetInsertPoint(clone);
      }
      ir.SetInsertPoint(clone);
      mapped = clone;
      I = clone;
    } else {
      return mapped;
    }

    for (auto op_it = I->op_begin(); op_it != I->op_end();) {
      auto &op = *op_it++;
      if (auto *op_inst = dyn_cast<Instruction>(op.get())) {
        ir.SetInsertPoint(I);
        op.set(getReload(op_inst, ir, name, reuse));
      }
    }

    // Restore the original insert point
    ir.SetInsertPoint(insBB, insPoint);
    return I;
  }

  return live;
}

void compiler::utils::Barrier::Run(llvm::ModuleAnalysisManager &mam) {
  bi_ = &mam.getResult<BuiltinInfoAnalysis>(module_);
  FindBarriers();

  if (barriers_.empty()) {
    // If there are no barriers, we can use the original function as the
    // single barrier region.
    barrier_graph.emplace_back();
    auto &node = barrier_graph.back();
    node.entry = &func_.getEntryBlock();
    node.id = kBarrier_FirstID;
    node.successor_ids.push_back(kBarrier_EndID);
    kernel_id_map_[kBarrier_FirstID] = &func_;
    return;
  }

  // If we found some barriers, we need to split up our kernel across them!
  {
    ModulePassManager pm;
    // It's convenient to create LCSSA PHI nodes to stop values defined
    // within a loop being stored to the barrier unnecessarily on every
    // iteration (if, for instance, the loop is entirely between two
    // barriers, but the value is used outside of that barrier region).
    pm.addPass(llvm::createModuleToFunctionPassAdaptor(LCSSAPass()));
    pm.run(module_, mam);
    mam.invalidate(module_, PreservedAnalyses::allInSet<CFGAnalyses>());
  }

  // Do the splitting first in case a value is used on both sides of a barrier
  // within the same basic block.
  SplitBlockwithBarrier();
  FindLiveVariables();

  // Tidy up the barrier struct, removing values that we can
  // reload/rematerialize on the other side of the barrier.
  // NB: We don't do this if any of the barriers is a work-group broadcast. In
  // the case that a broadcasted value is non-uniform (i.e., it depends on
  // work-item builtins), we must preserve it in the barrier struct! This is
  // because we can't rematerialize the local ID and broadcast that; we need
  // to broadcast the specific local ID for the broadcasted work-item.
  // This is very crude. We could either:
  // 1. Trace through all candidate values we want to remove and ensure they're
  // not being broadcasted.
  // 2. Add some more advanced rematerialization logic to substitute
  // rematerializable work-item functions with values specific to a given
  // work-item. Note that the builtins we rematerialize are ultimately up to
  // the BuiltinInfo to identify, so we can't assume anything here and would
  // have to defer back to the BuiltinInfo to do this correctly.
  if (llvm::none_of(barriers_, [this](llvm::CallInst *const CI) {
        auto Info = getWorkGroupCollectiveCall(CI, *bi_);
        return Info && Info->isBroadcast();
      })) {
    TidyLiveVariables();
  }

  MakeLiveVariableMemType();
  SeperateKernelWithBarrier();
}

void compiler::utils::Barrier::replaceSubkernel(Function *from, Function *to) {
  for (auto &k : kernel_id_map_) {
    if (k.second == from) {
      k.second = to;
    }
  }
}

/// @brief Find Barriers.
void compiler::utils::Barrier::FindBarriers() {
  SmallVector<std::pair<unsigned, CallInst *>, 8> orderedBarriers;

  // Check whether current function has barrier or not.
  for (BasicBlock &b : func_) {
    for (Instruction &bi : b) {
      // Check call instructions for barrier.
      if (CallInst *call_inst = dyn_cast<CallInst>(&bi)) {
        Function *callee = call_inst->getCalledFunction();
        if (callee != nullptr) {
          const auto B = bi_->analyzeBuiltin(*callee);
          if (BuiltinInfo::isMuxBuiltinWithWGBarrierID(B.ID)) {
            unsigned id = ~0u;
            auto *const id_param = call_inst->getOperand(0);
            if (auto *const id_param_c = dyn_cast<ConstantInt>(id_param)) {
              id = id_param_c->getZExtValue();
            }
            orderedBarriers.emplace_back(id, call_inst);
          }
        }
      }
    }
  }

  std::sort(orderedBarriers.begin(), orderedBarriers.end());
  for (const auto &barrier : orderedBarriers) {
    barriers_.push_back(barrier.second);
  }
}

/// @brief Split block with barrier.
void compiler::utils::Barrier::SplitBlockwithBarrier() {
  // If debugging, create stub functions in the module which will be invoked
  // before each barrier, and after each barrier, by every work item.
  Function *entry_stub = nullptr;
  Function *exit_stub = nullptr;
  if (is_debug_) {
    CallingConv::ID stub_cc;
    if (func_.getCallingConv() == CallingConv::SPIR_KERNEL) {
      stub_cc = CallingConv::SPIR_FUNC;
    } else {
      stub_cc = func_.getCallingConv();
    }
    entry_stub = MakeStubFunction("__barrier_entry", module_, stub_cc);
    exit_stub = MakeStubFunction("__barrier_exit", module_, stub_cc);
  }

  barrier_graph.emplace_back();
  auto &node = barrier_graph.back();
  node.entry = &func_.getEntryBlock();
  node.id = kBarrier_FirstID;

  unsigned barrier_id = kBarrier_StartNewID;
  for (CallInst *split_point : barriers_) {
    if (is_debug_) {
      assert(entry_stub != nullptr);  // Guaranteed as is_debug_ is const.
      assert(exit_stub != nullptr);   // Guaranteed as is_debug_ is const.

      // Create call instructions invoking debug stubs for every barrier. We
      // don't insert these into a basic block yet since we want to insert
      // them at a point where live variables have already been loaded. This
      // info won't be available till later.

      // ID identifying which barrier invoked stub used as argument to call.
      // This number monotonically increases from 0 for each barrier.
      auto id = ConstantInt::get(Type::getInt32Ty(module_.getContext()),
                                 barrier_id - kBarrier_StartNewID);
      // Call invoking entry stub
      auto entry_caller =
          CallInst::Create(entry_stub, id, "", (Instruction *)nullptr);
      entry_caller->setDebugLoc(split_point->getDebugLoc());
      entry_caller->setCallingConv(entry_stub->getCallingConv());

      // Call invoking exit stub
      auto exit_caller =
          CallInst::Create(exit_stub, id, "", (Instruction *)nullptr);
      exit_caller->setDebugLoc(split_point->getDebugLoc());
      exit_caller->setCallingConv(exit_stub->getCallingConv());

      // Store call instructions in map for later insertion
      barrier_stub_call_map_[barrier_id] =
          std::make_pair(entry_caller, exit_caller);
    }

    barrier_graph.emplace_back();
    auto &node = barrier_graph.back();
    node.barrier_inst = split_point;
    node.id = barrier_id++;
    node.schedule = getBarrierSchedule(*split_point);

    // Our scan implementation requires a linear work-item ordering, to loop
    // over all of the 'main' and 'tail' work-items in order.
    if (auto collective = getWorkGroupCollectiveCall(split_point, *bi_)) {
      if (collective->isScan()) {
        node.schedule = BarrierSchedule::Linear;
      }
    }

    split_point->getParent()->splitBasicBlock(split_point, "barrier");
  }

  // We have to gather the basic block data after splitting, because we
  // might not be processing barriers in program order, and things can get
  // awfully confused.
  for (auto &node : barrier_graph) {
    if (node.barrier_inst) {
      auto *const bb = node.barrier_inst->getParent();
      barrier_id_map_[bb] = node.id;
      barrier_successor_set_.insert(*predecessors(bb).begin());
      node.entry = bb;
    }
  }
}

/// @brief Generate an empty kernel that only duplicates the source kernel's
/// CFG
///
/// This is used to do a "dry run" of kernel splitting in order to obtain the
/// dominator tree, which is needed for correct identification of values that
/// cross the barrier.
///
/// @param[in] region the region to clone into the new kernel.
/// @param[out] bbmap a mapping of original blocks onto the empty clones.
/// @return the fake kernel
Function *compiler::utils::Barrier::GenerateFakeKernel(
    BarrierRegion &region, DenseMap<BasicBlock *, BasicBlock *> &bbmap) {
  LLVMContext &context = module_.getContext();

  // Make new kernel function.
  FunctionType *new_fty = FunctionType::get(Type::getVoidTy(context), false);
  Function *new_kernel =
      Function::Create(new_fty, Function::InternalLinkage, "tmp", &module_);
  ValueToValueMapTy vmap;

  for (auto *bb : region.blocks) {
    BasicBlock *new_bb = BasicBlock::Create(context, "", new_kernel);
    if (region.barrier_blocks.count(bb)) {
      ReturnInst::Create(context, nullptr, new_bb);
    } else {
      bb->getTerminator()->clone()->insertInto(new_bb, new_bb->end());
    }
    vmap[bb] = new_bb;
    bbmap[bb] = new_bb;
  }

  const RemapFlags remapFlags =
      RF_IgnoreMissingLocals | llvm::RF_ReuseAndMutateDistinctMDs;
  for (auto &f : *new_kernel) {
    auto *term = f.getTerminator();
    RemapInstruction(term, vmap, remapFlags);
  }

  return new_kernel;
}

/// @brief Obtain a set of Basic Blocks for an inter-barrier region
///
/// It traverses the CFG, following successors, until it hits a barrier,
/// building the region's internal data.
///
/// @param[out] region the region to process
void compiler::utils::Barrier::GatherBarrierRegionBlocks(
    BarrierRegion &region) {
  DenseSet<BasicBlock *> visited;
  region.blocks.push_back(region.entry);
  visited.insert(region.entry);
  size_t index = 0;
  while (index < region.blocks.size()) {
    BasicBlock *BB = region.blocks[index++];
    if (barrier_successor_set_.count(BB)) {
      region.barrier_blocks.insert(BB);
    } else {
      for (BasicBlock *succ : successors(BB)) {
        if (visited.insert(succ).second) {
          region.blocks.push_back(succ);
        }
      }
    }
  }
}

/// @brief Obtain a set of Values used in a region that cross a barrier
///
/// A value use crosses a barrier in the following cases:
/// * Its use is not in the same region as the defintion
/// * Its definition does not dominate the use
///
/// @param[in] region The inter-barrier region
/// @param[in] ignore set of values to ignore
void compiler::utils::Barrier::GatherBarrierRegionUses(
    BarrierRegion &region, DenseSet<Value *> &ignore) {
  DenseMap<BasicBlock *, BasicBlock *> bbmap;
  Function *fake_func = GenerateFakeKernel(region, bbmap);

  // We should check the dominance relation between definition bb of live
  // variables and user bb. If def bb does not dominate user bb, the user is
  // modified by live variable information.
  DominatorTree DT;
  DT.recalculate(*fake_func);

  for (auto *BB : region.blocks) {
    BasicBlock *BBclone = bbmap[BB];
    for (auto &I : *BB) {
      if (PHINode *pn = dyn_cast<PHINode>(&I)) {
        for (unsigned i = 0, e = pn->getNumIncomingValues(); i != e; i++) {
          Value *val = pn->getIncomingValue(i);
          if (CheckValidUse(val) && !ignore.count(val)) {
            if (auto *inst = dyn_cast<Instruction>(val)) {
              BasicBlock *incoming = pn->getIncomingBlock(i);
              BasicBlock *parent = inst->getParent();
              // If the incoming edge comes from outside the region, it is
              // going to get removed anyway, so disregard it
              if (bbmap.count(incoming)) {
                if (!bbmap.count(parent)) {
                  region.uses_ext.insert(val);
                } else if (!DT.dominates(bbmap[parent], bbmap[incoming])) {
                  region.uses_int.insert(val);
                }
              }
            }
          }
        }
      } else {
        for (Value *val : I.operands()) {
          if (CheckValidUse(val) && !ignore.count(val)) {
            if (auto *inst = dyn_cast<Instruction>(val)) {
              BasicBlock *parent = inst->getParent();
              if (!bbmap.count(parent)) {
                region.uses_ext.insert(val);
              } else if (!DT.dominates(bbmap[parent], BBclone)) {
                region.uses_int.insert(val);
              }
            }
          }
        }
      }
      if (CheckValidDef(&I) && !I.use_empty()) {
        region.defs.insert(&I);
      }
    }
  }
  DT.reset();
  fake_func->eraseFromParent();
}

/// @brief Find livein and liveout variables per each basic block.
void compiler::utils::Barrier::FindLiveVariables() {
  DenseSet<Value *> func_args;
  for (Argument &arg : func_.args()) {
    func_args.insert(&arg);
  }

#ifndef NDEBUG
  // Make sure there aren't any stray allocas outside the entry block.
  for (auto block = func_.begin(); ++block != func_.end();) {
    for (auto &inst : *block) {
      assert(!isa<AllocaInst>(inst) && "Alloca found outside entry block!");
    }
  }
#endif  // ndef NDEBUG

  // Put all the original allocas into the barrier struct, in case they get
  // indirectly referenced from the other side of a barrier.
  for (Instruction &bi : func_.front()) {
    if (isa<AllocaInst>(&bi)) {
      whole_live_variables_set_.insert(&bi);
    } else {
      continue;
    }
  }

  for (auto &region : barrier_graph) {
    GatherBarrierRegionBlocks(region);
    GatherBarrierRegionUses(region, func_args);
    whole_live_variables_set_.set_union(region.uses_int);
    whole_live_variables_set_.set_union(region.uses_ext);
  }
}

/// @brief Remove variables that are better recalculated than stored in the
///        barrier, for instance casts and vector splats.
void compiler::utils::Barrier::TidyLiveVariables() {
  const auto &dl = module_.getDataLayout();

  // Start off by doing a simple sweep of stuff that is better off not in the
  // barrier: vector splats, no-op/widening casts, and single/zero index GEPs
  // since we might as well put their source operand in the barrier, instead.
  SmallVector<Value *, 16> removals;
  SmallVector<Value *, 16> redirects;
  for (auto v : whole_live_variables_set_) {
    if (auto *const shuffle = dyn_cast<ShuffleVectorInst>(v)) {
      if (shuffle->isZeroEltSplat()) {
        // if we remove a vector splat, we have to make sure the scalar
        // source operand is in the barrier instead.
        Value *const op = shuffle->getOperand(0);
        if (auto *const ins = dyn_cast<InsertElementInst>(op)) {
          removals.push_back(v);

          Value *const src = ins->getOperand(1);
          // Put the source instruction in the barrier instead.
          // If it's not an instruction, it is probably a function argument.
          if (isa<Instruction>(src) && !IsTrivialGEP(src, redirects)) {
            redirects.push_back(src);
          }
        }
      }
    } else if (auto *const cast = dyn_cast<CastInst>(v)) {
      if (auto *const src = dyn_cast<Instruction>(cast->getOperand(0))) {
        if (cast->isNoopCast(dl) ||
            (cast->getSrcTy()->getScalarSizeInBits() <
             cast->getDestTy()->getScalarSizeInBits())) {
          removals.push_back(v);

          // Put the source instruction in the barrier instead.
          if (isa<Instruction>(src) && !IsTrivialGEP(src, redirects)) {
            redirects.push_back(src);
          }
        }
      } else {
        // No casts of non-instructions in the barrier, please..
        removals.push_back(v);
      }
    } else if (IsTrivialGEP(v, redirects)) {
      removals.push_back(v);
    }
  }

  // We put the redirects into the barrier first, so that if they in turn
  // turn out to be redundant, we can remove them again.
  whole_live_variables_set_.set_union(redirects);

  // Remove work item calls and casts of arguments or other barrier members.
  for (auto v : whole_live_variables_set_) {
    if (IsTrivialValue(v, 4u, *bi_)) {
      removals.push_back(v);
    } else if (auto *cast = dyn_cast<CastInst>(v)) {
      Value *op = cast->getOperand(0);
      if (whole_live_variables_set_.count(op)) {
        removals.push_back(v);
      }
    }
  }
  whole_live_variables_set_.set_subtract(removals);
}

/// @brief Pad the field types to an alignment by adding an int array if
/// needed
/// @param field_tys The vector of types representing the final structure
/// @param offset The current offset in the structure
/// @param alignment The required alignment
/// @return The new offset (or original offset if no padding needed)
unsigned compiler::utils::Barrier::PadTypeToAlignment(
    SmallVectorImpl<Type *> &field_tys, unsigned offset, unsigned alignment) {
  if (alignment) {
    // check if member is not already aligned
    const unsigned int remainder = offset % alignment;
    if (0 != remainder) {
      // calculate number of padding bytes
      const unsigned int padding = alignment - remainder;

      // Use a byte array to pad struct rather than trying to create
      // an arbitrary intNTy, since this may not be supported by the backend.
      const auto padByteType = Type::getInt8Ty(module_.getContext());
      const auto padByteArrayType = ArrayType::get(padByteType, padding);
      field_tys.push_back(padByteArrayType);

      // bump offset by padding size
      offset += padding;
    }
  }
  return offset;
}

/// @brief Make type for whole live variables.
void compiler::utils::Barrier::MakeLiveVariableMemType() {
  SmallVector<Type *, 128> field_tys;
  max_live_var_alignment = 0;

  const auto &dl = module_.getDataLayout();

  struct member_info {
    /// @brief The root `value` being stored.
    Value *value;
    /// @brief The member index of this member inside `value`, if `value` is a
    /// decomposed structure type. Zero otherwise.
    unsigned member_idx;
    /// @brief The type of `value`, or of the specific member of `value`.
    Type *type;
    /// @brief The alignment of the value being stored
    unsigned alignment;
    /// @brief The size of the value being stored
    unsigned size;
  };

  SmallVector<member_info, 8> barrier_members;
  barrier_members.reserve(whole_live_variables_set_.size());
  for (Value *live_var : whole_live_variables_set_) {
    LLVM_DEBUG(dbgs() << "whole live set:" << *live_var << '\n';
               dbgs() << "type:" << *(live_var->getType()) << '\n';);
    Type *field_ty = live_var->getType();

    Type *member_ty = nullptr;
    unsigned alignment = 0;
    // If allocainst is live variable, get element type of pointer type
    // from field_ty and remember alignment
    if (const auto *AI = dyn_cast<AllocaInst>(live_var)) {
      member_ty = AI->getAllocatedType();
      alignment = AI->getAlign().value();
    } else {
      member_ty = field_ty;
    }

    std::vector<Type *> member_tys = {member_ty};
    // If this is a struct type containing any scalable members, we must
    // decompose the value into its individual components.
    if (isStructWithScalables(member_ty)) {
      member_tys = cast<StructType>(member_ty)->elements().vec();
    }

    for (auto [idx, ty] : enumerate(member_tys)) {
      // For a scalable vector, we need the size of the equivalent fixed vector
      // based on its known minimum size.
      auto member_ty_fixed = ty;
      if (isa<ScalableVectorType>(ty)) {
        auto *const eltTy = multi_llvm::getVectorElementType(ty);
        auto n = multi_llvm::getVectorElementCount(ty).getKnownMinValue();
        member_ty_fixed = VectorType::get(eltTy, ElementCount::getFixed(n));
      }

      // Need to ensure that alloc alignment or preferred alignment is kept
      // in the new struct so pad as necessary.
      const unsigned size = dl.getTypeAllocSize(member_ty_fixed);
      alignment = std::max(dl.getPrefTypeAlign(ty).value(),
                           static_cast<AlignIntTy>(alignment));
      max_live_var_alignment = std::max(alignment, max_live_var_alignment);

      barrier_members.push_back(
          {live_var, static_cast<unsigned>(idx), ty, alignment, size});
    }
  }

  // sort the barrier members by decreasing alignment to minimise the amount
  // of padding required (use a stable sort so it's deterministic)
  std::stable_sort(barrier_members.begin(), barrier_members.end(),
                   [](const member_info &lhs, const member_info &rhs) -> bool {
                     return lhs.alignment > rhs.alignment;
                   });

  // Deal with non-scalable members first
  unsigned offset = 0;
  for (auto &member : barrier_members) {
    if (isa<ScalableVectorType>(member.type)) {
      continue;
    }

    offset = PadTypeToAlignment(field_tys, offset, member.alignment);

    // Check if the alloca has a debug info source variable attached. If
    // so record this and the matching byte offset into the struct.
#if LLVM_VERSION_GREATER_EQUAL(18, 0)
    auto DbgIntrinsics = findDbgDeclares(member.value);
#else
    auto DbgIntrinsics = FindDbgDeclareUses(member.value);
#endif
    for (auto DII : DbgIntrinsics) {
      if (auto dbgDeclare = dyn_cast<DbgDeclareInst>(DII)) {
        debug_intrinsics_.push_back(std::make_pair(dbgDeclare, offset));
      }
    }
#if LLVM_VERSION_GREATER_EQUAL(19, 0)
    const auto DVRDeclares = findDVRDeclares(member.value);
    for (auto *const DVRDeclare : DVRDeclares) {
      debug_variable_records_.push_back(std::make_pair(DVRDeclare, offset));
    }
#endif
    offset += member.size;
    live_variable_index_map_[std::make_pair(member.value, member.member_idx)] =
        field_tys.size();
    field_tys.push_back(member.type);
  }
  // Pad the end of the struct to the max alignment as we are creating an
  // array
  offset = PadTypeToAlignment(field_tys, offset, max_live_var_alignment);
  live_var_mem_size_fixed = offset;  // No more offsets required.

  // Now deal with any scalable members. We reset the offset to zero because
  // scalables are indexed bytewise starting from the beginning of the
  // variable-sized scalables section at the end of the struct.
  SmallVector<Type *, 128> field_tys_scalable;
  offset = 0;
  for (auto &member : barrier_members) {
    if (!isa<ScalableVectorType>(member.type)) {
      continue;
    }

    offset = PadTypeToAlignment(field_tys_scalable, offset, member.alignment);

    live_variable_scalables_map_[std::make_pair(member.value,
                                                member.member_idx)] = offset;
    offset += member.size;
    field_tys_scalable.push_back(member.type);
  }
  // Pad the end of the struct to the max alignment as we are creating an
  // array
  offset =
      PadTypeToAlignment(field_tys_scalable, offset, max_live_var_alignment);
  live_var_mem_size_scalable = offset;  // No more offsets required.

  LLVMContext &context = module_.getContext();
  // if the barrier contains scalables, add a flexible byte array on the end
  if (offset != 0) {
    live_var_mem_scalables_index = field_tys.size();
    field_tys.push_back(ArrayType::get(IntegerType::getInt8Ty(context), 0));
  }

  // Create struct type for live variable memory allocation; we create this
  // even when the type is empty. The big entry point pass depends on this
  // to detect that the barrier pass has been executed.
  SmallString<128> name;
  live_var_mem_ty_ = StructType::create(
      context, field_tys,
      (Twine(func_.getName() + "_live_mem_info")).toStringRef(name), false);

  name.clear();

  LLVM_DEBUG(dbgs() << "Barrier size: " << offset << "\n";
             dbgs() << "whole live set type:" << *(live_var_mem_ty_) << '\n';);
}

/// @brief Generate new kernel from an inter-barrier region such that no call
/// to barriers occur within it.
///
/// @param[in] region the inter-barrier region to create the kernel from
/// @return the new kernel
Function *compiler::utils::Barrier::GenerateNewKernel(BarrierRegion &region) {
  BasicBlock *entry_point = region.entry;
  LLVMContext &context = module_.getContext();

  LLVM_DEBUG(dbgs() << "\n"; unsigned i = 0; for (auto *d : region.blocks) {
    dbgs() << "entry block: " << entry_point->getName() << "\n";
    dbgs() << "region visited path [" << i++ << "] = " << d->getName()
           << "\n\n";
    dbgs() << *d << "\n\n";
  });

  SmallVector<Type *, 8> new_func_params;
  // First kernel adds original kernel's parameters.
  for (const auto &arg : func_.args()) {
    new_func_params.push_back(arg.getType());
  }

  // If we have a work group collective call, we need to create a new argument
  // so that the result can be passed in.
  const bool collective =
      getWorkGroupCollectiveCall(region.barrier_inst, *bi_).has_value();
  if (collective) {
    new_func_params.push_back(region.barrier_inst->getType());
  }

  // Add live variables' parameter as last if there are any.
  const bool hasBarrierStruct = !whole_live_variables_set_.empty() &&
                                region.schedule != BarrierSchedule::Once;
  if (hasBarrierStruct) {
    PointerType *pty = PointerType::get(live_var_mem_ty_, 0);
    new_func_params.push_back(pty);
  }

  // Make new kernel function.
  FunctionType *new_fty = FunctionType::get(Type::getInt32Ty(context),
                                            new_func_params, func_.isVarArg());
  Function *new_kernel =
      Function::Create(new_fty, Function::InternalLinkage,
                       func_.getName() + ".mux-barrier-region", &module_);

  // We don't use exceptions.
  new_kernel->setAttributes(func_.getAttributes());

  // We also want to always inline this function (unless it is noinline).
  if (!new_kernel->hasFnAttribute(Attribute::NoInline)) {
    new_kernel->addFnAttr(Attribute::AlwaysInline);
  }

  // copy the calling convention from the old function, except for
  // SPIR_KERNEL. SPIR_KERNELs need to be split into SPIR_FUNC
  CallingConv::ID new_kernel_cc;
  if (func_.getCallingConv() == CallingConv::SPIR_KERNEL) {
    new_kernel_cc = CallingConv::SPIR_FUNC;
  } else {
    new_kernel_cc = func_.getCallingConv();
  }
  new_kernel->setCallingConv(new_kernel_cc);

  // Copy the metadata into the new kernel ignoring any debug info.
  compiler::utils::copyFunctionMetadata(func_, *new_kernel);

  // We're not interested in these sub-kernels being registered as kernels.
  // While they're technically kernels, they're only ever called from our
  // actual wrapper entry point.
  compiler::utils::dropIsKernel(*new_kernel);

  live_variable_mem_t live_vars_defs_in_kernel;
  ValueToValueMapTy vmap;
  // First kernel follows original kernel's arguments first.
  Function::arg_iterator new_arg = new_kernel->arg_begin();
  for (const auto &arg : func_.args()) {
    vmap[&arg] = &*(new_arg++);
  }

  // Copy a region to the new kernel function.
  bool returns_from_kernel = false;
  for (auto *block : region.blocks) {
    BasicBlock *cloned_bb =
        CloneBasicBlock(block, vmap, "", live_vars_defs_in_kernel, new_kernel);
    vmap[block] = cloned_bb;

    // Remove last terminator from clone block with barrier.
    if (region.barrier_blocks.count(block)) {
      cloned_bb->getTerminator()->eraseFromParent();

      // Return the next barrier's id.
      const unsigned next_barrier_id =
          barrier_id_map_[block->getSingleSuccessor()];
      ConstantInt *barrier_id_cst =
          ConstantInt::get(Type::getInt32Ty(context), next_barrier_id);
      auto new_ret = ReturnInst::Create(context, barrier_id_cst, cloned_bb);

      // Barrier blocks should be unique.
      region.successor_ids.push_back(next_barrier_id);

      // Insert call to debug stub before return if debugging, this stub
      // signifies that we're about to enter the next barrier
      if (is_debug_) {
        // Look up entry call instruction in map
        CallInst *entry_call = barrier_stub_call_map_[next_barrier_id].first;

        // Check for null since if this is the final kernel there won't be
        // a next barrier to have an entry for.
        if (!entry_call) {
          continue;
        }

        // Check if the entry call already has a parent since there can be
        // multiple return instructions in a kernel, if it does then clone
        // the instruction first.
        if (nullptr == entry_call->getParent()) {
          entry_call->insertBefore(new_ret);
        } else {
          entry_call->clone()->insertBefore(new_ret);
        }
      }
    } else if (ReturnInst *ret =
                   dyn_cast<ReturnInst>(cloned_bb->getTerminator())) {
      // Change return instruction with end barrier number.
      ConstantInt *cst_zero =
          ConstantInt::get(Type::getInt32Ty(context), kBarrier_EndID);
      ReturnInst *new_ret = ReturnInst::Create(context, cst_zero, ret);
      ret->replaceAllUsesWith(new_ret);
      ret->eraseFromParent();

      // We can have multiple return points, but should only count it once.
      returns_from_kernel = true;
    }
  }
  if (returns_from_kernel) {
    region.successor_ids.push_back(kBarrier_EndID);
  }
  // Keep things consistent
  std::sort(region.successor_ids.begin(), region.successor_ids.end());

  // Update the incoming edges to phi nodes, and drop edges to basic blocks
  // that are not present in the new function.  Note that this must happen
  // after all the basic blocks have been cloned, so that we know how to
  // update the incoming edges to phi nodes that represent back edges.
  for (auto *block : region.blocks) {
    UpdateAndTrimPHINodeEdges(cast<BasicBlock>(vmap[block]), vmap);
  }

  BasicBlock *new_kernel_entry_block = &(new_kernel->getEntryBlock());
  Instruction *insert_point = new_kernel_entry_block->getFirstNonPHIOrDbg();
  auto *const cloned_barrier_call =
      region.barrier_inst ? insert_point : nullptr;

  // If we have a work group collective call, we need to remap its result from
  // the arguments list.
  if (collective) {
    vmap[insert_point] = &*(new_arg++);
  }

  // The entry kernel might have allocas in it that don't get removed,
  // so better make sure to insert after them.
  while (isa<AllocaInst>(insert_point)) {
    insert_point = insert_point->getNextNonDebugInstruction();
  }

  // It puts all the GEPs at the start of the kernel, but only once
  LiveValuesHelper live_values(
      *this, insert_point,
      hasBarrierStruct ? compiler::utils::getLastArgument(new_kernel)
                       : nullptr);

  // Load live variables and map them.
  // These variables are defined in a different kernel, so we insert the
  // relevant load instructions in the entry block of the kernel.
  {
    // Note that if our barrier is a work group collective, its operand will
    // probably still get reloaded here, even though it's going to get deleted,
    // so we hope that it gets optimized away later, in this case.
    for (const auto cur_live : region.uses_ext) {
      IRBuilder<> insertIR(insert_point);
      vmap[cur_live] = live_values.getReload(cur_live, insertIR, "_load", true);
    }
  }

  SmallVector<Instruction *, 8> allocas_and_intrinsics_to_remove;

  // Store only live variables that are defined in this kernel.
  //
  // We might like to store the variables at the point we hit the barrier.
  // However, this is not always possible because the value definition might
  // not dominate any or all of the exit blocks. Furthermore, if this value
  // is used again in the same kernel after looping around the barrier, we
  // have to be aware that the usage might be expecting the updated value.
  // (This can happen in nested loops, where the outer increment becomes a
  // conditional block.) Therefore, we put the store right after the
  // definition instead.
  for (const auto live_var : live_vars_defs_in_kernel) {
    // If allocainst is live variable and defined in this function, then
    // change the alloca to a GEP directly into the live variables struct
    // otherwise we store the value to the struct. This is needed because
    // it is possible for one live variable to reference another by
    // pointer. When we then save them to the live variable struct they
    // will point to the wrong address. By GEPping directly to the final
    // live struct we resolve this issue as it will always use the final
    // address.
    if (auto *alloca_inst = dyn_cast<AllocaInst>(live_var)) {
      // Check to see if it is still an alloca after vmap. If not we may
      // have processed it before and no work needs doing as we are using
      // the live variable struct directly.
      if (auto *new_alloca_inst = dyn_cast<AllocaInst>(vmap[alloca_inst])) {
        allocas_and_intrinsics_to_remove.push_back(new_alloca_inst);
        // Also remove any assume-like intrinsics that are users of this
        // alloca. These assumptions may not hold. For example, lifetime
        // intrinsics are definitely dangerous, as by directly replacing their
        // alloca operands with the address of the live variable struct, we are
        // telling LLVM that *all* accesses of the live variable struct also
        // start/end at that point, which is not true.
        // Similarly, llvm.assume and llvm.experimental.noalias.scope.decl may
        // hold for the alloca but not the live variables struct.
        for (auto *const user : alloca_inst->users()) {
          if (auto *const intrinsic = dyn_cast<IntrinsicInst>(user);
              intrinsic && intrinsic->isAssumeLikeIntrinsic()) {
            allocas_and_intrinsics_to_remove.push_back(intrinsic);
          }
        }
        // change the vmap to point to the GEP instead of the original alloca
        vmap[live_var] = live_values.getGEP(live_var);
      }
    } else {
      // Place the new store immediately after the definition, but if it's a
      // PHI node we have to make sure to put it after any other PHI nodes.
      Instruction *inst = cast<Instruction>(vmap[live_var]);
      Instruction *insert_point = inst->getNextNonDebugInstruction();
      while (isa<PHINode>(insert_point)) {
        insert_point = insert_point->getNextNonDebugInstruction();
      }
      IRBuilder<> B(insert_point);
      if (!isStructWithScalables(live_var->getType())) {
        auto *addr = live_values.getGEP(live_var);
        B.CreateStore(live_var, addr);
      } else {
        // Store this struct containing scalable members piece-wise
        auto member_tys = cast<StructType>(live_var->getType())->elements();
        for (auto [idx, ty] : enumerate(member_tys)) {
          auto *extract = B.CreateExtractValue(live_var, idx);
          auto *extract_addr = live_values.getGEP(extract);
          assert(extract_addr);
          B.CreateStore(extract, extract_addr);
        }
      }
    }
  }

  // Iterate instruction from insert point at entry basic block.
  insert_point = new_kernel_entry_block->getFirstNonPHIOrDbg();
  const RemapFlags remapFlags =
      RF_IgnoreMissingLocals | llvm::RF_ReuseAndMutateDistinctMDs;
  BasicBlock::iterator b_iter = insert_point->getIterator();
  while (b_iter != new_kernel_entry_block->end()) {
    RemapInstruction(&*b_iter, vmap, remapFlags);
    b_iter++;
  }

  // Remove barrier. We do this after creating stores so that if it's a work
  // group collective, it will have been processed as normal above and written
  // into the barrier struct where needed.
  if (cloned_barrier_call) {
    // When debugging insert a call to the exit debug stub at the insert
    // point, this location is important since all the live variables will
    // have been loaded by this point.
    if (is_debug_) {
      const unsigned barrier_id = barrier_id_map_[entry_point];
      // Get call instruction invoking exit stub from map
      CallInst *exit_caller = barrier_stub_call_map_[barrier_id].second;
      exit_caller->insertAfter(cloned_barrier_call);
      // Use updated debug info scope since call_inst will have had
      // this set by ModifyDebugInfoScopes()
      exit_caller->setDebugLoc(cloned_barrier_call->getDebugLoc());
    }
    if (collective) {
      cloned_barrier_call->replaceAllUsesWith(vmap[cloned_barrier_call]);
    }
    cloned_barrier_call->eraseFromParent();
  }

  // don't remap the first basicblock again..
  Function::iterator cfi = ++(new_kernel->begin());
  const Function::iterator cfie = new_kernel->end();
  for (; cfi != cfie; cfi++) {
    for (Instruction &cbi : *cfi) {
      RemapInstruction(&cbi, vmap, remapFlags);
    }
  }

  // Remove any allocas and their dependent intrinsics that have been replaced
  // by a GEP instruction
  for (auto *inst : allocas_and_intrinsics_to_remove) {
    inst->eraseFromParent();
  }

  // This needs resetting for the sake of any further new GEPs created
  live_values.gepBuilder.SetInsertPoint(
      new_kernel_entry_block->getFirstNonPHIOrDbg());

  // If there are definitions of live variable in this function, process it
  // here. As mentioned above regarding value stores, the user might want to
  // load the value after it has been updated. Therefore, we place the new
  // loads right before their uses.
  //
  // Potentially, this is not optimal, since it might create multiple loads.
  // Ideally we should use some kind of reachability query to determine if
  // the load can be placed before the store, and if not, PHI nodes could
  // be inserted instead to get the value directly from the new definition.
  //
  // It would be nice not to have to build the Dominator Tree here again,
  // since we already did it when we gathered the barrier crossing values.
  // The problem is it's a use/user pair that crosses a barrier, not just the
  // use itself. Some users may be dominated, and others not.
  //
  // NOTE it is impossible for any of these to be an Alloca.
  DominatorTree DT;
  DT.recalculate(*new_kernel);

  for (auto OldDef : region.uses_int) {
    Instruction *NewDef = cast<Instruction>(vmap[OldDef]);
    BasicBlock *DefBB = NewDef->getParent();

    for (auto use_it = NewDef->use_begin(); use_it != NewDef->use_end();) {
      auto &U = *use_it++;
      Instruction *UserInst = cast<Instruction>(U.getUser());
      BasicBlock *UserBB = UserInst->getParent();

      // Check whether user is in current function.
      if (UserBB->getParent() == new_kernel) {
        Instruction *load_insert = nullptr;

        // Check dominance relation between def bb and user bb.
        if (auto *PHI = dyn_cast<PHINode>(UserInst)) {
          BasicBlock *incoming = PHI->getIncomingBlock(U);
          if (!DT.dominates(DefBB, incoming)) {
            load_insert = incoming->getTerminator();
          }
        } else if (!DT.dominates(DefBB, UserBB)) {
          load_insert = UserInst;
        }

        if (load_insert) {
          IRBuilder<> loadIR(load_insert);
          U.set(live_values.getReload(OldDef, loadIR, "_reload"));
        }
      }
    }
  }

  // Removing incoming PHI node edges might have created some redundant ones.
  for (auto *BB : region.blocks) {
    BasicBlock *cBB = cast<BasicBlock>(vmap[BB]);
    for (auto I = cBB->begin(); I != cBB->end();) {
      if (auto *PHI = dyn_cast<PHINode>(&*(I++))) {
        if (auto *V = PHI->hasConstantValue()) {
          PHI->replaceAllUsesWith(V);
          PHI->eraseFromParent();
        }
      } else {
        break;
      }
    }
  }

  // Remap any remaining unmapped instructions coming from DT-based reloads
  for (auto &BB : *new_kernel) {
    for (Instruction &I : BB) {
      RemapInstruction(&I, vmap, remapFlags);
    }
  }

  LLVM_DEBUG(dbgs() << "new kernel function: " << new_kernel->getName()
                    << "\n";);
  return new_kernel;
}

/// @brief This function is a copy from llvm::CloneBasicBlock. In order to
/// update live variable information, some of codes are added.
///
/// @param[in] bb Basic block to copy.
/// @param[out] vmap Map for value for cloning.
/// @param[in] name_suffix Name for suffix.
/// @param[out] live_defs_info Live definitions' info current basic block.
/// @param[in] F Current function.
///
/// @return Return cloned basic block.
BasicBlock *compiler::utils::Barrier::CloneBasicBlock(
    BasicBlock *bb, ValueToValueMapTy &vmap, const Twine &name_suffix,
    live_variable_mem_t &live_defs_info, Function *F) {
  BasicBlock *new_bb = BasicBlock::Create(bb->getContext(), "", F);
  if (bb->hasName()) new_bb->setName(bb->getName() + name_suffix);

  // Loop over all instructions, and copy them over.
  for (Instruction &i : *bb) {
    // Don't clone over debug intrinsics since we're going to create them
    // manually later.
    if (isa<DbgDeclareInst>(&i)) {
      continue;
    }

    Instruction *new_inst = i.clone();
    if (i.hasName()) new_inst->setName(i.getName() + name_suffix);
    new_inst->insertInto(new_bb, new_bb->end());

    // Record live variables' defs which are in current kernel.
    if (whole_live_variables_set_.count(&i)) {
      live_defs_info.insert(&i);
    }

    vmap[&i] = new_inst;
  }
  return new_bb;
}

/// @brief Seperate kernel function with barrier boundary.
void compiler::utils::Barrier::SeperateKernelWithBarrier() {
  if (barriers_.empty()) return;

  for (auto &region : barrier_graph) {
    kernel_id_map_[region.id] = GenerateNewKernel(region);
  }

  // Record barrier information on metadata.
  SmallString<128> name;
  LLVMContext &context = module_.getContext();
  ValueAsMetadata *num_barriers_ = ValueAsMetadata::get(
      ConstantInt::get(Type::getInt32Ty(context), barriers_.size()));
  MDNode *num_barriers__md =
      MDNode::get(context, ArrayRef<Metadata *>(num_barriers_));
  NamedMDNode *barrier_md = module_.getOrInsertNamedMetadata(
      Twine(func_.getName() + "_barrier").toStringRef(name));
  barrier_md->addOperand(num_barriers__md);

  LLVM_DEBUG({
    for (const auto &kid : kernel_id_map_) {
      dbgs() << "1. kernel_id[" << kid.first << "] = " << kid.second->getName()
             << "\n";
    }

    for (unsigned i = kBarrier_FirstID;
         i < kernel_id_map_.size() + kBarrier_FirstID; i++) {
      dbgs() << "2. kernel_id[" << i << "] = " << kernel_id_map_[i]->getName()
             << "\n";
    }
    dbgs() << "\n\n" << module_ << "\n\n";
  });
}
