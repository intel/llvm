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

#include <compiler/utils/address_spaces.h>
#include <compiler/utils/builtin_info.h>
#include <compiler/utils/dma.h>
#include <compiler/utils/group_collective_helpers.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/pass_functions.h>
#include <compiler/utils/scheduling.h>
#include <compiler/utils/target_extension_types.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ModRef.h>
#include <multi_llvm/llvm_version.h>
#include <multi_llvm/multi_llvm.h>

#include <optional>

using namespace llvm;

namespace compiler {
namespace utils {

namespace SchedParamIndices {
enum {
  WI = 0,
  WG = 1,
  TOTAL = 2,
};
}

static Function *defineLocalWorkItemBuiltin(BIMuxInfoConcept &BI, BuiltinID ID,
                                            Module &M) {
  // Simple 'local' work-item getters and setters.
  bool IsSetter = false;
  bool HasRankArg = false;
  std::optional<WorkItemInfoStructField::Type> WIFieldIdx;
  switch (ID) {
    default:
      return nullptr;
    case eMuxBuiltinSetLocalId:
      IsSetter = true;
      LLVM_FALLTHROUGH;
    case eMuxBuiltinGetLocalId:
      HasRankArg = true;
      WIFieldIdx = WorkItemInfoStructField::local_id;
      break;
    case eMuxBuiltinSetSubGroupId:
      IsSetter = true;
      LLVM_FALLTHROUGH;
    case eMuxBuiltinGetSubGroupId:
      WIFieldIdx = WorkItemInfoStructField::sub_group_id;
      break;
    case eMuxBuiltinSetNumSubGroups:
      IsSetter = true;
      LLVM_FALLTHROUGH;
    case eMuxBuiltinGetNumSubGroups:
      WIFieldIdx = WorkItemInfoStructField::num_sub_groups;
      break;
    case eMuxBuiltinSetMaxSubGroupSize:
      IsSetter = true;
      LLVM_FALLTHROUGH;
    case eMuxBuiltinGetMaxSubGroupSize:
      WIFieldIdx = WorkItemInfoStructField::max_sub_group_size;
      break;
  }

  Function *F = M.getFunction(BuiltinInfo::getMuxBuiltinName(ID));
  assert(F && WIFieldIdx);

  // Gather up the list of scheduling parameters on this builtin
  const auto &SchedParams = BI.getFunctionSchedulingParameters(*F);
  assert(SchedParamIndices::WI < SchedParams.size());

  // Grab the work-item info argument
  const auto &SchedParam = SchedParams[SchedParamIndices::WI];
  auto *const StructTy = dyn_cast<StructType>(SchedParam.ParamPointeeTy);
  assert(SchedParam.ArgVal && StructTy == getWorkItemInfoStructTy(M) &&
         "Inconsistent scheduling parameter data");

  if (IsSetter) {
    populateStructSetterFunction(*F, *SchedParam.ArgVal, StructTy, *WIFieldIdx,
                                 HasRankArg);
  } else {
    populateStructGetterFunction(*F, *SchedParam.ArgVal, StructTy, *WIFieldIdx,
                                 HasRankArg);
  }

  return F;
}

static Function *defineLocalWorkGroupBuiltin(BIMuxInfoConcept &BI, BuiltinID ID,
                                             Module &M) {
  // Simple work-group getters
  bool HasRankArg = true;
  size_t DefaultVal = 0;
  std::optional<WorkGroupInfoStructField::Type> WGFieldIdx;
  switch (ID) {
    default:
      return nullptr;
    case eMuxBuiltinGetLocalSize:
      DefaultVal = 1;
      WGFieldIdx = WorkGroupInfoStructField::local_size;
      break;
    case eMuxBuiltinGetGroupId:
      DefaultVal = 0;
      WGFieldIdx = WorkGroupInfoStructField::group_id;
      break;
    case eMuxBuiltinGetNumGroups:
      DefaultVal = 1;
      WGFieldIdx = WorkGroupInfoStructField::num_groups;
      break;
    case eMuxBuiltinGetGlobalOffset:
      DefaultVal = 0;
      WGFieldIdx = WorkGroupInfoStructField::global_offset;
      break;
    case eMuxBuiltinGetWorkDim:
      DefaultVal = 1;
      HasRankArg = false;
      WGFieldIdx = WorkGroupInfoStructField::work_dim;
      break;
  }

  Function *F = M.getFunction(BuiltinInfo::getMuxBuiltinName(ID));
  assert(F && WGFieldIdx);

  // Gather up the list of scheduling parameters on this builtin
  const auto &SchedParams = BI.getFunctionSchedulingParameters(*F);
  assert(SchedParamIndices::WG < SchedParams.size());

  // Grab the work-group info argument
  const auto &SchedParam = SchedParams[SchedParamIndices::WG];
  auto *const StructTy = dyn_cast<StructType>(SchedParam.ParamPointeeTy);
  assert(SchedParam.ArgVal && StructTy == getWorkGroupInfoStructTy(M) &&
         "Inconsistent scheduling parameter data");

  populateStructGetterFunction(*F, *SchedParam.ArgVal, StructTy, *WGFieldIdx,
                               HasRankArg, DefaultVal);
  return F;
}

// FIXME: Assumes a sub-group size of 1.
static Function *defineSubGroupGroupOpBuiltin(Function &F,
                                              GroupCollective GroupOp,
                                              ArrayRef<Type *> OverloadInfo) {
  if (!GroupOp.isSubGroupScope()) {
    return nullptr;
  }

  auto *Arg = F.getArg(0);

  IRBuilder<> B(BasicBlock::Create(F.getContext(), "entry", &F));

  switch (GroupOp.Op) {
    default:
      llvm_unreachable("Unhandled group operation");
    case GroupCollective::OpKind::Any:
    case GroupCollective::OpKind::All:
    case GroupCollective::OpKind::Broadcast:
    case GroupCollective::OpKind::Reduction:
    case GroupCollective::OpKind::ScanInclusive:
      // In the trivial size=1 case, all of these operations just return the
      // argument back again
      B.CreateRet(Arg);
      break;
    case GroupCollective::OpKind::ScanExclusive: {
      // In the trivial size=1 case, exclusive scans return the identity.
      assert(!OverloadInfo.empty());
      auto *const IdentityVal =
          getIdentityVal(GroupOp.Recurrence, OverloadInfo[0]);
      assert(IdentityVal && "Unable to deduce identity val");
      B.CreateRet(IdentityVal);
      break;
    }
    case GroupCollective::OpKind::Shuffle:
    case GroupCollective::OpKind::ShuffleXor:
      // In the trivial size=1 case, all of these operations just return the
      // argument back again. Any computed shuffle index other than the only
      // one in the sub-group would be out of bounds anyway.
      B.CreateRet(Arg);
      break;
    case GroupCollective::OpKind::ShuffleUp: {
      auto *const Prev = F.getArg(0);
      auto *const Curr = F.getArg(1);
      auto *const Delta = F.getArg(2);
      // In the trivial size=1 case, negative delta is the desired index (since
      // we're subtracting it from zero). If it's greater than zero and less
      // than the size, we return 'current', else if it's less than zero and
      // greater than or equal to the negative size, we return 'prev'. So if
      // 'delta' is zero, return 'current', else return 'prev'. Anything else
      // is out of bounds so we can simplify things here.
      auto *const EqZero = B.CreateICmpEQ(Delta, B.getInt32(0), "eqzero");
      auto *const Sel = B.CreateSelect(EqZero, Curr, Prev, "sel");
      B.CreateRet(Sel);
      break;
    }
    case GroupCollective::OpKind::ShuffleDown: {
      auto *const Curr = F.getArg(0);
      auto *const Next = F.getArg(1);
      auto *const Delta = F.getArg(2);
      // In the trivial size=1 case, the delta is the desired index (since
      // we're adding it to zero). If it's less than the size, we return
      // 'current', else if it's greater or equal to the size but less than
      // twice the size, we return 'next'. So if 'delta' is zero, return
      // 'current', else return 'next'. Anything else is out of bounds so we
      // can simplify things here.
      auto *const EqZero = B.CreateICmpEQ(Delta, B.getInt32(0), "eqzero");
      auto *const Sel = B.CreateSelect(EqZero, Curr, Next, "sel");
      B.CreateRet(Sel);
      break;
    }
  }

  return &F;
}

static Value *createCallHelper(IRBuilder<> &B, Function &F,
                               ArrayRef<Value *> Args) {
  auto *const CI = B.CreateCall(&F, Args);
  CI->setAttributes(F.getAttributes());
  CI->setCallingConv(F.getCallingConv());
  return CI;
}

void BIMuxInfoConcept::setDefaultBuiltinAttributes(Function &F,
                                                   bool AlwaysInline) {
  // Many of our mux builtin functions are marked alwaysinline (unless they're
  // already marked noinline)
  if (AlwaysInline && !F.hasFnAttribute(Attribute::NoInline)) {
    F.addFnAttr(Attribute::AlwaysInline);
  }
  // We never use exceptions
  F.addFnAttr(Attribute::NoUnwind);
  // Recursion is not supported in ComputeMux
  F.addFnAttr(Attribute::NoRecurse);
}

Function *BIMuxInfoConcept::defineGetGlobalId(Module &M) {
  Function *F =
      M.getFunction(BuiltinInfo::getMuxBuiltinName(eMuxBuiltinGetGlobalId));
  assert(F);
  setDefaultBuiltinAttributes(*F);
  F->setLinkage(GlobalValue::InternalLinkage);

  // Create an IR builder with a single basic block in our function
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", F));

  auto *const MuxGetGroupIdFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetGroupId, M);
  auto *const MuxGetGlobalOffsetFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetGlobalOffset, M);
  auto *const MuxGetLocalIdFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalId, M);
  auto *const MuxGetLocalSizeFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalSize, M);
  assert(MuxGetGroupIdFn && MuxGetGlobalOffsetFn && MuxGetLocalIdFn &&
         MuxGetLocalSizeFn);

  // Pass on all arguments through to dependent builtins. We expect that each
  // function has identical prototypes, regardless of whether scheduling
  // parameters have been added
  const SmallVector<Value *, 4> Args(make_pointer_range(F->args()));

  auto *const GetGroupIdCall = createCallHelper(B, *MuxGetGroupIdFn, Args);
  auto *const GetGlobalOffsetCall =
      createCallHelper(B, *MuxGetGlobalOffsetFn, Args);
  auto *const GetLocalIdCall = createCallHelper(B, *MuxGetLocalIdFn, Args);
  auto *const GetLocalSizeCall = createCallHelper(B, *MuxGetLocalSizeFn, Args);

  // (get_group_id(i) * get_local_size(i))
  auto *Ret = B.CreateMul(GetGroupIdCall, GetLocalSizeCall);
  // (get_group_id(i) * get_local_size(i)) + get_local_id(i)
  Ret = B.CreateAdd(Ret, GetLocalIdCall);
  // get_global_id(i) = (get_group_id(i) * get_local_size(i)) +
  //                    get_local_id(i) + get_global_offset(i)
  Ret = B.CreateAdd(Ret, GetGlobalOffsetCall);

  // ... and return our result
  B.CreateRet(Ret);
  return F;
}

// FIXME: Assumes a sub-group size of 1.
Function *BIMuxInfoConcept::defineGetSubGroupSize(Function &F) {
  setDefaultBuiltinAttributes(F);
  F.setLinkage(GlobalValue::InternalLinkage);

  IRBuilder<> B(BasicBlock::Create(F.getContext(), "entry", &F));

  assert(F.getReturnType() == B.getInt32Ty());
  B.CreateRet(B.getInt32(1));

  return &F;
}

// FIXME: Assumes a sub-group size of 1.
Function *BIMuxInfoConcept::defineGetSubGroupLocalId(Function &F) {
  setDefaultBuiltinAttributes(F);
  F.setLinkage(GlobalValue::InternalLinkage);

  IRBuilder<> B(BasicBlock::Create(F.getContext(), "entry", &F));

  assert(F.getReturnType() == B.getInt32Ty());
  B.CreateRet(B.getInt32(0));

  return &F;
}

Function *BIMuxInfoConcept::defineGetGlobalSize(Module &M) {
  Function *F =
      M.getFunction(BuiltinInfo::getMuxBuiltinName(eMuxBuiltinGetGlobalSize));
  assert(F);
  setDefaultBuiltinAttributes(*F);
  F->setLinkage(GlobalValue::InternalLinkage);

  auto *const MuxGetNumGroupsFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetNumGroups, M);
  auto *const MuxGetLocalSizeFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalSize, M);
  assert(MuxGetNumGroupsFn && MuxGetLocalSizeFn);

  // create an IR builder with a single basic block in our function
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "", F));

  // Pass on all arguments through to dependent builtins. We expect that each
  // function has identical prototypes, regardless of whether scheduling
  // parameters have been added
  const SmallVector<Value *, 4> Args(make_pointer_range(F->args()));

  // call get_num_groups
  auto *const GetNumGroupsCall = createCallHelper(B, *MuxGetNumGroupsFn, Args);

  // call get_local_size
  auto *const GetLocalSizeCall = createCallHelper(B, *MuxGetLocalSizeFn, Args);

  // get_global_size(i) = get_num_groups(i) * get_local_size(i)
  auto *const Ret = B.CreateMul(GetNumGroupsCall, GetLocalSizeCall);

  // and return our result
  B.CreateRet(Ret);
  return F;
}

Function *BIMuxInfoConcept::defineGetLocalLinearId(Module &M) {
  Function *F = M.getFunction(
      BuiltinInfo::getMuxBuiltinName(eMuxBuiltinGetLocalLinearId));
  assert(F);
  setDefaultBuiltinAttributes(*F);
  F->setLinkage(GlobalValue::InternalLinkage);

  auto *const MuxGetLocalIdFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalId, M);
  auto *const MuxGetLocalSizeFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalSize, M);
  assert(MuxGetLocalIdFn && MuxGetLocalSizeFn);

  // Create a call to all the required builtins.
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "", F));

  // Pass on all arguments through to dependent builtins. Ignoring the index
  // parameters we'll add, we expect that each function has identical
  // prototypes, regardless of whether scheduling parameters have been added
  SmallVector<Value *, 4> Args(make_pointer_range(F->args()));

  SmallVector<Value *, 4> Idx0Args = {B.getInt32(0)};
  append_range(Idx0Args, Args);
  SmallVector<Value *, 4> Idx1Args = {B.getInt32(1)};
  append_range(Idx1Args, Args);
  SmallVector<Value *, 4> Idx2Args = {B.getInt32(2)};
  append_range(Idx2Args, Args);

  auto *const GetLocalIDXCall = createCallHelper(B, *MuxGetLocalIdFn, Idx0Args);
  auto *const GetLocalIDYCall = createCallHelper(B, *MuxGetLocalIdFn, Idx1Args);
  auto *const GetLocalIDZCall = createCallHelper(B, *MuxGetLocalIdFn, Idx2Args);

  auto *const GetLocalSizeXCall =
      createCallHelper(B, *MuxGetLocalSizeFn, Idx0Args);
  auto *const GetLocalSizeYCall =
      createCallHelper(B, *MuxGetLocalSizeFn, Idx1Args);

  // get_local_id(2) * get_local_size(1).
  auto *ZTerm = B.CreateMul(GetLocalIDZCall, GetLocalSizeYCall);
  // get_local_id(2) * get_local_size(1) * get_local_size(0).
  ZTerm = B.CreateMul(ZTerm, GetLocalSizeXCall);

  // get_local_id(1) * get_local_size(0).
  auto *const YTerm = B.CreateMul(GetLocalIDYCall, GetLocalSizeXCall);

  // get_local_id(2) * get_local_size(1) * get_local_size(0) +
  // get_local_id(1) * get_local_size(0).
  auto *Ret = B.CreateAdd(ZTerm, YTerm);
  // get_local_id(2) * get_local_size(1) * get_local_size(0) +
  // get_local_id(1) * get_local_size(0) + get_local_id(0).
  Ret = B.CreateAdd(Ret, GetLocalIDXCall);

  B.CreateRet(Ret);
  return F;
}

Function *BIMuxInfoConcept::defineGetGlobalLinearId(Module &M) {
  Function *F = M.getFunction(
      BuiltinInfo::getMuxBuiltinName(eMuxBuiltinGetGlobalLinearId));
  assert(F);
  setDefaultBuiltinAttributes(*F);
  F->setLinkage(GlobalValue::InternalLinkage);

  auto *const MuxGetGlobalIdFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetGlobalId, M);
  auto *const MuxGetGlobalOffsetFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetGlobalOffset, M);
  auto *const MuxGetGlobalSizeFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetGlobalSize, M);
  assert(MuxGetGlobalIdFn && MuxGetGlobalOffsetFn && MuxGetGlobalSizeFn);

  // Create a call to all the required builtins.
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "", F));

  // Pass on all arguments through to dependent builtins. Ignoring the index
  // parameters we'll add, we expect that each function has identical
  // prototypes, regardless of whether scheduling parameters have been added
  SmallVector<Value *, 4> Args(make_pointer_range(F->args()));

  SmallVector<Value *, 4> Idx0Args = {B.getInt32(0)};
  append_range(Idx0Args, Args);
  SmallVector<Value *, 4> Idx1Args = {B.getInt32(1)};
  append_range(Idx1Args, Args);
  SmallVector<Value *, 4> Idx2Args = {B.getInt32(2)};
  append_range(Idx2Args, Args);

  auto *const GetGlobalIDXCall =
      createCallHelper(B, *MuxGetGlobalIdFn, Idx0Args);
  auto *const GetGlobalIDYCall =
      createCallHelper(B, *MuxGetGlobalIdFn, Idx1Args);
  auto *const GetGlobalIDZCall =
      createCallHelper(B, *MuxGetGlobalIdFn, Idx2Args);

  auto *const GetGlobalOffsetXCall =
      createCallHelper(B, *MuxGetGlobalOffsetFn, Idx0Args);
  auto *const GetGlobalOffsetYCall =
      createCallHelper(B, *MuxGetGlobalOffsetFn, Idx1Args);
  auto *const GetGlobalOffsetZCall =
      createCallHelper(B, *MuxGetGlobalOffsetFn, Idx2Args);

  auto *const GetGlobalSizeXCall =
      createCallHelper(B, *MuxGetGlobalSizeFn, Idx0Args);
  auto *const GetGlobalSizeYCall =
      createCallHelper(B, *MuxGetGlobalSizeFn, Idx1Args);

  // global linear id is calculated as follows:
  // get_global_linear_id() =
  // (get_global_id(2) - get_global_offset(2)) * get_global_size(1) *
  // get_global_size(0) + (get_global_id(1) - get_global_offset(1)) *
  // get_global_size(0) + get_global_id(0) - get_global_offset(0).
  // =
  // ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) +
  // get_global_id(1) - get_global_offset(1)) * get_global_size(0) +
  // get_global_id(0) - get_global_offset(0).

  auto *ZTerm = B.CreateSub(GetGlobalIDZCall, GetGlobalOffsetZCall);
  // (get_global_id(2) - get_global_offset(2)) * get_global_size(1).
  ZTerm = B.CreateMul(ZTerm, GetGlobalSizeYCall);

  // get_global_id(1) - get_global_offset(1).
  auto *const YTerm = B.CreateSub(GetGlobalIDYCall, GetGlobalOffsetYCall);

  // (get_global_id(2) - get_global_offset(2)) * get_global_size(1) +
  // get_global_id(1) - get_global_offset(1)
  auto *YZTermsCombined = B.CreateAdd(ZTerm, YTerm);

  // ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) +
  // get_global_id(1) - get_global_offset(1)) * get_global_size(0).
  YZTermsCombined = B.CreateMul(YZTermsCombined, GetGlobalSizeXCall);

  // get_global_id(0) - get_global_offset(0).
  auto *const XTerm = B.CreateSub(GetGlobalIDXCall, GetGlobalOffsetXCall);

  // ((get_global_id(2) - get_global_offset(2)) * get_global_size(1) +
  // get_global_id(1) - get_global_offset(1)) * get_global_size(0) +
  // get_global_id(0) - get_global_offset(0).
  auto *const Ret = B.CreateAdd(XTerm, YZTermsCombined);

  B.CreateRet(Ret);
  return F;
}

Function *BIMuxInfoConcept::defineGetEnqueuedLocalSize(Module &M) {
  Function *F = M.getFunction(
      BuiltinInfo::getMuxBuiltinName(eMuxBuiltinGetEnqueuedLocalSize));
  assert(F);
  setDefaultBuiltinAttributes(*F);
  F->setLinkage(GlobalValue::InternalLinkage);

  auto *const MuxGetLocalSizeFn =
      getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalSize, M);
  assert(MuxGetLocalSizeFn);

  IRBuilder<> B(BasicBlock::Create(M.getContext(), "", F));

  // Pass on all arguments through to dependent builtins. We expect that each
  // function has identical prototypes, regardless of whether scheduling
  // parameters have been added
  const SmallVector<Value *, 4> Args(make_pointer_range(F->args()));

  // Since we don't support non-uniform subgroups
  // get_enqueued_local_size(x) == get_local_size(x).
  auto *const GetLocalSize = createCallHelper(B, *MuxGetLocalSizeFn, Args);

  B.CreateRet(GetLocalSize);
  return F;
}

Function *BIMuxInfoConcept::defineMemBarrier(Function &F, unsigned,
                                             unsigned SemanticsIdx) {
  // FIXME: We're ignoring some operands here. We're dropping the 'scope' but
  // our set of default set of targets can't make use of anything but a
  // single-threaded fence. We're also ignoring the kind of memory being
  // controlled by the barrier.
  // See CA-2997 and CA-3042 for related discussions.
  auto &M = *F.getParent();
  setDefaultBuiltinAttributes(F);
  F.setLinkage(GlobalValue::InternalLinkage);
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "", &F));

  // Grab the semantics argument.
  Value *Semantics = F.getArg(SemanticsIdx);
  // Mask out only the memory ordering value.
  Semantics = B.CreateAnd(Semantics, B.getInt32(MemSemanticsMask));

  // Don't insert this exit block just yet
  auto *const ExitBB = BasicBlock::Create(M.getContext(), "exit");

  auto *const DefaultBB =
      BasicBlock::Create(M.getContext(), "case.default", &F);
  auto *const Switch = B.CreateSwitch(Semantics, DefaultBB);

  const struct {
    StringRef Name;
    unsigned SwitchVal;
    AtomicOrdering Ordering;
  } Data[4] = {
      {"case.acquire", MemSemanticsAcquire, AtomicOrdering::Acquire},
      {"case.release", MemSemanticsRelease, AtomicOrdering::Release},
      {"case.acq_rel", MemSemanticsAcquireRelease,
       AtomicOrdering::AcquireRelease},
      {"case.seq_cst", MemSemanticsSequentiallyConsistent,
       AtomicOrdering::SequentiallyConsistent},
  };

  for (const auto &D : Data) {
    auto *const BB = BasicBlock::Create(M.getContext(), D.Name, &F);

    Switch->addCase(B.getInt32(D.SwitchVal), BB);
    B.SetInsertPoint(BB);
    B.CreateFence(D.Ordering, SyncScope::SingleThread);
    B.CreateBr(ExitBB);
  }

  // The default case assumes a 'relaxed' ordering and emits no fence
  // whatsoever.
  B.SetInsertPoint(DefaultBB);
  B.CreateBr(ExitBB);

  ExitBB->insertInto(&F);
  B.SetInsertPoint(ExitBB);
  B.CreateRetVoid();

  return &F;
}

static BasicBlock *copy1D(Module &M, BasicBlock &ParentBB, Value *DstPtr,
                          Value *SrcPtr, Value *NumBytes) {
  Type *const I8Ty = IntegerType::get(M.getContext(), 8);

  assert(SrcPtr->getType()->isPointerTy() &&
         "Mux DMA builtins are always byte-accessed");
  assert(DstPtr->getType()->isPointerTy() &&
         "Mux DMA builtins are always byte-accessed");

  compiler::utils::CreateLoopOpts opts;
  opts.IVs = {SrcPtr, DstPtr};
  opts.loopIVNames = {"dma.src", "dma.dst"};

  // This is a simple loop copy a byte at a time from SrcPtr to DstPtr.
  BasicBlock *ExitBB = compiler::utils::createLoop(
      &ParentBB, nullptr, ConstantInt::get(getSizeType(M), 0), NumBytes, opts,
      [&](BasicBlock *BB, Value *X, ArrayRef<Value *> IVsCurr,
          MutableArrayRef<Value *> IVsNext) {
        IRBuilder<> B(BB);
        Value *const CurrentDmaSrcPtr1DPhi = IVsCurr[0];
        Value *const CurrentDmaDstPtr1DPhi = IVsCurr[1];
        Value *load = B.CreateLoad(I8Ty, CurrentDmaSrcPtr1DPhi);
        B.CreateStore(load, CurrentDmaDstPtr1DPhi);
        IVsNext[0] = B.CreateGEP(I8Ty, CurrentDmaSrcPtr1DPhi,
                                 ConstantInt::get(X->getType(), 1));
        IVsNext[1] = B.CreateGEP(I8Ty, CurrentDmaDstPtr1DPhi,
                                 ConstantInt::get(X->getType(), 1));
        return BB;
      });

  return ExitBB;
}

static BasicBlock *copy2D(Module &M, BasicBlock &ParentBB, Value *DstPtr,
                          Value *SrcPtr, Value *LineSizeBytes,
                          Value *LineStrideDst, Value *LineStrideSrc,
                          Value *NumLines) {
  Type *const I8Ty = IntegerType::get(M.getContext(), 8);

  assert(SrcPtr->getType()->isPointerTy() &&
         "Mux DMA builtins are always byte-accessed");
  assert(DstPtr->getType()->isPointerTy() &&
         "Mux DMA builtins are always byte-accessed");

  compiler::utils::CreateLoopOpts opts;
  opts.IVs = {SrcPtr, DstPtr};
  opts.loopIVNames = {"dma.src", "dma.dst"};

  // This is a loop over the range of lines, calling a 1D copy on each line
  BasicBlock *ExitBB = compiler::utils::createLoop(
      &ParentBB, nullptr, ConstantInt::get(getSizeType(M), 0), NumLines, opts,
      [&](BasicBlock *block, Value *, ArrayRef<Value *> IVsCurr,
          MutableArrayRef<Value *> IVsNext) {
        IRBuilder<> loopIr(block);
        Value *CurrentDmaSrcPtrPhi = IVsCurr[0];
        Value *CurrentDmaDstPtrPhi = IVsCurr[1];

        IVsNext[0] = loopIr.CreateGEP(I8Ty, CurrentDmaSrcPtrPhi, LineStrideSrc);
        IVsNext[1] = loopIr.CreateGEP(I8Ty, CurrentDmaDstPtrPhi, LineStrideDst);
        return copy1D(M, *block, CurrentDmaDstPtrPhi, CurrentDmaSrcPtrPhi,
                      LineSizeBytes);
      });

  return ExitBB;
}

Function *BIMuxInfoConcept::defineDMA1D(Function &F) {
  Argument *const ArgDstPtr = F.getArg(0);
  Argument *const ArgSrcPtr = F.getArg(1);
  Argument *const ArgWidth = F.getArg(2);
  Argument *const ArgEvent = F.getArg(3);

  auto &M = *F.getParent();
  auto &Ctx = F.getContext();
  auto *const ExitBB = BasicBlock::Create(Ctx, "exit", &F);
  auto *const LoopEntryBB = BasicBlock::Create(Ctx, "loop_entry", &F, ExitBB);
  auto *const EntryBB = BasicBlock::Create(Ctx, "entry", &F, LoopEntryBB);

  auto *const GetLocalIDFn = getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalId, M);
  compiler::utils::buildThreadCheck(EntryBB, LoopEntryBB, ExitBB,
                                    *GetLocalIDFn);

  BasicBlock *const LoopExitBB =
      copy1D(M, *LoopEntryBB, ArgDstPtr, ArgSrcPtr, ArgWidth);
  IRBuilder<> LoopIRB(LoopExitBB);
  LoopIRB.CreateBr(ExitBB);

  IRBuilder<> ExitIRB(ExitBB);
  ExitIRB.CreateRet(ArgEvent);

  return &F;
}

Function *BIMuxInfoConcept::defineDMA2D(Function &F) {
  Argument *const ArgDstPtr = F.getArg(0);
  Argument *const ArcSrcPtr = F.getArg(1);
  Argument *const ArgWidth = F.getArg(2);
  Argument *const ArgDstStride = F.getArg(3);
  Argument *const ArgSrcStride = F.getArg(4);
  Argument *const ArgNumLines = F.getArg(5);
  Argument *const ArgEvent = F.getArg(6);

  auto &M = *F.getParent();
  auto &Ctx = F.getContext();
  auto *const ExitBB = BasicBlock::Create(Ctx, "exit", &F);
  auto *const LoopEntryBB = BasicBlock::Create(Ctx, "loop_entry", &F, ExitBB);
  auto *const EntryBB = BasicBlock::Create(Ctx, "entry", &F, LoopEntryBB);

  auto *const GetLocalIDFn = getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalId, M);
  compiler::utils::buildThreadCheck(EntryBB, LoopEntryBB, ExitBB,
                                    *GetLocalIDFn);

  // Create a loop around 1D DMA memcpy, adding strides each time.
  BasicBlock *const LoopExitBB =
      copy2D(M, *LoopEntryBB, ArgDstPtr, ArcSrcPtr, ArgWidth, ArgDstStride,
             ArgSrcStride, ArgNumLines);

  IRBuilder<> LoopIRB(LoopExitBB);
  LoopIRB.CreateBr(ExitBB);

  IRBuilder<> ExitIRB(ExitBB);
  ExitIRB.CreateRet(ArgEvent);

  return &F;
}

Function *BIMuxInfoConcept::defineDMA3D(Function &F) {
  Argument *const ArgDstPtr = F.getArg(0);
  Argument *const ArgSrcPtr = F.getArg(1);
  Argument *const ArgLineSize = F.getArg(2);
  Argument *const ArgDstLineStride = F.getArg(3);
  Argument *const ArgSrcLineStride = F.getArg(4);
  Argument *const ArgNumLinesPerPlane = F.getArg(5);
  Argument *const ArgDstPlaneStride = F.getArg(6);
  Argument *const ArgSrcPlaneStride = F.getArg(7);
  Argument *const ArgNumPlanes = F.getArg(8);
  Argument *const ArgEvent = F.getArg(9);

  auto &M = *F.getParent();
  auto &Ctx = F.getContext();
  Type *const I8Ty = IntegerType::get(Ctx, 8);

  auto *const ExitBB = BasicBlock::Create(Ctx, "exit", &F);
  auto *const LoopEntryBB = BasicBlock::Create(Ctx, "loop_entry", &F, ExitBB);
  auto *const EntryBB = BasicBlock::Create(Ctx, "entry", &F, LoopEntryBB);

  auto *const GetLocalIDFn = getOrDeclareMuxBuiltin(eMuxBuiltinGetLocalId, M);
  compiler::utils::buildThreadCheck(EntryBB, LoopEntryBB, ExitBB,
                                    *GetLocalIDFn);

  assert(ArgSrcPtr->getType()->isPointerTy() &&
         "Mux DMA builtins are always byte-accessed");
  assert(ArgDstPtr->getType()->isPointerTy() &&
         "Mux DMA builtins are always byte-accessed");

  compiler::utils::CreateLoopOpts opts;
  opts.IVs = {ArgSrcPtr, ArgDstPtr};
  opts.loopIVNames = {"dma.src", "dma.dst"};

  // Create a loop around 1D DMA memcpy, adding stride, local width each time.
  BasicBlock *LoopExitBB = compiler::utils::createLoop(
      LoopEntryBB, nullptr, ConstantInt::get(getSizeType(M), 0), ArgNumPlanes,
      opts,
      [&](BasicBlock *BB, Value *, ArrayRef<Value *> IVsCurr,
          MutableArrayRef<Value *> IVsNext) {
        IRBuilder<> loopIr(BB);
        Value *CurrentDmaPlaneSrcPtrPhi = IVsCurr[0];
        Value *CurrentDmaPlaneDstPtrPhi = IVsCurr[1];

        IVsNext[0] =
            loopIr.CreateGEP(I8Ty, CurrentDmaPlaneSrcPtrPhi, ArgSrcPlaneStride);
        IVsNext[1] =
            loopIr.CreateGEP(I8Ty, CurrentDmaPlaneDstPtrPhi, ArgDstPlaneStride);

        return copy2D(M, *BB, CurrentDmaPlaneDstPtrPhi,
                      CurrentDmaPlaneSrcPtrPhi, ArgLineSize, ArgDstLineStride,
                      ArgSrcLineStride, ArgNumLinesPerPlane);
      });

  IRBuilder<> LoopExitIRB(LoopExitBB);
  LoopExitIRB.CreateBr(ExitBB);

  IRBuilder<> ExitIRB(ExitBB);
  ExitIRB.CreateRet(ArgEvent);

  return &F;
}

Function *BIMuxInfoConcept::defineDMAWait(Function &F) {
  // By default this function is a simple return-void.
  IRBuilder<> B(BasicBlock::Create(F.getContext(), "entry", &F));
  B.CreateRetVoid();

  return &F;
}

Function *BIMuxInfoConcept::defineMuxBuiltin(BuiltinID ID, Module &M,
                                             ArrayRef<Type *> OverloadInfo) {
  assert(BuiltinInfo::isMuxBuiltinID(ID) && "Only handling mux builtins");
  Function *F = M.getFunction(BuiltinInfo::getMuxBuiltinName(ID, OverloadInfo));
  // FIXME: We'd ideally want to declare it here to reduce pass
  // inter-dependencies.
  assert(F && "Function should have been pre-declared");
  if (!F->isDeclaration()) {
    return F;
  }

  switch (ID) {
    default:
      break;
    case eMuxBuiltinGetGlobalId:
      return defineGetGlobalId(M);
    case eMuxBuiltinGetGlobalSize:
      return defineGetGlobalSize(M);
    case eMuxBuiltinGetLocalLinearId:
      return defineGetLocalLinearId(M);
    case eMuxBuiltinGetGlobalLinearId:
      return defineGetGlobalLinearId(M);
    case eMuxBuiltinGetEnqueuedLocalSize:
      return defineGetEnqueuedLocalSize(M);
    // Just handle the memory synchronization requirements of any barrier
    // builtin. We assume that the control requirements of work-group and
    // sub-group control barriers have been handled by earlier passes.
    case eMuxBuiltinMemBarrier:
      return defineMemBarrier(*F, 0, 1);
    case eMuxBuiltinSubGroupBarrier:
    case eMuxBuiltinWorkGroupBarrier:
      return defineMemBarrier(*F, 1, 2);
    case eMuxBuiltinDMARead1D:
    case eMuxBuiltinDMAWrite1D:
      return defineDMA1D(*F);
    case eMuxBuiltinDMARead2D:
    case eMuxBuiltinDMAWrite2D:
      return defineDMA2D(*F);
    case eMuxBuiltinDMARead3D:
    case eMuxBuiltinDMAWrite3D:
      return defineDMA3D(*F);
    case eMuxBuiltinDMAWait:
      return defineDMAWait(*F);
    case eMuxBuiltinGetSubGroupSize:
      return defineGetSubGroupSize(*F);
    case eMuxBuiltinGetSubGroupLocalId:
      return defineGetSubGroupLocalId(*F);
  }

  if (auto *const NewF = defineLocalWorkItemBuiltin(*this, ID, M)) {
    return NewF;
  }

  if (auto *const NewF = defineLocalWorkGroupBuiltin(*this, ID, M)) {
    return NewF;
  }

  if (auto GroupOp = BuiltinInfo::isMuxGroupCollective(ID)) {
    if (auto *const NewF =
            defineSubGroupGroupOpBuiltin(*F, *GroupOp, OverloadInfo)) {
      return NewF;
    }
  }

  return nullptr;
}

bool BIMuxInfoConcept::requiresSchedulingParameters(BuiltinID ID) {
  switch (ID) {
    default:
      return false;
    case eMuxBuiltinGetLocalId:
    case eMuxBuiltinSetLocalId:
    case eMuxBuiltinGetSubGroupId:
    case eMuxBuiltinSetSubGroupId:
    case eMuxBuiltinGetNumSubGroups:
    case eMuxBuiltinSetNumSubGroups:
    case eMuxBuiltinGetMaxSubGroupSize:
    case eMuxBuiltinSetMaxSubGroupSize:
    case eMuxBuiltinGetLocalLinearId:
      // Work-item struct only
      return true;
    case eMuxBuiltinGetWorkDim:
    case eMuxBuiltinGetGroupId:
    case eMuxBuiltinGetNumGroups:
    case eMuxBuiltinGetGlobalSize:
    case eMuxBuiltinGetLocalSize:
    case eMuxBuiltinGetGlobalOffset:
    case eMuxBuiltinGetEnqueuedLocalSize:
      // Work-group struct only
      return true;
    case eMuxBuiltinGetGlobalId:
    case eMuxBuiltinGetGlobalLinearId:
      // Work-item and work-group structs
      return true;
  }
}

Type *BIMuxInfoConcept::getRemappedTargetExtTy(Type *Ty, Module &M) {
  // We only map target extension types
  assert(Ty && Ty->isTargetExtTy() && "Only expecting target extension types");
  auto &Ctx = Ty->getContext();
  auto *TgtExtTy = cast<TargetExtType>(Ty);

  // Samplers are replaced by default with size_t.
  if (TgtExtTy == compiler::utils::tgtext::getSamplerTy(Ctx)) {
    return getSizeType(M);
  }

  // Events are replaced by default with size_t.
  if (TgtExtTy == compiler::utils::tgtext::getEventTy(Ctx)) {
    return getSizeType(M);
  }

  // *All* images are replaced by default with a pointer in the default address
  // space to the same structure type (i.e., regardless of image dimensions,
  // etc.)
  if (TgtExtTy->getName() == "spirv.Image") {
    return PointerType::getUnqual([&Ctx]() {
      const char *MuxImageTyName = "MuxImage";
      if (auto *STy = StructType::getTypeByName(Ctx, MuxImageTyName)) {
        return STy;
      }
      return StructType::create(Ctx, MuxImageTyName);
    }());
  }

  return nullptr;
}

Function *BIMuxInfoConcept::getOrDeclareMuxBuiltin(
    BuiltinID ID, Module &M, ArrayRef<Type *> OverloadInfo) {
  assert(BuiltinInfo::isMuxBuiltinID(ID) && "Only handling mux builtins");
  auto FnName = BuiltinInfo::getMuxBuiltinName(ID, OverloadInfo);
  if (auto *const F = M.getFunction(FnName)) {
    return F;
  }
  auto &Ctx = M.getContext();
  AttrBuilder AB(Ctx);
  auto *const SizeTy = getSizeType(M);
  auto *const Int32Ty = Type::getInt32Ty(Ctx);
  auto *const VoidTy = Type::getVoidTy(Ctx);

  Type *RetTy = nullptr;
  SmallVector<Type *, 4> ParamTys;
  SmallVector<std::string, 4> ParamNames;

  switch (ID) {
    // Ranked Getters
    case eMuxBuiltinGetLocalId:
    case eMuxBuiltinGetGlobalId:
    case eMuxBuiltinGetLocalSize:
    case eMuxBuiltinGetGlobalSize:
    case eMuxBuiltinGetGlobalOffset:
    case eMuxBuiltinGetNumGroups:
    case eMuxBuiltinGetGroupId:
    case eMuxBuiltinGetEnqueuedLocalSize:
      ParamTys.push_back(Int32Ty);
      ParamNames.push_back("idx");
      LLVM_FALLTHROUGH;
    // Unranked Getters
    case eMuxBuiltinGetWorkDim:
    case eMuxBuiltinGetSubGroupId:
    case eMuxBuiltinGetNumSubGroups:
    case eMuxBuiltinGetSubGroupSize:
    case eMuxBuiltinGetMaxSubGroupSize:
    case eMuxBuiltinGetSubGroupLocalId:
    case eMuxBuiltinGetLocalLinearId:
    case eMuxBuiltinGetGlobalLinearId: {
      // Some builtins return uint, others return size_t
      RetTy = (ID == eMuxBuiltinGetWorkDim || ID == eMuxBuiltinGetSubGroupId ||
               ID == eMuxBuiltinGetNumSubGroups ||
               ID == eMuxBuiltinGetSubGroupSize ||
               ID == eMuxBuiltinGetMaxSubGroupSize ||
               ID == eMuxBuiltinGetSubGroupLocalId)
                  ? Int32Ty
                  : SizeTy;
      // All of our mux getters are readonly - they may never write data
      AB.addMemoryAttr(MemoryEffects::readOnly());
      break;
    }
    // Ranked Setters
    case eMuxBuiltinSetLocalId:
      ParamTys.push_back(Int32Ty);
      ParamNames.push_back("idx");
      LLVM_FALLTHROUGH;
    // Unranked Setters
    case eMuxBuiltinSetSubGroupId:
    case eMuxBuiltinSetNumSubGroups:
    case eMuxBuiltinSetMaxSubGroupSize: {
      RetTy = VoidTy;
      ParamTys.push_back(ID == eMuxBuiltinSetLocalId ? SizeTy : Int32Ty);
      ParamNames.push_back("val");
      break;
    }
    case eMuxBuiltinMemBarrier: {
      RetTy = VoidTy;
      for (auto PName : {"scope", "semantics"}) {
        ParamTys.push_back(Int32Ty);
        ParamNames.push_back(PName);
      }
      AB.addAttribute(Attribute::NoMerge);
      AB.addAttribute(Attribute::NoDuplicate);
      AB.addAttribute(Attribute::Convergent);
      break;
    }
    case eMuxBuiltinSubGroupBarrier:
    case eMuxBuiltinWorkGroupBarrier: {
      RetTy = VoidTy;
      for (auto PName : {"id", "scope", "semantics"}) {
        ParamTys.push_back(Int32Ty);
        ParamNames.push_back(PName);
      }
      AB.addAttribute(Attribute::NoMerge);
      AB.addAttribute(Attribute::NoDuplicate);
      AB.addAttribute(Attribute::Convergent);
      break;
    }
    case eMuxBuiltinDMAWait:
      RetTy = VoidTy;
      // Num events
      ParamTys.push_back(Int32Ty);
      ParamNames.push_back("num_events");
      // The events list
      ParamTys.push_back(PointerType::getUnqual(Ctx));
      ParamNames.push_back("events");
      AB.addAttribute(Attribute::Convergent);
      break;
    case eMuxBuiltinDMARead1D:
    case eMuxBuiltinDMAWrite1D: {
      // We need to be told the target event type to declare this builtin.
      assert(!OverloadInfo.empty() && "Missing event type");
      auto *const EventTy = OverloadInfo[0];
      RetTy = EventTy;
      const bool IsRead = ID == eMuxBuiltinDMARead1D;

      PointerType *const LocalPtrTy =
          PointerType::get(Ctx, AddressSpace::Local);
      PointerType *const GlobalPtrTy =
          PointerType::get(Ctx, AddressSpace::Global);

      ParamTys.push_back(IsRead ? LocalPtrTy : GlobalPtrTy);
      ParamNames.push_back("dst");

      ParamTys.push_back(IsRead ? GlobalPtrTy : LocalPtrTy);
      ParamNames.push_back("src");

      ParamTys.push_back(SizeTy);
      ParamNames.push_back("num_bytes");

      ParamTys.push_back(EventTy);
      ParamNames.push_back("event");
      break;
    }
    case eMuxBuiltinDMARead2D:
    case eMuxBuiltinDMAWrite2D: {
      // We need to be told the target event type to declare this builtin.
      assert(!OverloadInfo.empty() && "Missing event type");
      auto *const EventTy = OverloadInfo[0];
      RetTy = EventTy;
      const bool IsRead = ID == eMuxBuiltinDMARead2D;

      PointerType *const LocalPtrTy =
          PointerType::get(Ctx, AddressSpace::Local);
      PointerType *const GlobalPtrTy =
          PointerType::get(Ctx, AddressSpace::Global);

      ParamTys.push_back(IsRead ? LocalPtrTy : GlobalPtrTy);
      ParamNames.push_back("dst");

      ParamTys.push_back(IsRead ? GlobalPtrTy : LocalPtrTy);
      ParamNames.push_back("src");

      for (auto &P : {"num_bytes", "dst_stride", "src_stride", "height"}) {
        ParamTys.push_back(SizeTy);
        ParamNames.push_back(P);
      }

      ParamTys.push_back(EventTy);
      ParamNames.push_back("event");
      break;
    }
    case eMuxBuiltinDMARead3D:
    case eMuxBuiltinDMAWrite3D: {
      // We need to be told the target event type to declare this builtin.
      assert(!OverloadInfo.empty() && "Missing event type");
      auto *const EventTy = OverloadInfo[0];
      RetTy = EventTy;
      const bool IsRead = ID == eMuxBuiltinDMARead3D;

      PointerType *const LocalPtrTy =
          PointerType::get(Ctx, AddressSpace::Local);
      PointerType *const GlobalPtrTy =
          PointerType::get(Ctx, AddressSpace::Global);

      ParamTys.push_back(IsRead ? LocalPtrTy : GlobalPtrTy);
      ParamNames.push_back("dst");

      ParamTys.push_back(IsRead ? GlobalPtrTy : LocalPtrTy);
      ParamNames.push_back("src");

      for (auto &P :
           {"num_bytes", "dst_line_stride", "src_line_stride", "height",
            "dst_plane_stride", "src_plane_stride", "depth"}) {
        ParamTys.push_back(SizeTy);
        ParamNames.push_back(P);
      }

      ParamTys.push_back(EventTy);
      ParamNames.push_back("event");
      break;
    }
    default:
      // Group builtins are more easily found using this helper rather than
      // explicitly enumerating each switch case.
      if (auto Group = BuiltinInfo::isMuxGroupCollective(ID)) {
        RetTy = OverloadInfo.front();
        AB.addAttribute(Attribute::Convergent);
        switch (Group->Op) {
          default:
            ParamTys.push_back(RetTy);
            ParamNames.push_back("val");
            break;
          case GroupCollective::OpKind::Broadcast:
            ParamTys.push_back(RetTy);
            ParamNames.push_back("val");
            // Broadcasts additionally add ID parameters
            if (Group->isSubGroupScope()) {
              ParamTys.push_back(Int32Ty);
              ParamNames.push_back("lid");
            } else {
              ParamTys.push_back(SizeTy);
              ParamNames.push_back("lidx");
              ParamTys.push_back(SizeTy);
              ParamNames.push_back("lidy");
              ParamTys.push_back(SizeTy);
              ParamNames.push_back("lidz");
            }
            break;
          case GroupCollective::OpKind::Shuffle:
            ParamTys.push_back(RetTy);
            ParamNames.push_back("val");
            ParamTys.push_back(Int32Ty);
            ParamNames.push_back("lid");
            break;
          case GroupCollective::OpKind::ShuffleXor:
            ParamTys.push_back(RetTy);
            ParamNames.push_back("val");
            ParamTys.push_back(Int32Ty);
            ParamNames.push_back("xor_val");
            break;
          case GroupCollective::OpKind::ShuffleUp:
            ParamTys.push_back(RetTy);
            ParamNames.push_back("prev");
            ParamTys.push_back(RetTy);
            ParamNames.push_back("curr");
            ParamTys.push_back(Int32Ty);
            ParamNames.push_back("delta");
            break;
          case GroupCollective::OpKind::ShuffleDown:
            ParamTys.push_back(RetTy);
            ParamNames.push_back("curr");
            ParamTys.push_back(RetTy);
            ParamNames.push_back("next");
            ParamTys.push_back(Int32Ty);
            ParamNames.push_back("delta");
            break;
        }
        // All work-group operations have a 'barrier id' operand as their first
        // parameter.
        if (Group->isWorkGroupScope()) {
          ParamTys.insert(ParamTys.begin(), Int32Ty);
          ParamNames.insert(ParamNames.begin(), "id");
        }
      } else {
        // Unknown mux builtin
        return nullptr;
      }
  }

  assert(RetTy);
  assert(ParamTys.size() == ParamNames.size());

  SmallVector<int, 4> SchedParamIdxs;
  // Fill up the scalar parameters with the default attributes.
  SmallVector<AttributeSet, 4> ParamAttrs(ParamTys.size(), AttributeSet());

  if (requiresSchedulingParameters(ID) &&
      getSchedulingParameterModuleMetadata(M)) {
    for (const auto &P : getMuxSchedulingParameters(M)) {
      ParamTys.push_back(P.ParamTy);
      ParamNames.push_back(P.ParamName);
      ParamAttrs.push_back(P.ParamAttrs);
      SchedParamIdxs.push_back(ParamTys.size() - 1);
    }
  }

  auto *const FnTy = FunctionType::get(RetTy, ParamTys, /*isVarArg*/ false);
  auto *const F = Function::Create(FnTy, Function::ExternalLinkage, FnName, &M);
  F->addFnAttrs(AB);

  // Add some extra attributes we know are always true.
  setDefaultBuiltinAttributes(*F);

  for (unsigned i = 0, e = ParamNames.size(); i != e; i++) {
    F->getArg(i)->setName(ParamNames[i]);
    auto AB = AttrBuilder(Ctx, ParamAttrs[i]);
    F->getArg(i)->addAttrs(AB);
  }

  setSchedulingParameterFunctionMetadata(*F, SchedParamIdxs);

  return F;
}

// By default we use two parameters:
// * one structure containing local work-group data
// * one structure containing non-local work-group data
SmallVector<BuiltinInfo::SchedParamInfo, 4>
BIMuxInfoConcept::getMuxSchedulingParameters(Module &M) {
  auto &Ctx = M.getContext();
  auto &DL = M.getDataLayout();
  AttributeSet DefaultAttrs;
  DefaultAttrs = DefaultAttrs.addAttribute(Ctx, Attribute::NonNull);
  DefaultAttrs = DefaultAttrs.addAttribute(Ctx, Attribute::NoAlias);

  BuiltinInfo::SchedParamInfo WIInfo;
  {
    auto *const WIInfoS = getWorkItemInfoStructTy(M);
    WIInfo.ID = SchedParamIndices::WI;
    WIInfo.ParamPointeeTy = WIInfoS;
    WIInfo.ParamTy = WIInfoS->getPointerTo();
    WIInfo.ParamName = "wi-info";
    WIInfo.ParamDebugName = WIInfoS->getStructName().str();
    WIInfo.PassedExternally = false;

    auto AB = AttrBuilder(Ctx, DefaultAttrs);
    AB.addAlignmentAttr(DL.getABITypeAlign(WIInfoS));
    AB.addDereferenceableAttr(DL.getTypeAllocSize(WIInfoS));
    WIInfo.ParamAttrs = AttributeSet::get(Ctx, AB);
  }

  BuiltinInfo::SchedParamInfo WGInfo;
  {
    auto *const WGInfoS = getWorkGroupInfoStructTy(M);
    WGInfo.ID = SchedParamIndices::WG;
    WGInfo.ParamPointeeTy = WGInfoS;
    WGInfo.ParamTy = WGInfoS->getPointerTo();
    WGInfo.ParamName = "wg-info";
    WGInfo.ParamDebugName = WGInfoS->getStructName().str();
    WGInfo.PassedExternally = true;

    auto AB = AttrBuilder(Ctx, DefaultAttrs);
    AB.addAlignmentAttr(DL.getABITypeAlign(WGInfoS));
    AB.addDereferenceableAttr(DL.getTypeAllocSize(WGInfoS));
    WGInfo.ParamAttrs = AttributeSet::get(Ctx, AB);
  }

  return {WIInfo, WGInfo};
}

SmallVector<BuiltinInfo::SchedParamInfo, 4>
BIMuxInfoConcept::getFunctionSchedulingParameters(Function &F) {
  // Query function metadata to determine whether this function has scheduling
  // parameters
  auto ParamIdxs = getSchedulingParameterFunctionMetadata(F);
  if (ParamIdxs.empty()) {
    return {};
  }

  auto SchedParamInfo = getMuxSchedulingParameters(*F.getParent());
  // We don't allow a function to have a subset of the global scheduling
  // parameters.
  assert(ParamIdxs.size() >= SchedParamInfo.size());
  // Set the concrete argument values on each of the scheduling parameter data.
  for (auto it : zip(SchedParamInfo, ParamIdxs)) {
    // Some scheduling parameters may not be present (returning an index of
    // -1), in which case skip their concrete argument values.
    if (std::get<1>(it) >= 0) {
      std::get<0>(it).ArgVal = F.getArg(std::get<1>(it));
    }
  }

  return SchedParamInfo;
}

Value *BIMuxInfoConcept::initializeSchedulingParamForWrappedKernel(
    const BuiltinInfo::SchedParamInfo &Info, IRBuilder<> &B, Function &IntoF,
    Function &) {
  // We only expect to have to initialize the work-item info. The work-group
  // info is straight passed through.
  (void)IntoF;
  assert(!Info.PassedExternally && Info.ID == SchedParamIndices::WI &&
         Info.ParamName == "wi-info" &&
         Info.ParamPointeeTy == getWorkItemInfoStructTy(*IntoF.getParent()));
  return B.CreateAlloca(Info.ParamPointeeTy,
                        /*ArraySize*/ nullptr, Info.ParamName);
}

std::optional<llvm::ConstantRange> BIMuxInfoConcept::getBuiltinRange(
    llvm::CallInst &CI, BuiltinID ID,
    std::array<std::optional<uint64_t>, 3> MaxLocalSizes,
    std::array<std::optional<uint64_t>, 3> MaxGlobalSizes) const {
  assert(CI.getCalledFunction() && CI.getType()->isIntegerTy() &&
         "Unexpected builtin");

  auto Bits = CI.getType()->getIntegerBitWidth();
  // Assume we're indexing the global sizes array.
  std::array<std::optional<uint64_t>, 3> *SizesPtr = &MaxGlobalSizes;

  switch (ID) {
    default:
      return std::nullopt;
    case eMuxBuiltinGetWorkDim:
      return ConstantRange::getNonEmpty(APInt(Bits, 1), APInt(Bits, 4));
    case eMuxBuiltinGetLocalId:
    case eMuxBuiltinGetLocalSize:
    case eMuxBuiltinGetEnqueuedLocalSize:
      // Use the local sizes array, and fall through to common handling.
      SizesPtr = &MaxLocalSizes;
      [[fallthrough]];
    case eMuxBuiltinGetGlobalSize: {
      auto *DimIdx = CI.getOperand(0);
      if (!isa<ConstantInt>(DimIdx)) {
        return std::nullopt;
      }
      const uint64_t DimVal = cast<ConstantInt>(DimIdx)->getZExtValue();
      if (DimVal >= SizesPtr->size()) {
        return std::nullopt;
      }
      const std::optional<uint64_t> Size = (*SizesPtr)[DimVal];
      if (!Size) {
        return std::nullopt;
      }
      // ID builtins range [0,size) (exclusive), and size builtins [1,size]
      // (inclusive). Thus offset the range by 1 at each low/high end when
      // returning the range for a size builtin.
      const int SizeAdjust = ID == eMuxBuiltinGetLocalSize ||
                             ID == eMuxBuiltinGetEnqueuedLocalSize ||
                             ID == eMuxBuiltinGetGlobalSize;
      return ConstantRange::getNonEmpty(APInt(Bits, SizeAdjust),
                                        APInt(Bits, Size.value() + SizeAdjust));
    }
  }
}

}  // namespace utils
}  // namespace compiler
