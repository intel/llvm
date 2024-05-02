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

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/cl_builtin_info.h>
#include <compiler/utils/group_collective_helpers.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/pass_functions.h>
#include <compiler/utils/scheduling.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringSwitch.h>

using namespace llvm;

namespace compiler {
namespace utils {

AnalysisKey BuiltinInfoAnalysis::Key;

BuiltinInfoAnalysis::BuiltinInfoAnalysis()
    : BICallback([](const Module &) -> BuiltinInfo {
        return BuiltinInfo(std::make_unique<CLBuiltinInfo>(nullptr));
      }) {}

Module *BuiltinInfo::getBuiltinsModule() {
  if (LangImpl) {
    return LangImpl->getBuiltinsModule();
  }
  // Mux builtins don't need a module.
  return nullptr;
}

std::pair<BuiltinID, std::vector<Type *>> BuiltinInfo::identifyMuxBuiltin(
    const Function &F) const {
  StringRef Name = F.getName();
  auto ID =
      StringSwitch<BuiltinID>(Name)
          .Case(MuxBuiltins::isftz, eMuxBuiltinIsFTZ)
          .Case(MuxBuiltins::usefast, eMuxBuiltinUseFast)
          .Case(MuxBuiltins::isembeddedprofile, eMuxBuiltinIsEmbeddedProfile)
          .Case(MuxBuiltins::get_global_size, eMuxBuiltinGetGlobalSize)
          .Case(MuxBuiltins::get_global_id, eMuxBuiltinGetGlobalId)
          .Case(MuxBuiltins::get_global_offset, eMuxBuiltinGetGlobalOffset)
          .Case(MuxBuiltins::get_local_size, eMuxBuiltinGetLocalSize)
          .Case(MuxBuiltins::get_local_id, eMuxBuiltinGetLocalId)
          .Case(MuxBuiltins::set_local_id, eMuxBuiltinSetLocalId)
          .Case(MuxBuiltins::get_sub_group_id, eMuxBuiltinGetSubGroupId)
          .Case(MuxBuiltins::set_sub_group_id, eMuxBuiltinSetSubGroupId)
          .Case(MuxBuiltins::get_num_groups, eMuxBuiltinGetNumGroups)
          .Case(MuxBuiltins::get_num_sub_groups, eMuxBuiltinGetNumSubGroups)
          .Case(MuxBuiltins::set_num_sub_groups, eMuxBuiltinSetNumSubGroups)
          .Case(MuxBuiltins::get_max_sub_group_size,
                eMuxBuiltinGetMaxSubGroupSize)
          .Case(MuxBuiltins::set_max_sub_group_size,
                eMuxBuiltinSetMaxSubGroupSize)
          .Case(MuxBuiltins::get_group_id, eMuxBuiltinGetGroupId)
          .Case(MuxBuiltins::get_work_dim, eMuxBuiltinGetWorkDim)
          .Case(MuxBuiltins::dma_read_1d, eMuxBuiltinDMARead1D)
          .Case(MuxBuiltins::dma_read_2d, eMuxBuiltinDMARead2D)
          .Case(MuxBuiltins::dma_read_3d, eMuxBuiltinDMARead3D)
          .Case(MuxBuiltins::dma_write_1d, eMuxBuiltinDMAWrite1D)
          .Case(MuxBuiltins::dma_write_2d, eMuxBuiltinDMAWrite2D)
          .Case(MuxBuiltins::dma_write_3d, eMuxBuiltinDMAWrite3D)
          .Case(MuxBuiltins::dma_wait, eMuxBuiltinDMAWait)
          .Case(MuxBuiltins::get_global_linear_id, eMuxBuiltinGetGlobalLinearId)
          .Case(MuxBuiltins::get_local_linear_id, eMuxBuiltinGetLocalLinearId)
          .Case(MuxBuiltins::get_enqueued_local_size,
                eMuxBuiltinGetEnqueuedLocalSize)
          .Case(MuxBuiltins::get_sub_group_size, eMuxBuiltinGetSubGroupSize)
          .Case(MuxBuiltins::get_sub_group_local_id,
                eMuxBuiltinGetSubGroupLocalId)
          .Case(MuxBuiltins::work_group_barrier, eMuxBuiltinWorkGroupBarrier)
          .Case(MuxBuiltins::sub_group_barrier, eMuxBuiltinSubGroupBarrier)
          .Case(MuxBuiltins::mem_barrier, eMuxBuiltinMemBarrier)
          .Default(eBuiltinInvalid);
  if (ID != eBuiltinInvalid) {
    switch (ID) {
      default:
        return {ID, {}};
      case eMuxBuiltinDMARead1D:
      case eMuxBuiltinDMARead2D:
      case eMuxBuiltinDMARead3D:
      case eMuxBuiltinDMAWrite1D:
      case eMuxBuiltinDMAWrite2D:
      case eMuxBuiltinDMAWrite3D:
        // Return the event type used by these builtins. The event type is
        // required to declare/define these builtins, so return it here for
        // the sake of completeness. The event type doesn't change the
        // builtins' name (i.e., it's not mangled) as it's required to be
        // consistent at any single snapshot of the module, though it may
        // change through time.
        return {ID, {F.getReturnType()}};
    }
  }

  // Now check for group functions, which are a bit more involved as there's
  // many of them and they're also mangled. We enforce that the mangling makes
  // sense, otherwise the builtin is declared as invalid.
  const bool IsSubgroupOp = Name.consume_front("__mux_sub_group_");
  const bool IsVecgroupOp = Name.consume_front("__mux_vec_group_");
  if (!IsSubgroupOp && !IsVecgroupOp &&
      !Name.consume_front("__mux_work_group_")) {
    return {eBuiltinInvalid, {}};
  }

#define SCOPED_GROUP_OP(OP)                 \
  (IsSubgroupOp   ? eMuxBuiltinSubgroup##OP \
   : IsVecgroupOp ? eMuxBuiltinVecgroup##OP \
                  : eMuxBuiltinWorkgroup##OP)

  // Most group operations have one argument, except for broadcasts. Despite
  // that, we don't mangle the indices as they're fixed.
  const unsigned NumExpectedMangledArgs = 1;

  if (Name.consume_front("any")) {
    ID = SCOPED_GROUP_OP(Any);
  } else if (Name.consume_front("all")) {
    ID = SCOPED_GROUP_OP(All);
  } else if (Name.consume_front("broadcast")) {
    ID = SCOPED_GROUP_OP(Broadcast);
  } else if (Name.consume_front("shuffle_up")) {
    if (!IsSubgroupOp) {
      return {eBuiltinInvalid, {}};
    }
    ID = eMuxBuiltinSubgroupShuffleUp;
  } else if (Name.consume_front("shuffle_down")) {
    if (!IsSubgroupOp) {
      return {eBuiltinInvalid, {}};
    }
    ID = eMuxBuiltinSubgroupShuffleDown;
  } else if (Name.consume_front("shuffle_xor")) {
    if (!IsSubgroupOp) {
      return {eBuiltinInvalid, {}};
    }
    ID = eMuxBuiltinSubgroupShuffleXor;
  } else if (Name.consume_front("shuffle")) {
    if (!IsSubgroupOp) {
      return {eBuiltinInvalid, {}};
    }
    ID = eMuxBuiltinSubgroupShuffle;
  } else if (Name.consume_front("reduce_")) {
    auto NextIdx = Name.find_first_of('_');
    std::string Group = Name.substr(0, NextIdx).str();
    Name = Name.drop_front(Group.size());

    if (Group == "logical") {
      Name = Name.drop_front();  // Drop the underscore
      auto NextIdx = Name.find_first_of('_');
      auto RealGroup = Name.substr(0, NextIdx);
      Group += "_" + RealGroup.str();
      Name = Name.drop_front(RealGroup.size());
    }

    ID = StringSwitch<BuiltinID>(Group)
             .Case("add", SCOPED_GROUP_OP(ReduceAdd))
             .Case("fadd", SCOPED_GROUP_OP(ReduceFAdd))
             .Case("mul", SCOPED_GROUP_OP(ReduceMul))
             .Case("fmul", SCOPED_GROUP_OP(ReduceFMul))
             .Case("smin", SCOPED_GROUP_OP(ReduceSMin))
             .Case("umin", SCOPED_GROUP_OP(ReduceUMin))
             .Case("fmin", SCOPED_GROUP_OP(ReduceFMin))
             .Case("smax", SCOPED_GROUP_OP(ReduceSMax))
             .Case("umax", SCOPED_GROUP_OP(ReduceUMax))
             .Case("fmax", SCOPED_GROUP_OP(ReduceFMax))
             .Case("and", SCOPED_GROUP_OP(ReduceAnd))
             .Case("or", SCOPED_GROUP_OP(ReduceOr))
             .Case("xor", SCOPED_GROUP_OP(ReduceXor))
             .Case("logical_and", SCOPED_GROUP_OP(ReduceLogicalAnd))
             .Case("logical_or", SCOPED_GROUP_OP(ReduceLogicalOr))
             .Case("logical_xor", SCOPED_GROUP_OP(ReduceLogicalXor))
             .Default(eBuiltinInvalid);
  } else if (Name.consume_front("scan_")) {
    const bool IsInclusive = Name.consume_front("inclusive_");
    if (!IsInclusive && !Name.consume_front("exclusive_")) {
      return {eBuiltinInvalid, {}};
    }

    auto NextIdx = Name.find_first_of('_');
    std::string Group = Name.substr(0, NextIdx).str();
    Name = Name.drop_front(Group.size());

    if (Group == "logical") {
      auto NextIdx = Name.find_first_of('_', /*From*/ 1);
      auto RealGroup = Name.substr(0, NextIdx);
      Group += RealGroup.str();
      Name = Name.drop_front(RealGroup.size());
    }

    ID = StringSwitch<BuiltinID>(Group)
             .Case("add", IsInclusive ? SCOPED_GROUP_OP(ScanAddInclusive)
                                      : SCOPED_GROUP_OP(ScanAddExclusive))
             .Case("fadd", IsInclusive ? SCOPED_GROUP_OP(ScanFAddInclusive)
                                       : SCOPED_GROUP_OP(ScanFAddExclusive))
             .Case("mul", IsInclusive ? SCOPED_GROUP_OP(ScanMulInclusive)
                                      : SCOPED_GROUP_OP(ScanMulExclusive))
             .Case("fmul", IsInclusive ? SCOPED_GROUP_OP(ScanFMulInclusive)
                                       : SCOPED_GROUP_OP(ScanFMulExclusive))
             .Case("smin", IsInclusive ? SCOPED_GROUP_OP(ScanSMinInclusive)
                                       : SCOPED_GROUP_OP(ScanSMinExclusive))
             .Case("umin", IsInclusive ? SCOPED_GROUP_OP(ScanUMinInclusive)
                                       : SCOPED_GROUP_OP(ScanUMinExclusive))
             .Case("fmin", IsInclusive ? SCOPED_GROUP_OP(ScanFMinInclusive)
                                       : SCOPED_GROUP_OP(ScanFMinExclusive))
             .Case("smax", IsInclusive ? SCOPED_GROUP_OP(ScanSMaxInclusive)
                                       : SCOPED_GROUP_OP(ScanSMaxExclusive))
             .Case("umax", IsInclusive ? SCOPED_GROUP_OP(ScanUMaxInclusive)
                                       : SCOPED_GROUP_OP(ScanUMaxExclusive))
             .Case("fmax", IsInclusive ? SCOPED_GROUP_OP(ScanFMaxInclusive)
                                       : SCOPED_GROUP_OP(ScanFMaxExclusive))
             .Case("and", IsInclusive ? SCOPED_GROUP_OP(ScanAndInclusive)
                                      : SCOPED_GROUP_OP(ScanAndExclusive))
             .Case("or", IsInclusive ? SCOPED_GROUP_OP(ScanOrInclusive)
                                     : SCOPED_GROUP_OP(ScanOrExclusive))
             .Case("xor", IsInclusive ? SCOPED_GROUP_OP(ScanXorInclusive)
                                      : SCOPED_GROUP_OP(ScanXorExclusive))
             .Case("logical_and",
                   IsInclusive ? SCOPED_GROUP_OP(ScanLogicalAndInclusive)
                               : SCOPED_GROUP_OP(ScanLogicalAndExclusive))
             .Case("logical_or", IsInclusive
                                     ? SCOPED_GROUP_OP(ScanLogicalOrInclusive)
                                     : SCOPED_GROUP_OP(ScanLogicalOrExclusive))
             .Case("logical_xor",
                   IsInclusive ? SCOPED_GROUP_OP(ScanLogicalXorInclusive)
                               : SCOPED_GROUP_OP(ScanLogicalXorExclusive))
             .Default(eBuiltinInvalid);
  }

  std::vector<Type *> OverloadInfo;
  if (ID != eBuiltinInvalid) {
    // Consume the rest of this group Op function name. If we can't identify a
    // series of mangled type names, this builtin is invalid.
    unsigned NumMangledArgs = 0;
    // Work-group builtins have an unmangled 'barrier ID' parameter first, which
    // we want to skip.
    const unsigned Offset = ID >= eFirstMuxWorkgroupCollectiveBuiltin &&
                            ID <= eLastMuxWorkgroupCollectiveBuiltin;
    while (!Name.empty()) {
      if (!Name.consume_front("_")) {
        return {eBuiltinInvalid, {}};
      }
      auto [Ty, NewName] = getDemangledTypeFromStr(Name, F.getContext());
      Name = NewName;

      auto ParamIdx = Offset + NumMangledArgs;
      if (ParamIdx >= F.arg_size() || Ty != F.getArg(ParamIdx)->getType()) {
        return {eBuiltinInvalid, {}};
      }

      ++NumMangledArgs;
      OverloadInfo.push_back(Ty);
    }
    if (NumMangledArgs != NumExpectedMangledArgs) {
      return {eBuiltinInvalid, {}};
    }
  }

  return {ID, OverloadInfo};
#undef SCOPED_GROUP_OP
}

BuiltinUniformity BuiltinInfo::isBuiltinUniform(const Builtin &B,
                                                const CallInst *CI,
                                                unsigned SimdDimIdx) const {
  switch (B.ID) {
    default:
      break;
    case eMuxBuiltinGetGlobalId:
    case eMuxBuiltinGetLocalId: {
      // We need to know the dimension requested from these builtins at compile
      // time to infer their uniformity.
      if (!CI || CI->arg_empty()) {
        return eBuiltinUniformityNever;
      }
      auto *Rank = dyn_cast<ConstantInt>(CI->getArgOperand(0));
      if (!Rank) {
        // The Rank is some function, which "might" evaluate to zero
        // sometimes, so we let the packetizer sort it out with some
        // conditional magic.
        // TODO Make sure this can never go haywire in weird edge cases.
        // Where we have one get_global_id() dependent on another, this is
        // not packetized correctly. Doing so is very hard!  We should
        // probably just fail to packetize in this case.  We might also be
        // able to return eBuiltinUniformityNever here, in cases where we can
        // prove that the value can never be zero.
        return eBuiltinUniformityMaybeInstanceID;
      }
      // Only vectorize on selected dimension. The value of get_global_id with
      // other ranks is uniform.
      if (Rank->getZExtValue() == SimdDimIdx) {
        return eBuiltinUniformityInstanceID;
      }

      return eBuiltinUniformityAlways;
    }
    case eMuxBuiltinGetSubGroupLocalId:
      return eBuiltinUniformityInstanceID;
    case eMuxBuiltinGetLocalLinearId:
    case eMuxBuiltinGetGlobalLinearId:
      // TODO: This is fine for vectorizing in the x-axis, but currently we do
      // not support vectorizing along y or z (see CA-2843).
      return SimdDimIdx ? eBuiltinUniformityNever
                        : eBuiltinUniformityInstanceID;
  }

  // Reductions and broadcasts are always uniform
  if (auto Info = isMuxGroupCollective(B.ID)) {
    if (Info->isAnyAll() || Info->isReduction() || Info->isBroadcast()) {
      return eBuiltinUniformityAlways;
    }
  }

  if (LangImpl) {
    return LangImpl->isBuiltinUniform(B, CI, SimdDimIdx);
  }
  return eBuiltinUniformityUnknown;
}

Builtin BuiltinInfo::analyzeBuiltin(const Function &F) const {
  // Handle LLVM intrinsics.
  if (F.isIntrinsic()) {
    int32_t Properties = eBuiltinPropertyNone;

    const Intrinsic::ID IntrID = (Intrinsic::ID)F.getIntrinsicID();
    const AttributeList AS = Intrinsic::getAttributes(F.getContext(), IntrID);
    const bool NoSideEffect = F.onlyReadsMemory();
    bool SafeIntrinsic = false;
    switch (IntrID) {
      default:
        SafeIntrinsic = false;
        break;
      case Intrinsic::smin:
      case Intrinsic::smax:
      case Intrinsic::umin:
      case Intrinsic::umax:
      case Intrinsic::abs:
      case Intrinsic::ctlz:
      case Intrinsic::cttz:
      case Intrinsic::sqrt:
      case Intrinsic::sin:
      case Intrinsic::cos:
      case Intrinsic::pow:
      case Intrinsic::exp:
      case Intrinsic::exp2:
      case Intrinsic::log:
      case Intrinsic::log10:
      case Intrinsic::log2:
      case Intrinsic::fma:
      case Intrinsic::fabs:
      case Intrinsic::minnum:
      case Intrinsic::maxnum:
      case Intrinsic::copysign:
      case Intrinsic::floor:
      case Intrinsic::ceil:
      case Intrinsic::trunc:
      case Intrinsic::rint:
      case Intrinsic::nearbyint:
      case Intrinsic::round:
      case Intrinsic::ctpop:
      case Intrinsic::fmuladd:
      case Intrinsic::fshl:
      case Intrinsic::fshr:
      case Intrinsic::sadd_sat:
      case Intrinsic::uadd_sat:
      case Intrinsic::ssub_sat:
      case Intrinsic::usub_sat:
      case Intrinsic::bitreverse:
        // All these function are overloadable and have both scalar and vector
        // versions.
        Properties |= eBuiltinPropertyVectorEquivalent;
        SafeIntrinsic = true;
        break;
      case Intrinsic::assume:
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::invariant_start:
      case Intrinsic::invariant_end:
      case Intrinsic::lifetime_start:
      case Intrinsic::lifetime_end:
      case Intrinsic::objectsize:
      case Intrinsic::ptr_annotation:
      case Intrinsic::var_annotation:
      case Intrinsic::experimental_noalias_scope_decl:
        SafeIntrinsic = true;
        break;
      case Intrinsic::memset:
      case Intrinsic::memcpy:
        Properties |= eBuiltinPropertyNoVectorEquivalent;
        Properties |= eBuiltinPropertySideEffects;
        break;
    }
    if (NoSideEffect || SafeIntrinsic) {
      Properties |= eBuiltinPropertyNoSideEffects;
      if (!AS.hasFnAttr(Attribute::NoDuplicate)) {
        Properties |= eBuiltinPropertySupportsInstantiation;
      }
    }
    return Builtin{F, eBuiltinUnknown, (BuiltinProperties)Properties};
  }

  auto [ID, OverloadInfo] = identifyMuxBuiltin(F);

  if (ID == eBuiltinInvalid) {
    // It's not a Mux builtin, so defer to the language implementation
    if (LangImpl) {
      return LangImpl->analyzeBuiltin(F);
    }
    return Builtin{F, ID, eBuiltinPropertyNone};
  }

  // Check that all overloadable builtins have returned some overloading
  // information, for API consistency.
  assert((!isOverloadableMuxBuiltinID(ID) || !OverloadInfo.empty()) &&
         "Inconsistency in overloadable builtin APIs");

  bool IsConvergent = false;
  unsigned Properties = eBuiltinPropertyNone;
  switch (ID) {
    default:
      break;
    case eMuxBuiltinMemBarrier:
      Properties = eBuiltinPropertySideEffects;
      break;
    case eMuxBuiltinSubGroupBarrier:
    case eMuxBuiltinWorkGroupBarrier:
      IsConvergent = true;
      Properties = eBuiltinPropertyExecutionFlow | eBuiltinPropertySideEffects;
      break;
    case eMuxBuiltinDMARead1D:
    case eMuxBuiltinDMARead2D:
    case eMuxBuiltinDMARead3D:
    case eMuxBuiltinDMAWrite1D:
    case eMuxBuiltinDMAWrite2D:
    case eMuxBuiltinDMAWrite3D:
    case eMuxBuiltinDMAWait:
      // Our DMA builtins, by default, rely on thread checks against specific
      // work-item IDs, so they must be convergent.
      IsConvergent = true;
      Properties = eBuiltinPropertyNoSideEffects;
      break;
    case eMuxBuiltinGetWorkDim:
    case eMuxBuiltinGetGroupId:
    case eMuxBuiltinGetGlobalSize:
    case eMuxBuiltinGetGlobalOffset:
    case eMuxBuiltinGetLocalSize:
    case eMuxBuiltinGetNumGroups:
    case eMuxBuiltinGetGlobalLinearId:
    case eMuxBuiltinGetLocalLinearId:
    case eMuxBuiltinGetGlobalId:
    case eMuxBuiltinGetSubGroupLocalId:
      Properties = eBuiltinPropertyWorkItem | eBuiltinPropertyRematerializable;
      break;
    case eMuxBuiltinGetLocalId:
      Properties = eBuiltinPropertyWorkItem | eBuiltinPropertyLocalID |
                   eBuiltinPropertyRematerializable;
      break;
    case eMuxBuiltinIsFTZ:
    case eMuxBuiltinIsEmbeddedProfile:
    case eMuxBuiltinUseFast:
      Properties = eBuiltinPropertyNoSideEffects;
      break;
  }

  // Group functions are convergent.
  if (isMuxGroupCollective(ID)) {
    IsConvergent = true;
  }

  if (!IsConvergent) {
    Properties |= eBuiltinPropertyKnownNonConvergent;
  }

  return Builtin{F, ID, (BuiltinProperties)Properties, OverloadInfo};
}

BuiltinCall BuiltinInfo::analyzeBuiltinCall(const CallInst &CI,
                                            unsigned SimdDimIdx) const {
  auto *const callee = CI.getCalledFunction();
  assert(callee && "Call instruction with no callee");
  const auto B = analyzeBuiltin(*callee);
  const auto U = isBuiltinUniform(B, &CI, SimdDimIdx);
  return BuiltinCall{B, CI, U};
}

Function *BuiltinInfo::getVectorEquivalent(const Builtin &B, unsigned Width,
                                           Module *M) {
  // We don't handle LLVM intrinsics here
  if (B.function.isIntrinsic()) {
    return nullptr;
  }

  if (LangImpl) {
    return LangImpl->getVectorEquivalent(B, Width, M);
  }
  return nullptr;
}

Function *BuiltinInfo::getScalarEquivalent(const Builtin &B, Module *M) {
  // We will first check to see if this is an LLVM intrinsic that has a scalar
  // equivalent.
  if (B.function.isIntrinsic()) {
    // Analyze the builtin. Some functions have no scalar equivalent.
    const auto Props = B.properties;
    if (!(Props & eBuiltinPropertyVectorEquivalent)) {
      return nullptr;
    }

    // Check the return type.
    auto *VecRetTy = dyn_cast<FixedVectorType>(B.function.getReturnType());
    if (!VecRetTy) {
      return nullptr;
    }

    auto IntrinsicID = B.function.getIntrinsicID();
    // Currently, we can only handle correctly intrinsics that have one
    // overloaded type, used for both the return type and all of the arguments.
    // TODO: More generic support for intrinsics with vector equivalents.
    for (Type *ArgTy : B.function.getFunctionType()->params()) {
      // If the argument isn't a vector, then it isn't going to get scalarized,
      // so don't worry about it.
      if (ArgTy->isVectorTy() && ArgTy != VecRetTy) {
        return nullptr;
      }
    }
    Type *ScalarType = VecRetTy->getElementType();
    // Get the scalar version of the intrinsic
    Function *ScalarIntrinsic =
        Intrinsic::getDeclaration(M, IntrinsicID, ScalarType);

    return ScalarIntrinsic;
  }

  if (LangImpl) {
    return LangImpl->getScalarEquivalent(B, M);
  }
  return nullptr;
}

Value *BuiltinInfo::emitBuiltinInline(Function *Builtin, IRBuilder<> &B,
                                      ArrayRef<Value *> Args) {
  if (LangImpl) {
    return LangImpl->emitBuiltinInline(Builtin, B, Args);
  }
  return nullptr;
}

std::optional<llvm::ConstantRange> BuiltinInfo::getBuiltinRange(
    CallInst &CI, std::array<std::optional<uint64_t>, 3> MaxLocalSizes,
    std::array<std::optional<uint64_t>, 3> MaxGlobalSizes) const {
  auto *F = CI.getCalledFunction();
  // Ranges only apply to integer types, and ensure that there's a named
  // function to analyze.
  if (!F || !F->hasName() || !CI.getType()->isIntegerTy()) {
    return std::nullopt;
  }

  // First, check mux builtins
  if (auto [ID, _] = identifyMuxBuiltin(*F); isMuxBuiltinID(ID)) {
    return MuxImpl->getBuiltinRange(CI, ID, MaxLocalSizes, MaxGlobalSizes);
  }

  // Next, ask the language builtin info
  if (LangImpl) {
    return LangImpl->getBuiltinRange(CI, MaxLocalSizes, MaxGlobalSizes);
  }

  return std::nullopt;
}

Instruction *BuiltinInfo::lowerBuiltinToMuxBuiltin(CallInst &CI) {
  if (LangImpl) {
    return LangImpl->lowerBuiltinToMuxBuiltin(CI, *MuxImpl);
  }
  // We shouldn't be mapping mux builtins to mux builtins, so we can stop here.
  return nullptr;
}

BuiltinID BuiltinInfo::getPrintfBuiltin() const {
  if (LangImpl) {
    return LangImpl->getPrintfBuiltin();
  }
  return eBuiltinInvalid;
}

bool BuiltinInfo::requiresSchedulingParameters(BuiltinID ID) {
  // Defer to mux for the scheduling parameters.
  return MuxImpl->requiresSchedulingParameters(ID);
}

Type *BuiltinInfo::getRemappedTargetExtTy(Type *Ty, Module &M) {
  // Defer to mux for the scheduling parameters.
  return MuxImpl->getRemappedTargetExtTy(Ty, M);
}

SmallVector<BuiltinInfo::SchedParamInfo, 4>
BuiltinInfo::getMuxSchedulingParameters(Module &M) {
  // Defer to mux for the scheduling parameters.
  return MuxImpl->getMuxSchedulingParameters(M);
}

SmallVector<BuiltinInfo::SchedParamInfo, 4>
BuiltinInfo::getFunctionSchedulingParameters(Function &F) {
  // Defer to mux for the scheduling parameters.
  return MuxImpl->getFunctionSchedulingParameters(F);
}

Value *BuiltinInfo::initializeSchedulingParamForWrappedKernel(
    const SchedParamInfo &Info, IRBuilder<> &B, Function &IntoF,
    Function &CalleeF) {
  return MuxImpl->initializeSchedulingParamForWrappedKernel(Info, B, IntoF,
                                                            CalleeF);
}

// This provides an extremely simple mangling scheme matching LLVM's intrinsic
// mangling system. It is only designed to be used with a specific set of types
// and is not a general-purpose mangler.
std::string BuiltinInfo::getMangledTypeStr(Type *Ty) {
  std::string Result;
  if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    const ElementCount EC = VTy->getElementCount();
    if (EC.isScalable()) {
      Result += "nx";
    }
    return "v" + utostr(EC.getKnownMinValue()) +
           getMangledTypeStr(VTy->getElementType());
  }

  if (Ty) {
    switch (Ty->getTypeID()) {
      default:
        break;
      case Type::HalfTyID:
        return "f16";
      case Type::BFloatTyID:
        return "bf16";
      case Type::FloatTyID:
        return "f32";
      case Type::DoubleTyID:
        return "f64";
      case Type::IntegerTyID:
        return "i" + utostr(cast<IntegerType>(Ty)->getBitWidth());
    }
  }
  llvm_unreachable("Unhandled type");
}

std::pair<Type *, StringRef> BuiltinInfo::getDemangledTypeFromStr(
    StringRef TyStr, LLVMContext &Ctx) {
  const bool IsScalable = TyStr.consume_front("nx");
  if (TyStr.consume_front("v")) {
    unsigned EC;
    if (TyStr.consumeInteger(10, EC)) {
      return {nullptr, TyStr};
    }
    if (auto [EltTy, NewTyStr] = getDemangledTypeFromStr(TyStr, Ctx); EltTy) {
      return {VectorType::get(EltTy, EC, IsScalable), NewTyStr};
    }
    return {nullptr, TyStr};
  }
  if (TyStr.consume_front("f16")) {
    return {Type::getHalfTy(Ctx), TyStr};
  }
  if (TyStr.consume_front("bf16")) {
    return {Type::getBFloatTy(Ctx), TyStr};
  }
  if (TyStr.consume_front("f32")) {
    return {Type::getFloatTy(Ctx), TyStr};
  }
  if (TyStr.consume_front("f64")) {
    return {Type::getDoubleTy(Ctx), TyStr};
  }
  unsigned IntBitWidth;
  if (TyStr.consume_front("i") && !TyStr.consumeInteger(10, IntBitWidth)) {
    return {IntegerType::get(Ctx, IntBitWidth), TyStr};
  }

  return {nullptr, TyStr};
}

std::string BuiltinInfo::getMuxBuiltinName(BuiltinID ID,
                                           ArrayRef<Type *> OverloadInfo) {
  assert(isMuxBuiltinID(ID));
  switch (ID) {
    default:
      break;
    case eMuxBuiltinIsFTZ:
      return MuxBuiltins::isftz;
    case eMuxBuiltinUseFast:
      return MuxBuiltins::usefast;
    case eMuxBuiltinIsEmbeddedProfile:
      return MuxBuiltins::isembeddedprofile;
    case eMuxBuiltinGetGlobalSize:
      return MuxBuiltins::get_global_size;
    case eMuxBuiltinGetGlobalId:
      return MuxBuiltins::get_global_id;
    case eMuxBuiltinGetGlobalOffset:
      return MuxBuiltins::get_global_offset;
    case eMuxBuiltinGetLocalSize:
      return MuxBuiltins::get_local_size;
    case eMuxBuiltinGetLocalId:
      return MuxBuiltins::get_local_id;
    case eMuxBuiltinSetLocalId:
      return MuxBuiltins::set_local_id;
    case eMuxBuiltinGetSubGroupId:
      return MuxBuiltins::get_sub_group_id;
    case eMuxBuiltinSetSubGroupId:
      return MuxBuiltins::set_sub_group_id;
    case eMuxBuiltinGetNumGroups:
      return MuxBuiltins::get_num_groups;
    case eMuxBuiltinGetNumSubGroups:
      return MuxBuiltins::get_num_sub_groups;
    case eMuxBuiltinSetNumSubGroups:
      return MuxBuiltins::set_num_sub_groups;
    case eMuxBuiltinGetMaxSubGroupSize:
      return MuxBuiltins::get_max_sub_group_size;
    case eMuxBuiltinSetMaxSubGroupSize:
      return MuxBuiltins::set_max_sub_group_size;
    case eMuxBuiltinGetGroupId:
      return MuxBuiltins::get_group_id;
    case eMuxBuiltinGetWorkDim:
      return MuxBuiltins::get_work_dim;
    case eMuxBuiltinDMARead1D:
      return MuxBuiltins::dma_read_1d;
    case eMuxBuiltinDMARead2D:
      return MuxBuiltins::dma_read_2d;
    case eMuxBuiltinDMARead3D:
      return MuxBuiltins::dma_read_3d;
    case eMuxBuiltinDMAWrite1D:
      return MuxBuiltins::dma_write_1d;
    case eMuxBuiltinDMAWrite2D:
      return MuxBuiltins::dma_write_2d;
    case eMuxBuiltinDMAWrite3D:
      return MuxBuiltins::dma_write_3d;
    case eMuxBuiltinDMAWait:
      return MuxBuiltins::dma_wait;
    case eMuxBuiltinGetGlobalLinearId:
      return MuxBuiltins::get_global_linear_id;
    case eMuxBuiltinGetLocalLinearId:
      return MuxBuiltins::get_local_linear_id;
    case eMuxBuiltinGetEnqueuedLocalSize:
      return MuxBuiltins::get_enqueued_local_size;
    case eMuxBuiltinGetSubGroupSize:
      return MuxBuiltins::get_sub_group_size;
    case eMuxBuiltinGetSubGroupLocalId:
      return MuxBuiltins::get_sub_group_local_id;
    case eMuxBuiltinMemBarrier:
      return MuxBuiltins::mem_barrier;
    case eMuxBuiltinWorkGroupBarrier:
      return MuxBuiltins::work_group_barrier;
    case eMuxBuiltinSubGroupBarrier:
      return MuxBuiltins::sub_group_barrier;
  }

    // A sneaky macro to do case statements on all scopes of a group operation.
    // Note that it is missing a leading 'case' and a trailing ':' to trick
    // clang-format into formatting it like a regular case statement.
#define CASE_GROUP_OP_ALL_SCOPES(OP)                      \
  eMuxBuiltinVecgroup##OP : case eMuxBuiltinSubgroup##OP: \
  case eMuxBuiltinWorkgroup##OP

  std::string BaseName = [](BuiltinID ID) {
    // For simplicity, return all group operations as 'work_group' and replace
    // the string with 'sub_group' or 'vec_group' post-hoc.
    switch (ID) {
      default:
        return "";
      case CASE_GROUP_OP_ALL_SCOPES(All):
        return "__mux_work_group_all";
      case CASE_GROUP_OP_ALL_SCOPES(Any):
        return "__mux_work_group_any";
      case CASE_GROUP_OP_ALL_SCOPES(Broadcast):
        return "__mux_work_group_broadcast";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceAdd):
        return "__mux_work_group_reduce_add";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFAdd):
        return "__mux_work_group_reduce_fadd";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceSMin):
        return "__mux_work_group_reduce_smin";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceUMin):
        return "__mux_work_group_reduce_umin";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFMin):
        return "__mux_work_group_reduce_fmin";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceSMax):
        return "__mux_work_group_reduce_smax";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceUMax):
        return "__mux_work_group_reduce_umax";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFMax):
        return "__mux_work_group_reduce_fmax";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceMul):
        return "__mux_work_group_reduce_mul";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFMul):
        return "__mux_work_group_reduce_fmul";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceAnd):
        return "__mux_work_group_reduce_and";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceOr):
        return "__mux_work_group_reduce_or";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceXor):
        return "__mux_work_group_reduce_xor";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalAnd):
        return "__mux_work_group_reduce_logical_and";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalOr):
        return "__mux_work_group_reduce_logical_or";
      case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalXor):
        return "__mux_work_group_reduce_logical_xor";
      case CASE_GROUP_OP_ALL_SCOPES(ScanAddInclusive):
        return "__mux_work_group_scan_inclusive_add";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFAddInclusive):
        return "__mux_work_group_scan_inclusive_fadd";
      case CASE_GROUP_OP_ALL_SCOPES(ScanAddExclusive):
        return "__mux_work_group_scan_exclusive_add";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFAddExclusive):
        return "__mux_work_group_scan_exclusive_fadd";
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMinInclusive):
        return "__mux_work_group_scan_inclusive_smin";
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMinInclusive):
        return "__mux_work_group_scan_inclusive_umin";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMinInclusive):
        return "__mux_work_group_scan_inclusive_fmin";
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMinExclusive):
        return "__mux_work_group_scan_exclusive_smin";
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMinExclusive):
        return "__mux_work_group_scan_exclusive_umin";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMinExclusive):
        return "__mux_work_group_scan_exclusive_fmin";
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMaxInclusive):
        return "__mux_work_group_scan_inclusive_smax";
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMaxInclusive):
        return "__mux_work_group_scan_inclusive_umax";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMaxInclusive):
        return "__mux_work_group_scan_inclusive_fmax";
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMaxExclusive):
        return "__mux_work_group_scan_exclusive_smax";
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMaxExclusive):
        return "__mux_work_group_scan_exclusive_umax";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMaxExclusive):
        return "__mux_work_group_scan_exclusive_fmax";
      case CASE_GROUP_OP_ALL_SCOPES(ScanMulInclusive):
        return "__mux_work_group_scan_inclusive_mul";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMulInclusive):
        return "__mux_work_group_scan_inclusive_fmul";
      case CASE_GROUP_OP_ALL_SCOPES(ScanMulExclusive):
        return "__mux_work_group_scan_exclusive_mul";
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMulExclusive):
        return "__mux_work_group_scan_exclusive_fmul";
      case CASE_GROUP_OP_ALL_SCOPES(ScanAndInclusive):
        return "__mux_work_group_scan_inclusive_and";
      case CASE_GROUP_OP_ALL_SCOPES(ScanAndExclusive):
        return "__mux_work_group_scan_exclusive_and";
      case CASE_GROUP_OP_ALL_SCOPES(ScanOrInclusive):
        return "__mux_work_group_scan_inclusive_or";
      case CASE_GROUP_OP_ALL_SCOPES(ScanOrExclusive):
        return "__mux_work_group_scan_exclusive_or";
      case CASE_GROUP_OP_ALL_SCOPES(ScanXorInclusive):
        return "__mux_work_group_scan_inclusive_xor";
      case CASE_GROUP_OP_ALL_SCOPES(ScanXorExclusive):
        return "__mux_work_group_scan_exclusive_xor";
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalAndInclusive):
        return "__mux_work_group_scan_inclusive_logical_and";
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalAndExclusive):
        return "__mux_work_group_scan_exclusive_logical_and";
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalOrInclusive):
        return "__mux_work_group_scan_inclusive_logical_or";
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalOrExclusive):
        return "__mux_work_group_scan_exclusive_logical_or";
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalXorInclusive):
        return "__mux_work_group_scan_inclusive_logical_xor";
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalXorExclusive):
        return "__mux_work_group_scan_exclusive_logical_xor";
      case eMuxBuiltinSubgroupShuffle:
        return "__mux_work_group_shuffle";
      case eMuxBuiltinSubgroupShuffleUp:
        return "__mux_work_group_shuffle_up";
      case eMuxBuiltinSubgroupShuffleDown:
        return "__mux_work_group_shuffle_down";
      case eMuxBuiltinSubgroupShuffleXor:
        return "__mux_work_group_shuffle_xor";
    }
  }(ID);

  if (!BaseName.empty()) {
    assert(!OverloadInfo.empty() &&
           "Must know how to overload group operation");
    if (ID >= eFirstMuxSubgroupCollectiveBuiltin &&
        ID <= eLastMuxSubgroupCollectiveBuiltin) {
      // Replace 'work' with 'sub'
      BaseName = BaseName.replace(6, 4, "sub");
    } else if (ID >= eFirstMuxVecgroupCollectiveBuiltin &&
               ID <= eLastMuxVecgroupCollectiveBuiltin) {
      // Replace 'work' with 'vec'
      BaseName = BaseName.replace(6, 4, "vec");
    }
    auto *const Ty = OverloadInfo.front();
    return BaseName + "_" + getMangledTypeStr(Ty);
  }
  llvm_unreachable("Unhandled mux builtin");
#undef CASE_GROUP_OP_ALL_SCOPES
}

Function *BuiltinInfo::defineMuxBuiltin(BuiltinID ID, Module &M,
                                        ArrayRef<Type *> OverloadInfo) {
  assert(isMuxBuiltinID(ID) && "Only handling mux builtins");
  // Check that all overloadable builtins have returned some overloading
  // information, for API consistency.
  assert((!isOverloadableMuxBuiltinID(ID) || !OverloadInfo.empty()) &&
         "Inconsistency in overloadable builtin APIs");

  Function *F = M.getFunction(getMuxBuiltinName(ID, OverloadInfo));
  // FIXME: We'd ideally want to declare it here to reduce pass
  // inter-dependencies.
  assert(F && "Function should have been pre-declared");
  if (!F->isDeclaration()) {
    return F;
  }
  // Defer to the mux implementation to define this builtin.
  return MuxImpl->defineMuxBuiltin(ID, M, OverloadInfo);
}

Function *BuiltinInfo::getOrDeclareMuxBuiltin(BuiltinID ID, Module &M,
                                              ArrayRef<Type *> OverloadInfo) {
  assert(isMuxBuiltinID(ID) && "Only handling mux builtins");
  // Check that all overloadable builtins have returned some overloading
  // information, for API consistency.
  assert((!isOverloadableMuxBuiltinID(ID) || !OverloadInfo.empty()) &&
         "Inconsistency in overloadable builtin APIs");
  // Defer to the mux implementation to get/declare this builtin.
  return MuxImpl->getOrDeclareMuxBuiltin(ID, M, OverloadInfo);
}

std::optional<GroupCollective> BuiltinInfo::isMuxGroupCollective(BuiltinID ID) {
  GroupCollective Collective;

  if (ID >= eFirstMuxSubgroupCollectiveBuiltin &&
      ID <= eLastMuxSubgroupCollectiveBuiltin) {
    Collective.Scope = GroupCollective::ScopeKind::SubGroup;
  } else if (ID >= eFirstMuxWorkgroupCollectiveBuiltin &&
             ID <= eLastMuxWorkgroupCollectiveBuiltin) {
    Collective.Scope = GroupCollective::ScopeKind::WorkGroup;
  } else if (ID >= eFirstMuxVecgroupCollectiveBuiltin &&
             ID <= eLastMuxVecgroupCollectiveBuiltin) {
    Collective.Scope = GroupCollective::ScopeKind::VectorGroup;
  } else {
    return std::nullopt;
  }

  // A sneaky macro to do case statements on all scopes of a group operation.
  // Note that it is missing a leading 'case' and a trailing ':' to trick
  // clang-format into formatting it like a regular case statement.
#define CASE_GROUP_OP_ALL_SCOPES(OP)                      \
  eMuxBuiltinVecgroup##OP : case eMuxBuiltinSubgroup##OP: \
  case eMuxBuiltinWorkgroup##OP

  switch (ID) {
    default:
      llvm_unreachable("Unhandled mux group builtin");
    case CASE_GROUP_OP_ALL_SCOPES(All):
      Collective.Op = GroupCollective::OpKind::All;
      break;
    case CASE_GROUP_OP_ALL_SCOPES(Any):
      Collective.Op = GroupCollective::OpKind::Any;
      break;
    case CASE_GROUP_OP_ALL_SCOPES(Broadcast):
      Collective.Op = GroupCollective::OpKind::Broadcast;
      break;
    case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalAnd):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalOr):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalXor):
      Collective.IsLogical = true;
      [[fallthrough]];
    case CASE_GROUP_OP_ALL_SCOPES(ReduceAdd):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceFAdd):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceMul):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceFMul):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceSMin):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceUMin):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceFMin):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceSMax):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceUMax):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceFMax):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceAnd):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceOr):
    case CASE_GROUP_OP_ALL_SCOPES(ReduceXor):
      Collective.Op = GroupCollective::OpKind::Reduction;
      break;
    case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalAndInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalOrInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalXorInclusive):
      Collective.IsLogical = true;
      [[fallthrough]];
    case CASE_GROUP_OP_ALL_SCOPES(ScanAddInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFAddInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanMulInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFMulInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanSMinInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanUMinInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFMinInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanSMaxInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanUMaxInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFMaxInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanAndInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanOrInclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanXorInclusive):
      Collective.Op = GroupCollective::OpKind::ScanInclusive;
      break;
    case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalAndExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalOrExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalXorExclusive):
      Collective.IsLogical = true;
      [[fallthrough]];
    case CASE_GROUP_OP_ALL_SCOPES(ScanAddExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFAddExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanMulExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFMulExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanSMinExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanUMinExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFMinExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanSMaxExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanUMaxExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanFMaxExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanAndExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanOrExclusive):
    case CASE_GROUP_OP_ALL_SCOPES(ScanXorExclusive):
      Collective.Op = GroupCollective::OpKind::ScanExclusive;
      break;
    case eMuxBuiltinSubgroupShuffle:
      Collective.Op = GroupCollective::OpKind::Shuffle;
      break;
    case eMuxBuiltinSubgroupShuffleUp:
      Collective.Op = GroupCollective::OpKind::ShuffleUp;
      break;
    case eMuxBuiltinSubgroupShuffleDown:
      Collective.Op = GroupCollective::OpKind::ShuffleDown;
      break;
    case eMuxBuiltinSubgroupShuffleXor:
      Collective.Op = GroupCollective::OpKind::ShuffleXor;
      break;
  }

  // Then the recurrence kind.
  if (Collective.Op == GroupCollective::OpKind::All) {
    Collective.Recurrence = RecurKind::And;
  } else if (Collective.Op == GroupCollective::OpKind::Any) {
    Collective.Recurrence = RecurKind::Or;
  } else if (Collective.Op == GroupCollective::OpKind::Reduction ||
             Collective.Op == GroupCollective::OpKind::ScanExclusive ||
             Collective.Op == GroupCollective::OpKind::ScanInclusive) {
    switch (ID) {
      case CASE_GROUP_OP_ALL_SCOPES(ReduceAdd):
      case CASE_GROUP_OP_ALL_SCOPES(ScanAddInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanAddExclusive):
        Collective.Recurrence = RecurKind::Add;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFAdd):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFAddInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFAddExclusive):
        Collective.Recurrence = RecurKind::FAdd;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceMul):
      case CASE_GROUP_OP_ALL_SCOPES(ScanMulInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanMulExclusive):
        Collective.Recurrence = RecurKind::Mul;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFMul):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMulInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMulExclusive):
        Collective.Recurrence = RecurKind::FMul;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceSMin):
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMinInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMinExclusive):
        Collective.Recurrence = RecurKind::SMin;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceUMin):
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMinInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMinExclusive):
        Collective.Recurrence = RecurKind::UMin;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFMin):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMinInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMinExclusive):
        Collective.Recurrence = RecurKind::FMin;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceSMax):
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMaxInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanSMaxExclusive):
        Collective.Recurrence = RecurKind::SMax;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceUMax):
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMaxInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanUMaxExclusive):
        Collective.Recurrence = RecurKind::UMax;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceFMax):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMaxInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanFMaxExclusive):
        Collective.Recurrence = RecurKind::FMax;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceAnd):
      case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalAnd):
      case CASE_GROUP_OP_ALL_SCOPES(ScanAndInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanAndExclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalAndInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalAndExclusive):
        Collective.Recurrence = RecurKind::And;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceOr):
      case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalOr):
      case CASE_GROUP_OP_ALL_SCOPES(ScanOrInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanOrExclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalOrInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalOrExclusive):
        Collective.Recurrence = RecurKind::Or;
        break;
      case CASE_GROUP_OP_ALL_SCOPES(ReduceXor):
      case CASE_GROUP_OP_ALL_SCOPES(ReduceLogicalXor):
      case CASE_GROUP_OP_ALL_SCOPES(ScanXorInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanXorExclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalXorInclusive):
      case CASE_GROUP_OP_ALL_SCOPES(ScanLogicalXorExclusive):
        Collective.Recurrence = RecurKind::Xor;
        break;
      default:
        llvm_unreachable("Unhandled mux group operation");
    }
  } else if (!Collective.isBroadcast() && !Collective.isShuffleLike()) {
    llvm_unreachable("Unhandled mux group operation");
  }

  return Collective;
#undef CASE_GROUP_OP_ALL_SCOPES
}

BuiltinID BuiltinInfo::getMuxGroupCollective(const GroupCollective &Group) {
#define SIMPLE_SCOPE_SWITCH(OP)                     \
  do {                                              \
    switch (Group.Scope) {                          \
      default:                                      \
        llvm_unreachable("Impossible scope kind");  \
      case GroupCollective::ScopeKind::SubGroup:    \
        return eMuxBuiltinSubgroup##OP;             \
      case GroupCollective::ScopeKind::WorkGroup:   \
        return eMuxBuiltinWorkgroup##OP;            \
      case GroupCollective::ScopeKind::VectorGroup: \
        return eMuxBuiltinVecgroup##OP;             \
    }                                               \
  } while (0)

#define COMPLEX_SCOPE_SWITCH(OP, SUFFIX)               \
  do {                                                 \
    switch (Group.Recurrence) {                        \
      default:                                         \
        llvm_unreachable("Unhandled recursion kind");  \
      case RecurKind::Add:                             \
        SIMPLE_SCOPE_SWITCH(OP##Add##SUFFIX);          \
      case RecurKind::Mul:                             \
        SIMPLE_SCOPE_SWITCH(OP##Mul##SUFFIX);          \
      case RecurKind::FAdd:                            \
        SIMPLE_SCOPE_SWITCH(OP##FAdd##SUFFIX);         \
      case RecurKind::FMul:                            \
        SIMPLE_SCOPE_SWITCH(OP##FMul##SUFFIX);         \
      case RecurKind::SMin:                            \
        SIMPLE_SCOPE_SWITCH(OP##SMin##SUFFIX);         \
      case RecurKind::UMin:                            \
        SIMPLE_SCOPE_SWITCH(OP##UMin##SUFFIX);         \
      case RecurKind::FMin:                            \
        SIMPLE_SCOPE_SWITCH(OP##FMin##SUFFIX);         \
      case RecurKind::SMax:                            \
        SIMPLE_SCOPE_SWITCH(OP##SMax##SUFFIX);         \
      case RecurKind::UMax:                            \
        SIMPLE_SCOPE_SWITCH(OP##UMax##SUFFIX);         \
      case RecurKind::FMax:                            \
        SIMPLE_SCOPE_SWITCH(OP##FMax##SUFFIX);         \
      case RecurKind::And:                             \
        if (Group.IsLogical) {                         \
          SIMPLE_SCOPE_SWITCH(OP##LogicalAnd##SUFFIX); \
        } else {                                       \
          SIMPLE_SCOPE_SWITCH(OP##And##SUFFIX);        \
        }                                              \
      case RecurKind::Or:                              \
        if (Group.IsLogical) {                         \
          SIMPLE_SCOPE_SWITCH(OP##LogicalOr##SUFFIX);  \
        } else {                                       \
          SIMPLE_SCOPE_SWITCH(OP##Or##SUFFIX);         \
        }                                              \
      case RecurKind::Xor:                             \
        if (Group.IsLogical) {                         \
          SIMPLE_SCOPE_SWITCH(OP##LogicalXor##SUFFIX); \
        } else {                                       \
          SIMPLE_SCOPE_SWITCH(OP##Xor##SUFFIX);        \
        }                                              \
    }                                                  \
  } while (0)

  switch (Group.Op) {
    case GroupCollective::OpKind::All:
      SIMPLE_SCOPE_SWITCH(All);
    case GroupCollective::OpKind::Any:
      SIMPLE_SCOPE_SWITCH(Any);
    case GroupCollective::OpKind::Broadcast:
      SIMPLE_SCOPE_SWITCH(Broadcast);
    case GroupCollective::OpKind::Reduction:
      COMPLEX_SCOPE_SWITCH(Reduce, );
    case GroupCollective::OpKind::ScanExclusive:
      COMPLEX_SCOPE_SWITCH(Scan, Exclusive);
    case GroupCollective::OpKind::ScanInclusive:
      COMPLEX_SCOPE_SWITCH(Scan, Inclusive);
      break;
    case GroupCollective::OpKind::Shuffle:
      return Group.isSubGroupScope() ? eMuxBuiltinSubgroupShuffle
                                     : eBuiltinInvalid;
    case GroupCollective::OpKind::ShuffleUp:
      return Group.isSubGroupScope() ? eMuxBuiltinSubgroupShuffleUp
                                     : eBuiltinInvalid;
    case GroupCollective::OpKind::ShuffleDown:
      return Group.isSubGroupScope() ? eMuxBuiltinSubgroupShuffleDown
                                     : eBuiltinInvalid;
    case GroupCollective::OpKind::ShuffleXor:
      return Group.isSubGroupScope() ? eMuxBuiltinSubgroupShuffleXor
                                     : eBuiltinInvalid;
  }
  return eBuiltinInvalid;
#undef COMPLEX_SCOPE_SWITCH
#undef SCOPE_SWITCH
}

bool BuiltinInfo::isOverloadableMuxBuiltinID(BuiltinID ID) {
  if (!isMuxBuiltinID(ID)) {
    return false;
  }
  switch (ID) {
    default:
      return isMuxGroupCollective(ID).has_value();
    case eMuxBuiltinDMARead1D:
    case eMuxBuiltinDMAWrite1D:
    case eMuxBuiltinDMARead2D:
    case eMuxBuiltinDMAWrite2D:
    case eMuxBuiltinDMARead3D:
    case eMuxBuiltinDMAWrite3D:
      return true;
  }
}

}  // namespace utils
}  // namespace compiler
