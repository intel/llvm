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
#include <compiler/utils/cl_builtin_info.h>
#include <compiler/utils/mangling.h>
#include <compiler/utils/metadata.h>
#include <compiler/utils/pass_functions.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/ValueMapper.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include <cmath>
#include <set>

// For compatibility with the Android NDK, we need to use the C ilogb function.
namespace stdcompat {
#ifdef __ANDROID__
// Note: This function accepts double only as its argument
using ::ilogb;
#else
using std::ilogb;
#endif  // __ANDROID__
}  // namespace stdcompat

namespace {
/// @brief Identifiers for recognized OpenCL builtins.
enum CLBuiltinID : compiler::utils::BuiltinID {
  // Non-standard Builtin Functions
  /// @brief Internal builtin 'convert_half_to_float'.
  eCLBuiltinConvertHalfToFloat = compiler::utils::eFirstTargetBuiltin,
  /// @brief Internal builtin 'convert_float_to_half'.
  eCLBuiltinConvertFloatToHalf,
  /// @brief Internal builtin 'convert_float_to_half_rte'
  eCLBuiltinConvertFloatToHalfRte,
  /// @brief Internal builtin 'convert_float_to_half_rtz'
  eCLBuiltinConvertFloatToHalfRtz,
  /// @brief Internal builtin 'convert_float_to_half_rtp'
  eCLBuiltinConvertFloatToHalfRtp,
  /// @brief Internal builtin 'convert_float_to_half_rtn'
  eCLBuiltinConvertFloatToHalfRtn,
  /// @brief Internal builtin 'convert_half_to_double'.
  eCLBuiltinConvertHalfToDouble,
  /// @brief Internal builtin 'convert_double_to_half'.
  eCLBuiltinConvertDoubleToHalf,
  /// @brief Internal builtin 'convert_double_to_half_rte'
  eCLBuiltinConvertDoubleToHalfRte,
  /// @brief Internal builtin 'convert_double_to_half_rtz'
  eCLBuiltinConvertDoubleToHalfRtz,
  /// @brief Internal builtin 'convert_double_to_half_rtp'
  eCLBuiltinConvertDoubleToHalfRtp,
  /// @brief Internal builtin 'convert_double_to_half_rtn'
  eCLBuiltinConvertDoubleToHalfRtn,

  // 6.2.3 Explicit Conversions
  /// @brief OpenCL builtin `convert_char`
  eCLBuiltinConvertChar,
  /// @brief OpenCL builtin `convert_short`
  eCLBuiltinConvertShort,
  /// @brief OpenCL builtin `convert_int`
  eCLBuiltinConvertInt,
  /// @brief OpenCL builtin `convert_long`
  eCLBuiltinConvertLong,
  /// @brief OpenCL builtin `convert_uchar`
  eCLBuiltinConvertUChar,
  /// @brief OpenCL builtin `convert_ushort`
  eCLBuiltinConvertUShort,
  /// @brief OpenCL builtin `convert_uint`
  eCLBuiltinConvertUInt,
  /// @brief OpenCL builtin `convert_ulong`
  eCLBuiltinConvertULong,

  // 6.12.1 Work-Item Functions
  /// @brief OpenCL builtin 'get_work_dim'.
  eCLBuiltinGetWorkDim,
  /// @brief OpenCL builtin 'get_group_id'.
  eCLBuiltinGetGroupId,
  /// @brief OpenCL builtin 'get_global_size'.
  eCLBuiltinGetGlobalSize,
  /// @brief OpenCL builtin 'get_global_offset'.
  eCLBuiltinGetGlobalOffset,
  /// @brief OpenCL builtin 'get_local_id'.
  eCLBuiltinGetLocalId,
  /// @brief OpenCL builtin 'get_local_size'.
  eCLBuiltinGetLocalSize,
  /// @brief OpenCL builtin 'get_enqueued_local_size'.
  eCLBuiltinGetEnqueuedLocalSize,
  /// @brief OpenCL builtin 'get_num_groups'.
  eCLBuiltinGetNumGroups,
  /// @brief OpenCL builtin 'get_global_id'.
  eCLBuiltinGetGlobalId,
  /// @brief OpenCL builtin 'get_local_linear_id' (OpenCL >= 2.0).
  eCLBuiltinGetLocalLinearId,
  /// @brief OpenCL builtin 'get_global_linear_id' (OpenCL >= 2.0).
  eCLBuiltinGetGlobalLinearId,
  /// @brief OpenCL builtin 'get_sub_group_local_id' (OpenCL >= 3.0).
  eCLBuiltinGetSubgroupLocalId,
  /// @brief OpenCL builtin 'get_sub_group_size' (OpenCL >= 3.0).
  eCLBuiltinGetSubgroupSize,
  /// @brief OpenCL builtin 'get_max_sub_group_size' (OpenCL >= 3.0).
  eCLBuiltinGetMaxSubgroupSize,
  /// @brief OpenCL builtin 'get_num_sub_groups' (OpenCL >= 3.0).
  eCLBuiltinGetNumSubgroups,
  /// @brief OpenCL builtin 'get_enqueued_num_sub_groups' (OpenCL >= 3.0).
  eCLBuiltinGetEnqueuedNumSubgroups,
  /// @brief OpenCL builtin 'get_sub_group_id' (OpenCL >= 3.0).
  eCLBuiltinGetSubgroupId,

  // 6.12.2 Math Functions
  /// @brief OpenCL builtin 'fmax'.
  eCLBuiltinFMax,
  /// @brief OpenCL builtin 'fmin'.
  eCLBuiltinFMin,
  /// @brief OpenCL builtin 'fract'.
  eCLBuiltinFract,
  /// @brief OpenCL builtin 'frexp'.
  eCLBuiltinFrexp,
  /// @brief OpenCL builtin 'lgamma_r'.
  eCLBuiltinLGammaR,
  /// @brief OpenCL builtin 'modf'.
  eCLBuiltinModF,
  /// @brief OpenCL builtin 'sincos'.
  eCLBuiltinSinCos,
  /// @brief OpenCL builtin 'remquo'.
  eCLBuiltinRemquo,

  // 6.12.3 Integer Functions
  /// @brief OpenCL builtin 'add_sat'.
  eCLBuiltinAddSat,
  /// @brief OpenCL builtin 'sub_sat'.
  eCLBuiltinSubSat,

  // 6.12.5 Geometric Builtin-in Functions
  /// @brief OpenCL builtin 'dot'.
  eCLBuiltinDot,
  /// @brief OpenCL builtin 'cross'.
  eCLBuiltinCross,
  /// @brief OpenCL builtin 'length'.
  eCLBuiltinLength,
  /// @brief OpenCL builtin 'distance'.
  eCLBuiltinDistance,
  /// @brief OpenCL builtin 'normalize'.
  eCLBuiltinNormalize,
  /// @brief OpenCL builtin 'fast_length'.
  eCLBuiltinFastLength,
  /// @brief OpenCL builtin 'fast_distance'.
  eCLBuiltinFastDistance,
  /// @brief OpenCL builtin 'fast_normalize'.
  eCLBuiltinFastNormalize,

  // 6.12.6 Relational Functions
  /// @brief OpenCL builtin 'all'.
  eCLBuiltinAll,
  /// @brief OpenCL builtin 'any'.
  eCLBuiltinAny,
  /// @brief OpenCL builtin 'isequal'.
  eCLBuiltinIsEqual,
  /// @brief OpenCL builtin 'isnotequal'.
  eCLBuiltinIsNotEqual,
  /// @brief OpenCL builtin 'isgreater'.
  eCLBuiltinIsGreater,
  /// @brief OpenCL builtin 'isgreaterequal'.
  eCLBuiltinIsGreaterEqual,
  /// @brief OpenCL builtin 'isless'.
  eCLBuiltinIsLess,
  /// @brief OpenCL builtin 'islessequal'.
  eCLBuiltinIsLessEqual,
  /// @brief OpenCL builtin 'islessgreater'.
  eCLBuiltinIsLessGreater,
  /// @brief OpenCL builtin 'isordered'.
  eCLBuiltinIsOrdered,
  /// @brief OpenCL builtin 'isunordered'.
  eCLBuiltinIsUnordered,
  /// @brief OpenCL builtin 'isfinite'.
  eCLBuiltinIsFinite,
  /// @brief OpenCL builtin 'isinf'.
  eCLBuiltinIsInf,
  /// @brief OpenCL builtin 'isnan'.
  eCLBuiltinIsNan,
  /// @brief OpenCL builtin 'isnormal'.
  eCLBuiltinIsNormal,
  /// @brief OpenCL builtin 'signbit'.
  eCLBuiltinSignBit,
  /// @brief OpenCL builtin `select`.
  eCLBuiltinSelect,

  // 6.12.8 Synchronization Functions
  /// @brief OpenCL builtin 'barrier'.
  eCLBuiltinBarrier,
  /// @brief OpenCL builtin 'mem_fence'.
  eCLBuiltinMemFence,
  /// @brief OpenCL builtin 'read_mem_fence'.
  eCLBuiltinReadMemFence,
  /// @brief OpenCL builtin 'write_mem_fence'.
  eCLBuiltinWriteMemFence,
  /// @brief OpenCL builtin 'atomic_work_item_fence'.
  eCLBuiltinAtomicWorkItemFence,
  /// @brief OpenCL builtin 'sub_group_barrier'.
  eCLBuiltinSubGroupBarrier,
  /// @brief OpenCL builtin 'work_group_barrier'.
  eCLBuiltinWorkGroupBarrier,

  // 6.12.10 Async Copies and Prefetch Functions
  /// @brief OpenCL builtin 'async_work_group_copy'.
  eCLBuiltinAsyncWorkGroupCopy,
  /// @brief OpenCL builtin 'async_work_group_strided_copy'.
  eCLBuiltinAsyncWorkGroupStridedCopy,
  /// @brief OpenCL builtin 'wait_group_events'.
  eCLBuiltinWaitGroupEvents,
  /// @brief OpenCL builtin 'async_work_group_copy_2D2D'.
  eCLBuiltinAsyncWorkGroupCopy2D2D,
  /// @brief OpenCL builtin 'async_work_group_copy_3D3D'.
  eCLBuiltinAsyncWorkGroupCopy3D3D,

  // 6.12.11 Atomic Functions
  /// @brief OpenCL builtins 'atomic_add', 'atom_add'.
  eCLBuiltinAtomicAdd,
  /// @brief OpenCL builtins 'atomic_sub', 'atom_sub'.
  eCLBuiltinAtomicSub,
  /// @brief OpenCL builtins 'atomic_xchg', 'atom_xchg'.
  eCLBuiltinAtomicXchg,
  /// @brief OpenCL builtins 'atomic_inc', 'atom_inc'.
  eCLBuiltinAtomicInc,
  /// @brief OpenCL builtins 'atomic_dec', 'atom_dec'.
  eCLBuiltinAtomicDec,
  /// @brief OpenCL builtins 'atomic_cmpxchg', 'atom_cmpxchg'.
  eCLBuiltinAtomicCmpxchg,
  /// @brief OpenCL builtins 'atomic_min', 'atom_min'.
  eCLBuiltinAtomicMin,
  /// @brief OpenCL builtins 'atomic_max', 'atom_max'.
  eCLBuiltinAtomicMax,
  /// @brief OpenCL builtins 'atomic_and', 'atom_and'.
  eCLBuiltinAtomicAnd,
  /// @brief OpenCL builtins 'atomic_or', 'atom_or'.
  eCLBuiltinAtomicOr,
  /// @brief OpenCL builtins 'atomic_xor', 'atom_xor'.
  eCLBuiltinAtomicXor,

  // 6.12.12 Miscellaneous Vector Functions
  eCLBuiltinShuffle,
  eCLBuiltinShuffle2,

  // 6.12.13 printf
  /// @brief OpenCL builtin 'printf'.
  eCLBuiltinPrintf,

  // 6.15.16 Work-group Collective Functions
  /// @brief OpenCL builtin 'work_group_all'.
  eCLBuiltinWorkgroupAll,
  /// @brief OpenCL builtin 'work_group_any'.
  eCLBuiltinWorkgroupAny,
  /// @brief OpenCL builtin 'work_group_broadcast'.
  eCLBuiltinWorkgroupBroadcast,
  /// @brief OpenCL builtin 'work_group_reduce_add'.
  eCLBuiltinWorkgroupReduceAdd,
  /// @brief OpenCL builtin 'work_group_reduce_min'.
  eCLBuiltinWorkgroupReduceMin,
  /// @brief OpenCL builtin 'work_group_reduce_max'.
  eCLBuiltinWorkgroupReduceMax,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_add'.
  eCLBuiltinWorkgroupScanAddInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_add'.
  eCLBuiltinWorkgroupScanAddExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_min'.
  eCLBuiltinWorkgroupScanMinInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_min'.
  eCLBuiltinWorkgroupScanMinExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_max'.
  eCLBuiltinWorkgroupScanMaxInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_max'.
  eCLBuiltinWorkgroupScanMaxExclusive,

  /// @brief OpenCL builtin 'work_group_reduce_mul'.
  eCLBuiltinWorkgroupReduceMul,
  /// @brief OpenCL builtin 'work_group_reduce_and'.
  eCLBuiltinWorkgroupReduceAnd,
  /// @brief OpenCL builtin 'work_group_reduce_or'.
  eCLBuiltinWorkgroupReduceOr,
  /// @brief OpenCL builtin 'work_group_reduce_xor'.
  eCLBuiltinWorkgroupReduceXor,
  /// @brief OpenCL builtin 'work_group_reduce_logical_and'.
  eCLBuiltinWorkgroupReduceLogicalAnd,
  /// @brief OpenCL builtin 'work_group_reduce_logical_or'.
  eCLBuiltinWorkgroupReduceLogicalOr,
  /// @brief OpenCL builtin 'work_group_reduce_logical_xor'.
  eCLBuiltinWorkgroupReduceLogicalXor,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_mul'.
  eCLBuiltinWorkgroupScanMulInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_mul'.
  eCLBuiltinWorkgroupScanMulExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_and'.
  eCLBuiltinWorkgroupScanAndInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_and'.
  eCLBuiltinWorkgroupScanAndExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_or'.
  eCLBuiltinWorkgroupScanOrInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_or'.
  eCLBuiltinWorkgroupScanOrExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_xor'.
  eCLBuiltinWorkgroupScanXorInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_xor'.
  eCLBuiltinWorkgroupScanXorExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_logical_and'.
  eCLBuiltinWorkgroupScanLogicalAndInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_logical_and'.
  eCLBuiltinWorkgroupScanLogicalAndExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_logical_or'.
  eCLBuiltinWorkgroupScanLogicalOrInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_logical_or'.
  eCLBuiltinWorkgroupScanLogicalOrExclusive,
  /// @brief OpenCL builtin 'work_group_scan_inclusive_logical_xor'.
  eCLBuiltinWorkgroupScanLogicalXorInclusive,
  /// @brief OpenCL builtin 'work_group_scan_exclusive_logical_xor'.
  eCLBuiltinWorkgroupScanLogicalXorExclusive,

  // 6.15.19 Subgroup Collective Functions
  /// @brief OpenCL builtin 'sub_group_all'.
  eCLBuiltinSubgroupAll,
  /// @brief OpenCL builtin 'sub_group_any'.
  eCLBuiltinSubgroupAny,
  /// @brief OpenCL builtin 'sub_group_broadcast'.
  eCLBuiltinSubgroupBroadcast,
  /// @brief OpenCL builtin 'sub_group_reduce_add'.
  eCLBuiltinSubgroupReduceAdd,
  /// @brief OpenCL builtin 'sub_group_reduce_min'.
  eCLBuiltinSubgroupReduceMin,
  /// @brief OpenCL builtin 'sub_group_reduce_max'.
  eCLBuiltinSubgroupReduceMax,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_add'.
  eCLBuiltinSubgroupScanAddInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_add'.
  eCLBuiltinSubgroupScanAddExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_min'.
  eCLBuiltinSubgroupScanMinInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_min'.
  eCLBuiltinSubgroupScanMinExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_max'.
  eCLBuiltinSubgroupScanMaxInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_max'.
  eCLBuiltinSubgroupScanMaxExclusive,

  /// @brief OpenCL builtin 'sub_group_reduce_mul'.
  eCLBuiltinSubgroupReduceMul,
  /// @brief OpenCL builtin 'sub_group_reduce_and'.
  eCLBuiltinSubgroupReduceAnd,
  /// @brief OpenCL builtin 'sub_group_reduce_or'.
  eCLBuiltinSubgroupReduceOr,
  /// @brief OpenCL builtin 'sub_group_reduce_xor'.
  eCLBuiltinSubgroupReduceXor,
  /// @brief OpenCL builtin 'sub_group_reduce_logical_and'.
  eCLBuiltinSubgroupReduceLogicalAnd,
  /// @brief OpenCL builtin 'sub_group_reduce_logical_or'.
  eCLBuiltinSubgroupReduceLogicalOr,
  /// @brief OpenCL builtin 'sub_group_reduce_logical_xor'.
  eCLBuiltinSubgroupReduceLogicalXor,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_mul'.
  eCLBuiltinSubgroupScanMulInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_mul'.
  eCLBuiltinSubgroupScanMulExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_and'.
  eCLBuiltinSubgroupScanAndInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_and'.
  eCLBuiltinSubgroupScanAndExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_or'.
  eCLBuiltinSubgroupScanOrInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_or'.
  eCLBuiltinSubgroupScanOrExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_xor'.
  eCLBuiltinSubgroupScanXorInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_xor'.
  eCLBuiltinSubgroupScanXorExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_logical_and'.
  eCLBuiltinSubgroupScanLogicalAndInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_logical_and'.
  eCLBuiltinSubgroupScanLogicalAndExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_logical_or'.
  eCLBuiltinSubgroupScanLogicalOrInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_logical_or'.
  eCLBuiltinSubgroupScanLogicalOrExclusive,
  /// @brief OpenCL builtin 'sub_group_scan_inclusive_logical_xor'.
  eCLBuiltinSubgroupScanLogicalXorInclusive,
  /// @brief OpenCL builtin 'sub_group_scan_exclusive_logical_xor'.
  eCLBuiltinSubgroupScanLogicalXorExclusive,

  // GLSL builtin functions
  eCLBuiltinCodeplayFindLSB,
  eCLBuiltinCodeplayFindMSB,
  eCLBuiltinCodeplayBitReverse,
  eCLBuiltinCodeplayFaceForward,
  eCLBuiltinCodeplayReflect,
  eCLBuiltinCodeplayRefract,
  eCLBuiltinCodeplayPackNormalizeChar4,
  eCLBuiltinCodeplayPackNormalizeUchar4,
  eCLBuiltinCodeplayPackNormalizeShort2,
  eCLBuiltinCodeplayPackNormalizeUshort2,
  eCLBuiltinCodeplayPackHalf2,
  eCLBuiltinCodeplayUnpackNormalize,
  eCLBuiltinCodeplayUnpackHalf2,

  // 6.12.7 Vector Data Load and Store Functions
  eCLBuiltinVLoad,
  eCLBuiltinVLoadHalf,
  eCLBuiltinVStore,
  eCLBuiltinVStoreHalf,

  // 6.3 Conversions & Type Casting Examples
  eCLBuiltinAs,
};
}  // namespace

namespace {
using namespace llvm;
using namespace compiler::utils;

// Returns whether the given integer is a valid vector width in OpenCL.
// Matches 2, 3, 4, 8, 16.
bool isValidVecWidth(unsigned w) {
  return (w == 3 || (w >= 2 && w <= 16 && llvm::isPowerOf2_32(w)));
}

/// @brief Copy global variables to a module on demand.
class GlobalValueMaterializer final : public llvm::ValueMaterializer {
 public:
  /// @brief Create a new global variable materializer.
  /// @param[in] M Module to materialize the variables in.
  GlobalValueMaterializer(Module &M) : DestM(M) {}

  /// @brief List of variables created during materialization.
  const std::vector<GlobalVariable *> &variables() const { return Variables; }

  /// @brief Materialize the given value.
  ///
  /// @param[in] V Value to materialize.
  ///
  /// @return A value that lives in the destination module, or nullptr if the
  /// given value could not be materialized (e.g. it is not a global variable).
  Value *materialize(Value *V) override final {
    GlobalVariable *GV = dyn_cast<GlobalVariable>(V);
    if (!GV) {
      return nullptr;
    }
    GlobalVariable *NewGV = DestM.getGlobalVariable(GV->getName());
    if (!NewGV) {
      NewGV = new GlobalVariable(
          DestM, GV->getValueType(), GV->isConstant(), GV->getLinkage(),
          (Constant *)nullptr, GV->getName(), (GlobalVariable *)nullptr,
          GV->getThreadLocalMode(), GV->getType()->getAddressSpace());
      NewGV->copyAttributesFrom(GV);
      Variables.push_back(GV);
    }
    return NewGV;
  }

 private:
  /// @brief Modules to materialize variables in.
  Module &DestM;
  /// @brief Materialized variables.
  std::vector<GlobalVariable *> Variables;
};
}  // namespace

namespace compiler {
namespace utils {
using namespace llvm;

std::unique_ptr<BILangInfoConcept> createCLBuiltinInfo(Module *Builtins) {
  return std::make_unique<CLBuiltinInfo>(Builtins);
}

CLBuiltinInfo::CLBuiltinInfo(Module *builtins)
    : Loader(std::make_unique<SimpleCLBuiltinLoader>(builtins)) {}

CLBuiltinInfo::~CLBuiltinInfo() = default;

/// @brief Create a call instruction to the given builtin and set the correct
/// calling convention.
///
/// This function is intended as a helper function for creating calls to
/// builtins. For each call generated we need to set the calling convention
/// manually, which can lead to code bloat. This function will create the call
/// instruction and then it will either copy the calling convention for the
/// called function (if possible) or set it to the default value of spir_func.
///
/// @param[in] B The IRBuilder to use when creating the CallInst
/// @param[in] Builtin The Function to call
/// @param[in] Args The call arguments
/// @param[in] NameStr The name for the new CallInst
/// @return The newly emitted CallInst
static CallInst *CreateBuiltinCall(IRBuilder<> &B, Function *Builtin,
                                   ArrayRef<Value *> Args,
                                   const Twine &NameStr = "") {
  CallInst *CI =
      B.CreateCall(Builtin->getFunctionType(), Builtin, Args, NameStr);
  CI->setCallingConv(Builtin->getCallingConv());
  return CI;
}

struct CLBuiltinEntry {
  /// @brief Identifier for the builtin function.
  BuiltinID ID;
  /// @brief OpenCL name of the builtin function.
  const char *OpenCLFnName;
  /// @brief Minimum OpenCL version that supports this builtin.
  uint32_t MinVer = OpenCLC10;
};

/// @brief Information about known OpenCL builtins.
static constexpr CLBuiltinEntry Builtins[] = {
    // Non-standard Builtin Functions
    {eCLBuiltinConvertHalfToFloat, "convert_half_to_float"},
    {eCLBuiltinConvertFloatToHalf, "convert_float_to_half"},
    {eCLBuiltinConvertFloatToHalfRte, "convert_float_to_half_rte"},
    {eCLBuiltinConvertFloatToHalfRtz, "convert_float_to_half_rtz"},
    {eCLBuiltinConvertFloatToHalfRtp, "convert_float_to_half_rtp"},
    {eCLBuiltinConvertFloatToHalfRtn, "convert_float_to_half_rtn"},
    {eCLBuiltinConvertHalfToDouble, "convert_half_to_double"},
    {eCLBuiltinConvertDoubleToHalf, "convert_double_to_half"},
    {eCLBuiltinConvertDoubleToHalfRte, "convert_double_to_half_rte"},
    {eCLBuiltinConvertDoubleToHalfRtz, "convert_double_to_half_rtz"},
    {eCLBuiltinConvertDoubleToHalfRtp, "convert_double_to_half_rtp"},
    {eCLBuiltinConvertDoubleToHalfRtn, "convert_double_to_half_rtn"},

    // 6.2.3 Explicit Conversions
    {eCLBuiltinConvertChar, "convert_char"},
    {eCLBuiltinConvertShort, "convert_short"},
    {eCLBuiltinConvertInt, "convert_int"},
    {eCLBuiltinConvertLong, "convert_long"},
    {eCLBuiltinConvertUChar, "convert_uchar"},
    {eCLBuiltinConvertUShort, "convert_ushort"},
    {eCLBuiltinConvertUInt, "convert_uint"},
    {eCLBuiltinConvertULong, "convert_ulong"},

    // 6.12.1 Work-Item Functions
    {eCLBuiltinGetWorkDim, "get_work_dim"},
    {eCLBuiltinGetGroupId, "get_group_id"},
    {eCLBuiltinGetGlobalSize, "get_global_size"},
    {eCLBuiltinGetGlobalOffset, "get_global_offset"},
    {eCLBuiltinGetLocalId, "get_local_id"},
    {eCLBuiltinGetLocalSize, "get_local_size"},
    {eCLBuiltinGetEnqueuedLocalSize, "get_enqueued_local_size"},
    {eCLBuiltinGetNumGroups, "get_num_groups"},
    {eCLBuiltinGetGlobalId, "get_global_id"},
    {eCLBuiltinGetLocalLinearId, "get_local_linear_id", OpenCLC20},
    {eCLBuiltinGetGlobalLinearId, "get_global_linear_id", OpenCLC20},
    {eCLBuiltinGetSubgroupLocalId, "get_sub_group_local_id", OpenCLC30},
    {eCLBuiltinGetSubgroupSize, "get_sub_group_size", OpenCLC30},
    {eCLBuiltinGetMaxSubgroupSize, "get_max_sub_group_size", OpenCLC30},
    {eCLBuiltinGetNumSubgroups, "get_num_sub_groups", OpenCLC30},
    {eCLBuiltinGetEnqueuedNumSubgroups, "get_enqueued_num_sub_groups",
     OpenCLC30},
    {eCLBuiltinGetSubgroupId, "get_sub_group_id", OpenCLC30},

    // 6.12.2 Math Functions
    {eCLBuiltinFMax, "fmax"},
    {eCLBuiltinFMin, "fmin"},
    {eCLBuiltinFract, "fract"},
    {eCLBuiltinFrexp, "frexp"},
    {eCLBuiltinLGammaR, "lgamma_r"},
    {eCLBuiltinModF, "modf"},
    {eCLBuiltinSinCos, "sincos"},
    {eCLBuiltinRemquo, "remquo"},

    // 6.12.3 Integer Functions
    {eCLBuiltinAddSat, "add_sat"},
    {eCLBuiltinSubSat, "sub_sat"},

    // 6.12.5 Geometric Functions
    {eCLBuiltinDot, "dot"},
    {eCLBuiltinCross, "cross"},
    {eCLBuiltinLength, "length"},
    {eCLBuiltinDistance, "distance"},
    {eCLBuiltinNormalize, "normalize"},
    {eCLBuiltinFastLength, "fast_length"},
    {eCLBuiltinFastDistance, "fast_distance"},
    {eCLBuiltinFastNormalize, "fast_normalize"},

    // 6.12.6 Relational Functions
    {eCLBuiltinAll, "all"},
    {eCLBuiltinAny, "any"},
    {eCLBuiltinIsEqual, "isequal"},
    {eCLBuiltinIsNotEqual, "isnotequal"},
    {eCLBuiltinIsGreater, "isgreater"},
    {eCLBuiltinIsGreaterEqual, "isgreaterequal"},
    {eCLBuiltinIsLess, "isless"},
    {eCLBuiltinIsLessEqual, "islessequal"},
    {eCLBuiltinIsLessGreater, "islessgreater"},
    {eCLBuiltinIsOrdered, "isordered"},
    {eCLBuiltinIsUnordered, "isunordered"},
    {eCLBuiltinIsFinite, "isfinite"},
    {eCLBuiltinIsInf, "isinf"},
    {eCLBuiltinIsNan, "isnan"},
    {eCLBuiltinIsNormal, "isnormal"},
    {eCLBuiltinSignBit, "signbit"},
    {eCLBuiltinSelect, "select"},

    // 6.12.8 Synchronization Functions
    {eCLBuiltinBarrier, "barrier"},
    {eCLBuiltinMemFence, "mem_fence"},
    {eCLBuiltinReadMemFence, "read_mem_fence"},
    {eCLBuiltinWriteMemFence, "write_mem_fence"},
    {eCLBuiltinAtomicWorkItemFence, "atomic_work_item_fence", OpenCLC20},
    {eCLBuiltinSubGroupBarrier, "sub_group_barrier", OpenCLC30},
    {eCLBuiltinWorkGroupBarrier, "work_group_barrier", OpenCLC20},

    // 6.12.10 Async Copies and Prefetch Functions
    {eCLBuiltinAsyncWorkGroupCopy, "async_work_group_copy"},
    {eCLBuiltinAsyncWorkGroupStridedCopy, "async_work_group_strided_copy"},
    {eCLBuiltinWaitGroupEvents, "wait_group_events"},
    {eCLBuiltinAsyncWorkGroupCopy2D2D, "async_work_group_copy_2D2D"},
    {eCLBuiltinAsyncWorkGroupCopy3D3D, "async_work_group_copy_3D3D"},

    // 6.12.11 Atomic Functions
    {eCLBuiltinAtomicAdd, "atom_add"},
    {eCLBuiltinAtomicSub, "atom_sub"},
    {eCLBuiltinAtomicXchg, "atom_xchg"},
    {eCLBuiltinAtomicInc, "atom_inc"},
    {eCLBuiltinAtomicDec, "atom_dec"},
    {eCLBuiltinAtomicCmpxchg, "atom_cmpxchg"},
    {eCLBuiltinAtomicMin, "atom_min"},
    {eCLBuiltinAtomicMax, "atom_max"},
    {eCLBuiltinAtomicAnd, "atom_and"},
    {eCLBuiltinAtomicOr, "atom_or"},
    {eCLBuiltinAtomicXor, "atom_xor"},
    {eCLBuiltinAtomicAdd, "atomic_add"},
    {eCLBuiltinAtomicSub, "atomic_sub"},
    {eCLBuiltinAtomicXchg, "atomic_xchg"},
    {eCLBuiltinAtomicInc, "atomic_inc"},
    {eCLBuiltinAtomicDec, "atomic_dec"},
    {eCLBuiltinAtomicCmpxchg, "atomic_cmpxchg"},
    {eCLBuiltinAtomicMin, "atomic_min"},
    {eCLBuiltinAtomicMax, "atomic_max"},
    {eCLBuiltinAtomicAnd, "atomic_and"},
    {eCLBuiltinAtomicOr, "atomic_or"},
    {eCLBuiltinAtomicXor, "atomic_xor"},

    // 6.11.12 Miscellaneous Vector Functions
    {eCLBuiltinShuffle, "shuffle"},
    {eCLBuiltinShuffle2, "shuffle2"},

    // 6.12.13 printf
    {eCLBuiltinPrintf, "printf"},

    // 6.15.16 Work-group Collective Functions
    {eCLBuiltinWorkgroupAll, "work_group_all", OpenCLC20},
    {eCLBuiltinWorkgroupAny, "work_group_any", OpenCLC20},
    {eCLBuiltinWorkgroupBroadcast, "work_group_broadcast", OpenCLC20},
    {eCLBuiltinWorkgroupReduceAdd, "work_group_reduce_add", OpenCLC20},
    {eCLBuiltinWorkgroupReduceMin, "work_group_reduce_min", OpenCLC20},
    {eCLBuiltinWorkgroupReduceMax, "work_group_reduce_max", OpenCLC20},
    {eCLBuiltinWorkgroupScanAddInclusive, "work_group_scan_inclusive_add",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanAddExclusive, "work_group_scan_exclusive_add",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanMinInclusive, "work_group_scan_inclusive_min",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanMinExclusive, "work_group_scan_exclusive_min",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanMaxInclusive, "work_group_scan_inclusive_max",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanMaxExclusive, "work_group_scan_exclusive_max",
     OpenCLC20},

    /// Provided by SPV_KHR_uniform_group_instructions.
    {eCLBuiltinWorkgroupReduceMul, "work_group_reduce_mul", OpenCLC20},
    {eCLBuiltinWorkgroupReduceAnd, "work_group_reduce_and", OpenCLC20},
    {eCLBuiltinWorkgroupReduceOr, "work_group_reduce_or", OpenCLC20},
    {eCLBuiltinWorkgroupReduceXor, "work_group_reduce_xor", OpenCLC20},
    {eCLBuiltinWorkgroupReduceLogicalAnd, "work_group_reduce_logical_and",
     OpenCLC20},
    {eCLBuiltinWorkgroupReduceLogicalOr, "work_group_reduce_logical_or",
     OpenCLC20},
    {eCLBuiltinWorkgroupReduceLogicalXor, "work_group_reduce_logical_xor",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanMulInclusive, "work_group_scan_inclusive_mul",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanMulExclusive, "work_group_scan_exclusive_mul",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanAndInclusive, "work_group_scan_inclusive_and",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanAndExclusive, "work_group_scan_exclusive_and",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanOrInclusive, "work_group_scan_inclusive_or",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanOrExclusive, "work_group_scan_exclusive_or",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanXorInclusive, "work_group_scan_inclusive_xor",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanXorExclusive, "work_group_scan_exclusive_xor",
     OpenCLC20},
    {eCLBuiltinWorkgroupScanLogicalAndInclusive,
     "work_group_scan_inclusive_logical_and", OpenCLC20},
    {eCLBuiltinWorkgroupScanLogicalAndExclusive,
     "work_group_scan_exclusive_logical_and", OpenCLC20},
    {eCLBuiltinWorkgroupScanLogicalOrInclusive,
     "work_group_scan_inclusive_logical_or", OpenCLC20},
    {eCLBuiltinWorkgroupScanLogicalOrExclusive,
     "work_group_scan_exclusive_logical_or", OpenCLC20},
    {eCLBuiltinWorkgroupScanLogicalXorInclusive,
     "work_group_scan_inclusive_logical_xor", OpenCLC20},
    {eCLBuiltinWorkgroupScanLogicalXorExclusive,
     "work_group_scan_exclusive_logical_xor", OpenCLC20},

    // 6.15.19 Subgroup Collective Functions
    {eCLBuiltinSubgroupAll, "sub_group_all", OpenCLC30},
    {eCLBuiltinSubgroupAny, "sub_group_any", OpenCLC30},
    {eCLBuiltinSubgroupBroadcast, "sub_group_broadcast", OpenCLC30},
    {eCLBuiltinSubgroupReduceAdd, "sub_group_reduce_add", OpenCLC30},
    {eCLBuiltinSubgroupReduceMin, "sub_group_reduce_min", OpenCLC30},
    {eCLBuiltinSubgroupReduceMax, "sub_group_reduce_max", OpenCLC30},
    {eCLBuiltinSubgroupScanAddInclusive, "sub_group_scan_inclusive_add",
     OpenCLC30},
    {eCLBuiltinSubgroupScanAddExclusive, "sub_group_scan_exclusive_add",
     OpenCLC30},
    {eCLBuiltinSubgroupScanMinInclusive, "sub_group_scan_inclusive_min",
     OpenCLC30},
    {eCLBuiltinSubgroupScanMinExclusive, "sub_group_scan_exclusive_min",
     OpenCLC30},
    {eCLBuiltinSubgroupScanMaxInclusive, "sub_group_scan_inclusive_max",
     OpenCLC30},
    {eCLBuiltinSubgroupScanMaxExclusive, "sub_group_scan_exclusive_max",
     OpenCLC30},
    /// Provided by SPV_KHR_uniform_group_instructions.
    {eCLBuiltinSubgroupReduceMul, "sub_group_reduce_mul", OpenCLC30},
    {eCLBuiltinSubgroupReduceAnd, "sub_group_reduce_and", OpenCLC30},
    {eCLBuiltinSubgroupReduceOr, "sub_group_reduce_or", OpenCLC30},
    {eCLBuiltinSubgroupReduceXor, "sub_group_reduce_xor", OpenCLC30},
    {eCLBuiltinSubgroupReduceLogicalAnd, "sub_group_reduce_logical_and",
     OpenCLC30},
    {eCLBuiltinSubgroupReduceLogicalOr, "sub_group_reduce_logical_or",
     OpenCLC30},
    {eCLBuiltinSubgroupReduceLogicalXor, "sub_group_reduce_logical_xor",
     OpenCLC30},
    {eCLBuiltinSubgroupScanMulInclusive, "sub_group_scan_inclusive_mul",
     OpenCLC30},
    {eCLBuiltinSubgroupScanMulExclusive, "sub_group_scan_exclusive_mul",
     OpenCLC30},
    {eCLBuiltinSubgroupScanAndInclusive, "sub_group_scan_inclusive_and",
     OpenCLC30},
    {eCLBuiltinSubgroupScanAndExclusive, "sub_group_scan_exclusive_and",
     OpenCLC30},
    {eCLBuiltinSubgroupScanOrInclusive, "sub_group_scan_inclusive_or",
     OpenCLC30},
    {eCLBuiltinSubgroupScanOrExclusive, "sub_group_scan_exclusive_or",
     OpenCLC30},
    {eCLBuiltinSubgroupScanXorInclusive, "sub_group_scan_inclusive_xor",
     OpenCLC30},
    {eCLBuiltinSubgroupScanXorExclusive, "sub_group_scan_exclusive_xor",
     OpenCLC30},
    {eCLBuiltinSubgroupScanLogicalAndInclusive,
     "sub_group_scan_inclusive_logical_and", OpenCLC30},
    {eCLBuiltinSubgroupScanLogicalAndExclusive,
     "sub_group_scan_exclusive_logical_and", OpenCLC30},
    {eCLBuiltinSubgroupScanLogicalOrInclusive,
     "sub_group_scan_inclusive_logical_or", OpenCLC30},
    {eCLBuiltinSubgroupScanLogicalOrExclusive,
     "sub_group_scan_exclusive_logical_or", OpenCLC30},
    {eCLBuiltinSubgroupScanLogicalXorInclusive,
     "sub_group_scan_inclusive_logical_xor", OpenCLC30},
    {eCLBuiltinSubgroupScanLogicalXorExclusive,
     "sub_group_scan_exclusive_logical_xor", OpenCLC30},

    // GLSL builtin functions
    {eCLBuiltinCodeplayFaceForward, "codeplay_face_forward"},
    {eCLBuiltinCodeplayReflect, "codeplay_reflect"},
    {eCLBuiltinCodeplayRefract, "codeplay_refract"},
    {eCLBuiltinCodeplayFindLSB, "codeplay_pack_find_lsb"},
    {eCLBuiltinCodeplayFindMSB, "codeplay_pack_find_msb"},
    {eCLBuiltinCodeplayBitReverse, "codeplay_pack_bit_reverse"},
    {eCLBuiltinCodeplayPackNormalizeChar4, "codeplay_pack_normalize_char4"},
    {eCLBuiltinCodeplayPackNormalizeUchar4, "codeplay_pack_normalize_uchar4"},
    {eCLBuiltinCodeplayPackNormalizeShort2, "codeplay_pack_normalize_short2"},
    {eCLBuiltinCodeplayPackNormalizeUshort2, "codeplay_pack_normalize_ushort2"},
    {eCLBuiltinCodeplayPackHalf2, "codeplay_pack_half2"},
    {eCLBuiltinCodeplayUnpackNormalize, "codeplay_unpack_normalize"},
    {eCLBuiltinCodeplayUnpackHalf2, "codeplay_unpack_half2"},

    {eBuiltinInvalid, nullptr},
    {eBuiltinUnknown, nullptr}};

////////////////////////////////////////////////////////////////////////////////

Function *CLBuiltinInfo::declareBuiltin(Module *M, BuiltinID ID, Type *RetTy,
                                        ArrayRef<Type *> ArgTys,
                                        ArrayRef<TypeQualifiers> ArgQuals,
                                        Twine Suffix) {
  // Determine the builtin function name.
  if (!M) {
    return nullptr;
  }
  std::string BuiltinName = getBuiltinName(ID).str();
  if (BuiltinName.empty()) {
    return nullptr;
  }

  // Add the optional suffix.
  SmallVector<char, 16> SuffixVec;
  Suffix.toVector(SuffixVec);
  if (!SuffixVec.empty()) {
    BuiltinName.append(SuffixVec.begin(), SuffixVec.end());
  }

  // Mangle the function name and look it up in the module.
  NameMangler Mangler(&M->getContext());
  const std::string MangledName =
      Mangler.mangleName(BuiltinName, ArgTys, ArgQuals);
  Function *Builtin = M->getFunction(MangledName);

  // Declare the builtin if necessary.
  if (!Builtin) {
    FunctionType *FT = FunctionType::get(RetTy, ArgTys, false);
    M->getOrInsertFunction(MangledName, FT);
    Builtin = M->getFunction(MangledName);
    Builtin->setCallingConv(CallingConv::SPIR_FUNC);
  }
  return Builtin;
}

BuiltinID CLBuiltinInfo::getPrintfBuiltin() const { return eCLBuiltinPrintf; }

Module *CLBuiltinInfo::getBuiltinsModule() {
  if (!Loader) {
    return nullptr;
  }
  return Loader->getBuiltinsModule();
}

Function *CLBuiltinInfo::materializeBuiltin(StringRef BuiltinName,
                                            Module *DestM,
                                            BuiltinMatFlags Flags) {
  // First try to find the builtin in the target module.
  if (DestM) {
    Function *Builtin = DestM->getFunction(BuiltinName);
    // If a builtin was found, it might be either a declaration or a definition.
    // If the definition flag (eBuiltinMatDefinition) is set, we can not return
    // just a declaration.
    if (Builtin &&
        (!(Flags & eBuiltinMatDefinition) || !Builtin->isDeclaration())) {
      return Builtin;
    }
  }

  if (!Loader) {
    return nullptr;
  }
  // Try to find the builtin in the builtins module
  return Loader->materializeBuiltin(BuiltinName, DestM, Flags);
}

BuiltinID CLBuiltinInfo::identifyBuiltin(const Function &F) const {
  NameMangler Mangler(nullptr);
  const StringRef Name = F.getName();
  const CLBuiltinEntry *entry = Builtins;
  const auto Version = getOpenCLVersion(*F.getParent());
  const StringRef DemangledName = Mangler.demangleName(Name);
  while (entry->ID != eBuiltinInvalid) {
    if (Version >= entry->MinVer && DemangledName == entry->OpenCLFnName) {
      return entry->ID;
    }
    entry++;
  }

  if (DemangledName == Name) {
    // The function name is not mangled and so it can not be an OpenCL builtin.
    return eBuiltinInvalid;
  }

  Lexer L(Mangler.demangleName(Name));
  if (L.Consume("vload")) {
    unsigned Width = 0;
    if (L.Consume("_half")) {
      // We have both `vload_half` and `vload_halfN` variants.
      if (!L.ConsumeInteger(Width) || isValidVecWidth(Width)) {
        // If there's nothing left to parse we're good to go.
        if (!L.Left()) {
          return eCLBuiltinVLoadHalf;
        }
      }
    } else if (L.ConsumeInteger(Width) && !L.Left() && isValidVecWidth(Width)) {
      // There are no scalar variants of this builtin.
      return eCLBuiltinVLoad;
    }
  } else if (L.Consume("vstore")) {
    unsigned Width = 0;
    if (L.Consume("_half")) {
      // We have both `vstore_half` and `vstore_halfN` variants.
      if (!L.ConsumeInteger(Width) || isValidVecWidth(Width)) {
        // Rounding modes are optional.
        L.Consume("_rte") || L.Consume("_rtz") || L.Consume("_rtp") ||
            L.Consume("_rtn");

        // If there's nothing left to parse we're good to go.
        if (!L.Left()) {
          return eCLBuiltinVStoreHalf;
        }
      }
    } else if (L.ConsumeInteger(Width) && !L.Left() && isValidVecWidth(Width)) {
      // There are no scalar variants of this builtin.
      return eCLBuiltinVStore;
    }
  } else if (L.Consume("as_")) {
    if (L.Consume("char") || L.Consume("uchar") || L.Consume("short") ||
        L.Consume("ushort") || L.Consume("int") || L.Consume("uint") ||
        L.Consume("long") || L.Consume("ulong") || L.Consume("float") ||
        L.Consume("double") || L.Consume("half")) {
      unsigned Width = 0;
      if (!L.ConsumeInteger(Width) || isValidVecWidth(Width)) {
        if (!L.Left()) {
          return eCLBuiltinAs;
        }
      }
    }
  }

  return eBuiltinUnknown;
}

llvm::StringRef CLBuiltinInfo::getBuiltinName(BuiltinID ID) const {
  const CLBuiltinEntry *entry = Builtins;
  while (entry->ID != eBuiltinInvalid) {
    if (ID == entry->ID) {
      return entry->OpenCLFnName;
    }
    entry++;
  }
  return llvm::StringRef();
}

BuiltinUniformity CLBuiltinInfo::isBuiltinUniform(const Builtin &,
                                                  const CallInst *CI,
                                                  unsigned) const {
  // Assume that builtins with side effects are varying.
  if (Function *Callee = CI->getCalledFunction()) {
    const auto Props = analyzeBuiltin(*Callee).properties;
    if (Props & eBuiltinPropertySideEffects) {
      return eBuiltinUniformityNever;
    }
  }

  return eBuiltinUniformityLikeInputs;
}

Builtin CLBuiltinInfo::analyzeBuiltin(const Function &Callee) const {
  const BuiltinID ID = identifyBuiltin(Callee);

  bool IsConvergent = false;
  unsigned Properties = eBuiltinPropertyNone;
  switch (ID) {
    default:
      // Assume convergence on unknown builtins.
      IsConvergent = true;
      break;
    case eBuiltinUnknown: {
      // Assume convergence on unknown builtins.
      IsConvergent = true;
      // If we know that this is an OpenCL builtin, but we don't have any
      // special information about it, we can determine if it has side effects
      // or not by its return type and its paramaters. This depends on being
      // able to identify all the "special" builtins, such as barriers and
      // fences.
      bool HasSideEffects = false;

      // Void functions have side effects
      if (Callee.getReturnType() == Type::getVoidTy(Callee.getContext())) {
        HasSideEffects = true;
      }
      // Functions that take pointers probably have side effects
      for (const auto &arg : Callee.args()) {
        if (arg.getType()->isPointerTy()) {
          HasSideEffects = true;
        }
      }
      Properties |= HasSideEffects ? eBuiltinPropertySideEffects
                                   : eBuiltinPropertyNoSideEffects;
    } break;
    case eCLBuiltinBarrier:
      IsConvergent = true;
      Properties |= eBuiltinPropertyExecutionFlow;
      Properties |= eBuiltinPropertySideEffects;
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
    case eCLBuiltinMemFence:
    case eCLBuiltinReadMemFence:
    case eCLBuiltinWriteMemFence:
      Properties |= eBuiltinPropertySupportsInstantiation;
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
    case eCLBuiltinPrintf:
      Properties |= eBuiltinPropertySideEffects;
      Properties |= eBuiltinPropertySupportsInstantiation;
      break;
    case eCLBuiltinAsyncWorkGroupCopy:
    case eCLBuiltinAsyncWorkGroupStridedCopy:
    case eCLBuiltinWaitGroupEvents:
    case eCLBuiltinAsyncWorkGroupCopy2D2D:
    case eCLBuiltinAsyncWorkGroupCopy3D3D:
      // Our implementation of these builtins uses thread checks against
      // specific work-item IDs, so they are convergent.
      IsConvergent = true;
      Properties |= eBuiltinPropertyNoSideEffects;
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
    case eCLBuiltinAtomicAdd:
    case eCLBuiltinAtomicSub:
    case eCLBuiltinAtomicXchg:
    case eCLBuiltinAtomicInc:
    case eCLBuiltinAtomicDec:
    case eCLBuiltinAtomicCmpxchg:
    case eCLBuiltinAtomicMin:
    case eCLBuiltinAtomicMax:
    case eCLBuiltinAtomicAnd:
    case eCLBuiltinAtomicOr:
    case eCLBuiltinAtomicXor:
      Properties |= eBuiltinPropertySideEffects;
      Properties |= eBuiltinPropertySupportsInstantiation;
      Properties |= eBuiltinPropertyAtomic;
      break;
    case eCLBuiltinGetWorkDim:
    case eCLBuiltinGetGroupId:
    case eCLBuiltinGetGlobalSize:
    case eCLBuiltinGetGlobalOffset:
    case eCLBuiltinGetNumGroups:
    case eCLBuiltinGetGlobalId:
    case eCLBuiltinGetLocalSize:
    case eCLBuiltinGetEnqueuedLocalSize:
    case eCLBuiltinGetLocalLinearId:
    case eCLBuiltinGetGlobalLinearId:
    case eCLBuiltinGetSubgroupLocalId:
      Properties |= eBuiltinPropertyWorkItem;
      Properties |= eBuiltinPropertyRematerializable;
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
    case eCLBuiltinGetLocalId:
      Properties |= eBuiltinPropertyWorkItem;
      Properties |= eBuiltinPropertyLocalID;
      Properties |= eBuiltinPropertyRematerializable;
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
    case eCLBuiltinDot:
    case eCLBuiltinCross:
    case eCLBuiltinFastDistance:
    case eCLBuiltinFastLength:
    case eCLBuiltinFastNormalize:
      Properties |= eBuiltinPropertyReduction;
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinDistance:
    case eCLBuiltinLength:
    case eCLBuiltinNormalize:
      Properties |= eBuiltinPropertyReduction;
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      // XXX The inline implementation seems to have precision issues. The dot
      // product can overflow to +inf which results in the wrong result.
      // See redmine #6427 and #9115
      // Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinIsEqual:
    case eCLBuiltinIsNotEqual:
    case eCLBuiltinIsGreater:
    case eCLBuiltinIsGreaterEqual:
    case eCLBuiltinIsLess:
    case eCLBuiltinIsLessEqual:
    case eCLBuiltinIsLessGreater:
    case eCLBuiltinIsOrdered:
    case eCLBuiltinIsUnordered:
    case eCLBuiltinIsFinite:
    case eCLBuiltinIsInf:
    case eCLBuiltinIsNan:
    case eCLBuiltinIsNormal:
    case eCLBuiltinSignBit:
      // Scalar variants return '0' or '1', vector variants '0' or '111...1'.
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      Properties |= eBuiltinPropertySupportsInstantiation;
      break;
    case eCLBuiltinAny:
    case eCLBuiltinAll:
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinFract:
    case eCLBuiltinModF:
    case eCLBuiltinSinCos:
      Properties |= eBuiltinPropertyPointerReturnEqualRetTy;
      break;
    case eCLBuiltinFrexp:
    case eCLBuiltinLGammaR:
    case eCLBuiltinRemquo:
      Properties |= eBuiltinPropertyPointerReturnEqualIntRetTy;
      break;
    case eCLBuiltinShuffle:
    case eCLBuiltinShuffle2:
      // While there are vector equivalents for these builtins, they require a
      // modified mask, so we cannot use them by simply packetizing their
      // arguments.
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinFMax:
    case eCLBuiltinFMin:
    case eCLBuiltinAddSat:
    case eCLBuiltinSubSat:
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinCodeplayFaceForward:
    case eCLBuiltinCodeplayReflect:
    case eCLBuiltinCodeplayRefract:
      Properties |= eBuiltinPropertyReduction;
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      break;
    case eCLBuiltinConvertChar:
    case eCLBuiltinConvertShort:
    case eCLBuiltinConvertInt:
    case eCLBuiltinConvertLong:
    case eCLBuiltinConvertUChar:
    case eCLBuiltinConvertUShort:
    case eCLBuiltinConvertUInt:
    case eCLBuiltinConvertULong:
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinVLoad:
    case eCLBuiltinVLoadHalf:
      Properties |= eBuiltinPropertyNoSideEffects;
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinVStore:
    case eCLBuiltinVStoreHalf:
      Properties |= eBuiltinPropertySideEffects;
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinSelect:
    case eCLBuiltinAs:
      // Some of these builtins do have vector equivalents, but since we can
      // emit all variants inline, we mark them as having none for simplicity.
      Properties |= eBuiltinPropertyNoVectorEquivalent;
      Properties |= eBuiltinPropertyCanEmitInline;
      break;
    case eCLBuiltinWorkGroupBarrier:
    case eCLBuiltinSubGroupBarrier:
      IsConvergent = true;
      LLVM_FALLTHROUGH;
    case eCLBuiltinAtomicWorkItemFence:
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
    case eCLBuiltinGetSubgroupSize:
    case eCLBuiltinGetMaxSubgroupSize:
    case eCLBuiltinGetNumSubgroups:
    case eCLBuiltinGetEnqueuedNumSubgroups:
    case eCLBuiltinGetSubgroupId:
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
      // Subgroup collectives
    case eCLBuiltinSubgroupAll:
    case eCLBuiltinSubgroupAny:
    case eCLBuiltinSubgroupBroadcast:
    case eCLBuiltinSubgroupReduceAdd:
    case eCLBuiltinSubgroupReduceMin:
    case eCLBuiltinSubgroupReduceMax:
    case eCLBuiltinSubgroupScanAddInclusive:
    case eCLBuiltinSubgroupScanAddExclusive:
    case eCLBuiltinSubgroupScanMinInclusive:
    case eCLBuiltinSubgroupScanMinExclusive:
    case eCLBuiltinSubgroupScanMaxInclusive:
    case eCLBuiltinSubgroupScanMaxExclusive:
    case eCLBuiltinSubgroupReduceMul:
    case eCLBuiltinSubgroupReduceAnd:
    case eCLBuiltinSubgroupReduceOr:
    case eCLBuiltinSubgroupReduceXor:
    case eCLBuiltinSubgroupReduceLogicalAnd:
    case eCLBuiltinSubgroupReduceLogicalOr:
    case eCLBuiltinSubgroupReduceLogicalXor:
    case eCLBuiltinSubgroupScanMulInclusive:
    case eCLBuiltinSubgroupScanMulExclusive:
    case eCLBuiltinSubgroupScanAndInclusive:
    case eCLBuiltinSubgroupScanAndExclusive:
    case eCLBuiltinSubgroupScanOrInclusive:
    case eCLBuiltinSubgroupScanOrExclusive:
    case eCLBuiltinSubgroupScanXorInclusive:
    case eCLBuiltinSubgroupScanXorExclusive:
    case eCLBuiltinSubgroupScanLogicalAndInclusive:
    case eCLBuiltinSubgroupScanLogicalAndExclusive:
    case eCLBuiltinSubgroupScanLogicalOrInclusive:
    case eCLBuiltinSubgroupScanLogicalOrExclusive:
    case eCLBuiltinSubgroupScanLogicalXorInclusive:
    case eCLBuiltinSubgroupScanLogicalXorExclusive:
      // Work-group collectives
    case eCLBuiltinWorkgroupAll:
    case eCLBuiltinWorkgroupAny:
    case eCLBuiltinWorkgroupBroadcast:
    case eCLBuiltinWorkgroupReduceAdd:
    case eCLBuiltinWorkgroupReduceMin:
    case eCLBuiltinWorkgroupReduceMax:
    case eCLBuiltinWorkgroupScanAddInclusive:
    case eCLBuiltinWorkgroupScanAddExclusive:
    case eCLBuiltinWorkgroupScanMinInclusive:
    case eCLBuiltinWorkgroupScanMinExclusive:
    case eCLBuiltinWorkgroupScanMaxInclusive:
    case eCLBuiltinWorkgroupScanMaxExclusive:
    case eCLBuiltinWorkgroupReduceMul:
    case eCLBuiltinWorkgroupReduceAnd:
    case eCLBuiltinWorkgroupReduceOr:
    case eCLBuiltinWorkgroupReduceXor:
    case eCLBuiltinWorkgroupReduceLogicalAnd:
    case eCLBuiltinWorkgroupReduceLogicalOr:
    case eCLBuiltinWorkgroupReduceLogicalXor:
    case eCLBuiltinWorkgroupScanMulInclusive:
    case eCLBuiltinWorkgroupScanMulExclusive:
    case eCLBuiltinWorkgroupScanAndInclusive:
    case eCLBuiltinWorkgroupScanAndExclusive:
    case eCLBuiltinWorkgroupScanOrInclusive:
    case eCLBuiltinWorkgroupScanOrExclusive:
    case eCLBuiltinWorkgroupScanXorInclusive:
    case eCLBuiltinWorkgroupScanXorExclusive:
    case eCLBuiltinWorkgroupScanLogicalAndInclusive:
    case eCLBuiltinWorkgroupScanLogicalAndExclusive:
    case eCLBuiltinWorkgroupScanLogicalOrInclusive:
    case eCLBuiltinWorkgroupScanLogicalOrExclusive:
    case eCLBuiltinWorkgroupScanLogicalXorInclusive:
    case eCLBuiltinWorkgroupScanLogicalXorExclusive:
      IsConvergent = true;
      Properties |= eBuiltinPropertyLowerToMuxBuiltin;
      break;
  }

  if (!IsConvergent) {
    Properties |= eBuiltinPropertyKnownNonConvergent;
  }

  return Builtin{Callee, ID, (BuiltinProperties)Properties};
}

Function *CLBuiltinInfo::getVectorEquivalent(const Builtin &B, unsigned Width,
                                             Module *M) {
  // Analyze the builtin. Some functions have no vector equivalent.
  const auto Props = B.properties;
  if (Props & eBuiltinPropertyNoVectorEquivalent) {
    return nullptr;
  }

  // Builtin functions have mangled names. If it's not mangled, there will be
  // no vector equivalent.
  NameMangler Mangler(&B.function.getContext());
  SmallVector<Type *, 4> BuiltinArgTypes, BuiltinPointeeTypes;
  SmallVector<TypeQualifiers, 4> BuiltinArgQuals;
  const StringRef BuiltinName =
      Mangler.demangleName(B.function.getName(), BuiltinArgTypes,
                           BuiltinPointeeTypes, BuiltinArgQuals);
  if (BuiltinName.empty()) {
    return nullptr;
  }

  // Determine the mangled name of the vector equivalent.
  // This means creating a list of qualified types for the arguments.
  SmallVector<Type *, 4> VectorTypes;
  SmallVector<TypeQualifiers, 4> VectorQuals;
  for (unsigned i = 0; i < BuiltinArgTypes.size(); i++) {
    Type *OldTy = BuiltinArgTypes[i];
    const TypeQualifiers OldQuals = BuiltinArgQuals[i];
    if (isa<FixedVectorType>(OldTy)) {
      return nullptr;
    }
    PointerType *OldPtrTy = dyn_cast<PointerType>(OldTy);
    if (OldPtrTy) {
      if (auto *const PtrRetPointeeTy =
              getPointerReturnPointeeTy(B.function, Props)) {
        [[maybe_unused]] auto *OldPointeeTy = BuiltinPointeeTypes[i];
        assert(OldPointeeTy && OldPointeeTy == PtrRetPointeeTy &&
               "Demangling inconsistency");
        if (!FixedVectorType::isValidElementType(PtrRetPointeeTy)) {
          return nullptr;
        }
        Type *NewEleTy = FixedVectorType::get(PtrRetPointeeTy, Width);
        Type *NewType = PointerType::get(NewEleTy, OldPtrTy->getAddressSpace());
        TypeQualifiers NewQuals;
        TypeQualifiers EleQuals = OldQuals;
        NewQuals.push_back(EleQuals.pop_front());  // Pointer qualifier
        NewQuals.push_back(eTypeQualNone);         // Vector qualifier
        NewQuals.push_back(EleQuals);

        VectorTypes.push_back(NewType);
        VectorQuals.push_back(NewQuals);

        continue;
      }
    }

    if (!FixedVectorType::isValidElementType(OldTy)) {
      return nullptr;
    }
    TypeQualifiers NewQuals;
    Type *NewType = FixedVectorType::get(OldTy, Width);
    NewQuals.push_back(eTypeQualNone);  // Vector qualifier
    NewQuals.push_back(OldQuals);       // Element qualifier

    VectorTypes.push_back(NewType);
    VectorQuals.push_back(NewQuals);
  }

  // Handle special builtin naming equivalents.
  std::string EquivNameBase = BuiltinName.str();
  StringRef FirstChunk;
  Lexer L(BuiltinName);
  if (L.ConsumeUntil('_', FirstChunk)) {
    const bool AsBuiltin = FirstChunk == "as";
    const bool ConvertBuiltin = FirstChunk == "convert";
    if (!L.Consume("_")) {
      return nullptr;
    }
    StringRef SecondChunkNoWidth;
    if (!L.ConsumeAlpha(SecondChunkNoWidth)) {
      return nullptr;
    }
    if (AsBuiltin || ConvertBuiltin) {
      // as_* and convert_* builtins have vector equivalents, with a vector
      // width suffix. Add the width suffix to the scalar builtin name.
      if (AsBuiltin && L.Left()) {
        return nullptr;
      }
      const Twine WidthText(Width);
      EquivNameBase.insert(L.CurrentPos(), WidthText.str());
    }
  }

  const std::string EquivName =
      Mangler.mangleName(EquivNameBase, VectorTypes, VectorQuals);

  // Lookup the vector equivalent and make sure the return type agrees.
  Function *VectorBuiltin = materializeBuiltin(EquivName, M);
  if (VectorBuiltin) {
    Type *RetTy = B.function.getReturnType();
    auto *VecRetTy = dyn_cast<FixedVectorType>(VectorBuiltin->getReturnType());
    if (!VecRetTy || (VecRetTy->getElementType() != RetTy) ||
        (VecRetTy->getNumElements() != Width)) {
      VectorBuiltin = nullptr;
    }
  }
  return VectorBuiltin;
}

Function *CLBuiltinInfo::getScalarEquivalent(const Builtin &B, Module *M) {
  // Analyze the builtin. Some functions have no scalar equivalent.
  const auto Props = B.properties;
  if (Props & eBuiltinPropertyNoVectorEquivalent) {
    return nullptr;
  }

  // Check the return type.
  auto *VecRetTy = dyn_cast<FixedVectorType>(B.function.getReturnType());
  if (!VecRetTy) {
    return nullptr;
  }

  // Builtin functions have mangled names. If it's not mangled, there will be
  // no scalar equivalent.
  NameMangler Mangler(&B.function.getContext());
  SmallVector<Type *, 4> BuiltinArgTypes, BuiltinPointeeTypes;
  SmallVector<TypeQualifiers, 4> BuiltinArgQuals;
  const StringRef BuiltinName =
      Mangler.demangleName(B.function.getName(), BuiltinArgTypes,
                           BuiltinPointeeTypes, BuiltinArgQuals);
  if (BuiltinName.empty()) {
    return nullptr;
  }

  // Determine the mangled name of the scalar equivalent.
  // This means creating a list of qualified types for the arguments.
  const unsigned Width = VecRetTy->getNumElements();
  SmallVector<Type *, 4> ScalarTypes;
  SmallVector<TypeQualifiers, 4> ScalarQuals;
  for (unsigned i = 0; i < BuiltinArgTypes.size(); i++) {
    Type *OldTy = BuiltinArgTypes[i];
    const TypeQualifiers OldQuals = BuiltinArgQuals[i];
    if (auto *OldVecTy = dyn_cast<FixedVectorType>(OldTy)) {
      if (OldVecTy->getNumElements() != Width) {
        return nullptr;
      }
      Type *NewTy = OldVecTy->getElementType();
      TypeQualifiers NewQuals = OldQuals;
      NewQuals.pop_front();

      ScalarTypes.push_back(NewTy);
      ScalarQuals.push_back(NewQuals);
    } else if (PointerType *OldPtrTy = dyn_cast<PointerType>(OldTy)) {
      Type *const PtrRetPointeeTy =
          getPointerReturnPointeeTy(B.function, Props);
      if (PtrRetPointeeTy && PtrRetPointeeTy->isVectorTy()) {
        [[maybe_unused]] auto *OldPointeeTy = BuiltinPointeeTypes[i];
        assert(OldPointeeTy && OldPointeeTy == PtrRetPointeeTy &&
               "Demangling inconsistency");
        auto *OldVecTy = cast<FixedVectorType>(PtrRetPointeeTy);
        Type *NewTy = PointerType::get(OldVecTy->getElementType(),
                                       OldPtrTy->getAddressSpace());
        TypeQualifiers NewQuals = OldQuals;
        const TypeQualifier PtrQual = NewQuals.pop_front();
        const TypeQualifier VecQual = NewQuals.pop_front();
        (void)VecQual;
        const TypeQualifier EleQual = NewQuals.pop_front();
        NewQuals.push_back(PtrQual);
        NewQuals.push_back(EleQual);
        ScalarTypes.push_back(NewTy);
        ScalarQuals.push_back(NewQuals);
      } else {
        ScalarTypes.push_back(OldTy);
        ScalarQuals.push_back(OldQuals);
      }
    } else {
      if (!OldTy) {
        return nullptr;
      }
      ScalarTypes.push_back(OldTy);
      ScalarQuals.push_back(OldQuals);
    }
  }

  // Handle special builtin naming equivalents.
  std::string EquivNameBase = BuiltinName.str();
  StringRef FirstChunk;
  Lexer L(BuiltinName);
  if (L.ConsumeUntil('_', FirstChunk)) {
    const bool AsBuiltin = FirstChunk == "as";
    const bool ConvertBuiltin = FirstChunk == "convert";
    if (!L.Consume("_")) {
      return nullptr;
    }
    StringRef SecondChunkNoWidth;
    if (!L.ConsumeAlpha(SecondChunkNoWidth)) {
      return nullptr;
    }
    if (AsBuiltin || ConvertBuiltin) {
      // as_* and convert_* builtins have scalar equivalents, with no width
      // suffix. Remove the width suffix from the vector builtin name.
      const unsigned WidthStart = L.CurrentPos();
      unsigned Width = 0;
      if (!L.ConsumeInteger(Width)) {
        return nullptr;
      }
      const unsigned WidthEnd = L.CurrentPos();
      EquivNameBase.erase(WidthStart, WidthEnd - WidthStart);
    }
  }

  const std::string EquivName =
      Mangler.mangleName(EquivNameBase, ScalarTypes, ScalarQuals);

  // Lookup the scalar equivalent and make sure the return type agrees.
  Function *ScalarBuiltin = materializeBuiltin(EquivName, M);
  if (!ScalarBuiltin) {
    return nullptr;
  }
  Type *RetTy = ScalarBuiltin->getReturnType();
  if (VecRetTy->getElementType() != RetTy) {
    return nullptr;
  }
  return ScalarBuiltin;
}

/// @brief Returns whether the parameter corresponding to given index to the
/// (assumed builtin) Function is known to possess the given qualifier.
/// @return true if the parameter is known to have the qualifier, false if not,
/// and None on error.
static std::optional<bool> paramHasTypeQual(const Function &F,
                                            unsigned ParamIdx,
                                            TypeQualifier Q) {
  // Demangle the function name to get the type qualifiers.
  SmallVector<Type *, 2> Types;
  SmallVector<TypeQualifiers, 2> Quals;
  NameMangler Mangler(&F.getContext());
  if (Mangler.demangleName(F.getName(), Types, Quals).empty()) {
    return std::nullopt;
  }

  if (ParamIdx >= Quals.size()) {
    return std::nullopt;
  }

  auto &Qual = Quals[ParamIdx];
  while (Qual.getCount()) {
    if (Qual.pop_front() == Q) {
      return true;
    }
  }
  return false;
}

Value *CLBuiltinInfo::emitBuiltinInline(Function *F, IRBuilder<> &B,
                                        ArrayRef<Value *> Args) {
  if (!F) {
    return nullptr;
  }

  // Handle 'common' builtins.
  const BuiltinID BuiltinID = identifyBuiltin(*F);
  if (BuiltinID != eBuiltinInvalid && BuiltinID != eBuiltinUnknown) {
    // Note we have to handle these specially since we need to deduce whether
    // the source operand is signed or not. It is not possible to do this based
    // solely on the BuiltinID.
    switch (BuiltinID) {
        // 6.2 Explicit Conversions
      case eCLBuiltinConvertChar:
      case eCLBuiltinConvertShort:
      case eCLBuiltinConvertInt:
      case eCLBuiltinConvertLong:
      case eCLBuiltinConvertUChar:
      case eCLBuiltinConvertUShort:
      case eCLBuiltinConvertUInt:
      case eCLBuiltinConvertULong:
        return emitBuiltinInlineConvert(F, BuiltinID, B, Args);
        // 6.12.3 Integer Functions
      case eCLBuiltinAddSat:
      case eCLBuiltinSubSat: {
        std::optional<bool> IsParamSignedOrNone =
            paramHasTypeQual(*F, 0, eTypeQualSignedInt);
        if (!IsParamSignedOrNone.has_value()) {
          return nullptr;
        }
        const bool IsSigned = *IsParamSignedOrNone;
        const Intrinsic::ID IntrinsicOpc = [=] {
          if (BuiltinID == eCLBuiltinSubSat) {
            return IsSigned ? Intrinsic::ssub_sat : Intrinsic::usub_sat;
          } else {
            return IsSigned ? Intrinsic::sadd_sat : Intrinsic::uadd_sat;
          }
        }();
        return emitBuiltinInlineAsLLVMBinaryIntrinsic(B, Args[0], Args[1],
                                                      IntrinsicOpc);
      }
      case eCLBuiltinVLoad: {
        NameMangler Mangler(&F->getContext());
        Lexer L(Mangler.demangleName(F->getName()));
        if (L.Consume("vload")) {
          unsigned Width = 0;
          if (L.ConsumeInteger(Width)) {
            return emitBuiltinInlineVLoad(F, Width, B, Args);
          }
        }
      } break;
      case eCLBuiltinVLoadHalf: {
        NameMangler Mangler(&F->getContext());
        const auto name = Mangler.demangleName(F->getName());
        if (name == "vload_half") {
          // TODO CA-4691 handle "vload_halfn"
          return emitBuiltinInlineVLoadHalf(F, B, Args);
        }
      } break;
      case eCLBuiltinVStore: {
        NameMangler Mangler(&F->getContext());
        Lexer L(Mangler.demangleName(F->getName()));
        if (L.Consume("vstore")) {
          unsigned Width = 0;
          if (L.ConsumeInteger(Width)) {
            return emitBuiltinInlineVStore(F, Width, B, Args);
          }
        }
      } break;
      case eCLBuiltinVStoreHalf: {
        NameMangler Mangler(&F->getContext());
        Lexer L(Mangler.demangleName(F->getName()));
        if (L.Consume("vstore_half")) {
          // TODO CA-4691 handle "vstore_halfn"
          return emitBuiltinInlineVStoreHalf(F, L.TextLeft(), B, Args);
        }
      } break;
      case eCLBuiltinSelect:
        return emitBuiltinInlineSelect(F, B, Args);
      case eCLBuiltinAs:
        return emitBuiltinInlineAs(F, B, Args);
      default:
        break;
    }
    return emitBuiltinInline(BuiltinID, B, Args);
  }

  return nullptr;
}

Value *CLBuiltinInfo::emitBuiltinInline(BuiltinID BuiltinID, IRBuilder<> &B,
                                        ArrayRef<Value *> Args) {
  switch (BuiltinID) {
    default:
      return nullptr;

    case eCLBuiltinDot:
    case eCLBuiltinCross:
    case eCLBuiltinLength:
    case eCLBuiltinDistance:
    case eCLBuiltinNormalize:
    case eCLBuiltinFastLength:
    case eCLBuiltinFastDistance:
    case eCLBuiltinFastNormalize:
      return emitBuiltinInlineGeometrics(BuiltinID, B, Args);
    // 6.12.2 Math Functions
    case eCLBuiltinFMax:
      return emitBuiltinInlineAsLLVMBinaryIntrinsic(B, Args[0], Args[1],
                                                    llvm::Intrinsic::maxnum);
    case eCLBuiltinFMin:
      return emitBuiltinInlineAsLLVMBinaryIntrinsic(B, Args[0], Args[1],
                                                    llvm::Intrinsic::minnum);
    // 6.12.6 Relational Functions
    case eCLBuiltinAll:
      return emitBuiltinInlineAll(B, Args);
    case eCLBuiltinAny:
      return emitBuiltinInlineAny(B, Args);
    case eCLBuiltinIsEqual:
    case eCLBuiltinIsNotEqual:
    case eCLBuiltinIsGreater:
    case eCLBuiltinIsGreaterEqual:
    case eCLBuiltinIsLess:
    case eCLBuiltinIsLessEqual:
    case eCLBuiltinIsLessGreater:
    case eCLBuiltinIsOrdered:
    case eCLBuiltinIsUnordered:
      return emitBuiltinInlineRelationalsWithTwoArguments(BuiltinID, B, Args);
    case eCLBuiltinIsFinite:
    case eCLBuiltinIsInf:
    case eCLBuiltinIsNan:
    case eCLBuiltinIsNormal:
    case eCLBuiltinSignBit:
      assert(Args.size() == 1 && "Invalid number of arguments");
      return emitBuiltinInlineRelationalsWithOneArgument(BuiltinID, B, Args[0]);
    // 6.12.12 Miscellaneous Vector Functions
    case eCLBuiltinShuffle:
    case eCLBuiltinShuffle2:
      return emitBuiltinInlineShuffle(BuiltinID, B, Args);

    case eCLBuiltinPrintf:
      return emitBuiltinInlinePrintf(BuiltinID, B, Args);
  }
}

Value *CLBuiltinInfo::emitBuiltinInlineGeometrics(BuiltinID BuiltinID,
                                                  IRBuilder<> &B,
                                                  ArrayRef<Value *> Args) {
  Value *Src = nullptr;
  switch (BuiltinID) {
    default:
      return nullptr;
    case eCLBuiltinDot:
      return emitBuiltinInlineDot(B, Args);
    case eCLBuiltinCross:
      return emitBuiltinInlineCross(B, Args);
    case eCLBuiltinLength:
    case eCLBuiltinFastLength:
      return emitBuiltinInlineLength(B, Args);
    case eCLBuiltinDistance:
    case eCLBuiltinFastDistance:
      if (Args.size() != 2) {
        return nullptr;
      }
      Src = B.CreateFSub(Args[0], Args[1], "distance");
      return emitBuiltinInlineLength(B, ArrayRef<Value *>(&Src, 1));
    case eCLBuiltinNormalize:
    case eCLBuiltinFastNormalize:
      return emitBuiltinInlineNormalize(B, Args);
  }
}

Value *CLBuiltinInfo::emitBuiltinInlineDot(IRBuilder<> &B,
                                           ArrayRef<Value *> Args) {
  if (Args.size() != 2) {
    return nullptr;
  }
  Value *Src0 = Args[0];
  Value *Src1 = Args[1];
  auto *SrcVecTy = dyn_cast<FixedVectorType>(Src0->getType());
  if (SrcVecTy) {
    Value *LHS0 = B.CreateExtractElement(Src0, B.getInt32(0), "lhs");
    Value *RHS0 = B.CreateExtractElement(Src1, B.getInt32(0), "rhs");
    Value *Sum = B.CreateFMul(LHS0, RHS0, "dot");
    for (unsigned i = 1; i < SrcVecTy->getNumElements(); i++) {
      Value *LHS = B.CreateExtractElement(Src0, B.getInt32(i), "lhs");
      Value *RHS = B.CreateExtractElement(Src1, B.getInt32(i), "rhs");
      Sum = B.CreateFAdd(Sum, B.CreateFMul(LHS, RHS, "dot"), "dot");
    }
    return Sum;
  } else {
    return B.CreateFMul(Src0, Src1, "dot");
  }
}

Value *CLBuiltinInfo::emitBuiltinInlineCross(IRBuilder<> &B,
                                             ArrayRef<Value *> Args) {
  if (Args.size() != 2) {
    return nullptr;
  }
  Value *Src0 = Args[0];
  Value *Src1 = Args[1];
  auto *RetTy = dyn_cast<FixedVectorType>(Src0->getType());
  if (!RetTy) {
    return nullptr;
  }
  const int SrcIndices[] = {1, 2, 2, 0, 0, 1};
  SmallVector<Value *, 4> Src0Lanes;
  SmallVector<Value *, 4> Src1Lanes;
  for (unsigned i = 0; i < 3; i++) {
    Src0Lanes.push_back(B.CreateExtractElement(Src0, B.getInt32(i)));
    Src1Lanes.push_back(B.CreateExtractElement(Src1, B.getInt32(i)));
  }

  Value *Result = UndefValue::get(RetTy);
  for (unsigned i = 0; i < 3; i++) {
    const int Idx0 = SrcIndices[(i * 2) + 0];
    const int Idx1 = SrcIndices[(i * 2) + 1];
    Value *Src0A = Src0Lanes[Idx0];
    Value *Src1A = Src1Lanes[Idx1];
    Value *TempA = B.CreateFMul(Src0A, Src1A);
    Value *Src0B = Src0Lanes[Idx1];
    Value *Src1B = Src1Lanes[Idx0];
    Value *TempB = B.CreateFMul(Src0B, Src1B);
    Value *Lane = B.CreateFSub(TempA, TempB);
    Result = B.CreateInsertElement(Result, Lane, B.getInt32(i));
  }
  if (RetTy->getNumElements() == 4) {
    Type *EleTy = RetTy->getElementType();
    Result = B.CreateInsertElement(Result, Constant::getNullValue(EleTy),
                                   B.getInt32(3));
  }
  return Result;
}

Value *CLBuiltinInfo::emitBuiltinInlineLength(IRBuilder<> &B,
                                              ArrayRef<Value *> Args) {
  if (Args.size() != 1) {
    return nullptr;
  }
  Value *Src0 = Args[0];
  Value *Src1 = Src0;

  NameMangler Mangler(&B.getContext());
  Type *SrcType = Src0->getType();
  auto *SrcVecType = dyn_cast<FixedVectorType>(SrcType);
  if (SrcVecType) {
    SrcType = SrcVecType->getElementType();
  }

  TypeQualifiers SrcQuals;
  SmallVector<Type *, 4> Tys;
  SmallVector<TypeQualifiers, 4> Quals;
  SrcQuals.push_back(eTypeQualNone);

  // Materialize 'sqrt', 'fabs' and 'isinf'.
  Tys.push_back(SrcType);
  Quals.push_back(SrcQuals);
  BasicBlock *BB = B.GetInsertBlock();
  if (!BB) {
    return nullptr;
  }
  Function *F = BB->getParent();
  if (!F) {
    return nullptr;
  }
  Module *M = F->getParent();
  if (!M) {
    return nullptr;
  }

  const std::string FabsName = Mangler.mangleName("fabs", Tys, Quals);
  Function *Fabs = materializeBuiltin(FabsName, M);
  if (!Fabs) {
    return nullptr;
  }
  if (!SrcVecType) {
    // The "length" of a scalar is just the absolute value.
    return CreateBuiltinCall(B, Fabs, Src0, "scalar_length");
  }

  const std::string SqrtName = Mangler.mangleName("sqrt", Tys, Quals);
  Function *Sqrt = materializeBuiltin(SqrtName, M);
  if (!Sqrt) {
    return nullptr;
  }

  const std::string IsInfName = Mangler.mangleName("isinf", Tys, Quals);
  Function *IsInf = materializeBuiltin(IsInfName, M);
  if (!IsInf) {
    return nullptr;
  }
  Tys.clear();
  Quals.clear();

  // Materialize 'fmax'.
  Tys.push_back(SrcType);
  Quals.push_back(SrcQuals);
  Tys.push_back(SrcType);
  Quals.push_back(SrcQuals);
  const std::string FmaxName = Mangler.mangleName("fmax", Tys, Quals);
  Function *Fmax = materializeBuiltin(FmaxName, M);
  if (!Fmax) {
    return nullptr;
  }

  // Emit length or distance inline.
  SmallVector<Value *, 4> Ops;
  Ops.push_back(Src0);
  Ops.push_back(Src1);
  Value *Result = emitBuiltinInline(eCLBuiltinDot, B, Ops);
  Result = CreateBuiltinCall(B, Sqrt, Result, "result");

  // Handle the case where the result is infinite.
  Value *AltResult = ConstantFP::get(SrcType, 0.0);
  if (SrcVecType) {
    for (unsigned i = 0; i < SrcVecType->getNumElements(); i++) {
      Value *SrcLane = B.CreateExtractElement(Src0, B.getInt32(i), "src_lane");
      SrcLane = CreateBuiltinCall(B, Fabs, SrcLane, "src_lane");
      AltResult =
          CreateBuiltinCall(B, Fmax, {SrcLane, AltResult}, "alt_result");
    }
  } else {
    Value *SrcLane = CreateBuiltinCall(B, Fabs, Src0, "src_lane");
    AltResult = CreateBuiltinCall(B, Fmax, {SrcLane, AltResult}, "alt_result");
  }
  Value *Cond = CreateBuiltinCall(B, IsInf, Result, "cond");
  Cond = B.CreateICmpEQ(Cond, B.getInt32(0), "cmp");
  Result = B.CreateSelect(Cond, Result, AltResult, "final_result");
  return Result;
}

Value *CLBuiltinInfo::emitBuiltinInlineNormalize(IRBuilder<> &B,
                                                 ArrayRef<Value *> Args) {
  if (Args.size() != 1) {
    return nullptr;
  }

  Value *Src0 = Args[0];

  NameMangler Mangler(&B.getContext());
  Type *SrcType = Src0->getType();
  auto *SrcVecType = dyn_cast<FixedVectorType>(SrcType);
  if (SrcVecType) {
    SrcType = SrcVecType->getElementType();
  }

  TypeQualifiers SrcQuals;
  SmallVector<Type *, 4> Tys;
  SmallVector<TypeQualifiers, 4> Quals;
  SrcQuals.push_back(eTypeQualNone);

  // Materialize 'rsqrt'.
  Tys.push_back(SrcType);
  Quals.push_back(SrcQuals);
  BasicBlock *BB = B.GetInsertBlock();
  if (!BB) {
    return nullptr;
  }
  Function *F = BB->getParent();
  if (!F) {
    return nullptr;
  }
  Module *M = F->getParent();
  if (!M) {
    return nullptr;
  }

  if (!SrcVecType) {
    // A normalized scalar is either 1.0 or -1.0, unless the input was NaN, or
    // in other words, just the sign.
    const std::string SignName = Mangler.mangleName("sign", Tys, Quals);
    Function *Sign = materializeBuiltin(SignName, M);
    if (!Sign) {
      return nullptr;
    }
    return CreateBuiltinCall(B, Sign, Src0, "scalar_normalize");
  }

  const std::string RSqrtName = Mangler.mangleName("rsqrt", Tys, Quals);
  Function *RSqrt = materializeBuiltin(RSqrtName, M);
  if (!RSqrt) {
    return nullptr;
  }

  // Call 'dot' on the input.
  SmallVector<Value *, 4> DotArgs;
  DotArgs.push_back(Src0);
  DotArgs.push_back(Src0);
  Value *Result = emitBuiltinInlineDot(B, DotArgs);
  Result = CreateBuiltinCall(B, RSqrt, Result, "normalize");
  if (SrcVecType) {
    Result = B.CreateVectorSplat(SrcVecType->getNumElements(), Result);
  }
  Result = B.CreateFMul(Result, Src0, "normalized");
  return Result;
}

static Value *emitAllAnyReduction(IRBuilder<> &B, ArrayRef<Value *> Args,
                                  Instruction::BinaryOps ReduceOp) {
  if (Args.size() != 1) {
    return nullptr;
  }
  Value *Arg0 = Args[0];
  IntegerType *EleTy = dyn_cast<IntegerType>(Arg0->getType()->getScalarType());
  if (!EleTy) {
    return nullptr;
  }

  // Reduce the MSB of all vector lanes.
  Value *ReducedVal = nullptr;
  auto *VecTy = dyn_cast<FixedVectorType>(Arg0->getType());
  if (VecTy) {
    ReducedVal = B.CreateExtractElement(Arg0, B.getInt32(0));
    for (unsigned i = 1; i < VecTy->getNumElements(); i++) {
      Value *Lane = B.CreateExtractElement(Arg0, B.getInt32(i));
      ReducedVal = B.CreateBinOp(ReduceOp, ReducedVal, Lane);
    }
  } else {
    ReducedVal = Arg0;
  }

  // Shift the MSB to return either 0 or 1.
  const unsigned ShiftAmount = EleTy->getPrimitiveSizeInBits() - 1;
  Value *ShiftAmountVal = ConstantInt::get(EleTy, ShiftAmount);
  Value *Result = B.CreateLShr(ReducedVal, ShiftAmountVal);
  return B.CreateZExtOrTrunc(Result, B.getInt32Ty());
}

Value *CLBuiltinInfo::emitBuiltinInlineAll(IRBuilder<> &B,
                                           ArrayRef<Value *> Args) {
  return emitAllAnyReduction(B, Args, Instruction::And);
}

Value *CLBuiltinInfo::emitBuiltinInlineAny(IRBuilder<> &B,
                                           ArrayRef<Value *> Args) {
  return emitAllAnyReduction(B, Args, Instruction::Or);
}

Value *CLBuiltinInfo::emitBuiltinInlineSelect(Function *F, IRBuilder<> &B,
                                              ArrayRef<Value *> Args) {
  if (F->arg_size() != 3) {
    return nullptr;
  }
  Value *FalseVal = Args[0];
  Value *TrueVal = Args[1];
  Value *Cond = Args[2];
  Type *RetTy = F->getReturnType();
  auto *VecRetTy = dyn_cast<FixedVectorType>(RetTy);
  Type *CondEleTy = Cond->getType()->getScalarType();
  const unsigned CondEleBits = CondEleTy->getPrimitiveSizeInBits();
  if (VecRetTy) {
    const unsigned SimdWidth = VecRetTy->getNumElements();
    Constant *ShiftAmount = ConstantInt::get(CondEleTy, CondEleBits - 1);
    Constant *VecShiftAmount = ConstantVector::getSplat(
        ElementCount::getFixed(SimdWidth), ShiftAmount);
    Value *Mask = B.CreateAShr(Cond, VecShiftAmount);
    Value *TrueValRaw = TrueVal;
    Value *FalseValRaw = FalseVal;
    if (VecRetTy->getElementType()->isFloatingPointTy()) {
      auto *RawType = FixedVectorType::getInteger(VecRetTy);
      TrueValRaw = B.CreateBitCast(TrueVal, RawType);
      FalseValRaw = B.CreateBitCast(FalseVal, RawType);
    }
    Value *Result = B.CreateXor(TrueValRaw, FalseValRaw);
    Result = B.CreateAnd(Result, Mask);
    Result = B.CreateXor(Result, FalseValRaw);
    if (Result->getType() != VecRetTy) {
      Result = B.CreateBitCast(Result, VecRetTy);
    }
    return Result;
  } else {
    Value *Cmp = B.CreateICmpNE(Cond, Constant::getNullValue(CondEleTy));
    return B.CreateSelect(Cmp, TrueVal, FalseVal);
  }
}

/// @brief Emit the body of a builtin function as a call to a binary LLVM
/// intrinsic. If one argument is a scalar type and the other a vector type,
/// the scalar argument is splatted to the vector type.
///
/// @param[in] B Builder used to emit instructions.
/// @param[in] LHS first argument to be passed to the intrinsic.
/// @param[in] RHS second argument to be passed to the intrinsic.
/// @param[in] ID the LLVM intrinsic ID.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineAsLLVMBinaryIntrinsic(
    IRBuilder<> &B, Value *LHS, Value *RHS, llvm::Intrinsic::ID ID) {
  const Triple TT(B.GetInsertBlock()->getModule()->getTargetTriple());
  if (TT.getArch() == Triple::arm || TT.getArch() == Triple::aarch64) {
    // fmin and fmax fail CTS on arm targets.
    // This is a HACK and should be removed when CA-3595 is resolved.
    return nullptr;
  }

  const auto *LHSTy = LHS->getType();
  const auto *RHSTy = RHS->getType();
  if (LHSTy->isVectorTy() != RHSTy->isVectorTy()) {
    auto VectorEC =
        multi_llvm::getVectorElementCount(LHSTy->isVectorTy() ? LHSTy : RHSTy);
    if (!LHS->getType()->isVectorTy()) {
      LHS = B.CreateVectorSplat(VectorEC, LHS);
    }
    if (!RHS->getType()->isVectorTy()) {
      RHS = B.CreateVectorSplat(VectorEC, RHS);
    }
  }
  return B.CreateBinaryIntrinsic(ID, LHS, RHS);
}

/// @brief Emit the body of the 'as_*' builtin function.
///
/// @param[in] F Function to emit the body inline.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineAs(Function *F, llvm::IRBuilder<> &B,
                                          llvm::ArrayRef<Value *> Args) {
  if (Args.size() != 1) {
    return nullptr;
  }
  Value *Src = Args[0];
  Type *SrcTy = Src->getType();
  Type *DstTy = F->getReturnType();
  auto *SrcVecTy = dyn_cast<FixedVectorType>(SrcTy);
  auto *DstVecTy = dyn_cast<FixedVectorType>(DstTy);
  Type *SrcEleTy = SrcVecTy ? SrcVecTy->getElementType() : nullptr;
  Type *DstEleTy = DstVecTy ? DstVecTy->getElementType() : nullptr;
  const unsigned SrcEleBits = SrcEleTy ? SrcEleTy->getPrimitiveSizeInBits() : 0;
  const unsigned DstEleBits = DstEleTy ? DstEleTy->getPrimitiveSizeInBits() : 0;
  const bool SrcDstHaveSameWidth =
      SrcEleTy && DstEleTy && (SrcEleBits == DstEleBits);
  const bool SrcVec3 = SrcVecTy && (SrcVecTy->getNumElements() == 3);
  const bool SrcVec4 = SrcVecTy && (SrcVecTy->getNumElements() == 4);
  const bool DstVec3 = DstVecTy && (DstVecTy->getNumElements() == 3);
  const bool DstVec4 = DstVecTy && (DstVecTy->getNumElements() == 4);
  bool LowerAsShuffle = false;
  if (SrcVec3 && !DstVec3) {
    if (!DstVec4 || !SrcDstHaveSameWidth) {
      return nullptr;
    }
    LowerAsShuffle = true;
  } else if (DstVec3 && !SrcVec3) {
    if (!SrcVec4 || !SrcDstHaveSameWidth) {
      return nullptr;
    }
    LowerAsShuffle = true;
  }

  // Lower some vec3 variants of as_* using vector shuffles.
  if (LowerAsShuffle) {
    SmallVector<Constant *, 4> Indices;
    for (unsigned i = 0; i < DstVecTy->getNumElements(); i++) {
      if (i < SrcVecTy->getNumElements()) {
        Indices.push_back(B.getInt32(i));
      } else {
        Indices.push_back(UndefValue::get(B.getInt32Ty()));
      }
    }
    Value *Mask = ConstantVector::get(Indices);
    Src = B.CreateShuffleVector(Src, UndefValue::get(SrcVecTy), Mask);
  }

  // Common case: as_* is a simple bitcast.
  return B.CreateBitCast(Src, DstTy, "as");
}

/// @brief Emit the body of the 'convert_*' builtin functions.
///
/// @param[in] F the function to emit inline.
/// @param[in] builtinID Builtin ID of the function.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineConvert(Function *F, BuiltinID builtinID,
                                               IRBuilder<> &B,
                                               ArrayRef<Value *> Args) {
  if (Args.size() != 1) {
    return nullptr;
  }
  Type *DstTy = nullptr;
  bool DstIsSigned = false;
  auto &Ctx = B.getContext();
  switch (builtinID) {
    case eCLBuiltinConvertChar:
      DstIsSigned = true;
      LLVM_FALLTHROUGH;
    case eCLBuiltinConvertUChar:
      DstTy = IntegerType::getInt8Ty(Ctx);
      break;
    case eCLBuiltinConvertShort:
      DstIsSigned = true;
      LLVM_FALLTHROUGH;
    case eCLBuiltinConvertUShort:
      DstTy = IntegerType::getInt16Ty(Ctx);
      break;
    case eCLBuiltinConvertInt:
      DstIsSigned = true;
      LLVM_FALLTHROUGH;
    case eCLBuiltinConvertUInt:
      DstTy = IntegerType::getInt32Ty(Ctx);
      break;
    case eCLBuiltinConvertLong:
      DstIsSigned = true;
      LLVM_FALLTHROUGH;
    case eCLBuiltinConvertULong:
      DstTy = IntegerType::getInt64Ty(Ctx);
      break;

    default:
      return nullptr;
  }
  if (!DstTy) {
    return nullptr;
  }

  Value *Src = Args[0];
  bool SrcIsSigned;
  if (Src->getType()->isFloatingPointTy()) {
    // All floating point types are signed
    SrcIsSigned = true;
  } else {
    auto IsParamSignedOrNone = paramHasTypeQual(*F, 0, eTypeQualSignedInt);
    if (!IsParamSignedOrNone) {
      return nullptr;
    }
    SrcIsSigned = *IsParamSignedOrNone;
  }

  auto Opcode = CastInst::getCastOpcode(Src, SrcIsSigned, DstTy, DstIsSigned);
  return B.CreateCast(Opcode, Src, DstTy, "inline_convert");
}

/// @brief Emit the body of the 'vloadN' builtin function.
///
/// @param[in] F Function to emit the body inline.
/// @param[in] Width Number of elements to load.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineVLoad(Function *F, unsigned Width,
                                             IRBuilder<> &B,
                                             ArrayRef<Value *> Args) {
  if (Width < 2) {
    return nullptr;
  }
  (void)F;

  Type *RetTy = F->getReturnType();
  assert(isa<FixedVectorType>(RetTy) && "vloadN must return a vector type");
  Type *EltTy = RetTy->getScalarType();

  Value *Ptr = Args[1];
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy) {
    return nullptr;
  }
  auto *DataTy = FixedVectorType::get(EltTy, Width);
  Value *Data = UndefValue::get(DataTy);

  // Emit the base pointer.
  Value *Offset = Args[0];
  IntegerType *OffsetTy = dyn_cast<IntegerType>(Offset->getType());
  if (!OffsetTy) {
    return nullptr;
  }
  Value *Stride = ConstantInt::get(OffsetTy, Width);
  Offset = B.CreateMul(Offset, Stride);
  Value *GEPBase = B.CreateGEP(EltTy, Ptr, Offset, "vload_base");

  if (Width == 3) {
    for (unsigned i = 0; i < Width; i++) {
      Value *Index = B.getInt32(i);
      Value *GEP = B.CreateGEP(EltTy, GEPBase, Index);
      Value *Lane = B.CreateLoad(EltTy, GEP, false, "vload");
      Data = B.CreateInsertElement(Data, Lane, Index, "vload_insert");
    }
  } else {
    PointerType *VecPtrTy = DataTy->getPointerTo(PtrTy->getAddressSpace());
    Value *VecBase = B.CreateBitCast(GEPBase, VecPtrTy, "vload_ptr");
    auto *Load = B.CreateLoad(DataTy, VecBase, false, "vload");

    const unsigned Align = DataTy->getScalarSizeInBits() / 8;
    Load->setAlignment(MaybeAlign(Align).valueOrOne());
    Data = Load;
  }

  return Data;
}

/// @brief Emit the body of the 'vstoreN' builtin function.
///
/// @param[in] F Function to emit the body inline.
/// @param[in] Width Number of elements to store.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineVStore(Function *F, unsigned Width,
                                              IRBuilder<> &B,
                                              ArrayRef<Value *> Args) {
  if (Width < 2) {
    return nullptr;
  }
  (void)F;

  Value *Data = Args[0];
  auto *VecDataTy = dyn_cast<FixedVectorType>(Data->getType());
  if (!VecDataTy || (VecDataTy->getNumElements() != Width)) {
    return nullptr;
  }

  Value *Ptr = Args[2];
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy) {
    return nullptr;
  }

  // Emit the base pointer.
  Value *Offset = Args[1];
  IntegerType *OffsetTy = dyn_cast<IntegerType>(Offset->getType());
  if (!OffsetTy) {
    return nullptr;
  }
  Value *Stride = ConstantInt::get(OffsetTy, Width);
  Offset = B.CreateMul(Offset, Stride);
  Value *GEPBase =
      B.CreateGEP(VecDataTy->getElementType(), Ptr, Offset, "vstore_base");

  // Emit store(s).
  StoreInst *Store = nullptr;
  if (Width == 3) {
    for (unsigned i = 0; i < Width; i++) {
      Value *Index = B.getInt32(i);
      Value *Lane = B.CreateExtractElement(Data, Index, "vstore_extract");
      Value *GEP = B.CreateGEP(VecDataTy->getElementType(), GEPBase, Index);
      Store = B.CreateStore(Lane, GEP, false);
    }
  } else {
    PointerType *VecPtrTy = VecDataTy->getPointerTo(PtrTy->getAddressSpace());
    Value *VecBase = B.CreateBitCast(GEPBase, VecPtrTy, "vstore_ptr");
    Store = B.CreateStore(Data, VecBase, false);

    const unsigned Align = VecDataTy->getScalarSizeInBits() / 8;
    Store->setAlignment(MaybeAlign(Align).valueOrOne());
  }
  return Store;
}

/// @brief Emit the body of the 'vload_half' builtin function.
///
/// @param[in] F Function to emit the body inline.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineVLoadHalf(Function *F, IRBuilder<> &B,
                                                 ArrayRef<Value *> Args) {
  if (F->getType()->isVectorTy()) {
    return nullptr;
  }

  // Cast the pointer to ushort*.
  Value *Ptr = Args[1];
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy) {
    return nullptr;
  }
  Type *U16Ty = B.getInt16Ty();
  Type *U16PtrTy = PointerType::get(U16Ty, PtrTy->getAddressSpace());
  Value *DataPtr = B.CreateBitCast(Ptr, U16PtrTy);

  // Emit the base pointer.
  Value *Offset = Args[0];
  DataPtr = B.CreateGEP(U16Ty, DataPtr, Offset, "vload_base");

  // Load a ushort.
  Value *Data = B.CreateLoad(B.getInt16Ty(), DataPtr, "vload_half");

  // Declare the conversion builtin.
  Module *M = F->getParent();
  Function *HalfToFloatFn =
      declareBuiltin(M, eCLBuiltinConvertHalfToFloat, B.getFloatTy(),
                     {B.getInt16Ty()}, {eTypeQualNone});
  if (!HalfToFloatFn) {
    return nullptr;
  }

  // Convert it to float.
  CallInst *CI = CreateBuiltinCall(B, HalfToFloatFn, {Data});
  CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// @brief Emit the body of the 'vstore_half' builtin function.
///
/// @param[in] F Function to emit the body inline.
/// @param[in] Mode Rounding mode to use, e.g. '_rte'.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineVStoreHalf(Function *F, StringRef Mode,
                                                  IRBuilder<> &B,
                                                  ArrayRef<Value *> Args) {
  Value *Data = Args[0];
  if (!Data || Data->getType()->isVectorTy()) {
    return nullptr;
  }

  // Declare the conversion builtin.
  BuiltinID ConvID;

  if (Data->getType() == B.getFloatTy()) {
    ConvID = StringSwitch<BuiltinID>(Mode)
                 .Case("", eCLBuiltinConvertFloatToHalf)
                 .Case("_rte", eCLBuiltinConvertFloatToHalfRte)
                 .Case("_rtz", eCLBuiltinConvertFloatToHalfRtz)
                 .Case("_rtp", eCLBuiltinConvertFloatToHalfRtp)
                 .Case("_rtn", eCLBuiltinConvertFloatToHalfRtn)
                 .Default(eBuiltinInvalid);
  } else {
    ConvID = StringSwitch<BuiltinID>(Mode)
                 .Case("", eCLBuiltinConvertDoubleToHalf)
                 .Case("_rte", eCLBuiltinConvertDoubleToHalfRte)
                 .Case("_rtz", eCLBuiltinConvertDoubleToHalfRtz)
                 .Case("_rtp", eCLBuiltinConvertDoubleToHalfRtp)
                 .Case("_rtn", eCLBuiltinConvertDoubleToHalfRtn)
                 .Default(eBuiltinInvalid);
  }
  if (ConvID == eBuiltinInvalid) {
    return nullptr;
  }
  Module *M = F->getParent();

  // Normally, the vstore_half functions take the number to store as a float.
  // However, if the double extension is enabled, it is also possible to use
  // double instead. This means that we might have to convert either a float or
  // a double to a half.
  Function *FloatToHalfFn = declareBuiltin(M, ConvID, B.getInt16Ty(),
                                           {Data->getType()}, {eTypeQualNone});
  if (!FloatToHalfFn) {
    return nullptr;
  }

  // Convert the data from float/double to half.
  CallInst *CI = CreateBuiltinCall(B, FloatToHalfFn, {Data});
  CI->setCallingConv(F->getCallingConv());
  Data = CI;

  // Cast the pointer to ushort*.
  Value *Ptr = Args[2];
  PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy) {
    return nullptr;
  }
  auto U16Ty = B.getInt16Ty();
  Type *U16PtrTy = PointerType::get(U16Ty, PtrTy->getAddressSpace());
  Value *DataPtr = B.CreateBitCast(Ptr, U16PtrTy);

  // Emit the base pointer.
  Value *Offset = Args[1];
  DataPtr = B.CreateGEP(U16Ty, DataPtr, Offset, "vstore_base");

  // Store the ushort.
  return B.CreateStore(Data, DataPtr, "vstore_half");
}

/// @brief Emit the body of a relational builtin function.
///
/// This function handles relational builtins that accept two arguments, such as
/// the comparison builtins.
///
/// @param[in] BuiltinID Identifier of the builtin to emit the body inline.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineRelationalsWithTwoArguments(
    BuiltinID BuiltinID, IRBuilder<> &B, ArrayRef<Value *> Args) {
  CmpInst::Predicate Pred = CmpInst::FCMP_FALSE;
  CmpInst::Predicate Pred2 = CmpInst::FCMP_FALSE;
  switch (BuiltinID) {
    default:
      return nullptr;
    case eCLBuiltinIsEqual:
      Pred = CmpInst::FCMP_OEQ;
      break;
    case eCLBuiltinIsNotEqual:
      Pred = CmpInst::FCMP_UNE;
      break;
    case eCLBuiltinIsGreater:
      Pred = CmpInst::FCMP_OGT;
      break;
    case eCLBuiltinIsGreaterEqual:
      Pred = CmpInst::FCMP_OGE;
      break;
    case eCLBuiltinIsLess:
      Pred = CmpInst::FCMP_OLT;
      break;
    case eCLBuiltinIsLessEqual:
      Pred = CmpInst::FCMP_OLE;
      break;
    case eCLBuiltinIsLessGreater:
      Pred = CmpInst::FCMP_OLT;
      Pred2 = CmpInst::FCMP_OGT;
      break;
    case eCLBuiltinIsOrdered:
      Pred = CmpInst::FCMP_ORD;
      break;
    case eCLBuiltinIsUnordered:
      Pred = CmpInst::FCMP_UNO;
      break;
  }

  if (Args.size() != 2) {
    return nullptr;
  }
  Value *Src0 = Args[0], *Src1 = Args[1];
  Value *Cmp = B.CreateFCmp(Pred, Src0, Src1, "relational");

  Type *ResultEleTy = nullptr;
  Type *Src0Ty = Src0->getType();
  if (Src0->getType() == B.getDoubleTy()) {
    // Special case because relational(doubleN, doubleN) returns longn while
    // relational(double, double) returns int.
    if (Src0Ty->isVectorTy()) {
      ResultEleTy = B.getInt64Ty();
    } else {
      ResultEleTy = B.getInt32Ty();
    }
  } else if (Src0->getType() == B.getHalfTy()) {
    // Special case because relational(HalfTyN, HalfTyN) returns i16 while
    // relational(HalfTy, HalfTy) returns int.
    if (Src0Ty->isVectorTy()) {
      ResultEleTy = B.getInt16Ty();
    } else {
      ResultEleTy = B.getInt32Ty();
    }
  } else {
    // All the other cases can be handled here.
    ResultEleTy = B.getIntNTy(Src0->getType()->getScalarSizeInBits());
  }
  Value *Result = nullptr;
  auto *SrcVecTy = dyn_cast<FixedVectorType>(Src0->getType());
  if (SrcVecTy) {
    auto *ResultVecTy =
        FixedVectorType::get(ResultEleTy, SrcVecTy->getNumElements());
    Result = B.CreateSExt(Cmp, ResultVecTy, "relational");
  } else {
    Result = B.CreateZExt(Cmp, ResultEleTy, "relational");
  }

  if (Pred2 != CmpInst::FCMP_FALSE) {
    Value *Cmp2 = B.CreateFCmp(Pred2, Src0, Src1, "relational");
    Value *True = SrcVecTy ? Constant::getAllOnesValue(Result->getType())
                           : ConstantInt::get(Result->getType(), 1);
    Result = B.CreateSelect(Cmp2, True, Result);
  }

  return Result;
}

/// @brief Emit the body of a relational builtin function.
///
/// This function handles relational builtins that accept a single argument,
/// such as the builtins checking if the argument is infinite or not.
///
/// @param[in] BuiltinID Identifier of the builtin to emit the body inline.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Arg Argument passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineRelationalsWithOneArgument(
    BuiltinID BuiltinID, IRBuilder<> &B, Value *Arg) {
  Value *Result = nullptr;
  // The types (and misc info) that we will be using
  Type *ArgTy = Arg->getType();
  const bool isVectorTy = ArgTy->isVectorTy();
  const unsigned Width =
      isVectorTy ? multi_llvm::getVectorNumElements(ArgTy) : 0;
  Type *ArgEleTy = isVectorTy ? multi_llvm::getVectorElementType(ArgTy) : ArgTy;
  Type *SignedTy = ArgEleTy == B.getFloatTy() ? B.getInt32Ty() : B.getInt64Ty();
  Type *ReturnTy = (ArgEleTy == B.getDoubleTy() && isVectorTy) ? B.getInt64Ty()
                                                               : B.getInt32Ty();

  if (ArgEleTy != B.getFloatTy() && ArgEleTy != B.getDoubleTy()) {
    return nullptr;
  }
  // Create all the masks we are going to be using
  Constant *ExponentMask = nullptr;
  Constant *MantissaMask = nullptr;
  Constant *NonSignMask = nullptr;
  Constant *Zero = nullptr;
  if (ArgEleTy == B.getFloatTy()) {
    ExponentMask = B.getInt32(0x7F800000u);
    MantissaMask = B.getInt32(0x007FFFFFu);
    NonSignMask = B.getInt32(0x7FFFFFFFu);
    Zero = B.getInt32(0u);
  } else if (ArgEleTy == B.getDoubleTy()) {
    ExponentMask = B.getInt64(0x7FF0000000000000u);
    MantissaMask = B.getInt64(0x000FFFFFFFFFFFFFu);
    NonSignMask = B.getInt64(0x7FFFFFFFFFFFFFFFu);
    Zero = B.getInt64(0u);
  }

  // For the vector versions, we need to create vector types and values
  if (isVectorTy) {
    SignedTy = FixedVectorType::get(SignedTy, Width);
    ReturnTy = FixedVectorType::get(ReturnTy, Width);
    const auto EC = ElementCount::getFixed(Width);
    ExponentMask = ConstantVector::getSplat(EC, ExponentMask);
    MantissaMask = ConstantVector::getSplat(EC, MantissaMask);
    NonSignMask = ConstantVector::getSplat(EC, NonSignMask);
    Zero = ConstantVector::getSplat(EC, Zero);
  }

  // We will be needing access to the argument as an integer (bitcast) value
  Value *STArg = B.CreateBitCast(Arg, SignedTy);

  // Emit the IR that will calculate the result
  switch (BuiltinID) {
    default:
      llvm_unreachable("Invalid Builtin ID");
      break;
    case eCLBuiltinIsFinite:
      Result = B.CreateAnd(STArg, NonSignMask);
      Result = B.CreateICmpSLT(Result, ExponentMask);
      break;
    case eCLBuiltinIsInf:
      Result = B.CreateAnd(STArg, NonSignMask);
      Result = B.CreateICmpEQ(Result, ExponentMask);
      break;
    case eCLBuiltinIsNan: {
      Result = B.CreateAnd(STArg, NonSignMask);
      // This checks if the exponent is all ones (the same as the ExponentMask)
      // and also if the significant (the mantissa) is not zero. If the mantissa
      // is zero then it would be infinite, not NaN.
      Value *ExponentAllOnes =
          B.CreateICmpEQ(ExponentMask, B.CreateAnd(ExponentMask, Result));
      Value *MantissaNotZero =
          B.CreateICmpSGT(B.CreateAnd(MantissaMask, Result), Zero);
      Result = B.CreateAnd(ExponentAllOnes, MantissaNotZero);
      break;
    }
    case eCLBuiltinIsNormal: {
      Result = B.CreateAnd(STArg, NonSignMask);
      Value *ExponentBitsNotAllSet = B.CreateICmpSLT(Result, ExponentMask);
      Value *ExponentBitsNonZero = B.CreateICmpSGT(Result, MantissaMask);
      Result = B.CreateAnd(ExponentBitsNotAllSet, ExponentBitsNonZero);
      break;
    }
    case eCLBuiltinSignBit:
      Result = B.CreateICmpSLT(STArg, Zero);
      break;
  }

  // Convert the i1 result from the comparison instruction to the type that the
  // builtin returns
  if (isVectorTy) {
    // 0 for false, -1 (all 1s) for true
    Result = B.CreateSExt(Result, ReturnTy);
  } else {
    // 0 for false, 1 for true
    Result = B.CreateZExt(Result, ReturnTy);
  }

  return Result;
}

/// @brief Emit the body of a vector shuffle builtin function.
///
/// @param[in] BuiltinID Identifier of the builtin to emit the body inline.
/// @param[in] B Builder used to emit instructions.
/// @param[in] Args Arguments passed to the function.
///
/// @return Value returned by the builtin implementation or null on failure.
Value *CLBuiltinInfo::emitBuiltinInlineShuffle(BuiltinID BuiltinID,
                                               IRBuilder<> &B,
                                               ArrayRef<Value *> Args) {
  // Make sure we have the correct number of arguments.
  assert(((BuiltinID == eCLBuiltinShuffle && Args.size() == 2) ||
          (BuiltinID == eCLBuiltinShuffle2 && Args.size() == 3)) &&
         "Wrong number of arguments!");

  // It is not worth splitting shuffle and shuffle2 into two functions as a lot
  // of the code is the same.
  const bool isShuffle2 = (BuiltinID == eCLBuiltinShuffle2);

  // Get the mask and the mask type.
  Value *Mask = Args[isShuffle2 ? 2 : 1];
  auto MaskVecTy = cast<FixedVectorType>(Mask->getType());
  IntegerType *MaskTy = cast<IntegerType>(MaskVecTy->getElementType());
  const int MaskWidth = MaskVecTy->getNumElements();

  // TODO: Support non-constant masks (in a less efficient way)
  if (!isa<Constant>(Mask)) {
    return nullptr;
  }

  // We need to mask the mask elements, since the OpenCL standard specifies that
  // we should only take the ilogb(2N-1)+1 least significant bits from each mask
  // element into consideration, where N the number of elements in the vector
  // according to vec_step.
  auto ShuffleTy = cast<FixedVectorType>(Args[0]->getType());
  const int Width = ShuffleTy->getNumElements();
  // Vectors for size 3 are not supported by the shuffle builtin.
  assert(Width != 3 && "Invalid vector width of 3!");
  const int N = (Width == 3 ? 4 : Width);
  const int SignificantBits =
      stdcompat::ilogb((2 * N) - 1) + (isShuffle2 ? 1 : 0);
  const unsigned BitMask = ~((~0u) << SignificantBits);
  Value *BitMaskV = ConstantVector::getSplat(ElementCount::getFixed(MaskWidth),
                                             ConstantInt::get(MaskTy, BitMask));
  // The builtin's mask may have different integer types, while the LLVM
  // instruction only supports i32.
  // Mask the mask.
  Value *MaskedMask = B.CreateAnd(Mask, BitMaskV, "mask");
  MaskedMask = B.CreateIntCast(
      MaskedMask, FixedVectorType::get(B.getInt32Ty(), MaskWidth), false);

  // Create the shufflevector instruction.
  Value *Arg1 = (isShuffle2 ? Args[1] : UndefValue::get(ShuffleTy));
  return B.CreateShuffleVector(Args[0], Arg1, MaskedMask, "shuffle");
}

Value *CLBuiltinInfo::emitBuiltinInlinePrintf(BuiltinID, IRBuilder<> &B,
                                              ArrayRef<Value *> Args) {
  Module &M = *(B.GetInsertBlock()->getModule());

  // Declare printf if needed.
  Function *Printf = M.getFunction("printf");
  if (!Printf) {
    PointerType *PtrTy = PointerType::getUnqual(B.getInt8Ty());
    FunctionType *PrintfTy = FunctionType::get(B.getInt32Ty(), {PtrTy}, true);
    Printf =
        Function::Create(PrintfTy, GlobalValue::ExternalLinkage, "printf", &M);
    Printf->setCallingConv(CallingConv::SPIR_FUNC);
  }

  return CreateBuiltinCall(B, Printf, Args);
}

// Must be kept in sync with our OpenCL headers!
enum : uint32_t {
  CLK_LOCAL_MEM_FENCE = 1,
  CLK_GLOBAL_MEM_FENCE = 2,
  // FIXME: We don't support image fences in our headers
};

// Must be kept in sync with our OpenCL headers!
enum : uint32_t {
  memory_scope_work_item = 1,
  memory_scope_sub_group = 2,
  memory_scope_work_group = 3,
  memory_scope_device = 4,
  memory_scope_all_svm_devices = 5,
  memory_scope_all_devices = 6,
};

// Must be kept in sync with our OpenCL headers!
enum : uint32_t {
  memory_order_relaxed = 0,
  memory_order_acquire = 1,
  memory_order_release = 2,
  memory_order_acq_rel = 3,
  memory_order_seq_cst = 4,
};

static std::optional<unsigned> parseMemFenceFlagsParam(Value *const P) {
  // Grab the 'flags' parameter.
  if (auto *const Flags = dyn_cast<ConstantInt>(P)) {
    // cl_mem_fence_flags is a bitfield and can be 0 or a combination of
    // CLK_(GLOBAL|LOCAL|IMAGE)_MEM_FENCE values ORed together.
    switch (Flags->getZExtValue()) {
      case 0:
        return std::nullopt;
      case CLK_LOCAL_MEM_FENCE:
        return BIMuxInfoConcept::MemSemanticsWorkGroupMemory;
      case CLK_GLOBAL_MEM_FENCE:
        return BIMuxInfoConcept::MemSemanticsCrossWorkGroupMemory;
      case CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE:
        return (BIMuxInfoConcept::MemSemanticsWorkGroupMemory |
                BIMuxInfoConcept::MemSemanticsCrossWorkGroupMemory);
      default:
        llvm_unreachable("unhandled memory fence flags");
    }
  }
  return std::nullopt;
}

static std::optional<unsigned> parseMemoryScopeParam(Value *const P) {
  if (auto *const Scope = dyn_cast<ConstantInt>(P)) {
    switch (Scope->getZExtValue()) {
      case memory_scope_work_item:
        return BIMuxInfoConcept::MemScopeWorkItem;
      case memory_scope_sub_group:
        return BIMuxInfoConcept::MemScopeSubGroup;
      case memory_scope_work_group:
        return BIMuxInfoConcept::MemScopeWorkGroup;
      case memory_scope_device:
        return BIMuxInfoConcept::MemScopeDevice;
      // 3.3.5. memory_scope_all_devices is an alias for
      // memory_scope_all_svm_devices.
      case memory_scope_all_devices:
      case memory_scope_all_svm_devices:
        return BIMuxInfoConcept::MemScopeCrossDevice;
      default:
        llvm_unreachable("unhandled memory scope");
    }
  }
  return std::nullopt;
}

static std::optional<unsigned> parseMemoryOrderParam(Value *const P) {
  if (auto *const Order = dyn_cast<ConstantInt>(P)) {
    switch (Order->getZExtValue()) {
      case memory_order_relaxed:
        return BIMuxInfoConcept::MemSemanticsRelaxed;
      case memory_order_acquire:
        return BIMuxInfoConcept::MemSemanticsAcquire;
      case memory_order_release:
        return BIMuxInfoConcept::MemSemanticsRelease;
      case memory_order_acq_rel:
        return BIMuxInfoConcept::MemSemanticsAcquireRelease;
      case memory_order_seq_cst:
        return BIMuxInfoConcept::MemSemanticsSequentiallyConsistent;
      default:
        llvm_unreachable("unhandled memory order");
    }
  }
  return std::nullopt;
}

// This function returns a mux builtin ID for the corresponding CL builtin ID
// when that lowering is straightforward and the function types of each builtin
// are identical.
static std::optional<BuiltinID> get1To1BuiltinLowering(BuiltinID CLBuiltinID) {
  switch (CLBuiltinID) {
    default:
      return std::nullopt;
    case eCLBuiltinGetWorkDim:
      return eMuxBuiltinGetWorkDim;
    case eCLBuiltinGetGroupId:
      return eMuxBuiltinGetGroupId;
    case eCLBuiltinGetGlobalSize:
      return eMuxBuiltinGetGlobalSize;
    case eCLBuiltinGetGlobalOffset:
      return eMuxBuiltinGetGlobalOffset;
    case eCLBuiltinGetLocalId:
      return eMuxBuiltinGetLocalId;
    case eCLBuiltinGetLocalSize:
      return eMuxBuiltinGetLocalSize;
    case eCLBuiltinGetEnqueuedLocalSize:
      return eMuxBuiltinGetEnqueuedLocalSize;
    case eCLBuiltinGetNumGroups:
      return eMuxBuiltinGetNumGroups;
    case eCLBuiltinGetGlobalId:
      return eMuxBuiltinGetGlobalId;
    case eCLBuiltinGetLocalLinearId:
      return eMuxBuiltinGetLocalLinearId;
    case eCLBuiltinGetGlobalLinearId:
      return eMuxBuiltinGetGlobalLinearId;
    case eCLBuiltinGetSubgroupSize:
      return eMuxBuiltinGetSubGroupSize;
    case eCLBuiltinGetMaxSubgroupSize:
      return eMuxBuiltinGetMaxSubGroupSize;
    case eCLBuiltinGetSubgroupLocalId:
      return eMuxBuiltinGetSubGroupLocalId;
    case eCLBuiltinGetNumSubgroups:
      return eMuxBuiltinGetNumSubGroups;
    case eCLBuiltinGetEnqueuedNumSubgroups:
      // Note - this is mapping to the same builtin as
      // eCLBuiltinGetNumSubgroups, as we don't currently support
      // non-uniform work-group sizes.
      return eMuxBuiltinGetNumSubGroups;
    case eCLBuiltinGetSubgroupId:
      return eMuxBuiltinGetSubGroupId;
  }
}

Instruction *CLBuiltinInfo::lowerBuiltinToMuxBuiltin(
    CallInst &CI, BIMuxInfoConcept &BIMuxImpl) {
  auto &M = *CI.getModule();
  auto *const F = CI.getCalledFunction();
  assert(F && "No calling function?");
  const auto ID = identifyBuiltin(*F);

  // Handle straightforward 1:1 mappings.
  if (auto MuxID = get1To1BuiltinLowering(ID)) {
    auto *const MuxBuiltinFn = BIMuxImpl.getOrDeclareMuxBuiltin(*MuxID, M);
    assert(MuxBuiltinFn && "Could not get/declare mux builtin");
    const SmallVector<Value *> Args(CI.args());
    auto *const NewCI = CallInst::Create(MuxBuiltinFn, Args, CI.getName(), &CI);
    NewCI->takeName(&CI);
    NewCI->setAttributes(MuxBuiltinFn->getAttributes());
    return NewCI;
  }

  IRBuilder<> B(&CI);
  LLVMContext &Ctx = M.getContext();
  auto *const I32Ty = Type::getInt32Ty(Ctx);

  auto CtrlBarrierID = eMuxBuiltinWorkGroupBarrier;
  unsigned DefaultMemScope = BIMuxInfoConcept::MemScopeWorkGroup;
  unsigned DefaultMemOrder =
      BIMuxInfoConcept::MemSemanticsSequentiallyConsistent;

  switch (ID) {
    default:
      // Sub-group and work-group builtins need lowering to their mux
      // equivalents.
      if (auto *const NewI = lowerGroupBuiltinToMuxBuiltin(CI, ID, BIMuxImpl)) {
        return NewI;
      }
      return nullptr;
    case eCLBuiltinSubGroupBarrier:
      CtrlBarrierID = eMuxBuiltinSubGroupBarrier;
      DefaultMemScope = BIMuxInfoConcept::MemScopeSubGroup;
      LLVM_FALLTHROUGH;
    case eCLBuiltinBarrier:
    case eCLBuiltinWorkGroupBarrier: {
      // Memory Scope which the barrier controls. Defaults to 'workgroup' or
      // 'subgroup' scope depending on the barrier, but sub_group_barrier and
      // work_group_barrier can optionally provide a scope.
      unsigned ScopeVal = DefaultMemScope;
      if ((ID == eCLBuiltinSubGroupBarrier ||
           ID == eCLBuiltinWorkGroupBarrier) &&
          F->arg_size() == 2) {
        if (auto Scope = parseMemoryScopeParam(CI.getOperand(1))) {
          ScopeVal = *Scope;
        }
      }

      const unsigned SemanticsVal =
          DefaultMemOrder |
          parseMemFenceFlagsParam(CI.getOperand(0)).value_or(0);

      auto *const CtrlBarrier =
          BIMuxImpl.getOrDeclareMuxBuiltin(CtrlBarrierID, M);

      auto *const BarrierID = ConstantInt::get(I32Ty, 0);
      auto *const Scope = ConstantInt::get(I32Ty, ScopeVal);
      auto *const Semantics = ConstantInt::get(I32Ty, SemanticsVal);
      auto *const NewCI = B.CreateCall(
          CtrlBarrier, {BarrierID, Scope, Semantics}, CI.getName());
      NewCI->setAttributes(CtrlBarrier->getAttributes());
      NewCI->takeName(&CI);
      return NewCI;
    }
    case eCLBuiltinAtomicWorkItemFence:
      // atomic_work_item_fence has two parameters which we can parse.
      DefaultMemOrder =
          parseMemoryOrderParam(CI.getOperand(1)).value_or(DefaultMemOrder);
      DefaultMemScope =
          parseMemoryScopeParam(CI.getOperand(2)).value_or(DefaultMemScope);
      LLVM_FALLTHROUGH;
    case eCLBuiltinMemFence:
    case eCLBuiltinReadMemFence:
    case eCLBuiltinWriteMemFence: {
      // The deprecated 'fence' builtins default to memory_scope_work_group and
      // have one possible order each.
      if (ID == eCLBuiltinMemFence) {
        DefaultMemOrder = BIMuxInfoConcept::MemSemanticsAcquireRelease;
      } else if (ID == eCLBuiltinReadMemFence) {
        DefaultMemOrder = BIMuxInfoConcept::MemSemanticsAcquire;
      } else if (ID == eCLBuiltinWriteMemFence) {
        DefaultMemOrder = BIMuxInfoConcept::MemSemanticsRelease;
      }
      const unsigned SemanticsVal =
          DefaultMemOrder |
          parseMemFenceFlagsParam(CI.getOperand(0)).value_or(0);
      auto *const MemBarrier =
          BIMuxImpl.getOrDeclareMuxBuiltin(eMuxBuiltinMemBarrier, M);
      auto *const Scope = ConstantInt::get(I32Ty, DefaultMemScope);
      auto *const Semantics = ConstantInt::get(I32Ty, SemanticsVal);
      auto *const NewCI =
          B.CreateCall(MemBarrier, {Scope, Semantics}, CI.getName());
      NewCI->setAttributes(MemBarrier->getAttributes());
      NewCI->takeName(&CI);
      return NewCI;
    }
    case eCLBuiltinAsyncWorkGroupCopy:
    case eCLBuiltinAsyncWorkGroupStridedCopy:
    case eCLBuiltinAsyncWorkGroupCopy2D2D:
    case eCLBuiltinAsyncWorkGroupCopy3D3D:
      return lowerAsyncBuiltinToMuxBuiltin(CI, ID, BIMuxImpl);
    case eCLBuiltinWaitGroupEvents: {
      auto *const MuxWait =
          BIMuxImpl.getOrDeclareMuxBuiltin(eMuxBuiltinDMAWait, M);
      assert(MuxWait && "Could not get/declare __mux_dma_wait");
      auto *const Count = CI.getArgOperand(0);
      auto *Events = CI.getArgOperand(1);

      assert(Events->getType()->isPointerTy() &&
             (Events->getType()->getPointerAddressSpace() ==
                  compiler::utils::AddressSpace::Private ||
              Events->getType()->getPointerAddressSpace() ==
                  compiler::utils::AddressSpace::Generic) &&
             "Pointer to event must be in address space 0 or 4.");

      Events = B.CreatePointerBitCastOrAddrSpaceCast(
          Events, PointerType::getUnqual(Ctx), "mux.events");
      auto *const NewCI = B.CreateCall(MuxWait, {Count, Events}, CI.getName());
      NewCI->setAttributes(MuxWait->getAttributes());
      NewCI->takeName(&CI);
      return NewCI;
    }
  }
}

Instruction *CLBuiltinInfo::lowerGroupBuiltinToMuxBuiltin(
    CallInst &CI, BuiltinID ID, BIMuxInfoConcept &BIMuxImpl) {
  auto &M = *CI.getModule();
  auto *const F = CI.getCalledFunction();
  assert(F && "No calling function?");

  // Some ops need extra checking to determine their mux ID:
  // * add/mul operations are split into integer/float
  // * min/max operations are split into signed/unsigned/float
  // So we set a 'base' builtin ID for these operations to the (unsigned)
  // integer variant and do a checking step afterwards where we refine the
  // builtin ID.
  bool RecheckOpType = false;
  BaseBuiltinID MuxBuiltinID = eBuiltinInvalid;
  switch (ID) {
    default:
      return nullptr;
    case eCLBuiltinSubgroupAll:
      MuxBuiltinID = eMuxBuiltinSubgroupAll;
      break;
    case eCLBuiltinSubgroupAny:
      MuxBuiltinID = eMuxBuiltinSubgroupAny;
      break;
    case eCLBuiltinSubgroupBroadcast:
      MuxBuiltinID = eMuxBuiltinSubgroupBroadcast;
      break;
    case eCLBuiltinSubgroupReduceAdd:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupReduceAdd;
      break;
    case eCLBuiltinSubgroupReduceMin:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupReduceUMin;
      break;
    case eCLBuiltinSubgroupReduceMax:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupReduceUMax;
      break;
    case eCLBuiltinSubgroupReduceMul:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupReduceMul;
      break;
    case eCLBuiltinSubgroupReduceAnd:
      MuxBuiltinID = eMuxBuiltinSubgroupReduceAnd;
      break;
    case eCLBuiltinSubgroupReduceOr:
      MuxBuiltinID = eMuxBuiltinSubgroupReduceOr;
      break;
    case eCLBuiltinSubgroupReduceXor:
      MuxBuiltinID = eMuxBuiltinSubgroupReduceXor;
      break;
    case eCLBuiltinSubgroupReduceLogicalAnd:
      MuxBuiltinID = eMuxBuiltinSubgroupReduceLogicalAnd;
      break;
    case eCLBuiltinSubgroupReduceLogicalOr:
      MuxBuiltinID = eMuxBuiltinSubgroupReduceLogicalOr;
      break;
    case eCLBuiltinSubgroupReduceLogicalXor:
      MuxBuiltinID = eMuxBuiltinSubgroupReduceLogicalXor;
      break;
    case eCLBuiltinSubgroupScanAddInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanAddInclusive;
      break;
    case eCLBuiltinSubgroupScanAddExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanAddExclusive;
      break;
    case eCLBuiltinSubgroupScanMinInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanUMinInclusive;
      break;
    case eCLBuiltinSubgroupScanMinExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanUMinExclusive;
      break;
    case eCLBuiltinSubgroupScanMaxInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanUMaxInclusive;
      break;
    case eCLBuiltinSubgroupScanMaxExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanUMaxExclusive;
      break;
    case eCLBuiltinSubgroupScanMulInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanMulInclusive;
      break;
    case eCLBuiltinSubgroupScanMulExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinSubgroupScanMulExclusive;
      break;
    case eCLBuiltinSubgroupScanAndInclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanAndInclusive;
      break;
    case eCLBuiltinSubgroupScanAndExclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanAndExclusive;
      break;
    case eCLBuiltinSubgroupScanOrInclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanOrInclusive;
      break;
    case eCLBuiltinSubgroupScanOrExclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanOrExclusive;
      break;
    case eCLBuiltinSubgroupScanXorInclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanXorInclusive;
      break;
    case eCLBuiltinSubgroupScanXorExclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanXorExclusive;
      break;
    case eCLBuiltinSubgroupScanLogicalAndInclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanLogicalAndInclusive;
      break;
    case eCLBuiltinSubgroupScanLogicalAndExclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanLogicalAndExclusive;
      break;
    case eCLBuiltinSubgroupScanLogicalOrInclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanLogicalOrInclusive;
      break;
    case eCLBuiltinSubgroupScanLogicalOrExclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanLogicalOrExclusive;
      break;
    case eCLBuiltinSubgroupScanLogicalXorInclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanLogicalXorInclusive;
      break;
    case eCLBuiltinSubgroupScanLogicalXorExclusive:
      MuxBuiltinID = eMuxBuiltinSubgroupScanLogicalXorExclusive;
      break;
    case eCLBuiltinWorkgroupAll:
      MuxBuiltinID = eMuxBuiltinWorkgroupAll;
      break;
    case eCLBuiltinWorkgroupAny:
      MuxBuiltinID = eMuxBuiltinWorkgroupAny;
      break;
    case eCLBuiltinWorkgroupBroadcast:
      MuxBuiltinID = eMuxBuiltinWorkgroupBroadcast;
      break;
    case eCLBuiltinWorkgroupReduceAdd:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceAdd;
      break;
    case eCLBuiltinWorkgroupReduceMin:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceUMin;
      break;
    case eCLBuiltinWorkgroupReduceMax:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceUMax;
      break;
    case eCLBuiltinWorkgroupReduceMul:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceMul;
      break;
    case eCLBuiltinWorkgroupReduceAnd:
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceAnd;
      break;
    case eCLBuiltinWorkgroupReduceOr:
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceOr;
      break;
    case eCLBuiltinWorkgroupReduceXor:
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceXor;
      break;
    case eCLBuiltinWorkgroupReduceLogicalAnd:
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceLogicalAnd;
      break;
    case eCLBuiltinWorkgroupReduceLogicalOr:
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceLogicalOr;
      break;
    case eCLBuiltinWorkgroupReduceLogicalXor:
      MuxBuiltinID = eMuxBuiltinWorkgroupReduceLogicalXor;
      break;
    case eCLBuiltinWorkgroupScanAddInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanAddInclusive;
      break;
    case eCLBuiltinWorkgroupScanAddExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanAddExclusive;
      break;
    case eCLBuiltinWorkgroupScanMinInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanUMinInclusive;
      break;
    case eCLBuiltinWorkgroupScanMinExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanUMinExclusive;
      break;
    case eCLBuiltinWorkgroupScanMaxInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanUMaxInclusive;
      break;
    case eCLBuiltinWorkgroupScanMaxExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanUMaxExclusive;
      break;
    case eCLBuiltinWorkgroupScanMulInclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanMulInclusive;
      break;
    case eCLBuiltinWorkgroupScanMulExclusive:
      RecheckOpType = true;
      MuxBuiltinID = eMuxBuiltinWorkgroupScanMulExclusive;
      break;
    case eCLBuiltinWorkgroupScanAndInclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanAndInclusive;
      break;
    case eCLBuiltinWorkgroupScanAndExclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanAndExclusive;
      break;
    case eCLBuiltinWorkgroupScanOrInclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanOrInclusive;
      break;
    case eCLBuiltinWorkgroupScanOrExclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanOrExclusive;
      break;
    case eCLBuiltinWorkgroupScanXorInclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanXorInclusive;
      break;
    case eCLBuiltinWorkgroupScanXorExclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanXorExclusive;
      break;
    case eCLBuiltinWorkgroupScanLogicalAndInclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanLogicalAndInclusive;
      break;
    case eCLBuiltinWorkgroupScanLogicalAndExclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanLogicalAndExclusive;
      break;
    case eCLBuiltinWorkgroupScanLogicalOrInclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanLogicalOrInclusive;
      break;
    case eCLBuiltinWorkgroupScanLogicalOrExclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanLogicalOrExclusive;
      break;
    case eCLBuiltinWorkgroupScanLogicalXorInclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanLogicalXorInclusive;
      break;
    case eCLBuiltinWorkgroupScanLogicalXorExclusive:
      MuxBuiltinID = eMuxBuiltinWorkgroupScanLogicalXorExclusive;
      break;
  }

  if (RecheckOpType) {
    // We've assumed (unsigned) integer operations, but we may actually have
    // signed integer, or floating point, operations. Refine the builtin ID to
    // the correct 'overload' now.
    compiler::utils::NameMangler Mangler(&F->getContext());
    SmallVector<Type *, 4> ArgumentTypes;
    SmallVector<compiler::utils::TypeQualifiers, 4> Qualifiers;

    Mangler.demangleName(F->getName(), ArgumentTypes, Qualifiers);

    assert(Qualifiers.size() == 1 && ArgumentTypes.size() == 1 &&
           "Unknown collective builtin");
    auto &Qual = Qualifiers[0];

    bool IsSignedInt = false;
    while (!IsSignedInt && Qual.getCount()) {
      IsSignedInt |= Qual.pop_front() == compiler::utils::eTypeQualSignedInt;
    }

    const bool IsFP = ArgumentTypes[0]->isFloatingPointTy();
    switch (MuxBuiltinID) {
      default:
        llvm_unreachable("unknown group operation for which to check the type");
      case eMuxBuiltinSubgroupReduceAdd:
        if (IsFP) MuxBuiltinID = eMuxBuiltinSubgroupReduceFAdd;
        break;
      case eMuxBuiltinSubgroupReduceMul:
        if (IsFP) MuxBuiltinID = eMuxBuiltinSubgroupReduceFMul;
        break;
      case eMuxBuiltinSubgroupReduceUMin:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinSubgroupReduceFMin;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinSubgroupReduceSMin;
        }
        break;
      case eMuxBuiltinSubgroupReduceUMax:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinSubgroupReduceFMax;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinSubgroupReduceSMax;
        }
        break;
      case eMuxBuiltinSubgroupScanAddInclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinSubgroupScanFAddInclusive;
        break;
      case eMuxBuiltinSubgroupScanAddExclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinSubgroupScanFAddExclusive;
        break;
      case eMuxBuiltinSubgroupScanMulInclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinSubgroupScanFMulInclusive;
        break;
      case eMuxBuiltinSubgroupScanMulExclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinSubgroupScanFMulExclusive;
        break;
      case eMuxBuiltinSubgroupScanUMinInclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanFMinInclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanSMinInclusive;
        }
        break;
      case eMuxBuiltinSubgroupScanUMinExclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanFMinExclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanSMinExclusive;
        }
        break;
      case eMuxBuiltinSubgroupScanUMaxInclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanFMaxInclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanSMaxInclusive;
        }
        break;
      case eMuxBuiltinSubgroupScanUMaxExclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanFMaxExclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinSubgroupScanSMaxExclusive;
        }
        break;
      case eMuxBuiltinWorkgroupReduceAdd:
        if (IsFP) MuxBuiltinID = eMuxBuiltinWorkgroupReduceFAdd;
        break;
      case eMuxBuiltinWorkgroupReduceMul:
        if (IsFP) MuxBuiltinID = eMuxBuiltinWorkgroupReduceFMul;
        break;
      case eMuxBuiltinWorkgroupReduceUMin:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinWorkgroupReduceFMin;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinWorkgroupReduceSMin;
        }
        break;
      case eMuxBuiltinWorkgroupReduceUMax:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinWorkgroupReduceFMax;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinWorkgroupReduceSMax;
        }
        break;
      case eMuxBuiltinWorkgroupScanAddInclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinWorkgroupScanFAddInclusive;
        break;
      case eMuxBuiltinWorkgroupScanAddExclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinWorkgroupScanFAddExclusive;
        break;
      case eMuxBuiltinWorkgroupScanMulInclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinWorkgroupScanFMulInclusive;
        break;
      case eMuxBuiltinWorkgroupScanMulExclusive:
        if (IsFP) MuxBuiltinID = eMuxBuiltinWorkgroupScanFMulExclusive;
        break;
      case eMuxBuiltinWorkgroupScanUMinInclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanFMinInclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanSMinInclusive;
        }
        break;
      case eMuxBuiltinWorkgroupScanUMinExclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanFMinExclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanSMinExclusive;
        }
        break;
      case eMuxBuiltinWorkgroupScanUMaxInclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanFMaxInclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanSMaxInclusive;
        }
        break;
      case eMuxBuiltinWorkgroupScanUMaxExclusive:
        if (IsFP) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanFMaxExclusive;
        } else if (IsSignedInt) {
          MuxBuiltinID = eMuxBuiltinWorkgroupScanSMaxExclusive;
        }
        break;
    }
  }

  const bool IsAnyAll = MuxBuiltinID == eMuxBuiltinSubgroupAny ||
                        MuxBuiltinID == eMuxBuiltinSubgroupAll ||
                        MuxBuiltinID == eMuxBuiltinWorkgroupAny ||
                        MuxBuiltinID == eMuxBuiltinWorkgroupAll;
  SmallVector<Type *, 2> OverloadInfo;
  if (!IsAnyAll) {
    OverloadInfo.push_back(CI.getOperand(0)->getType());
  } else {
    OverloadInfo.push_back(IntegerType::getInt1Ty(M.getContext()));
  }

  auto *const MuxBuiltinFn =
      BIMuxImpl.getOrDeclareMuxBuiltin(MuxBuiltinID, M, OverloadInfo);

  assert(MuxBuiltinFn && "Missing mux builtin");
  auto *const SizeTy = getSizeType(M);
  auto *const I32Ty = Type::getInt32Ty(M.getContext());

  SmallVector<Value *, 4> Args;
  if (MuxBuiltinID >= eFirstMuxWorkgroupCollectiveBuiltin &&
      MuxBuiltinID <= eLastMuxWorkgroupCollectiveBuiltin) {
    // Work-group operations have a barrier ID first.
    Args.push_back(ConstantInt::get(I32Ty, 0));
  }
  // Then the arg itself
  // If it's an any/all operation, we must first reduce to i1 because that's how
  // the mux builtins expect their arguments.
  auto *Val = CI.getOperand(0);
  if (!IsAnyAll) {
    Args.push_back(Val);
  } else {
    assert(Val->getType()->isIntegerTy());
    auto *NEZero =
        ICmpInst::Create(Instruction::ICmp, ICmpInst::ICMP_NE, Val,
                         ConstantInt::getNullValue(Val->getType()), "", &CI);
    Args.push_back(NEZero);
  }

  if (MuxBuiltinID == eMuxBuiltinSubgroupBroadcast) {
    // Pass on the ID parameter
    Args.push_back(CI.getOperand(1));
  }
  if (MuxBuiltinID == eMuxBuiltinWorkgroupBroadcast) {
    // The mux version always has three indices. Any missing ones are replaced
    // with zeros
    for (unsigned i = 0, e = CI.arg_size(); i != 3; i++) {
      Args.push_back(1 + i < e ? CI.getOperand(1 + i)
                               : ConstantInt::getNullValue(SizeTy));
    }
  }

  auto *const NewCI = CallInst::Create(MuxBuiltinFn, Args, CI.getName(), &CI);
  NewCI->takeName(&CI);
  NewCI->setAttributes(MuxBuiltinFn->getAttributes());

  if (!IsAnyAll) {
    return NewCI;
  }
  // For any/all we need to recreate the original i32 return value.
  return SExtInst::Create(Instruction::SExt, NewCI, CI.getType(), "sext", &CI);
}

Instruction *CLBuiltinInfo::lowerAsyncBuiltinToMuxBuiltin(
    CallInst &CI, BuiltinID ID, BIMuxInfoConcept &BIMuxImpl) {
  assert((ID == eCLBuiltinAsyncWorkGroupCopy ||
          ID == eCLBuiltinAsyncWorkGroupStridedCopy ||
          ID == eCLBuiltinAsyncWorkGroupCopy2D2D ||
          ID == eCLBuiltinAsyncWorkGroupCopy3D3D) &&
         "Invalid ID");

  IRBuilder<> B(&CI);
  auto &M = *CI.getModule();
  LLVMContext &Ctx = M.getContext();
  const auto &DL = M.getDataLayout();

  switch (ID) {
    default:
      llvm_unreachable("Unhandled builtin");
    case eCLBuiltinAsyncWorkGroupCopy:
    case eCLBuiltinAsyncWorkGroupStridedCopy: {
      NameMangler Mangler(&Ctx);

      // Do a full demangle to determing the pointer element type of the first
      // argument.
      SmallVector<Type *, 4> BuiltinArgTypes, BuiltinArgPointeeTypes;
      SmallVector<compiler::utils::TypeQualifiers, 4> BuiltinArgQuals;

      [[maybe_unused]] const StringRef BuiltinName = Mangler.demangleName(
          CI.getCalledFunction()->getName(), BuiltinArgTypes,
          BuiltinArgPointeeTypes, BuiltinArgQuals);
      assert(!BuiltinName.empty() && BuiltinArgTypes[0]->isPointerTy() &&
             BuiltinArgPointeeTypes[0] && "Could not demangle async builtin");

      auto *const DataTy = BuiltinArgPointeeTypes[0];
      const bool IsStrided = ID == eCLBuiltinAsyncWorkGroupStridedCopy;

      auto *const Dst = CI.getArgOperand(0);
      auto *const Src = CI.getArgOperand(1);
      auto *const NumElements = CI.getArgOperand(2);
      auto *const EventIn = CI.getArgOperand(3 + IsStrided);

      // Find out which way the DMA is going and declare the appropriate mux
      // builtin.
      const bool IsRead = Dst->getType()->getPointerAddressSpace() ==
                          compiler::utils::AddressSpace::Local;
      const auto ElementTypeWidthInBytes =
          DL.getTypeAllocSize(DataTy).getFixedValue();
      auto *const ElementSize =
          ConstantInt::get(NumElements->getType(), ElementTypeWidthInBytes);

      auto *const WidthInBytes =
          IsStrided ? ElementSize
                    : B.CreateMul(ElementSize, NumElements, "width.bytes");

      const BuiltinID MuxBuiltinID = [&] {
        if (IsRead) {
          return IsStrided ? eMuxBuiltinDMARead2D : eMuxBuiltinDMARead1D;
        } else {
          return IsStrided ? eMuxBuiltinDMAWrite2D : eMuxBuiltinDMAWrite1D;
        }
      }();

      auto *const MuxDMA =
          BIMuxImpl.getOrDeclareMuxBuiltin(MuxBuiltinID, M, EventIn->getType());
      assert(MuxDMA && "Could not get/declare mux dma read/write");

      CallInst *NewCI = nullptr;
      if (!IsStrided) {
        NewCI = B.CreateCall(MuxDMA, {Dst, Src, WidthInBytes, EventIn},
                             "mux.out.event");
      } else {
        // The stride from async_work_group_strided_copy is in elements, but the
        // stride in the __mux builtins are in bytes so we need to scale the
        // value.
        auto *const Stride = CI.getArgOperand(3);
        auto *const StrideInBytes =
            B.CreateMul(ElementSize, Stride, "stride.bytes");

        // For async_work_group_strided_copy, the stride only applies to the
        // global memory, as we are doing scatters/gathers.
        auto *const DstStride = IsRead ? ElementSize : StrideInBytes;
        auto *const SrcStride = IsRead ? StrideInBytes : ElementSize;

        NewCI = B.CreateCall(MuxDMA,
                             {Dst, Src, WidthInBytes, DstStride, SrcStride,
                              NumElements, EventIn},
                             "mux.out.event");
      }
      NewCI->setAttributes(MuxDMA->getAttributes());
      NewCI->takeName(&CI);
      return NewCI;
    }
    case eCLBuiltinAsyncWorkGroupCopy2D2D: {
      // Unpack the arguments for ease of access.
      auto *const Dst = CI.getArgOperand(0);
      auto *const DstOffset = CI.getArgOperand(1);
      auto *const Src = CI.getArgOperand(2);
      auto *const SrcOffset = CI.getArgOperand(3);
      auto *const NumBytesPerEl = CI.getArgOperand(4);
      auto *const NumElsPerLine = CI.getArgOperand(5);
      auto *const NumLines = CI.getArgOperand(6);
      auto *const SrcTotalLineLength = CI.getArgOperand(7);
      auto *const DstTotalLineLength = CI.getArgOperand(8);
      auto *const EventIn = CI.getArgOperand(9);

      // Find out which way the DMA is going and declare the appropriate mux
      // builtin.
      const bool IsRead = Dst->getType()->getPointerAddressSpace() ==
                          compiler::utils::AddressSpace::Local;
      auto *const MuxDMA = BIMuxImpl.getOrDeclareMuxBuiltin(
          IsRead ? eMuxBuiltinDMARead2D : eMuxBuiltinDMAWrite2D, M,
          EventIn->getType());
      assert(MuxDMA && "Could not get/declare mux dma read/write");

      auto *const DstOffsetBytes = B.CreateMul(DstOffset, NumBytesPerEl);
      auto *const SrcOffsetBytes = B.CreateMul(SrcOffset, NumBytesPerEl);
      auto *const LineSizeBytes = B.CreateMul(NumElsPerLine, NumBytesPerEl);
      auto *const ByteTy = B.getInt8Ty();
      auto *const DstWithOffset = B.CreateGEP(ByteTy, Dst, DstOffsetBytes);
      auto *const SrcWithOffset = B.CreateGEP(ByteTy, Src, SrcOffsetBytes);
      auto *const SrcStrideBytes =
          B.CreateMul(SrcTotalLineLength, NumBytesPerEl);
      auto *const DstStrideBytes =
          B.CreateMul(DstTotalLineLength, NumBytesPerEl);
      auto *const NewCI = B.CreateCall(
          MuxDMA, {DstWithOffset, SrcWithOffset, LineSizeBytes, DstStrideBytes,
                   SrcStrideBytes, NumLines, EventIn});
      NewCI->setAttributes(MuxDMA->getAttributes());
      NewCI->takeName(&CI);
      return NewCI;
    }
    case eCLBuiltinAsyncWorkGroupCopy3D3D: {
      auto *const Dst = CI.getArgOperand(0);
      auto *const DstOffset = CI.getArgOperand(1);
      auto *const Src = CI.getArgOperand(2);
      auto *const SrcOffset = CI.getArgOperand(3);
      auto *const NumBytesPerEl = CI.getArgOperand(4);
      auto *const NumElsPerLine = CI.getArgOperand(5);
      auto *const NumLines = CI.getArgOperand(6);
      auto *const NumPlanes = CI.getArgOperand(7);
      auto *const SrcTotalLineLength = CI.getArgOperand(8);
      auto *const SrcTotalPlaneArea = CI.getArgOperand(9);
      auto *const DstTotalLineLength = CI.getArgOperand(10);
      auto *const DstTotalPlaneArea = CI.getArgOperand(11);
      auto *const EventIn = CI.getArgOperand(12);

      // Find out which way the DMA is going and declare the appropriate mux
      // builtin.
      const bool IsRead = Dst->getType()->getPointerAddressSpace() ==
                          compiler::utils::AddressSpace::Local;
      auto *const MuxDMA = BIMuxImpl.getOrDeclareMuxBuiltin(
          IsRead ? eMuxBuiltinDMARead3D : eMuxBuiltinDMAWrite3D, M,
          EventIn->getType());
      assert(MuxDMA && "Could not get/declare mux dma read/write");

      auto *const DstOffsetBytes = B.CreateMul(DstOffset, NumBytesPerEl);
      auto *const SrcOffsetBytes = B.CreateMul(SrcOffset, NumBytesPerEl);
      auto *const LineSizeBytes = B.CreateMul(NumElsPerLine, NumBytesPerEl);
      auto *const ByteTy = B.getInt8Ty();
      auto *const DstWithOffset = B.CreateGEP(ByteTy, Dst, DstOffsetBytes);
      auto *const SrcWithOffset = B.CreateGEP(ByteTy, Src, SrcOffsetBytes);
      auto *const SrcLineStrideBytes =
          B.CreateMul(SrcTotalLineLength, NumBytesPerEl);
      auto *const DstLineStrideBytes =
          B.CreateMul(DstTotalLineLength, NumBytesPerEl);
      auto *const SrcPlaneStrideBytes =
          B.CreateMul(SrcTotalPlaneArea, NumBytesPerEl);
      auto *const DstPlaneStrideBytes =
          B.CreateMul(DstTotalPlaneArea, NumBytesPerEl);
      auto *const NewCI =
          B.CreateCall(MuxDMA, {DstWithOffset, SrcWithOffset, LineSizeBytes,
                                DstLineStrideBytes, SrcLineStrideBytes,
                                NumLines, DstPlaneStrideBytes,
                                SrcPlaneStrideBytes, NumPlanes, EventIn});
      NewCI->setAttributes(MuxDMA->getAttributes());
      NewCI->takeName(&CI);
      return NewCI;
    }
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////

Function *CLBuiltinLoader::materializeBuiltin(StringRef BuiltinName,
                                              Module *DestM,
                                              BuiltinMatFlags Flags) {
  auto *const BuiltinModule = this->getBuiltinsModule();

  // Retrieve it from the builtin module.
  if (!BuiltinModule) {
    return nullptr;
  }
  Function *SrcBuiltin = BuiltinModule->getFunction(BuiltinName);
  if (!SrcBuiltin) {
    return nullptr;
  }

  // The user only wants a declaration.
  if (!(Flags & eBuiltinMatDefinition)) {
    if (!DestM) {
      return SrcBuiltin;
    } else {
      FunctionType *FT = dyn_cast<FunctionType>(SrcBuiltin->getFunctionType());
      Function *BuiltinDecl = cast<Function>(
          DestM->getOrInsertFunction(BuiltinName, FT).getCallee());
      BuiltinDecl->copyAttributesFrom(SrcBuiltin);
      BuiltinDecl->setCallingConv(SrcBuiltin->getCallingConv());
      return BuiltinDecl;
    }
  }

  // Materialize the builtin and its callees.
  std::set<Function *> Callees;
  std::vector<Function *> Worklist;
  Worklist.push_back(SrcBuiltin);
  while (!Worklist.empty()) {
    // Materialize the first function in the work list.
    Function *Current = Worklist.front();
    Worklist.erase(Worklist.begin());
    if (!Callees.insert(Current).second) {
      continue;
    }
    if (!BuiltinModule->materialize(Current)) {
      return nullptr;
    }

    // Find any callees in the function and add them to the list.
    for (BasicBlock &BB : *Current) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);
        if (!CI) {
          continue;
        }
        Function *callee = CI->getCalledFunction();
        if (!callee) {
          continue;
        }
        Worklist.push_back(callee);
      }
    }
  }

  if (!DestM) {
    return SrcBuiltin;
  }

  // Copy builtin and callees to the target module if requested by the user.
  ValueToValueMapTy ValueMap;
  SmallVector<ReturnInst *, 4> Returns;
  // Avoid linking errors.
  const GlobalValue::LinkageTypes Linkage = GlobalValue::LinkOnceAnyLinkage;

  // Declare the callees in the module if they don't already exist.
  for (Function *Callee : Callees) {
    Function *NewCallee = DestM->getFunction(Callee->getName());
    if (!NewCallee) {
      FunctionType *FT = Callee->getFunctionType();
      NewCallee = Function::Create(FT, Linkage, Callee->getName(), DestM);
    } else {
      NewCallee->setLinkage(Linkage);
    }
    Function::arg_iterator NewArgI = NewCallee->arg_begin();
    for (Argument &Arg : Callee->args()) {
      NewArgI->setName(Arg.getName());
      ValueMap[&Arg] = &*(NewArgI++);
    }
    NewCallee->copyAttributesFrom(Callee);
    ValueMap[Callee] = NewCallee;
  }

  // Clone the callees' bodies into the module.
  GlobalValueMaterializer Materializer(*DestM);
  for (Function *Callee : Callees) {
    if (Callee->isDeclaration()) {
      continue;
    }
    Function *NewCallee = cast<Function>(ValueMap[Callee]);
    assert(DestM);
    const auto CloneType = DestM == Callee->getParent()
                               ? CloneFunctionChangeType::LocalChangesOnly
                               : CloneFunctionChangeType::DifferentModule;
    CloneFunctionInto(NewCallee, Callee, ValueMap, CloneType, Returns, "",
                      nullptr, nullptr, &Materializer);
    Returns.clear();
  }

  // Clone global variable initializers.
  for (GlobalVariable *var : Materializer.variables()) {
    GlobalVariable *newVar = dyn_cast_or_null<GlobalVariable>(ValueMap[var]);
    if (!newVar) {
      return nullptr;
    }
    Constant *oldInit = var->getInitializer();
    Constant *newInit = MapValue(oldInit, ValueMap);
    newVar->setInitializer(newInit);
  }

  return cast<Function>(ValueMap[SrcBuiltin]);
}
}  // namespace utils
}  // namespace compiler
