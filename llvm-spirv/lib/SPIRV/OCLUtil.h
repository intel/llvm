//===- OCLUtil.h - OCL Utilities declarations -------------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file declares OCL utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_OCLUTIL_H
#define SPIRV_OCLUTIL_H

#include "SPIRVInternal.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Path.h"

#include <atomic>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
using namespace SPIRV;
using namespace llvm;
using namespace spv;

namespace OCLUtil {

///////////////////////////////////////////////////////////////////////////////
//
// Enums
//
///////////////////////////////////////////////////////////////////////////////

enum OCLMemFenceKind {
  OCLMF_Local = 1,
  OCLMF_Global = 2,
  OCLMF_Image = 4,
};

// This enum declares extra constants for OpenCL mem_fence flag. It includes
// combinations of local/global/image flags.
enum OCLMemFenceExtendedKind {
  OCLMFEx_Local = OCLMF_Local,
  OCLMFEx_Global = OCLMF_Global,
  OCLMFEx_Local_Global = OCLMF_Global | OCLMF_Local,
  OCLMFEx_Image = OCLMF_Image,
  OCLMFEx_Image_Local = OCLMF_Image | OCLMF_Local,
  OCLMFEx_Image_Global = OCLMF_Image | OCLMF_Global,
  OCLMFEx_Image_Local_Global = OCLMF_Image | OCLMF_Global | OCLMF_Local,
};

enum OCLScopeKind {
  OCLMS_work_item,
  OCLMS_work_group,
  OCLMS_device,
  OCLMS_all_svm_devices,
  OCLMS_sub_group,
};

// The enum below declares constants corresponding to memory synchronization
// operations constants defined in
// https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/memory_order.html
// To avoid any inconsistence here, constants are explicitly initialized with
// the corresponding constants from 'std::memory_order' enum.
enum OCLMemOrderKind {
  OCLMO_relaxed = std::memory_order::memory_order_relaxed,
  OCLMO_acquire = std::memory_order::memory_order_acquire,
  OCLMO_release = std::memory_order::memory_order_release,
  OCLMO_acq_rel = std::memory_order::memory_order_acq_rel,
  OCLMO_seq_cst = std::memory_order::memory_order_seq_cst
};

enum IntelFPGAMemoryAccessesVal {
  BurstCoalesce = 0x1,
  CacheSizeFlag = 0x2,
  DontStaticallyCoalesce = 0x4,
  PrefetchFlag = 0x8
};

///////////////////////////////////////////////////////////////////////////////
//
// Types
//
///////////////////////////////////////////////////////////////////////////////

typedef SPIRVMap<OCLMemFenceKind, MemorySemanticsMask> OCLMemFenceMap;

typedef SPIRVMap<OCLMemFenceExtendedKind, MemorySemanticsMask>
    OCLMemFenceExtendedMap;

typedef SPIRVMap<OCLMemOrderKind, unsigned, MemorySemanticsMask> OCLMemOrderMap;

typedef SPIRVMap<OCLScopeKind, Scope> OCLMemScopeMap;

typedef SPIRVMap<std::string, SPIRVGroupOperationKind>
    SPIRSPIRVGroupOperationMap;

typedef SPIRVMap<std::string, SPIRVFPRoundingModeKind>
    SPIRSPIRVFPRoundingModeMap;

typedef SPIRVMap<std::string, Op, SPIRVInstruction> OCLSPIRVBuiltinMap;

class OCL12Builtin;
typedef SPIRVMap<std::string, Op, OCL12Builtin> OCL12SPIRVBuiltinMap;

typedef SPIRVMap<std::string, SPIRVBuiltinVariableKind>
    SPIRSPIRVBuiltinVariableMap;

/// Tuple of literals for atomic_work_item_fence (flag, order, scope)
typedef std::tuple<unsigned, OCLMemOrderKind, OCLScopeKind>
    AtomicWorkItemFenceLiterals;

/// Tuple of literals for work_group_barrier or sub_group_barrier
///     (flag, mem_scope, exec_scope)
typedef std::tuple<unsigned, OCLScopeKind, OCLScopeKind> BarrierLiterals;

class OCLOpaqueType;
typedef SPIRVMap<std::string, Op, OCLOpaqueType> OCLOpaqueTypeOpCodeMap;

/// Information for translating OCL builtin.
struct OCLBuiltinTransInfo {
  std::string UniqName;
  std::string MangledName;
  std::string Postfix; // Postfix to be added
  /// Postprocessor of operands
  std::function<void(std::vector<Value *> &)> PostProc;
  Type *RetTy;      // Return type of the translated function
  bool IsRetSigned; // When RetTy is int, determines if extensions
                    // on it should be a sext or zet.
  OCLBuiltinTransInfo() : RetTy(nullptr), IsRetSigned(false) {
    PostProc = [](std::vector<Value *> &) {};
  }
};

///////////////////////////////////////////////////////////////////////////////
//
// Constants
//
///////////////////////////////////////////////////////////////////////////////
namespace kOCLBuiltinName {
const static char All[] = "all";
const static char Any[] = "any";
#define _SPIRV_OP(x, y)                                                        \
  const static char ArbitraryFloat##x##INTEL[] = "intel_arbitrary_float_" #y;
_SPIRV_OP(Cast, cast)
_SPIRV_OP(CastFromInt, cast_from_int)
_SPIRV_OP(CastToInt, cast_to_int)
_SPIRV_OP(Add, add)
_SPIRV_OP(Sub, sub)
_SPIRV_OP(Mul, mul)
_SPIRV_OP(Div, div)
_SPIRV_OP(GT, gt)
_SPIRV_OP(GE, ge)
_SPIRV_OP(LT, lt)
_SPIRV_OP(LE, le)
_SPIRV_OP(EQ, eq)
_SPIRV_OP(Recip, recip)
_SPIRV_OP(RSqrt, rsqrt)
_SPIRV_OP(Cbrt, cbrt)
_SPIRV_OP(Hypot, hypot)
_SPIRV_OP(Sqrt, sqrt)
_SPIRV_OP(Log, log)
_SPIRV_OP(Log2, log2)
_SPIRV_OP(Log10, log10)
_SPIRV_OP(Log1p, log1p)
_SPIRV_OP(Exp, exp)
_SPIRV_OP(Exp2, exp2)
_SPIRV_OP(Exp10, exp10)
_SPIRV_OP(Expm1, expm1)
_SPIRV_OP(Sin, sin)
_SPIRV_OP(Cos, cos)
_SPIRV_OP(SinCos, sincos)
_SPIRV_OP(SinPi, sinpi)
_SPIRV_OP(CosPi, cospi)
_SPIRV_OP(SinCosPi, sincospi)
_SPIRV_OP(ASin, asin)
_SPIRV_OP(ASinPi, asinpi)
_SPIRV_OP(ACos, acos)
_SPIRV_OP(ACosPi, acospi)
_SPIRV_OP(ATan, atan)
_SPIRV_OP(ATanPi, atanpi)
_SPIRV_OP(ATan2, atan2)
_SPIRV_OP(Pow, pow)
_SPIRV_OP(PowR, powr)
_SPIRV_OP(PowN, pown)
#undef _SPIRV_OP
const static char AsyncWorkGroupCopy[] = "async_work_group_copy";
const static char AsyncWorkGroupStridedCopy[] = "async_work_group_strided_copy";
const static char AtomPrefix[] = "atom_";
const static char AtomCmpXchg[] = "atom_cmpxchg";
const static char AtomicPrefix[] = "atomic_";
const static char AtomicCmpXchg[] = "atomic_cmpxchg";
const static char AtomicCmpXchgStrong[] = "atomic_compare_exchange_strong";
const static char AtomicCmpXchgStrongExplicit[] =
    "atomic_compare_exchange_strong_explicit";
const static char AtomicCmpXchgWeak[] = "atomic_compare_exchange_weak";
const static char AtomicCmpXchgWeakExplicit[] =
    "atomic_compare_exchange_weak_explicit";
const static char AtomicInit[] = "atomic_init";
const static char AtomicWorkItemFence[] = "atomic_work_item_fence";
const static char Barrier[] = "barrier";
const static char Clamp[] = "clamp";
const static char ConvertPrefix[] = "convert_";
const static char Dot[] = "dot";
const static char EnqueueKernel[] = "enqueue_kernel";
const static char FixedSqrtINTEL[] = "intel_arbitrary_fixed_sqrt";
const static char FixedRecipINTEL[] = "intel_arbitrary_fixed_recip";
const static char FixedRsqrtINTEL[] = "intel_arbitrary_fixed_rsqrt";
const static char FixedSinINTEL[] = "intel_arbitrary_fixed_sin";
const static char FixedCosINTEL[] = "intel_arbitrary_fixed_cos";
const static char FixedSinCosINTEL[] = "intel_arbitrary_fixed_sincos";
const static char FixedSinPiINTEL[] = "intel_arbitrary_fixed_sinpi";
const static char FixedCosPiINTEL[] = "intel_arbitrary_fixed_cospi";
const static char FixedSinCosPiINTEL[] = "intel_arbitrary_fixed_sincospi";
const static char FixedLogINTEL[] = "intel_arbitrary_fixed_log";
const static char FixedExpINTEL[] = "intel_arbitrary_fixed_exp";
const static char FMax[] = "fmax";
const static char FMin[] = "fmin";
const static char FPGARegIntel[] = "__builtin_intel_fpga_reg";
const static char GetFence[] = "get_fence";
const static char GetImageArraySize[] = "get_image_array_size";
const static char GetImageChannelOrder[] = "get_image_channel_order";
const static char GetImageChannelDataType[] = "get_image_channel_data_type";
const static char GetImageDepth[] = "get_image_depth";
const static char GetImageDim[] = "get_image_dim";
const static char GetImageHeight[] = "get_image_height";
const static char GetImageWidth[] = "get_image_width";
const static char IsFinite[] = "isfinite";
const static char IsNan[] = "isnan";
const static char IsNormal[] = "isnormal";
const static char IsInf[] = "isinf";
const static char Max[] = "max";
const static char MemFence[] = "mem_fence";
const static char ReadMemFence[] = "read_mem_fence";
const static char WriteMemFence[] = "write_mem_fence";
const static char Min[] = "min";
const static char Mix[] = "mix";
const static char NDRangePrefix[] = "ndrange_";
const static char Pipe[] = "pipe";
const static char ReadImage[] = "read_image";
const static char ReadPipe[] = "read_pipe";
const static char ReadPipeBlockingINTEL[] = "read_pipe_bl";
const static char RoundingPrefix[] = "_r";
const static char Sampled[] = "sampled_";
const static char SampledReadImage[] = "sampled_read_image";
const static char Signbit[] = "signbit";
const static char SmoothStep[] = "smoothstep";
const static char Step[] = "step";
const static char SubGroupPrefix[] = "sub_group_";
const static char SubGroupBarrier[] = "sub_group_barrier";
const static char SubPrefix[] = "sub_";
const static char ToGlobal[] = "to_global";
const static char ToLocal[] = "to_local";
const static char ToPrivate[] = "to_private";
const static char VLoadPrefix[] = "vload";
const static char VLoadAPrefix[] = "vloada";
const static char VLoadHalf[] = "vload_half";
const static char VStorePrefix[] = "vstore";
const static char VStoreAPrefix[] = "vstorea";
const static char WaitGroupEvent[] = "wait_group_events";
const static char WriteImage[] = "write_image";
const static char WorkGroupBarrier[] = "work_group_barrier";
const static char WritePipe[] = "write_pipe";
const static char WritePipeBlockingINTEL[] = "write_pipe_bl";
const static char WorkGroupPrefix[] = "work_group_";
const static char WorkGroupAll[] = "work_group_all";
const static char WorkGroupAny[] = "work_group_any";
const static char SubGroupAll[] = "sub_group_all";
const static char SubGroupAny[] = "sub_group_any";
const static char WorkPrefix[] = "work_";
const static char SubgroupBlockReadINTELPrefix[] = "intel_sub_group_block_read";
const static char SubgroupBlockWriteINTELPrefix[] =
    "intel_sub_group_block_write";
const static char SubgroupImageMediaBlockINTELPrefix[] =
    "intel_sub_group_media_block";
const static char LDEXP[] = "ldexp";
} // namespace kOCLBuiltinName

/// Offset for OpenCL image channel order enumeration values.
const unsigned int OCLImageChannelOrderOffset = 0x10B0;

/// Offset for OpenCL image channel data type enumeration values.
const unsigned int OCLImageChannelDataTypeOffset = 0x10D0;

/// OCL 1.x atomic memory order when translated to 2.0 atomics.
const OCLMemOrderKind OCLLegacyAtomicMemOrder = OCLMO_relaxed;

/// OCL 1.x atomic memory scope when translated to 2.0 atomics.
const OCLScopeKind OCLLegacyAtomicMemScope = OCLMS_work_group;

namespace kOCLVer {
const unsigned CL12 = 102000;
const unsigned CL20 = 200000;
const unsigned CL21 = 201000;
const unsigned CL30 = 300000;
} // namespace kOCLVer

namespace OclExt {
// clang-format off
enum Kind {
#define _SPIRV_OP(x) x,
  _SPIRV_OP(cl_images)
  _SPIRV_OP(cl_doubles)
  _SPIRV_OP(cl_khr_int64_base_atomics)
  _SPIRV_OP(cl_khr_int64_extended_atomics)
  _SPIRV_OP(cl_khr_fp16)
  _SPIRV_OP(cl_khr_gl_sharing)
  _SPIRV_OP(cl_khr_gl_event)
  _SPIRV_OP(cl_khr_d3d10_sharing)
  _SPIRV_OP(cl_khr_media_sharing)
  _SPIRV_OP(cl_khr_d3d11_sharing)
  _SPIRV_OP(cl_khr_global_int32_base_atomics)
  _SPIRV_OP(cl_khr_global_int32_extended_atomics)
  _SPIRV_OP(cl_khr_local_int32_base_atomics)
  _SPIRV_OP(cl_khr_local_int32_extended_atomics)
  _SPIRV_OP(cl_khr_byte_addressable_store)
  _SPIRV_OP(cl_khr_3d_image_writes)
  _SPIRV_OP(cl_khr_gl_msaa_sharing)
  _SPIRV_OP(cl_khr_depth_images)
  _SPIRV_OP(cl_khr_gl_depth_images)
  _SPIRV_OP(cl_khr_subgroups)
  _SPIRV_OP(cl_khr_mipmap_image)
  _SPIRV_OP(cl_khr_mipmap_image_writes)
  _SPIRV_OP(cl_khr_egl_event)
  _SPIRV_OP(cl_khr_srgb_image_writes)
  _SPIRV_OP(cl_khr_extended_bit_ops)
#undef _SPIRV_OP
};
// clang-format on
} // namespace OclExt
namespace kOCLSubgroupsAVCIntel {
const static char Prefix[] = "intel_sub_group_avc_";
const static char MCEPrefix[] = "intel_sub_group_avc_mce_";
const static char IMEPrefix[] = "intel_sub_group_avc_ime_";
const static char REFPrefix[] = "intel_sub_group_avc_ref_";
const static char SICPrefix[] = "intel_sub_group_avc_sic_";
const static char TypePrefix[] = "opencl.intel_sub_group_avc_";
} // namespace kOCLSubgroupsAVCIntel

///////////////////////////////////////////////////////////////////////////////
//
// Functions
//
///////////////////////////////////////////////////////////////////////////////

/// Get instruction index for SPIR-V extended instruction for OpenCL.std
///   extended instruction set.
/// \param MangledName The mangled name of OpenCL builtin function.
/// \param DemangledName The demangled name of OpenCL builtin function if
///   not empty.
/// \return instruction index of extended instruction if the OpenCL builtin
///   function is translated to an extended instruction, otherwise ~0U.
unsigned getExtOp(StringRef MangledName, StringRef DemangledName = "");

/// Get literal arguments of call of atomic_work_item_fence.
AtomicWorkItemFenceLiterals getAtomicWorkItemFenceLiterals(CallInst *CI);

/// Get literal arguments of call of work_group_barrier or sub_group_barrier.
BarrierLiterals getBarrierLiterals(CallInst *CI);

/// Get number of memory order arguments for atomic builtin function.
size_t getAtomicBuiltinNumMemoryOrderArgs(StringRef Name);

/// Get number of memory order arguments for spirv atomic builtin function.
size_t getSPIRVAtomicBuiltinNumMemoryOrderArgs(Op OC);

/// Return true for OpenCL builtins which do compute operations
/// (like add, sub, min, max, inc, dec, ...) atomically
bool isComputeAtomicOCLBuiltin(StringRef DemangledName);

/// Get OCL version from metadata opencl.ocl.version.
/// \param AllowMulti Allows multiple operands if true.
/// \return OCL version encoded as Major*10^5+Minor*10^3+Rev,
/// e.g. 201000 for OCL 2.1, 200000 for OCL 2.0, 102000 for OCL 1.2,
/// 0 if metadata not found.
/// If there are multiple operands, check they are identical.
unsigned getOCLVersion(Module *M, bool AllowMulti = false);

/// Encode OpenCL version as Major*10^5+Minor*10^3+Rev.
unsigned encodeOCLVer(unsigned short Major, unsigned char Minor,
                      unsigned char Rev);

/// Decode OpenCL version which is encoded as Major*10^5+Minor*10^3+Rev
std::tuple<unsigned short, unsigned char, unsigned char>
decodeOCLVer(unsigned Ver);

/// Decode a MDNode assuming it contains three integer constants.
void decodeMDNode(MDNode *N, unsigned &X, unsigned &Y, unsigned &Z);

/// Get full path from debug info metadata
/// Return empty string if the path is not available.
template <typename T> std::string getFullPath(const T *Scope) {
  if (!Scope)
    return std::string();
  std::string Filename = Scope->getFilename().str();
  if (sys::path::is_absolute(Filename))
    return Filename;
  SmallString<16> DirName = Scope->getDirectory();
  sys::path::append(DirName, sys::path::Style::posix, Filename);
  return DirName.str().str();
}

/// Decode OpenCL vector type hint MDNode and encode it as SPIR-V execution
/// mode VecTypeHint.
unsigned transVecTypeHint(MDNode *Node);

/// Decode SPIR-V encoding of vector type hint execution mode.
Type *decodeVecTypeHint(LLVMContext &C, unsigned Code);

SPIRAddressSpace getOCLOpaqueTypeAddrSpace(Op OpCode);
SPIR::TypeAttributeEnum getOCLOpaqueTypeAddrSpace(SPIR::TypePrimitiveEnum Prim);

inline unsigned mapOCLMemSemanticToSPIRV(unsigned MemFenceFlag,
                                         OCLMemOrderKind Order) {
  return OCLMemOrderMap::map(Order) | mapBitMask<OCLMemFenceMap>(MemFenceFlag);
}

inline unsigned mapOCLMemFenceFlagToSPIRV(unsigned MemFenceFlag) {
  return mapBitMask<OCLMemFenceMap>(MemFenceFlag);
}

inline std::pair<unsigned, OCLMemOrderKind>
mapSPIRVMemSemanticToOCL(unsigned Sema) {
  return std::make_pair(
      rmapBitMask<OCLMemFenceMap>(Sema),
      OCLMemOrderMap::rmap(extractSPIRVMemOrderSemantic(Sema)));
}

inline OCLMemOrderKind mapSPIRVMemOrderToOCL(unsigned Sema) {
  return OCLMemOrderMap::rmap(extractSPIRVMemOrderSemantic(Sema));
}

/// Mutate call instruction to call OpenCL builtin function.
CallInst *mutateCallInstOCL(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &)> ArgMutate,
    AttributeList *Attrs = nullptr);

/// Mutate call instruction to call OpenCL builtin function.
Instruction *mutateCallInstOCL(
    Module *M, CallInst *CI,
    std::function<std::string(CallInst *, std::vector<Value *> &, Type *&RetTy)>
        ArgMutate,
    std::function<Instruction *(CallInst *)> RetMutate,
    AttributeList *Attrs = nullptr, bool TakeFuncName = false);

/// Check if instruction is bitcast from spirv.ConstantSampler to spirv.Sampler
bool isSamplerInitializer(Instruction *Inst);

/// Check if instruction is bitcast from spirv.ConstantPipeStorage
/// to spirv.PipeStorage
bool isPipeStorageInitializer(Instruction *Inst);

/// Check (isSamplerInitializer || isPipeStorageInitializer)
bool isSpecialTypeInitializer(Instruction *Inst);

bool isPipeOrAddressSpaceCastBI(const StringRef MangledName);
bool isEnqueueKernelBI(const StringRef MangledName);
bool isKernelQueryBI(const StringRef MangledName);

/// Check that the type is the sampler_t
bool isSamplerTy(Type *Ty);

// Checks if the binary operator is an unfused fmul + fadd instruction.
bool isUnfusedMulAdd(BinaryOperator *B);

// Get data and vector size postfix for sugroup_block_{read|write} builtins
// as specified by cl_intel_subgroups* extensions.
// Scalar data assumed to be represented as vector of one element.
std::string getIntelSubgroupBlockDataPostfix(unsigned ElementBitSize,
                                             unsigned VectorNumElements);

void insertImageNameAccessQualifier(SPIRVAccessQualifierKind Acc,
                                    std::string &Name);
} // namespace OCLUtil

using namespace OCLUtil;
namespace SPIRV {

template <class KeyTy, class ValTy, class Identifier = void>
Instruction *
getOrCreateSwitchFunc(StringRef MapName, Value *V,
                      const SPIRVMap<KeyTy, ValTy, Identifier> &Map,
                      bool IsReverse, Optional<int> DefaultCase,
                      Instruction *InsertPoint, int KeyMask = 0) {
  static_assert(std::is_convertible<KeyTy, int>::value &&
                    std::is_convertible<ValTy, int>::value,
                "Can map only integer values");
  Type *Ty = V->getType();
  assert(Ty && Ty->isIntegerTy() && "Can't map non-integer types");
  Module *M = InsertPoint->getModule();
  Function *F = getOrCreateFunction(M, Ty, Ty, MapName);
  if (!F->empty()) // The switch function already exists. just call it.
    return addCallInst(M, MapName, Ty, V, nullptr, InsertPoint);

  F->setLinkage(GlobalValue::PrivateLinkage);

  LLVMContext &Ctx = M->getContext();
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  IRBuilder<> IRB(BB);
  SwitchInst *SI;
  F->arg_begin()->setName("key");
  if (KeyMask) {
    Value *MaskV = ConstantInt::get(Type::getInt32Ty(Ctx), KeyMask);
    Value *NewKey = IRB.CreateAnd(MaskV, F->arg_begin());
    NewKey->setName("key.masked");
    SI = IRB.CreateSwitch(NewKey, BB);
  } else {
    SI = IRB.CreateSwitch(F->arg_begin(), BB);
  }

  if (!DefaultCase) {
    BasicBlock *DefaultBB = BasicBlock::Create(Ctx, "default", F);
    IRBuilder<> DefaultIRB(DefaultBB);
    DefaultIRB.CreateUnreachable();
    SI->setDefaultDest(DefaultBB);
  }

  Map.foreach ([&](int Key, int Val) {
    if (IsReverse)
      std::swap(Key, Val);
    BasicBlock *CaseBB = BasicBlock::Create(Ctx, "case." + Twine(Key), F);
    IRBuilder<> CaseIRB(CaseBB);
    CaseIRB.CreateRet(CaseIRB.getInt32(Val));
    SI->addCase(IRB.getInt32(Key), CaseBB);
    if (Key == DefaultCase)
      SI->setDefaultDest(CaseBB);
  });
  assert(SI->getDefaultDest() != BB && "Invalid default destination in switch");
  return addCallInst(M, MapName, Ty, V, nullptr, InsertPoint);
}

/// Performs conversion from OpenCL memory_scope into SPIR-V Scope.
///
/// Supports both constant and non-constant values. To handle the latter case,
/// function with switch..case statement will be inserted into module which
/// \arg InsertBefore belongs to (in order to perform mapping at runtime)
///
/// \param [in] MemScope memory_scope value which needs to be translated
/// \param [in] DefaultCase default value for switch..case construct if
///             dynamic mapping is used
/// \param [in] InsertBefore insertion point for call into conversion function
///             which is generated if \arg MemScope is not a constant
/// \returns \c Value corresponding to SPIR-V Scope equivalent to OpenCL
///          memory_scope passed in \arg MemScope
Value *transOCLMemScopeIntoSPIRVScope(Value *MemScope,
                                      Optional<int> DefaultCase,
                                      Instruction *InsertBefore);

/// Performs conversion from OpenCL memory_order into SPIR-V Memory Semantics.
///
/// Supports both constant and non-constant values. To handle the latter case,
/// function with switch..case statement will be inserted into module which
/// \arg InsertBefore belongs to (in order to perform mapping at runtime)
///
/// \param [in] MemOrder memory_scope value which needs to be translated
/// \param [in] DefaultCase default value for switch..case construct if
///             dynamic mapping is used
/// \param [in] InsertBefore insertion point for call into conversion function
///             which is generated if \arg MemOrder is not a constant
/// \returns \c Value corresponding to SPIR-V Memory Semantics equivalent to
///          OpenCL memory_order passed in \arg MemOrder
Value *transOCLMemOrderIntoSPIRVMemorySemantics(Value *MemOrder,
                                                Optional<int> DefaultCase,
                                                Instruction *InsertBefore);

/// Performs conversion from SPIR-V Scope into OpenCL memory_scope.
///
/// Supports both constant and non-constant values. To handle the latter case,
/// function with switch..case statement will be inserted into module which
/// \arg InsertBefore belongs to (in order to perform mapping at runtime)
///
/// \param [in] MemScope Scope value which needs to be translated
/// \param [in] InsertBefore insertion point for call into conversion function
///             which is generated if \arg MemScope is not a constant
/// \returns \c Value corresponding to  OpenCL memory_scope equivalent to SPIR-V
///          Scope passed in \arg MemScope
Value *transSPIRVMemoryScopeIntoOCLMemoryScope(Value *MemScope,
                                               Instruction *InsertBefore);

/// Performs conversion from SPIR-V Memory Semantics into OpenCL memory_order.
///
/// Supports both constant and non-constant values. To handle the latter case,
/// function with switch..case statement will be inserted into module which
/// \arg InsertBefore belongs to (in order to perform mapping at runtime)
///
/// \param [in] MemorySemantics Memory Semantics value which needs to be
///             translated
/// \param [in] InsertBefore insertion point for call into conversion function
///             which is generated if \arg MemorySemantics is not a constant
/// \returns \c Value corresponding to  OpenCL memory_order equivalent to SPIR-V
///          Memory Semantics passed in \arg MemorySemantics
Value *transSPIRVMemorySemanticsIntoOCLMemoryOrder(Value *MemorySemantics,
                                                   Instruction *InsertBefore);

/// Performs conversion from SPIR-V Memory Semantics into OpenCL
/// mem_fence_flags.
///
/// Supports both constant and non-constant values. To handle the latter case,
/// function with switch..case statement will be inserted into module which
/// \arg InsertBefore belongs to (in order to perform mapping at runtime)
///
/// \param [in] MemorySemantics Memory Semantics value which needs to be
///             translated
/// \param [in] InsertBefore insertion point for call into conversion function
///             which is generated if \arg MemorySemantics is not a constant
/// \returns \c Value corresponding to  OpenCL mem_fence_flags equivalent to
///          SPIR-V Memory Semantics passed in \arg MemorySemantics
Value *transSPIRVMemorySemanticsIntoOCLMemFenceFlags(Value *MemorySemantics,
                                                     Instruction *InsertBefore);

class SPIRVSubgroupsAVCIntelInst;
typedef SPIRVMap<std::string, Op, SPIRVSubgroupsAVCIntelInst>
    OCLSPIRVSubgroupAVCIntelBuiltinMap;

typedef SPIRVMap<AtomicRMWInst::BinOp, Op> LLVMSPIRVAtomicRmwOpCodeMap;

class SPIRVFixedPointIntelInst;
template <>
inline void SPIRVMap<std::string, Op, SPIRVFixedPointIntelInst>::init() {
#define _SPIRV_OP(x, y) add("intel_arbitrary_fixed_" #x, OpFixed##y##INTEL);
  _SPIRV_OP(sqrt, Sqrt)
  _SPIRV_OP(recip, Recip)
  _SPIRV_OP(rsqrt, Rsqrt)
  _SPIRV_OP(sin, Sin)
  _SPIRV_OP(cos, Cos)
  _SPIRV_OP(sincos, SinCos)
  _SPIRV_OP(sinpi, SinPi)
  _SPIRV_OP(cospi, CosPi)
  _SPIRV_OP(sincospi, SinCosPi)
  _SPIRV_OP(log, Log)
  _SPIRV_OP(exp, Exp)
#undef _SPIRV_OP
}
typedef SPIRVMap<std::string, Op, SPIRVFixedPointIntelInst>
    SPIRVFixedPointIntelMap;

class SPIRVArbFloatIntelInst;
template <>
inline void SPIRVMap<std::string, Op, SPIRVArbFloatIntelInst>::init() {
#define _SPIRV_OP(x, y)                                                        \
  add("intel_arbitrary_float_" #y, OpArbitraryFloat##x##INTEL);
  _SPIRV_OP(Cast, cast)
  _SPIRV_OP(CastFromInt, cast_from_int)
  _SPIRV_OP(CastToInt, cast_to_int)
  _SPIRV_OP(Add, add)
  _SPIRV_OP(Sub, sub)
  _SPIRV_OP(Mul, mul)
  _SPIRV_OP(Div, div)
  _SPIRV_OP(GT, gt)
  _SPIRV_OP(GE, ge)
  _SPIRV_OP(LT, lt)
  _SPIRV_OP(LE, le)
  _SPIRV_OP(EQ, eq)
  _SPIRV_OP(Recip, recip)
  _SPIRV_OP(RSqrt, rsqrt)
  _SPIRV_OP(Cbrt, cbrt)
  _SPIRV_OP(Hypot, hypot)
  _SPIRV_OP(Sqrt, sqrt)
  _SPIRV_OP(Log, log)
  _SPIRV_OP(Log2, log2)
  _SPIRV_OP(Log10, log10)
  _SPIRV_OP(Log1p, log1p)
  _SPIRV_OP(Exp, exp)
  _SPIRV_OP(Exp2, exp2)
  _SPIRV_OP(Exp10, exp10)
  _SPIRV_OP(Expm1, expm1)
  _SPIRV_OP(Sin, sin)
  _SPIRV_OP(Cos, cos)
  _SPIRV_OP(SinCos, sincos)
  _SPIRV_OP(SinPi, sinpi)
  _SPIRV_OP(CosPi, cospi)
  _SPIRV_OP(SinCosPi, sincospi)
  _SPIRV_OP(ASin, asin)
  _SPIRV_OP(ASinPi, asinpi)
  _SPIRV_OP(ACos, acos)
  _SPIRV_OP(ACosPi, acospi)
  _SPIRV_OP(ATan, atan)
  _SPIRV_OP(ATanPi, atanpi)
  _SPIRV_OP(ATan2, atan2)
  _SPIRV_OP(Pow, pow)
  _SPIRV_OP(PowR, powr)
  _SPIRV_OP(PowN, pown)
#undef _SPIRV_OP
}
typedef SPIRVMap<std::string, Op, SPIRVArbFloatIntelInst> SPIRVArbFloatIntelMap;

} // namespace SPIRV

#endif // SPIRV_OCLUTIL_H
