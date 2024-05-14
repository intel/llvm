//===- SPIRVToOCL.h - Converts SPIR-V to LLVM ------------------*- C++ -*-===//
//
//                     The LLVM/SPIR-V Translator
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
/// \file
///
/// This file contains declaration of SPIRVToOCL class which implements
/// common transform of SPIR-V builtins to OCL builtins.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRVTOOCL_H
#define SPIRVTOOCL_H

#include "OCLUtil.h"
#include "SPIRVBuiltinHelper.h"
#include "SPIRVInternal.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

#include <string>

namespace SPIRV {

class SPIRVToOCLBase : public InstVisitor<SPIRVToOCLBase>,
                       protected BuiltinCallHelper {
public:
  SPIRVToOCLBase()
      : BuiltinCallHelper(ManglingRules::OpenCL, translateOpaqueType),
        M(nullptr), Ctx(nullptr) {}
  virtual ~SPIRVToOCLBase() {}

  virtual bool runSPIRVToOCL(Module &M) = 0;

  void visitCallInst(CallInst &CI);

  // SPIR-V reader should translate vector casts into OCL built-ins because
  // such conversions are not defined neither by OpenCL C/C++ nor
  // by SPIR 1.2/2.0 standards. So, it is safer to convert such casts into
  // appropriate calls to conversion built-ins defined by the standards.
  void visitCastInst(CastInst &CI);

  /// Transform __spirv_ImageQuerySize[Lod] into vector of the same length
  /// containing {[get_image_width | get_image_dim], get_image_array_size}
  /// for all images except image1d_t which is always converted into
  /// get_image_width returning scalar result.
  void visitCallSPIRVImageQuerySize(CallInst *CI);

  /// Transform __spirv_(NonUniform)Group* to {work_group|sub_group}_*.
  ///
  /// Special handling of work_group_broadcast.
  ///   __spirv_GroupBroadcast(a, vec3(x, y, z))
  ///     =>
  ///   work_group_broadcast(a, x, y, z)
  ///
  /// Special handling of sub_group_all, sub_group_any,
  /// sub_group_non_uniform_all, sub_group_non_uniform_any, sub_group_ballot,
  /// sub_group_clustered_logical_[and/or/xor].
  ///   retTy func(i1 arg)
  ///     =>
  ///   retTy func(i32 arg)
  ///
  /// Special handling of sub_group_all, sub_group_any,
  /// sub_group_non_uniform_all,
  /// sub_group_non_uniform_any, sub_group_non_uniform_all_equal.
  ///   i1 func
  ///     =>
  ///   i32 func
  void visitCallSPIRVGroupBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_{PipeOpName} to OCL pipe builtin functions.
  void visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_OpOpSubgroupImageMediaBlockReadINTEL =>
  ///  intel_sub_group_media_block_read
  ///           __spirv_OpSubgroupImageMediaBlockWriteINTEL =>
  ///  intel_sub_group_media_block_write
  void visitCallSPIRVImageMediaBlockBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_OpGenericCastToPtrExplicit_To{Global|Local|Private} to
  /// to_{global|local|private} OCL builtin.
  void visitCallGenericCastToPtrExplicitBuiltIn(CallInst *CI, Op OC);

  /// Transform __spirv_OpBuildINDRange_{1|2|3}D to
  /// ndrange_{1|2|3}D OCL builtin.
  void visitCallBuildNDRangeBuiltIn(CallInst *CI, Op OC,
                                    StringRef DemangledName);

  /// Transform __spirv_*Convert_R{ReturnType}{_sat}{_rtp|_rtn|_rtz|_rte} to
  /// convert_{ReturnType}_{sat}{_rtp|_rtn|_rtz|_rte}
  /// example:  <2 x i8> __spirv_SatConvertUToS(<2 x i32>) =>
  ///   convert_uchar2_sat(int2)
  void visitCallSPIRVCvtBuiltin(CallInst *CI, Op OC, StringRef DemangledName);

  /// Transform
  ///   __spirv_AsyncGroupCopy(ScopeWorkGroup, dst, src, n, stride, event)
  ///   => async_work_group_strided_copy(dst, src, n, stride, event)
  void visitCallAsyncWorkGroupCopy(CallInst *CI, Op OC);

  /// Transform __spirv_GroupWaitEvents(Scope, NumEvents, EventsList)
  ///   => wait_group_events(NumEvents, EventsList)
  void visitCallGroupWaitEvents(CallInst *CI, Op OC);

  /// Transform __spirv_ImageSampleExplicitLod__{ReturnType} to read_imade
  void visitCallSPIRVImageSampleExplicitLodBuiltIn(CallInst *CI, Op OC);

  /// Transform __spirv_ImageWrite to write_image
  void visitCallSPIRVImageWriteBuiltIn(CallInst *CI, Op OC);

  /// Transform __spirv_ImageRead to read_image
  void visitCallSPIRVImageReadBuiltIn(CallInst *CI, Op OC);

  /// Transform __spirv_ImageQueryOrder to get_image_channel_order
  //            __spirv_ImageQueryFormat to get_image_channel_data_type
  void visitCallSPIRVImageQueryBuiltIn(CallInst *CI, Op OC);

  /// Transform subgroup Intel opcodes
  /// example: __spirv_SubgroupBlockWriteINTEL
  ///    =>    intel_sub_group_block_write_ul
  void visitCallSPIRVSubgroupINTELBuiltIn(CallInst *CI, Op OC);

  /// Transform AVC INTEL Evaluate opcodes
  /// example: __spirv_SubgroupAvcImeEvaluateWithSingleReference
  ///    => intel_sub_group_avc_ime_evaluate_with_single_reference
  void visitCallSPIRVAvcINTELEvaluateBuiltIn(CallInst *CI, Op OC);

  /// Transform AVC INTEL general opcodes
  /// example: __spirv_SubgroupAvcMceGetDefaultInterBaseMultiReferencePenalty
  ///   =>
  ///   intel_sub_group_avc_mce_get_default_inter_base_multi_reference_penalty
  void visitCallSPIRVAvcINTELInstructionBuiltin(CallInst *CI, Op OC);

  ///  Transform __spirv_GenericPtrMemSemantics to:
  ///  %0 = call spirv_func i32 @_Z9get_fence
  ///  %1 = shl i31 %0, 8
  void visitCallSPIRVGenericPtrMemSemantics(CallInst *CI);

  /// Transform __spirv_ConvertFToBF16INTELDv(N)_f to:
  /// intel_convert_bfloat16(N)_as_ushort(N)Dv(N)_f;
  /// and transform __spirv_ConvertBF16ToFINTELDv(N)_s to:
  /// intel_convert_as_bfloat16(N)_float(N)Dv(N)_t;
  /// where N is vector size
  void visitCallSPIRVBFloat16Conversions(CallInst *CI, Op OC);

  /// Transform __spirv_* builtins to OCL 2.0 builtins.
  /// No change with arguments.
  void visitCallSPIRVBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_* builtins (originates from builtin variables) to OCL
  /// builtins.
  /// No change with arguments.
  /// e.g.
  /// _Z33__spirv_BuiltInGlobalInvocationIdi(x) -> get_global_id(x)
  void visitCallSPIRVBuiltin(CallInst *CI, SPIRVBuiltinVariableKind Kind);

  /// Transform __spirv_ocl* instructions (OpenCL Extended Instruction Set)
  /// to OpenCL builtins.
  void visitCallSPIRVOCLExt(CallInst *CI, OCLExtOpKind Kind);

  /// Transform __spirv_ocl_vstore* to corresponding vstore OpenCL instruction
  void visitCallSPIRVVStore(CallInst *CI, OCLExtOpKind Kind);

  /// Transform __spirv_ocl_vloadn to OpenCL vload[2|4|8|16]
  void visitCallSPIRVVLoadn(CallInst *CI, OCLExtOpKind Kind);

  /// Transform __spirv_ocl_printf to (i8 addrspace(2)*, ...) @printf
  void visitCallSPIRVPrintf(CallInst *CI, OCLExtOpKind Kind);

  /// Get prefix work_/sub_ for OCL group builtin functions.
  /// Assuming the first argument of \param CI is a constant integer for
  /// workgroup/subgroup scope enums.
  std::string getGroupBuiltinPrefix(CallInst *CI);

  /// Transform __spirv_OpAtomicCompareExchange and
  /// __spirv_OpAtomicCompareExchangeWeak
  virtual void visitCallSPIRVAtomicCmpExchg(CallInst *CI) = 0;

  /// Transform __spirv_OpAtomicIIncrement/OpAtomicIDecrement to:
  /// - OCL2.0: atomic_fetch_add_explicit/atomic_fetch_sub_explicit
  /// - OCL1.2: atomic_inc/atomic_dec
  virtual void visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_Atomic* to atomic_*.
  ///   __spirv_Atomic*(atomic_op, scope, sema, ops, ...) =>
  ///      atomic_*(atomic_op, ops, ..., order(sema), map(scope))
  virtual void visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_MemoryBarrier to:
  /// - OCL2.0: atomic_work_item_fence.__spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  /// - OCL1.2: mem_fence
  virtual void visitCallSPIRVMemoryBarrier(CallInst *CI) = 0;

  /// Transform __spirv_ControlBarrier to:
  /// - OCL2.0: work_group_barrier or sub_group barrier
  /// - OCL1.2: barrier
  virtual void visitCallSPIRVControlBarrier(CallInst *CI) = 0;

  /// Transform split __spirv_ControlBarrierArriveINTEL and
  /// __spirv_ControlBarrierWaitINTEL barrier to:
  /// - OCL2.0: overload with a memory_scope argument
  /// - OCL1.2: overload with no memory_scope argument
  virtual void visitCallSPIRVSplitBarrierINTEL(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_EnqueueKernel to __enqueue_kernel
  virtual void visitCallSPIRVEnqueueKernel(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_Any and __spirv_All to OpenCL builtin.
  void visitCallSPIRVAnyAll(CallInst *CI, Op OC);

  /// Transform relational builtin, e.g. __spirv_IsNan, to OpenCL builtin.
  void visitCallSPIRVRelational(CallInst *CI, Op OC);

  /// Conduct generic mutations for all atomic builtins
  virtual CallInst *mutateCommonAtomicArguments(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_Opcode to ocl-version specific builtin name
  /// using separate maps for OpenCL 1.2 and OpenCL 2.0
  virtual void mutateAtomicName(CallInst *CI, Op OC) = 0;

  // Transform FP atomic opcode to corresponding OpenCL function name
  virtual std::string mapFPAtomicName(Op OC) = 0;

  void translateOpaqueTypes();

private:
  /// Transform uniform group opcode to corresponding OpenCL function name,
  /// example: GroupIAdd(Reduce) => group_iadd => work_group_reduce_add |
  /// sub_group_reduce_add
  std::string getUniformArithmeticBuiltinName(CallInst *CI, Op OC);
  /// Transform non-uniform group opcode to corresponding OpenCL function name,
  /// example: GroupNonUniformIAdd(Reduce) => group_non_uniform_iadd =>
  /// sub_group_non_uniform_reduce_add
  std::string getNonUniformArithmeticBuiltinName(CallInst *CI, Op OC);
  /// Transform ballot bit count opcode to corresponding OpenCL function name,
  /// example: GroupNonUniformBallotBitCount(Reduce) =>
  /// group_ballot_bit_count_iadd => sub_group_ballot_bit_count
  std::string getBallotBuiltinName(CallInst *CI, Op OC);
  /// Transform OpGroupNonUniformRotateKHR to corresponding OpenCL function
  /// name.
  std::string getRotateBuiltinName(CallInst *CI, Op OC);
  /// Transform group opcode to corresponding OpenCL function name
  std::string groupOCToOCLBuiltinName(CallInst *CI, Op OC);
  /// Transform SPV-IR image opaque type into OpenCL representation,
  /// example: spirv.Image._void_1_0_0_0_0_0_1 => opencl.image2d_wo_t
  static std::string
  getOCLImageOpaqueType(SmallVector<std::string, 8> &Postfixes);
  /// Transform SPV-IR pipe opaque type into OpenCL representation,
  /// example: spirv.Pipe._0 => opencl.pipe_ro_t
  static std::string
  getOCLPipeOpaqueType(SmallVector<std::string, 8> &Postfixes);

  static std::string translateOpaqueType(StringRef STName);

  /// Mutate the call instruction based on (optional) image operands at position
  /// ImOpArgIndex. The new function name will be based on NewFuncName, and the
  /// type suffix based on Ty and whether the image operand was signed.
  BuiltinCallMutator mutateCallImageOperands(CallInst *CI,
                                             StringRef NewFuncName, Type *Ty,
                                             unsigned ImOpArgIndex);

protected:
  Module *M;
  LLVMContext *Ctx;
};

class SPIRVToOCLLegacy : public ModulePass {
protected:
  SPIRVToOCLLegacy(char &ID) : ModulePass(ID) {}

public:
  bool runOnModule(Module &M) override = 0;
};

class SPIRVToOCL12Base : public SPIRVToOCLBase {
public:
  bool runSPIRVToOCL(Module &M) override;

  /// Transform __spirv_MemoryBarrier to atomic_work_item_fence.
  ///   __spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  void visitCallSPIRVMemoryBarrier(CallInst *CI) override;

  /// Transform __spirv_ControlBarrier to barrier.
  ///   __spirv_ControlBarrier(execScope, memScope, sema) =>
  ///       barrier(flag(sema))
  void visitCallSPIRVControlBarrier(CallInst *CI) override;

  /// Transform split __spirv_ControlBarrierArriveINTEL and
  /// __spirv_ControlBarrierWaitINTEL barrier to overloads without a
  /// memory_scope argument.
  void visitCallSPIRVSplitBarrierINTEL(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomic functions. It firstly conduct generic
  /// mutations for all builtins and then mutate some of them seperately
  void visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicIIncrement / OpAtomicIDecrement to
  /// atomic_inc / atomic_dec
  void visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicUMin/SMin/UMax/SMax into
  /// atomic_min/atomic_max, as there is no distinction in OpenCL 1.2
  /// between signed and unsigned version of those functions
  void visitCallSPIRVAtomicUMinUMax(CallInst *CI, Op OC);

  /// Transform __spirv_OpAtomicLoad to atomic_add(*ptr, 0)
  void visitCallSPIRVAtomicLoad(CallInst *CI);

  /// Transform __spirv_OpAtomicStore to atomic_xchg(*ptr, value)
  void visitCallSPIRVAtomicStore(CallInst *CI);

  /// Transform __spirv_OpAtomicFlagClear to atomic_xchg(*ptr, 0)
  /// with ignoring the result
  void visitCallSPIRVAtomicFlagClear(CallInst *CI);

  /// Transform __spirv_OpAtomicFlagTestAndTest to
  /// (bool)atomic_xchg(*ptr, 1)
  void visitCallSPIRVAtomicFlagTestAndSet(CallInst *CI);

  /// Transform __spirv_OpAtomicCompareExchange/Weak into atomic_cmpxchg
  /// OpAtomicCompareExchangeWeak is not "weak" at all, but instead has
  /// the same semantics as OpAtomicCompareExchange.
  void visitCallSPIRVAtomicCmpExchg(CallInst *CI) override;

  /// Trigger assert, since OpenCL 1.2 doesn't support enqueue_kernel
  void visitCallSPIRVEnqueueKernel(CallInst *CI, Op OC) override;

  /// Conduct generic mutations for all atomic builtins
  CallInst *mutateCommonAtomicArguments(CallInst *CI, Op OC) override;

  /// Transform atomic builtin name into correct ocl-dependent name
  void mutateAtomicName(CallInst *CI, Op OC) override;

  // Transform FP atomic opcode to corresponding OpenCL function name
  std::string mapFPAtomicName(Op OC) override;

  /// Transform SPIR-V atomic instruction opcode into OpenCL 1.2 builtin name.
  /// Depending on the type, the return name starts with "atomic_" for 32-bit
  /// types or with "atom_" for 64-bit types, as specified by
  /// cl_khr_int64_base_atomics and cl_khr_int64_extended_atomics extensions.
  std::string mapAtomicName(Op OC, Type *Ty);
};

class SPIRVToOCL12Pass : public llvm::PassInfoMixin<SPIRVToOCL12Pass>,
                         public SPIRVToOCL12Base {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runSPIRVToOCL(M) ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

class SPIRVToOCL12Legacy : public SPIRVToOCL12Base, public SPIRVToOCLLegacy {
public:
  SPIRVToOCL12Legacy() : SPIRVToOCLLegacy(ID) {
    initializeSPIRVToOCL12LegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;

  static char ID;
};

class SPIRVToOCL20Base : public SPIRVToOCLBase {
public:
  bool runSPIRVToOCL(Module &M) override;

  /// Transform __spirv_MemoryBarrier to atomic_work_item_fence.
  ///   __spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  void visitCallSPIRVMemoryBarrier(CallInst *CI) override;

  /// Transform __spirv_ControlBarrier to work_group_barrier/sub_group_barrier.
  /// If execution scope is ScopeWorkgroup:
  ///    __spirv_ControlBarrier(execScope, memScope, sema) =>
  ///         work_group_barrier(flag(sema), map(memScope))
  /// Otherwise:
  ///    __spirv_ControlBarrier(execScope, memScope, sema) =>
  ///         sub_group_barrier(flag(sema), map(memScope))
  void visitCallSPIRVControlBarrier(CallInst *CI) override;

  /// Transform split __spirv_ControlBarrierArriveINTEL and
  /// __spirv_ControlBarrierWaitINTEL barrier to overloads with a
  /// memory_scope argument.
  void visitCallSPIRVSplitBarrierINTEL(CallInst *CI, Op OC) override;

  /// Transform __spirv_Atomic* to atomic_*.
  ///   __spirv_Atomic*(atomic_op, scope, sema, ops, ...) =>
  ///      atomic_*(generic atomic_op, ops, ..., order(sema), map(scope))
  void visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) override;

  /// Transform __spirv_OpAtomicIIncrement / OpAtomicIDecrement to
  /// atomic_fetch_add_explicit / atomic_fetch_sub_explicit
  void visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) override;

  /// Transform __spirv_EnqueueKernel to __enqueue_kernel
  void visitCallSPIRVEnqueueKernel(CallInst *CI, Op OC) override;

  /// Conduct generic mutations for all atomic builtins
  CallInst *mutateCommonAtomicArguments(CallInst *CI, Op OC) override;

  /// Transform atomic builtin name into correct ocl-dependent name
  void mutateAtomicName(CallInst *CI, Op OC) override;

  // Transform FP atomic opcode to corresponding OpenCL function name
  std::string mapFPAtomicName(Op OC) override;

  /// Transform __spirv_OpAtomicCompareExchange/Weak into
  /// atomic_compare_exchange_strong_explicit
  /// OpAtomicCompareExchangeWeak is not "weak" at all, but instead has
  /// the same semantics as OpAtomicCompareExchange.
  void visitCallSPIRVAtomicCmpExchg(CallInst *CI) override;
};

class SPIRVToOCL20Pass : public llvm::PassInfoMixin<SPIRVToOCL20Pass>,
                         public SPIRVToOCL20Base {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runSPIRVToOCL(M) ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

class SPIRVToOCL20Legacy : public SPIRVToOCLLegacy, public SPIRVToOCL20Base {
public:
  SPIRVToOCL20Legacy() : SPIRVToOCLLegacy(ID) {
    initializeSPIRVToOCL20LegacyPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
  static char ID;
};

/// Add passes for translating SPIR-V Instructions to the desired
/// representation in LLVM IR (such as OpenCL builtins or SPIR-V Friendly IR).
void addSPIRVBIsLoweringPass(ModulePassManager &PassMgr,
                             SPIRV::BIsRepresentation BIsRep);

} // namespace SPIRV

#endif // SPIRVTOOCL_H
