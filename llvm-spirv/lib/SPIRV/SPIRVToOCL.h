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

#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"

#include <string>

namespace SPIRV {

class SPIRVToOCLBase : public InstVisitor<SPIRVToOCLBase> {
public:
  SPIRVToOCLBase() : M(nullptr), Ctx(nullptr) {}

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
  void visitCallSPRIVImageQuerySize(CallInst *CI);

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

  /// Transform __spirv_* builtins to OCL 2.0 builtins.
  /// No change with arguments.
  void visitCallSPIRVBuiltin(CallInst *CI, Op OC);

  /// Get prefix work_/sub_ for OCL group builtin functions.
  /// Assuming the first argument of \param CI is a constant integer for
  /// workgroup/subgroup scope enums.
  std::string getGroupBuiltinPrefix(CallInst *CI);

  /// Transform __spirv_OpAtomicCompareExchange and
  /// __spirv_OpAtomicCompareExchangeWeak
  virtual Instruction *visitCallSPIRVAtomicCmpExchg(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_OpAtomicIIncrement/OpAtomicIDecrement to:
  /// - OCL2.0: atomic_fetch_add_explicit/atomic_fetch_sub_explicit
  /// - OCL1.2: atomic_inc/atomic_dec
  virtual Instruction *visitCallSPIRVAtomicIncDec(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_Atomic* to atomic_*.
  ///   __spirv_Atomic*(atomic_op, scope, sema, ops, ...) =>
  ///      atomic_*(atomic_op, ops, ..., order(sema), map(scope))
  virtual Instruction *visitCallSPIRVAtomicBuiltin(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_MemoryBarrier to:
  /// - OCL2.0: atomic_work_item_fence.__spirv_MemoryBarrier(scope, sema) =>
  ///       atomic_work_item_fence(flag(sema), order(sema), map(scope))
  /// - OCL1.2: mem_fence
  virtual void visitCallSPIRVMemoryBarrier(CallInst *CI) = 0;

  /// Transform __spirv_ControlBarrier to:
  /// - OCL2.0: work_group_barrier or sub_group barrier
  /// - OCL1.2: barrier
  virtual void visitCallSPIRVControlBarrier(CallInst *CI) = 0;

  /// Conduct generic mutations for all atomic builtins
  virtual CallInst *mutateCommonAtomicArguments(CallInst *CI, Op OC) = 0;

  /// Transform __spirv_Opcode to ocl-version specific builtin name
  /// using separate maps for OpenCL 1.2 and OpenCL 2.0
  virtual Instruction *mutateAtomicName(CallInst *CI, Op OC) = 0;

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
  /// Transform group opcode to corresponding OpenCL function name
  std::string groupOCToOCLBuiltinName(CallInst *CI, Op OC);

  Module *M;
  LLVMContext *Ctx;
};

class SPIRVToOCLLegacy : public ModulePass {
protected:
  SPIRVToOCLLegacy(char &ID) : ModulePass(ID) {}

public:
  virtual bool runOnModule(Module &M) = 0;
};

} // namespace SPIRV
