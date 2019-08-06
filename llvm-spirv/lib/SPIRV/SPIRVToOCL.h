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
#include "llvm/PassSupport.h"

#include <cstring>

namespace SPIRV {
class SPIRVToOCL : public ModulePass, public InstVisitor<SPIRVToOCL> {
protected:
  SPIRVToOCL() : ModulePass(ID), M(nullptr), Ctx(nullptr) {}

public:
  virtual bool runOnModule(Module &M) = 0;

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

  /// Transform __spirv_Group* to {work_group|sub_group}_*.
  ///
  /// Special handling of work_group_broadcast.
  ///   __spirv_GroupBroadcast(a, vec3(x, y, z))
  ///     =>
  ///   work_group_broadcast(a, x, y, z)
  ///
  /// Transform OpenCL group builtin function names from group_
  /// to workgroup_ and sub_group_.
  /// Insert group operation part: reduce_/inclusive_scan_/exclusive_scan_
  /// Transform the operation part:
  ///    fadd/iadd/sadd => add
  ///    fmax/smax => max
  ///    fmin/smin => min
  /// Keep umax/umin unchanged.
  void visitCallSPIRVGroupBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_{PipeOpName} to OCL pipe builtin functions.
  void visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC);

  /// Transform __spirv_* builtins to OCL 2.0 builtins.
  /// No change with arguments.
  void visitCallSPIRVBuiltin(CallInst *CI, Op OC);

  /// Translate mangled atomic type name: "atomic_" =>
  ///   MangledAtomicTypeNamePrefix
  void translateMangledAtomicTypeName();

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

  static char ID;

protected:
  Module *M;
  LLVMContext *Ctx;
};
} // namespace SPIRV
