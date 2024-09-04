//=- OCLToSPIRV.h - OpenCL to SPIR-V builtin preprocessing pass -*- C++ -*-=//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2022 The Khronos Group Inc.
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
// Neither the names of The Khronos Group, nor the names of its
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
// This file implements preprocessing of OpenCL C built-in functions into SPIR-V
// friendly IR form for further translation into SPIR-V
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_OCLTOSPIRV_H
#define SPIRV_OCLTOSPIRV_H

#include "OCLUtil.h"
#include "SPIRVBuiltinHelper.h"

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace SPIRV {

class OCLTypeToSPIRVBase;

class OCLToSPIRVBase : public InstVisitor<OCLToSPIRVBase>, BuiltinCallHelper {
public:
  OCLToSPIRVBase()
      : BuiltinCallHelper(ManglingRules::SPIRV), Ctx(nullptr), CLVer(0),
        OCLTypeToSPIRVPtr(nullptr) {}
  virtual ~OCLToSPIRVBase() {}
  bool runOCLToSPIRV(Module &M);

  virtual void visitCallInst(CallInst &CI);

  /// Transform barrier/work_group_barrier/sub_group_barrier
  ///     to __spirv_ControlBarrier.
  /// barrier(flag) =>
  ///   __spirv_ControlBarrier(workgroup, workgroup, map(flag))
  /// work_group_barrier(scope, flag) =>
  ///   __spirv_ControlBarrier(workgroup, map(scope), map(flag))
  /// sub_group_barrier(scope, flag) =>
  ///   __spirv_ControlBarrier(subgroup, map(scope), map(flag))
  void visitCallBarrier(CallInst *CI);

  /// Erase useless convert functions.
  /// \return true if the call instruction is erased.
  bool eraseUselessConvert(CallInst *Call, StringRef MangledName,
                           StringRef DeMangledName);

  /// Transform convert_ to
  ///   __spirv_{CastOpName}_R{TargeTyName}{_sat}{_rt[p|n|z|e]}
  void visitCallConvert(CallInst *CI, StringRef MangledName,
                        StringRef DemangledName);

  /// Transform async_work_group{_strided}_copy.
  /// async_work_group_copy(dst, src, n, event)
  ///   => async_work_group_strided_copy(dst, src, n, 1, event)
  /// async_work_group_strided_copy(dst, src, n, stride, event)
  ///   => __spirv_AsyncGroupCopy(ScopeWorkGroup, dst, src, n, stride, event)
  void visitCallAsyncWorkGroupCopy(CallInst *CI, StringRef DemangledName);

  /// Transform OCL builtin function to SPIR-V builtin function.
  void transBuiltin(CallInst *CI, OCLBuiltinTransInfo &Info);

  /// Transform atomic_work_item_fence/mem_fence to __spirv_MemoryBarrier.
  /// func(flag, order, scope) =>
  ///   __spirv_MemoryBarrier(map(scope), map(flag)|map(order))
  void transMemoryBarrier(CallInst *CI, AtomicWorkItemFenceLiterals);

  /// Transform all to __spirv_Op(All|Any).  Note that the types mismatch so
  // some extra code is emitted to convert between the two.
  void visitCallAllAny(spv::Op OC, CallInst *CI);

  /// Transform atomic_* to __spirv_Atomic*.
  /// atomic_x(ptr_arg, args, order, scope) =>
  ///   __spirv_AtomicY(ptr_arg, map(order), map(scope), args)
  void transAtomicBuiltin(CallInst *CI, OCLBuiltinTransInfo &Info);

  /// Transform atomic_work_item_fence to __spirv_MemoryBarrier.
  /// atomic_work_item_fence(flag, order, scope) =>
  ///   __spirv_MemoryBarrier(map(scope), map(flag)|map(order))
  void visitCallAtomicWorkItemFence(CallInst *CI);

  /// Transform atomic_compare_exchange call.
  /// In atomic_compare_exchange, the expected value parameter is a pointer.
  /// However in SPIR-V it is a value. The transformation adds a load
  /// instruction, result of which is passed to atomic_compare_exchange as
  /// argument.
  /// The transformation adds a store instruction after the call, to update the
  /// value in expected with the value pointed to by object. Though, it is not
  /// necessary in case they are equal, this approach makes result code simpler.
  /// Also ICmp instruction is added, because the call must return result of
  /// comparison.
  /// \returns the call instruction of atomic_compare_exchange_strong.
  CallInst *visitCallAtomicCmpXchg(CallInst *CI);

  /// Transform atomic_init.
  /// atomic_init(p, x) => store p, x
  void visitCallAtomicInit(CallInst *CI);

  /// Transform legacy OCL 1.x atomic builtins to SPIR-V builtins for extensions
  ///   cl_khr_int64_base_atomics
  ///   cl_khr_int64_extended_atomics
  /// Do nothing if the called function is not a legacy atomic builtin.
  void visitCallAtomicLegacy(CallInst *CI, StringRef MangledName,
                             StringRef DemangledName);

  /// Transform OCL 2.0 C++11 atomic builtins to SPIR-V builtins.
  /// Do nothing if the called function is not a C++11 atomic builtin.
  void visitCallAtomicCpp11(CallInst *CI, StringRef MangledName,
                            StringRef DemangledName);

  /// Transform OCL builtin function to SPIR-V builtin function.
  /// Assuming there is a simple name mapping without argument changes.
  /// Should be called at last.
  void visitCallBuiltinSimple(CallInst *CI, StringRef MangledName,
                              StringRef DemangledName);

  /// Transform get_image_{width|height|depth|dim}.
  /// get_image_xxx(...) =>
  ///   dimension = __spirv_ImageQuerySizeLod_R{ReturnType}(...);
  ///   return dimension.{x|y|z};
  void visitCallGetImageSize(CallInst *CI, StringRef DemangledName);

  /// Transform {work|sub}_group_x =>
  ///   __spirv_{OpName}
  ///
  /// Special handling of work_group_broadcast.
  ///   work_group_broadcast(a, x, y, z)
  ///     =>
  ///   __spirv_GroupBroadcast(a, vec3(x, y, z))

  void visitCallGroupBuiltin(CallInst *CI, StringRef DemangledName);

  /// Transform mem_fence to __spirv_MemoryBarrier.
  /// mem_fence(flag) => __spirv_MemoryBarrier(Workgroup, map(flag))
  void visitCallMemFence(CallInst *CI, StringRef DemangledName);

  void visitCallNDRange(CallInst *CI, StringRef DemangledName);

  /// Transform read_image with sampler arguments.
  /// read_image(image, sampler, ...) =>
  ///   sampled_image = __spirv_SampledImage(image, sampler);
  ///   return __spirv_ImageSampleExplicitLod_R{ReturnType}(sampled_image, ...);
  void visitCallReadImageWithSampler(CallInst *CI, StringRef MangledName,
                                     StringRef DemangledName);

  /// Transform read_image with msaa image arguments.
  /// Sample argument must be acoded as Image Operand.
  void visitCallReadImageMSAA(CallInst *CI, StringRef MangledName);

  /// Transform {read|write}_image without sampler arguments.
  void visitCallReadWriteImage(CallInst *CI, StringRef DemangledName);

  /// Transform to_{global|local|private}.
  ///
  /// T* a = ...;
  /// addr T* b = to_addr(a);
  ///   =>
  /// i8* x = cast<i8*>(a);
  /// addr i8* y = __spirv_GenericCastToPtr_ToAddr(x);
  /// addr T* b = cast<addr T*>(y);
  void visitCallToAddr(CallInst *CI, StringRef DemangledName);

  /// Transform return type of relatinal built-in functions like isnan, isfinite
  /// to boolean values.
  void visitCallRelational(CallInst *CI, StringRef DemangledName);

  /// Transform vector load/store functions to SPIR-V extended builtin
  ///   functions
  /// {vload|vstore{a}}{_half}{n}{_rte|_rtz|_rtp|_rtn} =>
  ///   __spirv_ocl_{ExtendedInstructionOpCodeName}__R{ReturnType}
  void visitCallVecLoadStore(CallInst *CI, StringRef MangledName,
                             StringRef DemangledName);

  /// Transforms get_mem_fence built-in to SPIR-V function and aligns result
  /// values with SPIR 1.2. get_mem_fence(ptr) => __spirv_GenericPtrMemSemantics
  /// GenericPtrMemSemantics valid values are 0x100, 0x200 and 0x300, where is
  /// SPIR 1.2 defines them as 0x1, 0x2 and 0x3, so this function adjusts
  /// GenericPtrMemSemantics results to SPIR 1.2 values.
  void visitCallGetFence(CallInst *CI, StringRef DemangledName);

  /// Transforms OpDot instructions with a scalar type to a fmul instruction
  void visitCallDot(CallInst *CI);

  /// Transforms OpDot instructions with a vector or scalar (packed vector) type
  /// to dot or dot_acc_sat instructions
  void visitCallDot(CallInst *CI, StringRef MangledName,
                    StringRef DemangledName);

  /// Transform clock_read_* calls to OpReadClockKHR instructions.
  void visitCallClockRead(CallInst *CI, StringRef MangledName,
                          StringRef DemangledName);

  /// Fixes for built-in functions with vector+scalar arguments that are
  /// translated to the SPIR-V instructions where all arguments must have the
  /// same type.
  void visitCallScalToVec(CallInst *CI, StringRef MangledName,
                          StringRef DemangledName);

  /// Transform get_image_channel_{order|data_type} built-in functions to
  ///   __spirv_ocl_{ImageQueryOrder|ImageQueryFormat}
  void visitCallGetImageChannel(CallInst *CI, StringRef DemangledName,
                                unsigned int Offset);

  /// Transform enqueue_kernel and kernel query built-in functions to
  /// spirv-friendly format filling arguments, required for device-side enqueue
  /// instructions, but missed in the original call
  void visitCallEnqueueKernel(CallInst *CI, StringRef DemangledName);
  void visitCallKernelQuery(CallInst *CI, StringRef DemangledName);

  /// For cl_intel_subgroups block read built-ins:
  void visitSubgroupBlockReadINTEL(CallInst *CI);

  /// For cl_intel_subgroups block write built-ins:
  void visitSubgroupBlockWriteINTEL(CallInst *CI);

  /// For cl_intel_media_block_io built-ins:
  void visitSubgroupImageMediaBlockINTEL(CallInst *CI, StringRef DemangledName);
  // For cl_intel_device_side_avc_motion_estimation built-ins
  void visitSubgroupAVCBuiltinCall(CallInst *CI, StringRef DemangledName);
  void visitSubgroupAVCWrapperBuiltinCall(CallInst *CI, Op WrappedOC,
                                          StringRef DemangledName);
  void visitSubgroupAVCBuiltinCallWithSampler(CallInst *CI,
                                              StringRef DemangledName);

  /// For cl_intel_split_work_group_barrier built-ins:
  void visitCallSplitBarrierINTEL(CallInst *CI, StringRef DemangledName);

  void visitCallLdexp(CallInst *CI, StringRef MangledName,
                      StringRef DemangledName);

  /// For cl_intel_convert_bfloat16_as_ushort
  void visitCallConvertBFloat16AsUshort(CallInst *CI, StringRef DemangledName);
  /// For cl_intel_convert_as_bfloat16_float
  void visitCallConvertAsBFloat16Float(CallInst *CI, StringRef DemangledName);

  void setOCLTypeToSPIRV(OCLTypeToSPIRVBase *OCLTypeToSPIRV) {
    OCLTypeToSPIRVPtr = OCLTypeToSPIRV;
  }
  OCLTypeToSPIRVBase *getOCLTypeToSPIRV() { return OCLTypeToSPIRVPtr; }

private:
  LLVMContext *Ctx;
  unsigned CLVer; /// OpenCL version as major*10+minor
  std::set<Instruction *> ValuesToDelete;
  OCLTypeToSPIRVBase *OCLTypeToSPIRVPtr;

  ConstantInt *addInt32(int I) { return getInt32(M, I); }
  ConstantInt *addSizet(uint64_t I) { return getSizet(M, I); }

  /// Get vector width from OpenCL vload* function name.
  SPIRVWord getVecLoadWidth(const std::string &DemangledName);

  /// Transform OpenCL vload/vstore function name.
  void transVecLoadStoreName(std::string &DemangledName,
                             const std::string &Stem, bool AlwaysN);

  void processSubgroupBlockReadWriteINTEL(CallInst *CI,
                                          OCLBuiltinTransInfo &Info,
                                          const Type *DataTy);
};

class OCLToSPIRVLegacy : public OCLToSPIRVBase, public llvm::ModulePass {
public:
  OCLToSPIRVLegacy() : ModulePass(ID) {
    initializeOCLToSPIRVLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  static char ID;
};

class OCLToSPIRVPass : public OCLToSPIRVBase,
                       public llvm::PassInfoMixin<OCLToSPIRVPass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace SPIRV

#endif // SPIRV_OCLTOSPIRV_H
