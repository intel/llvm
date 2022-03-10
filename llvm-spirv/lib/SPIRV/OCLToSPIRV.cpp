//===- OCLToSPIRV.cpp - Transform OCL to SPIR-V builtins --------*- C++ -*-===//
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
// This file implements preprocessing of OpenCL C built-in functions into SPIR-V
// friendly IR form for further translation into SPIR-V
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "ocl-to-spv"

#include "OCLTypeToSPIRV.h"
#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <set>

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {
static size_t getOCLCpp11AtomicMaxNumOps(StringRef Name) {
  return StringSwitch<size_t>(Name)
      .Cases("load", "flag_test_and_set", "flag_clear", 3)
      .Cases("store", "exchange", 4)
      .StartsWith("compare_exchange", 6)
      .StartsWith("fetch", 4)
      .Default(0);
}

class OCLToSPIRVBase : public InstVisitor<OCLToSPIRVBase> {
public:
  OCLToSPIRVBase() : M(nullptr), Ctx(nullptr), CLVer(0) {}
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
  void visitCallReadImageWithSampler(CallInst *CI, StringRef MangledName);

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

  void visitCallLdexp(CallInst *CI, StringRef MangledName,
                      StringRef DemangledName);

  void setOCLTypeToSPIRV(OCLTypeToSPIRVBase *OCLTypeToSPIRV) {
    OCLTypeToSPIRVPtr = OCLTypeToSPIRV;
  }
  OCLTypeToSPIRVBase *getOCLTypeToSPIRV() { return OCLTypeToSPIRVPtr; }

private:
  Module *M;
  LLVMContext *Ctx;
  unsigned CLVer; /// OpenCL version as major*10+minor
  std::set<Value *> ValuesToDelete;
  OCLTypeToSPIRVBase *OCLTypeToSPIRVPtr;

  ConstantInt *addInt32(int I) { return getInt32(M, I); }
  ConstantInt *addSizet(uint64_t I) { return getSizet(M, I); }

  /// Get vector width from OpenCL vload* function name.
  SPIRVWord getVecLoadWidth(const std::string &DemangledName) {
    SPIRVWord Width = 0;
    if (DemangledName == "vloada_half")
      Width = 1;
    else {
      unsigned Loc = 5;
      if (DemangledName.find("vload_half") == 0)
        Loc = 10;
      else if (DemangledName.find("vloada_half") == 0)
        Loc = 11;

      std::stringstream SS(DemangledName.substr(Loc));
      SS >> Width;
    }
    return Width;
  }

  /// Transform OpenCL vload/vstore function name.
  void transVecLoadStoreName(std::string &DemangledName,
                             const std::string &Stem, bool AlwaysN) {
    auto HalfStem = Stem + "_half";
    auto HalfStemR = HalfStem + "_r";
    if (!AlwaysN && DemangledName == HalfStem)
      return;
    if (!AlwaysN && DemangledName.find(HalfStemR) == 0) {
      DemangledName = HalfStemR;
      return;
    }
    if (DemangledName.find(HalfStem) == 0) {
      auto OldName = DemangledName;
      DemangledName = HalfStem + "n";
      if (OldName.find("_r") != std::string::npos)
        DemangledName += "_r";
      return;
    }
    if (DemangledName.find(Stem) == 0) {
      DemangledName = Stem + "n";
      return;
    }
  }
};

class OCLToSPIRVLegacy : public OCLToSPIRVBase, public llvm::ModulePass {
public:
  OCLToSPIRVLegacy() : ModulePass(ID) {
    initializeOCLToSPIRVLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    setOCLTypeToSPIRV(&getAnalysis<OCLTypeToSPIRVLegacy>());
    return runOCLToSPIRV(M);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OCLTypeToSPIRVLegacy>();
  }

  static char ID;
};

class OCLToSPIRVPass : public OCLToSPIRVBase,
                       public llvm::PassInfoMixin<OCLToSPIRVBase> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    setOCLTypeToSPIRV(&MAM.getResult<OCLTypeToSPIRVPass>(M));
    return runOCLToSPIRV(M) ? llvm::PreservedAnalyses::none()
                            : llvm::PreservedAnalyses::all();
  }
};

char OCLToSPIRVLegacy::ID = 0;

bool OCLToSPIRVBase::runOCLToSPIRV(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();
  auto Src = getSPIRVSource(&Module);
  // This is a pre-processing pass, which transform LLVM IR module to a more
  // suitable form for the SPIR-V translation: it is specifically designed to
  // handle OpenCL C built-in functions and shouldn't be launched for other
  // source languages
  if (std::get<0>(Src) != spv::SourceLanguageOpenCL_C)
    return false;

  CLVer = std::get<1>(Src);

  LLVM_DEBUG(dbgs() << "Enter OCLToSPIRV:\n");

  visit(*M);

  for (auto &I : ValuesToDelete)
    if (auto Inst = dyn_cast<Instruction>(I))
      Inst->eraseFromParent();
  for (auto &I : ValuesToDelete)
    if (auto GV = dyn_cast<GlobalValue>(I))
      GV->eraseFromParent();

  eraseUselessFunctions(M); // remove unused functions declarations
  LLVM_DEBUG(dbgs() << "After OCLToSPIRV:\n" << *M);

  verifyRegularizationPass(*M, "OCLToSPIRV");

  return true;
}

// The order of handling OCL builtin functions is important.
// Workgroup functions need to be handled before pipe functions since
// there are functions fall into both categories.
void OCLToSPIRVBase::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto F = CI.getCalledFunction();
  if (!F)
    return;

  auto MangledName = F->getName();
  StringRef DemangledName;
  if (!oclIsBuiltin(MangledName, DemangledName))
    return;

  LLVM_DEBUG(dbgs() << "DemangledName: " << DemangledName << '\n');
  if (DemangledName.find(kOCLBuiltinName::NDRangePrefix) == 0) {
    visitCallNDRange(&CI, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::All) {
    visitCallAllAny(OpAll, &CI);
    return;
  }
  if (DemangledName == kOCLBuiltinName::Any) {
    visitCallAllAny(OpAny, &CI);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::AsyncWorkGroupCopy) == 0 ||
      DemangledName.find(kOCLBuiltinName::AsyncWorkGroupStridedCopy) == 0) {
    visitCallAsyncWorkGroupCopy(&CI, DemangledName);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::AtomicPrefix) == 0 ||
      DemangledName.find(kOCLBuiltinName::AtomPrefix) == 0) {

    // Compute atomic builtins do not support floating types.
    if (CI.getType()->isFloatingPointTy() &&
        isComputeAtomicOCLBuiltin(DemangledName))
      return;

    auto PCI = &CI;
    if (DemangledName == kOCLBuiltinName::AtomicInit) {
      visitCallAtomicInit(PCI);
      return;
    }
    if (DemangledName == kOCLBuiltinName::AtomicWorkItemFence) {
      visitCallAtomicWorkItemFence(PCI);
      return;
    }
    if (DemangledName == kOCLBuiltinName::AtomicCmpXchgWeak ||
        DemangledName == kOCLBuiltinName::AtomicCmpXchgStrong ||
        DemangledName == kOCLBuiltinName::AtomicCmpXchgWeakExplicit ||
        DemangledName == kOCLBuiltinName::AtomicCmpXchgStrongExplicit) {
      assert((CLVer == kOCLVer::CL20 || CLVer == kOCLVer::CL30) &&
             "Wrong version of OpenCL");
      PCI = visitCallAtomicCmpXchg(PCI);
    }
    visitCallAtomicLegacy(PCI, MangledName, DemangledName);
    visitCallAtomicCpp11(PCI, MangledName, DemangledName);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::ConvertPrefix) == 0) {
    visitCallConvert(&CI, MangledName, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::GetImageWidth ||
      DemangledName == kOCLBuiltinName::GetImageHeight ||
      DemangledName == kOCLBuiltinName::GetImageDepth ||
      DemangledName == kOCLBuiltinName::GetImageDim ||
      DemangledName == kOCLBuiltinName::GetImageArraySize) {
    visitCallGetImageSize(&CI, DemangledName);
    return;
  }
  if ((DemangledName.find(kOCLBuiltinName::WorkGroupPrefix) == 0 &&
       DemangledName != kOCLBuiltinName::WorkGroupBarrier) ||
      DemangledName == kOCLBuiltinName::WaitGroupEvent ||
      (DemangledName.find(kOCLBuiltinName::SubGroupPrefix) == 0 &&
       DemangledName != kOCLBuiltinName::SubGroupBarrier)) {
    visitCallGroupBuiltin(&CI, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::MemFence ||
      DemangledName == kOCLBuiltinName::ReadMemFence ||
      DemangledName == kOCLBuiltinName::WriteMemFence) {
    visitCallMemFence(&CI, DemangledName);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::ReadImage) == 0) {
    if (MangledName.find(kMangledName::Sampler) != StringRef::npos) {
      visitCallReadImageWithSampler(&CI, MangledName);
      return;
    }
    if (MangledName.find("msaa") != StringRef::npos) {
      visitCallReadImageMSAA(&CI, MangledName);
      return;
    }
  }
  if (DemangledName.find(kOCLBuiltinName::ReadImage) == 0 ||
      DemangledName.find(kOCLBuiltinName::WriteImage) == 0) {
    visitCallReadWriteImage(&CI, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::ToGlobal ||
      DemangledName == kOCLBuiltinName::ToLocal ||
      DemangledName == kOCLBuiltinName::ToPrivate) {
    visitCallToAddr(&CI, DemangledName);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::VLoadPrefix) == 0 ||
      DemangledName.find(kOCLBuiltinName::VStorePrefix) == 0) {
    visitCallVecLoadStore(&CI, MangledName, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::IsFinite ||
      DemangledName == kOCLBuiltinName::IsInf ||
      DemangledName == kOCLBuiltinName::IsNan ||
      DemangledName == kOCLBuiltinName::IsNormal ||
      DemangledName == kOCLBuiltinName::Signbit) {
    visitCallRelational(&CI, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::WorkGroupBarrier ||
      DemangledName == kOCLBuiltinName::Barrier ||
      DemangledName == kOCLBuiltinName::SubGroupBarrier) {
    visitCallBarrier(&CI);
    return;
  }
  if (DemangledName == kOCLBuiltinName::GetFence) {
    visitCallGetFence(&CI, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::Dot &&
      !(CI.getOperand(0)->getType()->isVectorTy())) {
    visitCallDot(&CI);
    return;
  }
  if (DemangledName == kOCLBuiltinName::FMin ||
      DemangledName == kOCLBuiltinName::FMax ||
      DemangledName == kOCLBuiltinName::Min ||
      DemangledName == kOCLBuiltinName::Max ||
      DemangledName == kOCLBuiltinName::Step ||
      DemangledName == kOCLBuiltinName::SmoothStep ||
      DemangledName == kOCLBuiltinName::Clamp ||
      DemangledName == kOCLBuiltinName::Mix) {
    visitCallScalToVec(&CI, MangledName, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::GetImageChannelDataType) {
    visitCallGetImageChannel(&CI, DemangledName, OCLImageChannelDataTypeOffset);
    return;
  }
  if (DemangledName == kOCLBuiltinName::GetImageChannelOrder) {
    visitCallGetImageChannel(&CI, DemangledName, OCLImageChannelOrderOffset);
    return;
  }
  if (isEnqueueKernelBI(MangledName)) {
    visitCallEnqueueKernel(&CI, DemangledName);
    return;
  }
  if (isKernelQueryBI(MangledName)) {
    visitCallKernelQuery(&CI, DemangledName);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::SubgroupBlockReadINTELPrefix) == 0) {
    visitSubgroupBlockReadINTEL(&CI);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::SubgroupBlockWriteINTELPrefix) == 0) {
    visitSubgroupBlockWriteINTEL(&CI);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::SubgroupImageMediaBlockINTELPrefix) ==
      0) {
    visitSubgroupImageMediaBlockINTEL(&CI, DemangledName);
    return;
  }
  // Handle 'cl_intel_device_side_avc_motion_estimation' extension built-ins
  if (DemangledName.find(kOCLSubgroupsAVCIntel::Prefix) == 0 ||
      // Workaround for a bug in the extension specification
      DemangledName.find("intel_sub_group_ime_ref_window_size") == 0) {
    if (MangledName.find(kMangledName::Sampler) != StringRef::npos)
      visitSubgroupAVCBuiltinCallWithSampler(&CI, DemangledName);
    else
      visitSubgroupAVCBuiltinCall(&CI, DemangledName);
    return;
  }
  if (DemangledName.find(kOCLBuiltinName::LDEXP) == 0) {
    visitCallLdexp(&CI, MangledName, DemangledName);
    return;
  }
  visitCallBuiltinSimple(&CI, MangledName, DemangledName);
}

void OCLToSPIRVBase::visitCallNDRange(CallInst *CI, StringRef DemangledName) {
  assert(DemangledName.find(kOCLBuiltinName::NDRangePrefix) == 0);
  StringRef LenStr = DemangledName.substr(8, 1);
  auto Len = atoi(LenStr.data());
  assert(Len >= 1 && Len <= 3);
  // SPIR-V ndrange structure requires 3 members in the following order:
  //   global work offset
  //   global work size
  //   local work size
  // The arguments need to add missing members.
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        for (size_t I = 1, E = Args.size(); I != E; ++I)
          Args[I] = getScalarOrArray(Args[I], Len, CI);
        switch (Args.size()) {
        case 2: {
          // Has global work size.
          auto T = Args[1]->getType();
          auto C = getScalarOrArrayConstantInt(CI, T, Len, 0);
          Args.push_back(C);
          Args.push_back(C);
        } break;
        case 3: {
          // Has global and local work size.
          auto T = Args[1]->getType();
          Args.push_back(getScalarOrArrayConstantInt(CI, T, Len, 0));
        } break;
        case 4: {
          // Move offset arg to the end
          auto OffsetPos = Args.begin() + 1;
          Value *OffsetVal = *OffsetPos;
          Args.erase(OffsetPos);
          Args.push_back(OffsetVal);
        } break;
        default:
          assert(0 && "Invalid number of arguments");
        }
        // Translate ndrange_ND into differently named SPIR-V
        // decorated functions because they have array arugments
        // of different dimension which mangled the same way.
        std::string Postfix("_");
        Postfix += LenStr;
        Postfix += 'D';
        return getSPIRVFuncName(OpBuildNDRange, Postfix);
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallAsyncWorkGroupCopy(CallInst *CI,
                                                 StringRef DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        if (DemangledName == OCLUtil::kOCLBuiltinName::AsyncWorkGroupCopy) {
          Args.insert(Args.begin() + 3, addSizet(1));
        }
        Args.insert(Args.begin(), addInt32(ScopeWorkgroup));
        return getSPIRVFuncName(OpGroupAsyncCopy);
      },
      &Attrs);
}

CallInst *OCLToSPIRVBase::visitCallAtomicCmpXchg(CallInst *CI) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Value *Expected = nullptr;
  CallInst *NewCI = nullptr;
  mutateCallInstOCL(
      M, CI,
      [&](CallInst *CI, std::vector<Value *> &Args, Type *&RetTy) {
        Expected = Args[1]; // temporary save second argument.
        Args[1] = new LoadInst(Args[1]->getType()->getPointerElementType(),
                               Args[1], "exp", false, CI);
        RetTy = Args[2]->getType();
        assert(Args[0]->getType()->getPointerElementType()->isIntegerTy() &&
               Args[1]->getType()->isIntegerTy() &&
               Args[2]->getType()->isIntegerTy() &&
               "In SPIR-V 1.0 arguments of OpAtomicCompareExchange must be "
               "an integer type scalars");
        return kOCLBuiltinName::AtomicCmpXchgStrong;
      },
      [&](CallInst *NCI) -> Instruction * {
        NewCI = NCI;
        Instruction *Store = new StoreInst(NCI, Expected, NCI->getNextNode());
        return new ICmpInst(Store->getNextNode(), CmpInst::ICMP_EQ, NCI,
                            NCI->getArgOperand(1));
      },
      &Attrs);
  return NewCI;
}

void OCLToSPIRVBase::visitCallAtomicInit(CallInst *CI) {
  auto ST = new StoreInst(CI->getArgOperand(1), CI->getArgOperand(0), CI);
  ST->takeName(CI);
  CI->dropAllReferences();
  CI->eraseFromParent();
}

void OCLToSPIRVBase::visitCallAllAny(spv::Op OC, CallInst *CI) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();

  auto Args = getArguments(CI);
  assert(Args.size() == 1);

  auto *ArgTy = Args[0]->getType();
  auto Zero = Constant::getNullValue(Args[0]->getType());

  auto *Cmp = CmpInst::Create(CmpInst::ICmp, CmpInst::ICMP_SLT, Args[0], Zero,
                              "cast", CI);

  if (!isa<VectorType>(ArgTy)) {
    auto *Cast = CastInst::CreateZExtOrBitCast(Cmp, Type::getInt32Ty(*Ctx), "",
                                               Cmp->getNextNode());
    CI->replaceAllUsesWith(Cast);
    CI->eraseFromParent();
  } else {
    mutateCallInstSPIRV(
        M, CI,
        [&](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
          Args[0] = Cmp;
          Ret = Type::getInt1Ty(*Ctx);

          return getSPIRVFuncName(OC);
        },
        [&](CallInst *CI) -> Instruction * {
          return CastInst::CreateZExtOrBitCast(CI, Type::getInt32Ty(*Ctx), "",
                                               CI->getNextNode());
        },
        &Attrs);
  }
}

void OCLToSPIRVBase::visitCallAtomicWorkItemFence(CallInst *CI) {
  transMemoryBarrier(CI, getAtomicWorkItemFenceLiterals(CI));
}

void OCLToSPIRVBase::visitCallMemFence(CallInst *CI, StringRef DemangledName) {
  OCLMemOrderKind MO = StringSwitch<OCLMemOrderKind>(DemangledName)
                           .Case(kOCLBuiltinName::ReadMemFence, OCLMO_acquire)
                           .Case(kOCLBuiltinName::WriteMemFence, OCLMO_release)
                           .Default(OCLMO_acq_rel); // kOCLBuiltinName::MemFence
  transMemoryBarrier(
      CI,
      std::make_tuple(cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue(),
                      MO, OCLMS_work_group));
}

void OCLToSPIRVBase::transMemoryBarrier(CallInst *CI,
                                        AtomicWorkItemFenceLiterals Lit) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(2);
        Args[0] = addInt32(map<Scope>(std::get<2>(Lit)));
        Args[1] = addInt32(
            mapOCLMemSemanticToSPIRV(std::get<0>(Lit), std::get<1>(Lit)));
        return getSPIRVFuncName(OpMemoryBarrier);
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallAtomicLegacy(CallInst *CI, StringRef MangledName,
                                           StringRef DemangledName) {
  StringRef Stem = DemangledName;
  if (Stem.startswith("atom_"))
    Stem = Stem.drop_front(strlen("atom_"));
  else if (Stem.startswith("atomic_"))
    Stem = Stem.drop_front(strlen("atomic_"));
  else
    return;

  std::string Sign;
  std::string Postfix;
  std::string Prefix;
  if (Stem == "add" || Stem == "sub" || Stem == "and" || Stem == "or" ||
      Stem == "xor" || Stem == "min" || Stem == "max") {
    if ((Stem == "min" || Stem == "max") &&
        isMangledTypeUnsigned(MangledName.back()))
      Sign = 'u';
    Prefix = "fetch_";
    Postfix = "_explicit";
  } else if (Stem == "xchg") {
    Stem = "exchange";
    Postfix = "_explicit";
  } else if (Stem == "cmpxchg") {
    Stem = "compare_exchange_strong";
    Postfix = "_explicit";
  } else if (Stem == "inc" || Stem == "dec") {
    // do nothing
  } else
    return;

  OCLBuiltinTransInfo Info;
  Info.UniqName = "atomic_" + Prefix + Sign + Stem.str() + Postfix;
  std::vector<int> PostOps;
  PostOps.push_back(OCLLegacyAtomicMemOrder);
  if (Stem.startswith("compare_exchange"))
    PostOps.push_back(OCLLegacyAtomicMemOrder);
  PostOps.push_back(OCLLegacyAtomicMemScope);

  Info.PostProc = [=](std::vector<Value *> &Ops) {
    for (auto &I : PostOps) {
      Ops.push_back(addInt32(I));
    }
  };
  transAtomicBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallAtomicCpp11(CallInst *CI, StringRef MangledName,
                                          StringRef DemangledName) {
  StringRef Stem = DemangledName;
  if (Stem.startswith("atomic_"))
    Stem = Stem.drop_front(strlen("atomic_"));
  else
    return;

  std::string NewStem(Stem);
  std::vector<int> PostOps;
  if (Stem.startswith("store") || Stem.startswith("load") ||
      Stem.startswith("exchange") || Stem.startswith("compare_exchange") ||
      Stem.startswith("fetch") || Stem.startswith("flag")) {
    if ((Stem.startswith("fetch_min") || Stem.startswith("fetch_max")) &&
        containsUnsignedAtomicType(MangledName))
      NewStem.insert(NewStem.begin() + strlen("fetch_"), 'u');

    if (!Stem.endswith("_explicit")) {
      NewStem = NewStem + "_explicit";
      PostOps.push_back(OCLMO_seq_cst);
      if (Stem.startswith("compare_exchange"))
        PostOps.push_back(OCLMO_seq_cst);
      PostOps.push_back(OCLMS_device);
    } else {
      auto MaxOps =
          getOCLCpp11AtomicMaxNumOps(Stem.drop_back(strlen("_explicit")));
      if (CI->arg_size() < MaxOps)
        PostOps.push_back(OCLMS_device);
    }
  } else if (Stem == "work_item_fence") {
    // do nothing
  } else
    return;

  OCLBuiltinTransInfo Info;
  Info.UniqName = std::string("atomic_") + NewStem;
  Info.PostProc = [=](std::vector<Value *> &Ops) {
    for (auto &I : PostOps) {
      Ops.push_back(addInt32(I));
    }
  };

  transAtomicBuiltin(CI, Info);
}

void OCLToSPIRVBase::transAtomicBuiltin(CallInst *CI,
                                        OCLBuiltinTransInfo &Info) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *CI, std::vector<Value *> &Args) -> std::string {
        Info.PostProc(Args);
        // Order of args in OCL20:
        // object, 0-2 other args, 1-2 order, scope
        const size_t NumOrder =
            getAtomicBuiltinNumMemoryOrderArgs(Info.UniqName);
        const size_t ArgsCount = Args.size();
        const size_t ScopeIdx = ArgsCount - 1;
        const size_t OrderIdx = ScopeIdx - NumOrder;

        Args[ScopeIdx] =
            transOCLMemScopeIntoSPIRVScope(Args[ScopeIdx], OCLMS_device, CI);

        for (size_t I = 0; I < NumOrder; ++I) {
          Args[OrderIdx + I] = transOCLMemOrderIntoSPIRVMemorySemantics(
              Args[OrderIdx + I], OCLMO_seq_cst, CI);
        }
        // Order of args in SPIR-V:
        // object, scope, 1-2 order, 0-2 other args
        std::swap(Args[1], Args[ScopeIdx]);
        if (OrderIdx > 2) {
          // For atomic_compare_exchange the swap above puts Comparator/Expected
          // argument just where it should be, so don't move the last argument
          // then.
          int Offset =
              Info.UniqName.find("atomic_compare_exchange") == 0 ? 1 : 0;
          std::rotate(Args.begin() + 2, Args.begin() + OrderIdx,
                      Args.end() - Offset);
        }
        llvm::Type *AtomicBuiltinsReturnType =
            CI->getCalledFunction()->getReturnType();
        auto IsFPType = [](llvm::Type *ReturnType) {
          return ReturnType->isHalfTy() || ReturnType->isFloatTy() ||
                 ReturnType->isDoubleTy();
        };
        auto SPIRVFunctionName =
            getSPIRVFuncName(OCLSPIRVBuiltinMap::map(Info.UniqName));
        if (!IsFPType(AtomicBuiltinsReturnType))
          return SPIRVFunctionName;
        // Translate FP-typed atomic builtins. Currently we only need to
        // translate atomic_fetch_[add, max, min] and atomic_fetch_[add, max,
        // min]_explicit to related float instructions
        auto SPIRFunctionNameForFloatAtomics =
            llvm::StringSwitch<std::string>(SPIRVFunctionName)
                .Case("__spirv_AtomicIAdd", "__spirv_AtomicFAddEXT")
                .Case("__spirv_AtomicSMax", "__spirv_AtomicFMaxEXT")
                .Case("__spirv_AtomicSMin", "__spirv_AtomicFMinEXT")
                .Default("others");
        return SPIRFunctionNameForFloatAtomics == "others"
                   ? SPIRVFunctionName
                   : SPIRFunctionNameForFloatAtomics;
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallBarrier(CallInst *CI) {
  auto Lit = getBarrierLiterals(CI);
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(3);
        // Execution scope
        Args[0] = addInt32(map<Scope>(std::get<2>(Lit)));
        // Memory scope
        Args[1] = addInt32(map<Scope>(std::get<1>(Lit)));
        // Use sequential consistent memory order by default.
        // But if the flags argument is set to 0, we use
        // None(Relaxed) memory order.
        unsigned MemFenceFlag = std::get<0>(Lit);
        OCLMemOrderKind MemOrder = MemFenceFlag ? OCLMO_seq_cst : OCLMO_relaxed;
        Args[2] = addInt32(mapOCLMemSemanticToSPIRV(
            MemFenceFlag, MemOrder)); // Memory semantics
        return getSPIRVFuncName(OpControlBarrier);
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallConvert(CallInst *CI, StringRef MangledName,
                                      StringRef DemangledName) {
  if (eraseUselessConvert(CI, MangledName, DemangledName))
    return;
  Op OC = OpNop;
  auto TargetTy = CI->getType();
  auto SrcTy = CI->getArgOperand(0)->getType();
  if (auto *VecTy = dyn_cast<VectorType>(TargetTy))
    TargetTy = VecTy->getElementType();
  if (auto *VecTy = dyn_cast<VectorType>(SrcTy))
    SrcTy = VecTy->getElementType();
  auto IsTargetInt = isa<IntegerType>(TargetTy);

  std::string TargetTyName(
      DemangledName.substr(strlen(kOCLBuiltinName::ConvertPrefix)));
  auto FirstUnderscoreLoc = TargetTyName.find('_');
  if (FirstUnderscoreLoc != std::string::npos)
    TargetTyName = TargetTyName.substr(0, FirstUnderscoreLoc);
  TargetTyName = std::string("_R") + TargetTyName;

  std::string Sat = DemangledName.find("_sat") != StringRef::npos ? "_sat" : "";
  auto TargetSigned = DemangledName[8] != 'u';
  if (isa<IntegerType>(SrcTy)) {
    bool Signed = isLastFuncParamSigned(MangledName);
    if (IsTargetInt) {
      if (!Sat.empty() && TargetSigned != Signed) {
        OC = Signed ? OpSatConvertSToU : OpSatConvertUToS;
        Sat = "";
      } else
        OC = Signed ? OpSConvert : OpUConvert;
    } else
      OC = Signed ? OpConvertSToF : OpConvertUToF;
  } else {
    if (IsTargetInt) {
      OC = TargetSigned ? OpConvertFToS : OpConvertFToU;
    } else
      OC = OpFConvert;
  }
  auto Loc = DemangledName.find("_rt");
  std::string Rounding;
  if (Loc != StringRef::npos && !(isa<IntegerType>(SrcTy) && IsTargetInt)) {
    Rounding = DemangledName.substr(Loc, 4).str();
  }
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return getSPIRVFuncName(OC, TargetTyName + Sat + Rounding);
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallGroupBuiltin(CallInst *CI,
                                           StringRef OrigDemangledName) {
  auto F = CI->getCalledFunction();
  std::vector<int> PreOps;
  std::string DemangledName{OrigDemangledName};

  if (DemangledName == kOCLBuiltinName::WorkGroupBarrier)
    return;
  if (DemangledName == kOCLBuiltinName::WaitGroupEvent) {
    PreOps.push_back(ScopeWorkgroup);
  } else if (DemangledName.find(kOCLBuiltinName::WorkGroupPrefix) == 0) {
    DemangledName.erase(0, strlen(kOCLBuiltinName::WorkPrefix));
    PreOps.push_back(ScopeWorkgroup);
  } else if (DemangledName.find(kOCLBuiltinName::SubGroupPrefix) == 0) {
    DemangledName.erase(0, strlen(kOCLBuiltinName::SubPrefix));
    PreOps.push_back(ScopeSubgroup);
  } else
    return;

  if (DemangledName != kOCLBuiltinName::WaitGroupEvent) {
    StringRef FuncName = DemangledName;
    FuncName = FuncName.drop_front(strlen(kSPIRVName::GroupPrefix));
    SPIRSPIRVGroupOperationMap::foreachConditional(
        [&](const std::string &S, SPIRVGroupOperationKind G) {
          if (!FuncName.startswith(S))
            return true; // continue
          PreOps.push_back(G);
          StringRef Op =
              StringSwitch<StringRef>(FuncName)
                  .StartsWith("ballot", "group_ballot_bit_count_")
                  .StartsWith("non_uniform", kSPIRVName::GroupNonUniformPrefix)
                  .Default(kSPIRVName::GroupPrefix);
          // clustered functions are handled with non uniform group opcodes
          StringRef ClusteredOp =
              FuncName.contains("clustered_") ? "non_uniform_" : "";
          StringRef LogicalOp = FuncName.contains("logical_") ? "logical_" : "";
          StringRef GroupOp = StringSwitch<StringRef>(FuncName)
                                  .Case("ballot_bit_count", "add")
                                  .Case("ballot_inclusive_scan", "add")
                                  .Case("ballot_exclusive_scan", "add")
                                  .Default(FuncName.take_back(
                                      3)); // assumes op is three characters
          GroupOp.consume_front("_");      // when op is two characters
          assert(!GroupOp.empty() && "Invalid OpenCL group builtin function");
          char OpTyC = 0;
          auto OpTy = F->getReturnType();
          if (OpTy->isFloatingPointTy())
            OpTyC = 'f';
          else if (OpTy->isIntegerTy()) {
            auto NeedSign = GroupOp == "max" || GroupOp == "min";
            if (!NeedSign)
              OpTyC = 'i';
            else {
              // clustered reduce args are (type, uint)
              // other operation args are (type)
              auto MangledName = F->getName();
              auto MangledTyC = ClusteredOp.empty()
                                    ? MangledName.back()
                                    : MangledName.take_back(2).front();
              if (isMangledTypeSigned(MangledTyC))
                OpTyC = 's';
              else
                OpTyC = 'u';
            }
          } else
            llvm_unreachable("Invalid OpenCL group builtin argument type");

          DemangledName = Op.str() + ClusteredOp.str() + LogicalOp.str() +
                          OpTyC + GroupOp.str();
          return false; // break out of loop
        });
  }

  const bool IsElect = DemangledName == "group_elect";
  const bool IsAllOrAny = (DemangledName.find("_all") != std::string::npos ||
                           DemangledName.find("_any") != std::string::npos);
  const bool IsAllEqual = DemangledName.find("_all_equal") != std::string::npos;
  const bool IsBallot = DemangledName == "group_ballot";
  const bool IsInverseBallot = DemangledName == "group_inverse_ballot";
  const bool IsBallotBitExtract = DemangledName == "group_ballot_bit_extract";
  const bool IsLogical = DemangledName.find("_logical") != std::string::npos;

  const bool HasBoolReturnType = IsElect || IsAllOrAny || IsAllEqual ||
                                 IsInverseBallot || IsBallotBitExtract ||
                                 IsLogical;
  const bool HasBoolArg = (IsAllOrAny && !IsAllEqual) || IsBallot || IsLogical;

  auto Consts = getInt32(M, PreOps);
  OCLBuiltinTransInfo Info;
  if (HasBoolReturnType)
    Info.RetTy = Type::getInt1Ty(*Ctx);
  Info.UniqName = DemangledName;
  Info.PostProc = [=](std::vector<Value *> &Ops) {
    if (HasBoolArg) {
      IRBuilder<> IRB(CI);
      Ops[0] =
          IRB.CreateICmpNE(Ops[0], ConstantInt::get(Type::getInt32Ty(*Ctx), 0));
    }
    size_t E = Ops.size();
    if (DemangledName == "group_broadcast" && E > 2) {
      assert(E == 3 || E == 4);
      makeVector(CI, Ops, std::make_pair(Ops.begin() + 1, Ops.end()));
    }
    Ops.insert(Ops.begin(), Consts.begin(), Consts.end());
  };
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::transBuiltin(CallInst *CI, OCLBuiltinTransInfo &Info) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Op OC = OpNop;
  unsigned ExtOp = ~0U;
  SPIRVBuiltinVariableKind BVKind = BuiltInMax;
  if (StringRef(Info.UniqName).startswith(kSPIRVName::Prefix))
    return;
  if (OCLSPIRVBuiltinMap::find(Info.UniqName, &OC)) {
    if (OC == OpImageRead) {
      // There are several read_image* functions defined by OpenCL C spec, but
      // all of them use the same SPIR-V Instruction - some of them might only
      // differ by return type, so, we need to include return type into the
      // mangling scheme to get them differentiated.
      //
      // Example: int4 read_imagei(image2d_t, sampler_t, int2)
      //          uint4 read_imageui(image2d_t, sampler_t, int2)
      // Both functions above are represented by the same SPIR-V
      // instruction: argument types are the same, only return type is
      // different
      Info.UniqName = getSPIRVFuncName(OC, CI->getType());
    } else {
      Info.UniqName = getSPIRVFuncName(OC);
    }
  } else if ((ExtOp = getExtOp(Info.MangledName, Info.UniqName)) != ~0U)
    Info.UniqName = getSPIRVExtFuncName(SPIRVEIS_OpenCL, ExtOp);
  else if (SPIRSPIRVBuiltinVariableMap::find(Info.UniqName, &BVKind)) {
    // Map OCL work item builtins to SPV-IR work item builtins.
    // e.g. get_global_id() --> __spirv_BuiltinGlobalInvocationId()
    Info.UniqName = getSPIRVFuncName(BVKind);
  } else
    return;
  if (!Info.RetTy)
    mutateCallInstSPIRV(
        M, CI,
        [=](CallInst *, std::vector<Value *> &Args) {
          Info.PostProc(Args);
          return Info.UniqName + Info.Postfix;
        },
        &Attrs);
  else
    mutateCallInstSPIRV(
        M, CI,
        [=](CallInst *, std::vector<Value *> &Args, Type *&RetTy) {
          Info.PostProc(Args);
          RetTy = Info.RetTy;
          return Info.UniqName + Info.Postfix;
        },
        [=](CallInst *NewCI) -> Instruction * {
          if (NewCI->getType()->isIntegerTy() && CI->getType()->isIntegerTy())
            return CastInst::CreateIntegerCast(NewCI, CI->getType(),
                                               Info.IsRetSigned, "", CI);
          else
            return CastInst::CreatePointerBitCastOrAddrSpaceCast(
                NewCI, CI->getType(), "", CI);
        },
        &Attrs);
}

void OCLToSPIRVBase::visitCallReadImageMSAA(CallInst *CI,
                                            StringRef MangledName) {
  assert(MangledName.find("msaa") != StringRef::npos);
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.insert(Args.begin() + 2, getInt32(M, ImageOperandsSampleMask));
        return getSPIRVFuncName(OpImageRead,
                                std::string(kSPIRVPostfix::ExtDivider) +
                                    getPostfixForReturnType(CI));
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallReadImageWithSampler(CallInst *CI,
                                                   StringRef MangledName) {
  assert(MangledName.find(kMangledName::Sampler) != StringRef::npos);
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  bool IsRetScalar = !CI->getType()->isVectorTy();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
        auto ImageTy = OCLTypeToSPIRVPtr->getAdaptedType(Args[0]);
        if (isOCLImageType(ImageTy))
          ImageTy = getSPIRVImageTypeFromOCL(M, ImageTy);
        auto SampledImgTy = getSPIRVTypeByChangeBaseTypeName(
            M, ImageTy, kSPIRVTypeName::Image, kSPIRVTypeName::SampledImg);
        Value *SampledImgArgs[] = {Args[0], Args[1]};
        auto SampledImg = addCallInstSPIRV(
            M, getSPIRVFuncName(OpSampledImage), SampledImgTy, SampledImgArgs,
            nullptr, CI, kSPIRVName::TempSampledImage);

        Args[0] = SampledImg;
        Args.erase(Args.begin() + 1, Args.begin() + 2);

        switch (Args.size()) {
        case 2: // no lod
          Args.push_back(getInt32(M, ImageOperandsMask::ImageOperandsLodMask));
          Args.push_back(getFloat32(M, 0.f));
          break;
        case 3: // explicit lod
          Args.insert(Args.begin() + 2,
                      getInt32(M, ImageOperandsMask::ImageOperandsLodMask));
          break;
        case 4: // gradient
          Args.insert(Args.begin() + 2,
                      getInt32(M, ImageOperandsMask::ImageOperandsGradMask));
          break;
        default:
          assert(0 && "read_image* with unhandled number of args!");
        }

        // SPIR-V instruction always returns 4-element vector
        if (IsRetScalar)
          Ret = FixedVectorType::get(Ret, 4);
        return getSPIRVFuncName(OpImageSampleExplicitLod,
                                std::string(kSPIRVPostfix::ExtDivider) +
                                    getPostfixForReturnType(Ret));
      },
      [&](CallInst *CI) -> Instruction * {
        if (IsRetScalar)
          return ExtractElementInst::Create(CI, getSizet(M, 0), "",
                                            CI->getNextNode());
        return CI;
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallGetImageSize(CallInst *CI,
                                           StringRef DemangledName) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  StringRef TyName;
  SmallVector<StringRef, 4> SubStrs;
  auto IsImg = isOCLImageType(CI->getArgOperand(0)->getType(), &TyName);
  (void)IsImg;
  assert(IsImg);
  std::string ImageTyName = getImageBaseTypeName(TyName);
  auto Desc = map<SPIRVTypeImageDescriptor>(ImageTyName);
  unsigned Dim = getImageDimension(Desc.Dim) + Desc.Arrayed;
  assert(Dim > 0 && "Invalid image dimension.");
  mutateCallInstSPIRV(
      M, CI,
      [&](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
        assert(Args.size() == 1);
        Ret = CI->getType()->isIntegerTy(64) ? Type::getInt64Ty(*Ctx)
                                             : Type::getInt32Ty(*Ctx);
        if (Dim > 1)
          Ret = FixedVectorType::get(Ret, Dim);
        if (Desc.Dim == DimBuffer)
          return getSPIRVFuncName(OpImageQuerySize, CI->getType());
        else {
          Args.push_back(getInt32(M, 0));
          return getSPIRVFuncName(OpImageQuerySizeLod, CI->getType());
        }
      },
      [&](CallInst *NCI) -> Instruction * {
        if (Dim == 1)
          return NCI;
        if (DemangledName == kOCLBuiltinName::GetImageDim) {
          if (Desc.Dim == Dim3D) {
            auto ZeroVec = ConstantVector::getSplat(
                ElementCount::getFixed(3),
                Constant::getNullValue(
                    cast<VectorType>(NCI->getType())->getElementType()));
            Constant *Index[] = {getInt32(M, 0), getInt32(M, 1), getInt32(M, 2),
                                 getInt32(M, 3)};
            return new ShuffleVectorInst(NCI, ZeroVec,
                                         ConstantVector::get(Index), "", CI);

          } else if (Desc.Dim == Dim2D && Desc.Arrayed) {
            Constant *Index[] = {getInt32(M, 0), getInt32(M, 1)};
            Constant *Mask = ConstantVector::get(Index);
            return new ShuffleVectorInst(NCI, UndefValue::get(NCI->getType()),
                                         Mask, NCI->getName(), CI);
          }
          return NCI;
        }
        unsigned I = StringSwitch<unsigned>(DemangledName)
                         .Case(kOCLBuiltinName::GetImageWidth, 0)
                         .Case(kOCLBuiltinName::GetImageHeight, 1)
                         .Case(kOCLBuiltinName::GetImageDepth, 2)
                         .Case(kOCLBuiltinName::GetImageArraySize, Dim - 1);
        return ExtractElementInst::Create(NCI, getUInt32(M, I), "",
                                          NCI->getNextNode());
      },
      &Attrs);
}

/// Remove trivial conversion functions
bool OCLToSPIRVBase::eraseUselessConvert(CallInst *CI, StringRef MangledName,
                                         StringRef DemangledName) {
  auto TargetTy = CI->getType();
  auto SrcTy = CI->getArgOperand(0)->getType();
  if (auto *VecTy = dyn_cast<VectorType>(TargetTy))
    TargetTy = VecTy->getElementType();
  if (auto *VecTy = dyn_cast<VectorType>(SrcTy))
    SrcTy = VecTy->getElementType();
  if (TargetTy == SrcTy) {
    if (isa<IntegerType>(TargetTy) &&
        DemangledName.find("_sat") != StringRef::npos &&
        isLastFuncParamSigned(MangledName) != (DemangledName[8] != 'u'))
      return false;
    CI->getArgOperand(0)->takeName(CI);
    SPIRVDBG(dbgs() << "[regularizeOCLConvert] " << *CI << " <- "
                    << *CI->getArgOperand(0) << '\n');
    CI->replaceAllUsesWith(CI->getArgOperand(0));
    ValuesToDelete.insert(CI);
    ValuesToDelete.insert(CI->getCalledFunction());
    return true;
  }
  return false;
}

void OCLToSPIRVBase::visitCallBuiltinSimple(CallInst *CI, StringRef MangledName,
                                            StringRef DemangledName) {
  OCLBuiltinTransInfo Info;
  Info.MangledName = MangledName.str();
  Info.UniqName = DemangledName.str();
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallReadWriteImage(CallInst *CI,
                                             StringRef DemangledName) {
  OCLBuiltinTransInfo Info;
  if (DemangledName.find(kOCLBuiltinName::ReadImage) == 0)
    Info.UniqName = kOCLBuiltinName::ReadImage;

  if (DemangledName.find(kOCLBuiltinName::WriteImage) == 0) {
    Info.UniqName = kOCLBuiltinName::WriteImage;
    Info.PostProc = [&](std::vector<Value *> &Args) {
      if (Args.size() == 4) // write with lod
      {
        auto Lod = Args[2];
        Args.erase(Args.begin() + 2);
        Args.push_back(getInt32(M, ImageOperandsMask::ImageOperandsLodMask));
        Args.push_back(Lod);
      }
    };
  }

  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallToAddr(CallInst *CI, StringRef DemangledName) {
  auto AddrSpace =
      static_cast<SPIRAddressSpace>(CI->getType()->getPointerAddressSpace());
  OCLBuiltinTransInfo Info;
  Info.UniqName = DemangledName.str();
  Info.Postfix = std::string(kSPIRVPostfix::Divider) + "To" +
                 SPIRAddrSpaceCapitalizedNameMap::map(AddrSpace);
  auto StorageClass = addInt32(SPIRSPIRVAddrSpaceMap::map(AddrSpace));
  Info.RetTy = getInt8PtrTy(cast<PointerType>(CI->getType()));
  Info.PostProc = [=](std::vector<Value *> &Ops) {
    auto P = Ops.back();
    Ops.pop_back();
    Ops.push_back(castToInt8Ptr(P, CI));
    Ops.push_back(StorageClass);
  };
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallRelational(CallInst *CI,
                                         StringRef DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Op OC = OpNop;
  OCLSPIRVBuiltinMap::find(DemangledName.str(), &OC);
  std::string SPIRVName = getSPIRVFuncName(OC);
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
        Ret = Type::getInt1Ty(*Ctx);
        if (CI->getOperand(0)->getType()->isVectorTy())
          Ret = FixedVectorType::get(
              Type::getInt1Ty(*Ctx),
              cast<FixedVectorType>(CI->getOperand(0)->getType())
                  ->getNumElements());
        return SPIRVName;
      },
      [=](CallInst *NewCI) -> Instruction * {
        Value *False = nullptr, *True = nullptr;
        if (NewCI->getType()->isVectorTy()) {
          Type *IntTy = Type::getInt32Ty(*Ctx);
          if (cast<FixedVectorType>(NewCI->getOperand(0)->getType())
                  ->getElementType()
                  ->isDoubleTy())
            IntTy = Type::getInt64Ty(*Ctx);
          if (cast<FixedVectorType>(NewCI->getOperand(0)->getType())
                  ->getElementType()
                  ->isHalfTy())
            IntTy = Type::getInt16Ty(*Ctx);
          Type *VTy = FixedVectorType::get(
              IntTy, cast<FixedVectorType>(NewCI->getType())->getNumElements());
          False = Constant::getNullValue(VTy);
          True = Constant::getAllOnesValue(VTy);
        } else {
          False = getInt32(M, 0);
          True = getInt32(M, 1);
        }
        return SelectInst::Create(NewCI, True, False, "", NewCI->getNextNode());
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallVecLoadStore(CallInst *CI, StringRef MangledName,
                                           StringRef OrigDemangledName) {
  std::vector<int> PreOps;
  std::string DemangledName{OrigDemangledName};
  if (DemangledName.find(kOCLBuiltinName::VLoadPrefix) == 0 &&
      DemangledName != kOCLBuiltinName::VLoadHalf) {
    SPIRVWord Width = getVecLoadWidth(DemangledName);
    SPIRVDBG(spvdbgs() << "[visitCallVecLoadStore] DemangledName: "
                       << DemangledName << " Width: " << Width << '\n');
    PreOps.push_back(Width);
  } else if (DemangledName.find(kOCLBuiltinName::RoundingPrefix) !=
             std::string::npos) {
    auto R = SPIRSPIRVFPRoundingModeMap::map(DemangledName.substr(
        DemangledName.find(kOCLBuiltinName::RoundingPrefix) + 1, 3));
    PreOps.push_back(R);
  }

  if (DemangledName.find(kOCLBuiltinName::VLoadAPrefix) == 0)
    transVecLoadStoreName(DemangledName, kOCLBuiltinName::VLoadAPrefix, true);
  else
    transVecLoadStoreName(DemangledName, kOCLBuiltinName::VLoadPrefix, false);

  if (DemangledName.find(kOCLBuiltinName::VStoreAPrefix) == 0)
    transVecLoadStoreName(DemangledName, kOCLBuiltinName::VStoreAPrefix, true);
  else
    transVecLoadStoreName(DemangledName, kOCLBuiltinName::VStorePrefix, false);

  auto Consts = getInt32(M, PreOps);
  OCLBuiltinTransInfo Info;
  Info.MangledName = MangledName.str();
  Info.UniqName = DemangledName;
  if (DemangledName.find(kOCLBuiltinName::VLoadPrefix) == 0)
    Info.Postfix =
        std::string(kSPIRVPostfix::ExtDivider) + getPostfixForReturnType(CI);
  Info.PostProc = [=](std::vector<Value *> &Ops) {
    Ops.insert(Ops.end(), Consts.begin(), Consts.end());
  };
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallGetFence(CallInst *CI, StringRef DemangledName) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Op OC = OpNop;
  OCLSPIRVBuiltinMap::find(DemangledName.str(), &OC);
  std::string SPIRVName = getSPIRVFuncName(OC);
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
        return SPIRVName;
      },
      [=](CallInst *NewCI) -> Instruction * {
        return BinaryOperator::CreateLShr(NewCI, getInt32(M, 8), "", CI);
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallDot(CallInst *CI) {
  IRBuilder<> Builder(CI);
  Value *FMulVal = Builder.CreateFMul(CI->getOperand(0), CI->getOperand(1));
  CI->replaceAllUsesWith(FMulVal);
  CI->eraseFromParent();
}

void OCLToSPIRVBase::visitCallScalToVec(CallInst *CI, StringRef MangledName,
                                        StringRef DemangledName) {
  // Check if all arguments have the same type - it's simple case.
  auto Uniform = true;
  auto IsArg0Vector = isa<VectorType>(CI->getOperand(0)->getType());
  for (unsigned I = 1, E = CI->arg_size(); Uniform && (I != E); ++I) {
    Uniform = isa<VectorType>(CI->getOperand(I)->getType()) == IsArg0Vector;
  }
  if (Uniform) {
    visitCallBuiltinSimple(CI, MangledName, DemangledName);
    return;
  }

  std::vector<unsigned int> VecPos;
  std::vector<unsigned int> ScalarPos;
  if (DemangledName == kOCLBuiltinName::FMin ||
      DemangledName == kOCLBuiltinName::FMax ||
      DemangledName == kOCLBuiltinName::Min ||
      DemangledName == kOCLBuiltinName::Max) {
    VecPos.push_back(0);
    ScalarPos.push_back(1);
  } else if (DemangledName == kOCLBuiltinName::Clamp) {
    VecPos.push_back(0);
    ScalarPos.push_back(1);
    ScalarPos.push_back(2);
  } else if (DemangledName == kOCLBuiltinName::Mix) {
    VecPos.push_back(0);
    VecPos.push_back(1);
    ScalarPos.push_back(2);
  } else if (DemangledName == kOCLBuiltinName::Step) {
    VecPos.push_back(1);
    ScalarPos.push_back(0);
  } else if (DemangledName == kOCLBuiltinName::SmoothStep) {
    VecPos.push_back(2);
    ScalarPos.push_back(0);
    ScalarPos.push_back(1);
  }

  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.resize(VecPos.size() + ScalarPos.size());
        for (auto I : VecPos) {
          Args[I] = CI->getOperand(I);
        }
        auto VecElemCount =
            cast<VectorType>(CI->getOperand(VecPos[0])->getType())
                ->getElementCount();
        for (auto I : ScalarPos) {
          Instruction *Inst = InsertElementInst::Create(
              UndefValue::get(CI->getOperand(VecPos[0])->getType()),
              CI->getOperand(I), getInt32(M, 0), "", CI);
          Value *NewVec = new ShuffleVectorInst(
              Inst, UndefValue::get(CI->getOperand(VecPos[0])->getType()),
              ConstantVector::getSplat(VecElemCount, getInt32(M, 0)), "", CI);

          Args[I] = NewVec;
        }
        return getSPIRVExtFuncName(SPIRVEIS_OpenCL,
                                   getExtOp(MangledName, DemangledName));
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallGetImageChannel(CallInst *CI,
                                              StringRef DemangledName,
                                              unsigned int Offset) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  Op OC = OpNop;
  OCLSPIRVBuiltinMap::find(DemangledName.str(), &OC);
  std::string SPIRVName = getSPIRVFuncName(OC);
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
        return SPIRVName;
      },
      [=](CallInst *NewCI) -> Instruction * {
        return BinaryOperator::CreateAdd(NewCI, getInt32(M, Offset), "", CI);
      },
      &Attrs);
}
void OCLToSPIRVBase::visitCallEnqueueKernel(CallInst *CI,
                                            StringRef DemangledName) {
  const DataLayout &DL = M->getDataLayout();
  bool HasEvents = DemangledName.find("events") != StringRef::npos;

  // SPIRV OpEnqueueKernel instruction has 10+ arguments.
  SmallVector<Value *, 16> Args;

  // Copy all arguments before block invoke function pointer
  // which match with what Clang 6.0 produced
  const unsigned BlockFIdx = HasEvents ? 6 : 3;
  Args.assign(CI->arg_begin(), CI->arg_begin() + BlockFIdx);

  // If no event arguments in original call, add dummy ones
  if (!HasEvents) {
    Args.push_back(getInt32(M, 0));           // dummy num events
    Args.push_back(getOCLNullClkEventPtr(M)); // dummy wait events
    Args.push_back(getOCLNullClkEventPtr(M)); // dummy ret event
  }

  // Invoke: Pointer to invoke function
  Value *BlockFunc = CI->getArgOperand(BlockFIdx);
  Args.push_back(cast<Function>(getUnderlyingObject(BlockFunc)));

  // Param: Pointer to block literal
  Value *BlockLiteral = CI->getArgOperand(BlockFIdx + 1);
  Args.push_back(BlockLiteral);

  // Param Size: Size of block literal structure
  // Param Aligment: Aligment of block literal structure
  // TODO: these numbers should be obtained from block literal structure
  Type *ParamType = getUnderlyingObject(BlockLiteral)->getType();
  if (PointerType *PT = dyn_cast<PointerType>(ParamType))
    ParamType = PT->getPointerElementType();
  Args.push_back(getInt32(M, DL.getTypeStoreSize(ParamType)));
  Args.push_back(getInt32(M, DL.getPrefTypeAlignment(ParamType)));

  // Local sizes arguments: Sizes of block invoke arguments
  // Clang 6.0 and higher generates local size operands as an array,
  // so we need to unpack them
  if (DemangledName.find("_varargs") != StringRef::npos) {
    const unsigned LocalSizeArrayIdx = HasEvents ? 9 : 6;
    auto *LocalSizeArray =
        cast<GetElementPtrInst>(CI->getArgOperand(LocalSizeArrayIdx));
    auto *LocalSizeArrayTy =
        cast<ArrayType>(LocalSizeArray->getSourceElementType());
    const uint64_t LocalSizeNum = LocalSizeArrayTy->getNumElements();
    for (unsigned I = 0; I < LocalSizeNum; ++I)
      Args.push_back(GetElementPtrInst::Create(
          LocalSizeArray->getSourceElementType(), // Pointee type
          LocalSizeArray->getPointerOperand(),    // Alloca
          {getInt32(M, 0), getInt32(M, I)},       // Indices
          "", CI));
  }

  StringRef NewName = "__spirv_EnqueueKernel__";
  FunctionType *FT =
      FunctionType::get(CI->getType(), getTypes(Args), false /*isVarArg*/);
  Function *NewF =
      Function::Create(FT, GlobalValue::ExternalLinkage, NewName, M);
  NewF->setCallingConv(CallingConv::SPIR_FUNC);
  CallInst *NewCall = CallInst::Create(NewF, Args, "", CI);
  NewCall->setCallingConv(NewF->getCallingConv());
  CI->replaceAllUsesWith(NewCall);
  CI->eraseFromParent();
}

void OCLToSPIRVBase::visitCallKernelQuery(CallInst *CI,
                                          StringRef DemangledName) {
  const DataLayout &DL = M->getDataLayout();
  bool HasNDRange = DemangledName.find("_for_ndrange_impl") != StringRef::npos;
  // BIs with "_for_ndrange_impl" suffix has NDRange argument first, and
  // Invoke argument following. For other BIs Invoke function is the first arg
  const unsigned BlockFIdx = HasNDRange ? 1 : 0;
  Value *BlockFVal = CI->getArgOperand(BlockFIdx)->stripPointerCasts();

  auto *BlockF = cast<Function>(getUnderlyingObject(BlockFVal));

  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInst(
      M, CI,
      [=](CallInst *CI, std::vector<Value *> &Args) {
        Value *Param = *Args.rbegin();
        Type *ParamType = getUnderlyingObject(Param)->getType();
        if (PointerType *PT = dyn_cast<PointerType>(ParamType)) {
          ParamType = PT->getPointerElementType();
        }
        // Last arg corresponds to SPIRV Param operand.
        // Insert Invoke in front of Param.
        // Add Param Size and Param Align at the end.
        Args[BlockFIdx] = BlockF;
        Args.push_back(getInt32(M, DL.getTypeStoreSize(ParamType)));
        Args.push_back(getInt32(M, DL.getPrefTypeAlignment(ParamType)));

        Op Opcode = OCLSPIRVBuiltinMap::map(DemangledName.str());
        // Adding "__" postfix, so in case we have multiple such
        // functions and their names will have numerical postfix,
        // then the numerical postfix will be droped and we will get
        // correct function name.
        return getSPIRVFuncName(Opcode, kSPIRVName::Postfix);
      },
      /*BuiltinFuncMangleInfo*/ nullptr, &Attrs);
}

// Add postfix to overloaded intel subgroup block read/write builtins
// so new functions can be distinguished.
static void processSubgroupBlockReadWriteINTEL(CallInst *CI,
                                               OCLBuiltinTransInfo &Info,
                                               const Type *DataTy, Module *M) {
  unsigned VectorNumElements = 1;
  if (auto *VecTy = dyn_cast<FixedVectorType>(DataTy))
    VectorNumElements = VecTy->getNumElements();
  unsigned ElementBitSize = DataTy->getScalarSizeInBits();
  Info.Postfix = "_";
  Info.Postfix +=
      getIntelSubgroupBlockDataPostfix(ElementBitSize, VectorNumElements);
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [&Info](CallInst *, std::vector<Value *> &Args) {
        Info.PostProc(Args);
        return Info.UniqName + Info.Postfix;
      },
      &Attrs);
}

// The intel_sub_group_block_read built-ins are overloaded to support both
// buffers and images, but need to be mapped to distinct SPIR-V instructions.
// Additionally, for block reads, need to distinguish between scalar block
// reads and vector block reads.
void OCLToSPIRVBase::visitSubgroupBlockReadINTEL(CallInst *CI) {
  OCLBuiltinTransInfo Info;
  if (isOCLImageType(CI->getArgOperand(0)->getType()))
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupImageBlockReadINTEL);
  else
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupBlockReadINTEL);
  Type *DataTy = CI->getType();
  processSubgroupBlockReadWriteINTEL(CI, Info, DataTy, M);
}

// The intel_sub_group_block_write built-ins are similarly overloaded to support
// both buffers and images but need to be mapped to distinct SPIR-V
// instructions.
void OCLToSPIRVBase::visitSubgroupBlockWriteINTEL(CallInst *CI) {
  OCLBuiltinTransInfo Info;
  if (isOCLImageType(CI->getArgOperand(0)->getType()))
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupImageBlockWriteINTEL);
  else
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupBlockWriteINTEL);
  assert(!CI->arg_empty() &&
         "Intel subgroup block write should have arguments");
  unsigned DataArg = CI->arg_size() - 1;
  Type *DataTy = CI->getArgOperand(DataArg)->getType();
  processSubgroupBlockReadWriteINTEL(CI, Info, DataTy, M);
}

void OCLToSPIRVBase::visitSubgroupImageMediaBlockINTEL(
    CallInst *CI, StringRef DemangledName) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  spv::Op OpCode = DemangledName.rfind("read") != StringRef::npos
                       ? spv::OpSubgroupImageMediaBlockReadINTEL
                       : spv::OpSubgroupImageMediaBlockWriteINTEL;
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        // Moving the last argument to the beginning.
        std::rotate(Args.begin(), Args.end() - 1, Args.end());
        return getSPIRVFuncName(OpCode, CI->getType());
      },
      &Attrs);
}

static const char *getSubgroupAVCIntelOpKind(StringRef Name) {
  return StringSwitch<const char *>(Name.data())
      .StartsWith(kOCLSubgroupsAVCIntel::IMEPrefix, "ime")
      .StartsWith(kOCLSubgroupsAVCIntel::REFPrefix, "ref")
      .StartsWith(kOCLSubgroupsAVCIntel::SICPrefix, "sic");
}

static const char *getSubgroupAVCIntelTyKind(Type *Ty) {
  auto *STy = cast<StructType>(cast<PointerType>(Ty)->getPointerElementType());
  auto TName = STy->getName();
  return TName.endswith("_payload_t") ? "payload" : "result";
}

static Type *getSubgroupAVCIntelMCEType(Module *M, std::string &TName) {
  auto Ty = StructType::getTypeByName(M->getContext(), TName);
  if (Ty)
    return Ty;

  return StructType::create(M->getContext(), TName);
}

static Op getSubgroupAVCIntelMCEOpCodeForWrapper(StringRef DemangledName) {
  if (DemangledName.size() <= strlen(kOCLSubgroupsAVCIntel::MCEPrefix))
    return OpNop; // this is not a VME built-in

  std::string MCEName{DemangledName};
  MCEName.replace(0, strlen(kOCLSubgroupsAVCIntel::MCEPrefix),
                  kOCLSubgroupsAVCIntel::MCEPrefix);
  Op MCEOC = OpNop;
  OCLSPIRVSubgroupAVCIntelBuiltinMap::find(MCEName, &MCEOC);
  return MCEOC;
}

// Handles Subgroup AVC Intel extension generic built-ins.
void OCLToSPIRVBase::visitSubgroupAVCBuiltinCall(CallInst *CI,
                                                 StringRef DemangledName) {
  Op OC = OpNop;
  std::string FName{DemangledName};
  std::string Prefix = kOCLSubgroupsAVCIntel::Prefix;

  // Update names for built-ins mapped on two or more SPIRV instructions
  if (FName.find(Prefix + "ime_get_streamout_major_shape_") == 0) {
    auto PTy = cast<PointerType>(CI->getArgOperand(0)->getType());
    auto *STy = cast<StructType>(PTy->getPointerElementType());
    assert(STy->hasName() && "Invalid Subgroup AVC Intel built-in call");
    FName += (STy->getName().contains("single")) ? "_single_reference"
                                                 : "_dual_reference";
  } else if (FName.find(Prefix + "sic_configure_ipe") == 0) {
    FName += (CI->arg_size() == 8) ? "_luma" : "_luma_chroma";
  }

  OCLSPIRVSubgroupAVCIntelBuiltinMap::find(FName, &OC);
  if (OC == OpNop) {
    if (Op MCEOC = getSubgroupAVCIntelMCEOpCodeForWrapper(DemangledName))
      // The called function is a VME wrapper built-in
      return visitSubgroupAVCWrapperBuiltinCall(CI, MCEOC, DemangledName);
    else
      // The called function isn't a VME built-in
      return;
  }

  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return getSPIRVFuncName(OC);
      },
      &Attrs);
}

// Handles Subgroup AVC Intel extension wrapper built-ins.
// 'IME', 'REF' and 'SIC' sets contain wrapper built-ins which don't have
// corresponded instructions in SPIRV and should be translated to a
// conterpart from 'MCE' with conversion for an argument and result (if needed).
void OCLToSPIRVBase::visitSubgroupAVCWrapperBuiltinCall(
    CallInst *CI, Op WrappedOC, StringRef DemangledName) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  std::string Prefix = kOCLSubgroupsAVCIntel::Prefix;

  // Find 'to_mce' conversion function.
  // The operand required conversion is always the last one.
  const char *OpKind = getSubgroupAVCIntelOpKind(DemangledName);
  const char *TyKind = getSubgroupAVCIntelTyKind(
      CI->getArgOperand(CI->arg_size() - 1)->getType());
  std::string MCETName =
      std::string(kOCLSubgroupsAVCIntel::TypePrefix) + "mce_" + TyKind + "_t";
  auto *MCETy =
      PointerType::get(getSubgroupAVCIntelMCEType(M, MCETName), SPIRAS_Private);
  std::string ToMCEFName = Prefix + OpKind + "_convert_to_mce_" + TyKind;
  Op ToMCEOC = OpNop;
  OCLSPIRVSubgroupAVCIntelBuiltinMap::find(ToMCEFName, &ToMCEOC);
  assert(ToMCEOC != OpNop && "Invalid Subgroup AVC Intel built-in call");

  if (std::strcmp(TyKind, "payload") == 0) {
    // Wrapper built-ins which take the 'payload_t' argument return it as
    // the result: two conversion calls required.
    std::string FromMCEFName =
        Prefix + "mce_convert_to_" + OpKind + "_" + TyKind;
    Op FromMCEOC = OpNop;
    OCLSPIRVSubgroupAVCIntelBuiltinMap::find(FromMCEFName, &FromMCEOC);
    assert(FromMCEOC != OpNop && "Invalid Subgroup AVC Intel built-in call");

    mutateCallInstSPIRV(
        M, CI,
        [=](CallInst *, std::vector<Value *> &Args, Type *&Ret) {
          Ret = MCETy;
          // Create conversion function call for the last operand
          Args[Args.size() - 1] =
              addCallInstSPIRV(M, getSPIRVFuncName(ToMCEOC), MCETy,
                               Args[Args.size() - 1], nullptr, CI, "");

          return getSPIRVFuncName(WrappedOC);
        },
        [=](CallInst *NewCI) -> Instruction * {
          // Create conversion function call for the return result
          return addCallInstSPIRV(M, getSPIRVFuncName(FromMCEOC), CI->getType(),
                                  NewCI, nullptr, CI, "");
        },
        &Attrs);
  } else {
    // Wrapper built-ins which take the 'result_t' argument requires only one
    // conversion for the argument
    mutateCallInstSPIRV(
        M, CI,
        [=](CallInst *, std::vector<Value *> &Args) {
          // Create conversion function call for the last
          // operand
          Args[Args.size() - 1] =
              addCallInstSPIRV(M, getSPIRVFuncName(ToMCEOC), MCETy,
                               Args[Args.size() - 1], nullptr, CI, "");

          return getSPIRVFuncName(WrappedOC);
        },
        &Attrs);
  }
}

// Handles Subgroup AVC Intel extension built-ins which take sampler as
// an argument (their SPIR-V counterparts take OpTypeVmeImageIntel instead)
void OCLToSPIRVBase::visitSubgroupAVCBuiltinCallWithSampler(
    CallInst *CI, StringRef DemangledName) {
  std::string FName{DemangledName};
  std::string Prefix = kOCLSubgroupsAVCIntel::Prefix;

  // Update names for built-ins mapped on two or more SPIRV instructions
  if (FName.find(Prefix + "ref_evaluate_with_multi_reference") == 0 ||
      FName.find(Prefix + "sic_evaluate_with_multi_reference") == 0) {
    FName += (CI->arg_size() == 5) ? "_interlaced" : "";
  }

  Op OC = OpNop;
  OCLSPIRVSubgroupAVCIntelBuiltinMap::find(FName, &OC);
  if (OC == OpNop)
    return; // this is not a VME built-in

  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstSPIRV(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        auto SamplerIt = std::find_if(Args.begin(), Args.end(), [](Value *V) {
          return OCLUtil::isSamplerTy(V->getType());
        });
        assert(SamplerIt != Args.end() &&
               "Invalid Subgroup AVC Intel built-in call");
        auto *SamplerVal = *SamplerIt;
        Args.erase(SamplerIt);

        for (unsigned I = 0, E = Args.size(); I < E; ++I) {
          if (!isOCLImageType(Args[I]->getType()))
            continue;

          auto *ImageTy = OCLTypeToSPIRVPtr->getAdaptedType(Args[I]);
          if (isOCLImageType(ImageTy))
            ImageTy = getSPIRVImageTypeFromOCL(M, ImageTy);
          auto *SampledImgTy = getSPIRVTypeByChangeBaseTypeName(
              M, ImageTy, kSPIRVTypeName::Image, kSPIRVTypeName::VmeImageINTEL);

          Value *SampledImgArgs[] = {Args[I], SamplerVal};
          Args[I] = addCallInstSPIRV(M, getSPIRVFuncName(OpVmeImageINTEL),
                                     SampledImgTy, SampledImgArgs, nullptr, CI,
                                     kSPIRVName::TempSampledImage);
        }
        return getSPIRVFuncName(OC);
      },
      &Attrs);
}

void OCLToSPIRVBase::visitCallLdexp(CallInst *CI, StringRef MangledName,
                                    StringRef DemangledName) {
  auto Args = getArguments(CI);
  if (Args.size() == 2) {
    Type *Type0 = Args[0]->getType();
    Type *Type1 = Args[1]->getType();
    // For OpenCL built-in math functions 'halfn ldexp(halfn x, int k)',
    // 'floatn ldexp(floatn x, int k)' and 'doublen ldexp (doublen x, int k)',
    // convert scalar arg to vector to keep consistency with SPIRV spec.
    // Regarding to SPIRV OpenCL Extended Instruction set, k operand must have
    // the same component count as Result Type and x operands
    if (auto *FixedVecType0 = dyn_cast<FixedVectorType>(Type0)) {
      auto ScalarTypeID = Type0->getScalarType()->getTypeID();
      if ((ScalarTypeID == llvm::Type::FloatTyID ||
           ScalarTypeID == llvm::Type::DoubleTyID ||
           ScalarTypeID == llvm::Type::HalfTyID) &&
          Type1->isIntegerTy()) {
        IRBuilder<> IRB(CI);
        unsigned Width = FixedVecType0->getNumElements();
        CI->setOperand(1, IRB.CreateVectorSplat(Width, CI->getArgOperand(1)));
      }
    }
  }
  visitCallBuiltinSimple(CI, MangledName, DemangledName);
}

} // namespace SPIRV

INITIALIZE_PASS_BEGIN(OCLToSPIRVLegacy, "ocl-to-spv",
                      "Transform OCL 2.0 to SPIR-V", false, false)
INITIALIZE_PASS_DEPENDENCY(OCLTypeToSPIRVLegacy)
INITIALIZE_PASS_END(OCLToSPIRVLegacy, "ocl-to-spv",
                    "Transform OCL 2.0 to SPIR-V", false, false)

ModulePass *llvm::createOCLToSPIRVLegacy() { return new OCLToSPIRVLegacy(); }
