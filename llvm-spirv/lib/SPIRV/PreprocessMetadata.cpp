//===- PreprocessMetadata.cpp -                                   - C++ -*-===//
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
// This file implements preprocessing of LLVM IR metadata in order to perform
// further translation to SPIR-V.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "clmdtospv"

#include "PreprocessMetadata.h"
#include "OCLUtil.h"
#include "SPIRVInternal.h"
#include "SPIRVMDBuilder.h"
#include "SPIRVMDWalker.h"
#include "VectorComputeUtil.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm;
using namespace SPIRV;
using namespace OCLUtil;

namespace SPIRV {

cl::opt<bool> EraseOCLMD("spirv-erase-cl-md", cl::init(true),
                         cl::desc("Erase OpenCL metadata"));

char PreprocessMetadataLegacy::ID = 0;

bool PreprocessMetadataLegacy::runOnModule(Module &Module) {
  return runPreprocessMetadata(Module);
}

llvm::PreservedAnalyses
PreprocessMetadataPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  return runPreprocessMetadata(M) ? llvm::PreservedAnalyses::none()
                                  : llvm::PreservedAnalyses::all();
}

bool PreprocessMetadataBase::runPreprocessMetadata(Module &Module) {
  M = &Module;
  Ctx = &M->getContext();

  LLVM_DEBUG(dbgs() << "Enter PreprocessMetadata:\n");
  visit(M);
  LLVM_DEBUG(dbgs() << "After PreprocessMetadata:\n" << *M);

  verifyRegularizationPass(*M, "PreprocessMetadata");

  return true;
}

void PreprocessMetadataBase::preprocessCXXStructorList(
    SPIRVMDBuilder::NamedMDWrapper &EM, GlobalVariable *V,
    ExecutionMode EMode) {
  auto *List = dyn_cast_or_null<ConstantArray>(V->getInitializer());
  if (!List)
    return;

  for (Value *V : List->operands()) {
    auto *Structor = cast<ConstantStruct>(V);

    // Each entry in the list is a struct containing 3 members:
    // (priority, function, data), with function being the entry point.
    auto *Kernel = cast<Function>(Structor->getOperand(1));

    EM.addOp().add(Kernel).add(EMode).done();
  }
}

void PreprocessMetadataBase::visit(Module *M) {
  SPIRVMDBuilder B(*M);
  SPIRVMDWalker W(*M);

  preprocessOCLMetadata(M, &B, &W);
  preprocessVectorComputeMetadata(M, &B, &W);

  // Create metadata representing (empty so far) list
  // of OpExecutionMode instructions
  auto EM = B.addNamedMD(kSPIRVMD::ExecutionMode); // !spirv.ExecutionMode = {}

  // Process special variables in LLVM IR module.
  if (auto *GV = M->getGlobalVariable("llvm.global_ctors"))
    preprocessCXXStructorList(EM, GV, spv::ExecutionModeInitializer);

  // Add execution modes for kernels. We take it from metadata attached to
  // the kernel functions.
  for (Function &Kernel : *M) {
    if (Kernel.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Specifing execution modes for the Kernel and adding it to the list
    // of ExecutionMode instructions.

    // !{void (i32 addrspace(1)*)* @kernel, i32 17, i32 X, i32 Y, i32 Z}
    if (MDNode *WGSize = Kernel.getMetadata(kSPIR2MD::WGSize)) {
      assert(WGSize->getNumOperands() >= 1 && WGSize->getNumOperands() <= 3 &&
             "reqd_work_group_size does not have between 1 and 3 operands.");
      SmallVector<unsigned, 3> DecodedVals = decodeMDNode(WGSize);
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeLocalSize)
          .add(DecodedVals[0])
          .add(DecodedVals.size() >= 2 ? DecodedVals[1] : 1)
          .add(DecodedVals.size() == 3 ? DecodedVals[2] : 1)
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 18, i32 X, i32 Y, i32 Z}
    if (MDNode *WGSizeHint = Kernel.getMetadata(kSPIR2MD::WGSizeHint)) {
      assert(WGSizeHint->getNumOperands() >= 1 &&
             WGSizeHint->getNumOperands() <= 3 &&
             "work_group_size_hint does not have between 1 and 3 operands.");
      SmallVector<unsigned, 3> DecodedVals = decodeMDNode(WGSizeHint);
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeLocalSizeHint)
          .add(DecodedVals[0])
          .add(DecodedVals.size() >= 2 ? DecodedVals[1] : 1)
          .add(DecodedVals.size() == 3 ? DecodedVals[2] : 1)
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 30, i32 hint}
    if (MDNode *VecTypeHint = Kernel.getMetadata(kSPIR2MD::VecTyHint)) {
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeVecTypeHint)
          .add(transVecTypeHint(VecTypeHint))
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 35, i32 size}
    if (MDNode *ReqdSubgroupSize = Kernel.getMetadata(kSPIR2MD::SubgroupSize)) {
      // A primary named subgroup size is encoded as
      // the metadata intel_reqd_sub_group_size with value 0.
      auto Val = getMDOperandAsInt(ReqdSubgroupSize, 0);
      if (Val == 0)
        EM.addOp()
            .add(&Kernel)
            .add(spv::internal::ExecutionModeNamedSubgroupSizeINTEL)
            .add(/* PrimarySubgroupSizeINTEL = */ 0U)
            .done();
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeSubgroupSize)
          .add(Val)
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 max_work_group_size, i32 X,
    //         i32 Y, i32 Z}
    if (MDNode *MaxWorkgroupSizeINTEL =
            Kernel.getMetadata(kSPIR2MD::MaxWGSize)) {
      assert(MaxWorkgroupSizeINTEL->getNumOperands() == 3 &&
             "max_work_group_size does not have 3 operands.");
      SmallVector<unsigned, 3> DecodedVals =
          decodeMDNode(MaxWorkgroupSizeINTEL);
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeMaxWorkgroupSizeINTEL)
          .add(DecodedVals[0])
          .add(DecodedVals[1])
          .add(DecodedVals[2])
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 no_global_work_offset}
    if (Kernel.getMetadata(kSPIR2MD::NoGlobalOffset)) {
      EM.addOp().add(&Kernel).add(spv::ExecutionModeNoGlobalOffsetINTEL).done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 max_global_work_dim, i32 dim}
    if (MDNode *MaxWorkDimINTEL = Kernel.getMetadata(kSPIR2MD::MaxWGDim)) {
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeMaxWorkDimINTEL)
          .add(getMDOperandAsInt(MaxWorkDimINTEL, 0))
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 num_simd_work_items, i32 num}
    if (MDNode *NumSIMDWorkitemsINTEL = Kernel.getMetadata(kSPIR2MD::NumSIMD)) {
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeNumSIMDWorkitemsINTEL)
          .add(getMDOperandAsInt(NumSIMDWorkitemsINTEL, 0))
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 scheduler_target_fmax_mhz,
    //   i32 num}
    if (MDNode *SchedulerTargetFmaxMhzINTEL =
            Kernel.getMetadata(kSPIR2MD::FmaxMhz)) {
      EM.addOp()
          .add(&Kernel)
          .add(spv::ExecutionModeSchedulerTargetFmaxMhzINTEL)
          .add(getMDOperandAsInt(SchedulerTargetFmaxMhzINTEL, 0))
          .done();
    }

    // !{void (i32 addrspace(1)*)* @kernel, i32 ip_interface, i32 interface}
    if (MDNode *Interface =
            Kernel.getMetadata(kSPIR2MD::IntelFPGAIPInterface)) {
      std::set<std::string> InterfaceStrSet;
      for (size_t I = 0; I != Interface->getNumOperands(); ++I)
        InterfaceStrSet.insert(getMDOperandAsString(Interface, I).str());

      // ip_interface metadata will either have Register Map metadata or
      // Streaming metadata.
      //
      // Register Map mode metadata:
      // Not 'WaitForDoneWrite' mode (to be mapped on '0' literal)
      // !ip_interface !N
      // !N = !{!"csr"}
      // 'WaitForDoneWrite' mode (to be mapped on '1' literal)
      // !ip_interface !N
      // !N = !{!"csr", !"wait_for_done_write"}
      if (InterfaceStrSet.find("csr") != InterfaceStrSet.end()) {
        int32_t InterfaceMode = 0;
        if (InterfaceStrSet.find("wait_for_done_write") !=
            InterfaceStrSet.end())
          InterfaceMode = 1;
        EM.addOp()
            .add(&Kernel)
            .add(spv::ExecutionModeRegisterMapInterfaceINTEL)
            .add(InterfaceMode)
            .done();
      }

      // Streaming mode metadata be like:
      // Not 'stall free' mode (to be mapped on '0' literal)
      // !ip_interface !N
      // !N = !{!"streaming"}
      // 'stall free' mode (to be mapped on '1' literal)
      // !ip_interface !N
      // !N = !{!"streaming", !"stall_free_return"}
      if (InterfaceStrSet.find("streaming") != InterfaceStrSet.end()) {
        int32_t InterfaceMode = 0;
        if (InterfaceStrSet.find("stall_free_return") != InterfaceStrSet.end())
          InterfaceMode = 1;
        EM.addOp()
            .add(&Kernel)
            .add(spv::ExecutionModeStreamingInterfaceINTEL)
            .add(InterfaceMode)
            .done();
      }
    }
  }
}

void PreprocessMetadataBase::preprocessOCLMetadata(Module *M, SPIRVMDBuilder *B,
                                                   SPIRVMDWalker *W) {
  unsigned CLVer = getOCLVersion(M, true);
  if (CLVer == 0)
    return;
  // Preprocess OpenCL-specific metadata
  // !spirv.Source = !{!x}
  // !{x} = !{i32 3, i32 102000}
  B->addNamedMD(kSPIRVMD::Source)
      .addOp()
      .add(CLVer == kOCLVer::CL21 ? spv::SourceLanguageOpenCL_CPP
                                  : spv::SourceLanguageOpenCL_C)
      .add(CLVer)
      .done();
  if (EraseOCLMD)
    B->eraseNamedMD(kSPIR2MD::OCLVer).eraseNamedMD(kSPIR2MD::SPIRVer);

  // !spirv.MemoryModel = !{!x}
  // !{x} = !{i32 1, i32 2}
  Triple TT(M->getTargetTriple());
  assert(isSupportedTriple(TT) && "Invalid triple");
  B->addNamedMD(kSPIRVMD::MemoryModel)
      .addOp()
      .add(TT.isArch32Bit() ? spv::AddressingModelPhysical32
                            : spv::AddressingModelPhysical64)
      .add(spv::MemoryModelOpenCL)
      .done();

  // Add source extensions
  // !spirv.SourceExtension = !{!x, !y, ...}
  // !x = {!"cl_khr_..."}
  // !y = {!"cl_khr_..."}
  auto Exts = getNamedMDAsStringSet(M, kSPIR2MD::Extensions);
  if (!Exts.empty()) {
    auto N = B->addNamedMD(kSPIRVMD::SourceExtension);
    for (auto &I : Exts)
      N.addOp().add(I).done();
  }
  if (EraseOCLMD)
    B->eraseNamedMD(kSPIR2MD::Extensions).eraseNamedMD(kSPIR2MD::OptFeatures);

  if (EraseOCLMD)
    B->eraseNamedMD(kSPIR2MD::FPContract);
}

void PreprocessMetadataBase::preprocessVectorComputeMetadata(Module *M,
                                                             SPIRVMDBuilder *B,
                                                             SPIRVMDWalker *W) {
  using namespace VectorComputeUtil;

  auto EM = B->addNamedMD(kSPIRVMD::ExecutionMode);

  for (auto &F : *M) {
    if (F.getCallingConv() != CallingConv::SPIR_KERNEL)
      continue;

    // Add VC float control execution modes
    // RoundMode and FloatMode are always same for all types in VC
    // While Denorm could be different for double, float and half
    auto Attrs = F.getAttributes();
    if (Attrs.hasFnAttr(kVCMetadata::VCFloatControl)) {
      SPIRVWord Mode = 0;
      Attrs
          .getFnAttr(kVCMetadata::VCFloatControl)
          .getValueAsString()
          .getAsInteger(0, Mode);
      spv::ExecutionMode ExecRoundMode =
          FPRoundingModeExecModeMap::map(getFPRoundingMode(Mode));
      spv::ExecutionMode ExecFloatMode =
          FPOperationModeExecModeMap::map(getFPOperationMode(Mode));
      VCFloatTypeSizeMap::foreach ([&](VCFloatType FloatType,
                                       unsigned TargetWidth) {
        EM.addOp().add(&F).add(ExecRoundMode).add(TargetWidth).done();
        EM.addOp().add(&F).add(ExecFloatMode).add(TargetWidth).done();
        EM.addOp()
            .add(&F)
            .add(FPDenormModeExecModeMap::map(getFPDenormMode(Mode, FloatType)))
            .add(TargetWidth)
            .done();
      });
    }
    if (Attrs.hasFnAttr(kVCMetadata::VCSLMSize)) {
      SPIRVWord SLMSize = 0;
      Attrs.getFnAttr(kVCMetadata::VCSLMSize)
          .getValueAsString()
          .getAsInteger(0, SLMSize);
      EM.addOp()
          .add(&F)
          .add(spv::ExecutionModeSharedLocalMemorySizeINTEL)
          .add(SLMSize)
          .done();
    }
    if (Attrs.hasFnAttr(kVCMetadata::VCFCEntry)) {
      EM.addOp()
          .add(&F)
          .add(spv::internal::ExecutionModeFastCompositeKernelINTEL)
          .done();
    }

    if (Attrs.hasFnAttr(kVCMetadata::VCNamedBarrierCount)) {
      SPIRVWord NBarrierCnt = 0;
      Attrs.getFnAttr(kVCMetadata::VCNamedBarrierCount)
          .getValueAsString()
          .getAsInteger(0, NBarrierCnt);
      EM.addOp()
          .add(&F)
          .add(spv::ExecutionModeNamedBarrierCountINTEL)
          .add(NBarrierCnt)
          .done();
    }
  }
}

} // namespace SPIRV

INITIALIZE_PASS(PreprocessMetadataLegacy, "preprocess-metadata",
                "Transform LLVM IR metadata to SPIR-V metadata format", false,
                false)

ModulePass *llvm::createPreprocessMetadataLegacy() {
  return new PreprocessMetadataLegacy();
}
