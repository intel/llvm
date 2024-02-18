//===- SPIR.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// Base ABI and target codegen info implementation common between SPIR and
// SPIR-V.
//===----------------------------------------------------------------------===//

namespace {
class CommonSPIRABIInfo : public DefaultABIInfo {
public:
  CommonSPIRABIInfo(CodeGenTypes &CGT) : DefaultABIInfo(CGT) { setCCs(); }

  ABIArgInfo classifyKernelArgumentType(QualType Ty) const;

  // Add new functions rather than overload existing so that these public APIs
  // can't be blindly misused with wrong calling convention.
  ABIArgInfo classifyRegcallReturnType(QualType RetTy) const;
  ABIArgInfo classifyRegcallArgumentType(QualType RetTy) const;

  void computeInfo(CGFunctionInfo &FI) const override;

private:
  void setCCs();
};

ABIArgInfo CommonSPIRABIInfo::classifyKernelArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (getContext().getLangOpts().SYCLIsDevice && isAggregateTypeForABI(Ty)) {
    // Pass all aggregate types allowed by Sema by value.
    return getNaturalAlignIndirect(Ty);
  }

  return DefaultABIInfo::classifyArgumentType(Ty);
}

void CommonSPIRABIInfo::computeInfo(CGFunctionInfo &FI) const {
  llvm::CallingConv::ID CC = FI.getCallingConvention();
  bool IsRegCall = CC == llvm::CallingConv::X86_RegCall;

  if (!getCXXABI().classifyReturnType(FI)) {
    CanQualType RetT = FI.getReturnType();
    FI.getReturnInfo() =
        IsRegCall ? classifyRegcallReturnType(RetT) : classifyReturnType(RetT);
  }

  for (auto &Arg : FI.arguments()) {
    if (CC == llvm::CallingConv::SPIR_KERNEL) {
      Arg.info = classifyKernelArgumentType(Arg.type);
    } else {
      Arg.info = IsRegCall ? classifyRegcallArgumentType(Arg.type)
                           : classifyArgumentType(Arg.type);
    }
  }
}

// The two functions below are based on AMDGPUABIInfo, but without any
// restriction on the maximum number of arguments passed via registers.
// SPIRV BEs are expected to further adjust the calling convention as
// needed (use stack or byval-like passing) for some of the arguments.

ABIArgInfo CommonSPIRABIInfo::classifyRegcallReturnType(QualType RetTy) const {
  if (isAggregateTypeForABI(RetTy)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // returned by value.
    if (!getRecordArgABI(RetTy, getCXXABI())) {
      // Ignore empty structs/unions.
      if (isEmptyRecord(getContext(), RetTy, true))
        return ABIArgInfo::getIgnore();

      // Lower single-element structs to just return a regular value.
      if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
        return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

      if (const RecordType *RT = RetTy->getAs<RecordType>()) {
        const RecordDecl *RD = RT->getDecl();
        if (RD->hasFlexibleArrayMember())
          return classifyReturnType(RetTy);
      }

      // Pack aggregates <= 8 bytes into a single vector register or pair.
      // TODO make this parameterizeable/adjustable depending on spir target
      // triple abi component.
      uint64_t Size = getContext().getTypeSize(RetTy);
      if (Size <= 16)
        return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));

      if (Size <= 32)
        return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

      if (Size <= 64) {
        llvm::Type *I32Ty = llvm::Type::getInt32Ty(getVMContext());
        return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
      }
      return ABIArgInfo::getDirect();
    }
  }
  // Otherwise just do the default thing.
  return classifyReturnType(RetTy);
}

ABIArgInfo CommonSPIRABIInfo::classifyRegcallArgumentType(QualType Ty) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (auto RAA = getRecordArgABI(Ty, getCXXABI()))
      return getNaturalAlignIndirect(Ty, RAA == CGCXXABI::RAA_DirectInMemory);

    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();

    // Lower single-element structs to just pass a regular value. TODO: We
    // could do reasonable-size multiple-element structs too, using getExpand(),
    // though watch out for things like bitfields.
    if (const Type *SeltTy = isSingleElementStruct(Ty, getContext()))
      return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));

    if (const RecordType *RT = Ty->getAs<RecordType>()) {
      const RecordDecl *RD = RT->getDecl();
      if (RD->hasFlexibleArrayMember())
        return classifyArgumentType(Ty);
    }

    // Pack aggregates <= 8 bytes into single vector register or pair.
    // TODO make this parameterizeable/adjustable depending on spir target
    // triple abi component.
    uint64_t Size = getContext().getTypeSize(Ty);
    if (Size <= 64) {
      if (Size <= 16)
        return ABIArgInfo::getDirect(llvm::Type::getInt16Ty(getVMContext()));

      if (Size <= 32)
        return ABIArgInfo::getDirect(llvm::Type::getInt32Ty(getVMContext()));

      // XXX: Should this be i64 instead, and should the limit increase?
      llvm::Type *I32Ty = llvm::Type::getInt32Ty(getVMContext());
      return ABIArgInfo::getDirect(llvm::ArrayType::get(I32Ty, 2));
    }
    return ABIArgInfo::getDirect();
  }

  // Otherwise just do the default thing.
  return classifyArgumentType(Ty);
}


class SPIRVABIInfo : public CommonSPIRABIInfo {
public:
  SPIRVABIInfo(CodeGenTypes &CGT) : CommonSPIRABIInfo(CGT) {}
  void computeInfo(CGFunctionInfo &FI) const override;

private:
  ABIArgInfo classifyKernelArgumentType(QualType Ty) const;
};
} // end anonymous namespace
namespace {
class CommonSPIRTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  CommonSPIRTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<CommonSPIRABIInfo>(CGT)) {}
  CommonSPIRTargetCodeGenInfo(std::unique_ptr<ABIInfo> ABIInfo)
      : TargetCodeGenInfo(std::move(ABIInfo)) {}

  LangAS getASTAllocaAddressSpace() const override {
    return getLangASFromTargetAS(
        getABIInfo().getDataLayout().getAllocaAddrSpace());
  }

  unsigned getOpenCLKernelCallingConv() const override;
  llvm::Type *getOpenCLType(CodeGenModule &CGM, const Type *T) const override;

  bool shouldEmitStaticExternCAliases() const override;
};
class SPIRVTargetCodeGenInfo : public CommonSPIRTargetCodeGenInfo {
public:
  SPIRVTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT)
      : CommonSPIRTargetCodeGenInfo(std::make_unique<SPIRVABIInfo>(CGT)) {}
  void setCUDAKernelCallingConvention(const FunctionType *&FT) const override;
};
} // End anonymous namespace.

void CommonSPIRABIInfo::setCCs() {
  assert(getRuntimeCC() == llvm::CallingConv::C);
  RuntimeCC = llvm::CallingConv::SPIR_FUNC;
}

ABIArgInfo SPIRVABIInfo::classifyKernelArgumentType(QualType Ty) const {
  if (getContext().getLangOpts().CUDAIsDevice) {
    // Coerce pointer arguments with default address space to CrossWorkGroup
    // pointers for HIPSPV/CUDASPV. When the language mode is HIP/CUDA, the
    // SPIRTargetInfo maps cuda_device to SPIR-V's CrossWorkGroup address space.
    llvm::Type *LTy = CGT.ConvertType(Ty);
    auto DefaultAS = getContext().getTargetAddressSpace(LangAS::Default);
    auto GlobalAS = getContext().getTargetAddressSpace(LangAS::cuda_device);
    auto *PtrTy = llvm::dyn_cast<llvm::PointerType>(LTy);
    if (PtrTy && PtrTy->getAddressSpace() == DefaultAS) {
      LTy = llvm::PointerType::get(PtrTy->getContext(), GlobalAS);
      return ABIArgInfo::getDirect(LTy, 0, nullptr, false);
    }

    // Force copying aggregate type in kernel arguments by value when
    // compiling CUDA targeting SPIR-V. This is required for the object
    // copied to be valid on the device.
    // This behavior follows the CUDA spec
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-function-argument-processing,
    // and matches the NVPTX implementation.
    if (isAggregateTypeForABI(Ty))
      return getNaturalAlignIndirect(Ty, /* byval */ true);
  }
  return classifyArgumentType(Ty);
}

void SPIRVABIInfo::computeInfo(CGFunctionInfo &FI) const {
  // The logic is same as in DefaultABIInfo with an exception on the kernel
  // arguments handling.
  llvm::CallingConv::ID CC = FI.getCallingConvention();

  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  for (auto &I : FI.arguments()) {
    if (CC == llvm::CallingConv::SPIR_KERNEL) {
      I.info = classifyKernelArgumentType(I.type);
    } else {
      I.info = classifyArgumentType(I.type);
    }
  }
}

namespace clang {
namespace CodeGen {
void computeSPIRKernelABIInfo(CodeGenModule &CGM, CGFunctionInfo &FI) {
  if (CGM.getTarget().getTriple().isSPIRV())
    SPIRVABIInfo(CGM.getTypes()).computeInfo(FI);
  else
    CommonSPIRABIInfo(CGM.getTypes()).computeInfo(FI);
}
}
}

unsigned CommonSPIRTargetCodeGenInfo::getOpenCLKernelCallingConv() const {
  return llvm::CallingConv::SPIR_KERNEL;
}

bool CommonSPIRTargetCodeGenInfo::shouldEmitStaticExternCAliases() const {
  return false;
}

void SPIRVTargetCodeGenInfo::setCUDAKernelCallingConvention(
    const FunctionType *&FT) const {
  // Convert HIP kernels to SPIR-V kernels.
  if (getABIInfo().getContext().getLangOpts().HIP) {
    FT = getABIInfo().getContext().adjustFunctionType(
        FT, FT->getExtInfo().withCallingConv(CC_OpenCLKernel));
    return;
  }
}

/// Construct a SPIR-V target extension type for the given OpenCL image type.
static llvm::Type *getSPIRVImageType(llvm::LLVMContext &Ctx, StringRef BaseType,
                                     StringRef OpenCLName,
                                     unsigned AccessQualifier) {
  // These parameters compare to the operands of OpTypeImage (see
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#OpTypeImage
  // for more details). The first 6 integer parameters all default to 0, and
  // will be changed to 1 only for the image type(s) that set the parameter to
  // one. The 7th integer parameter is the access qualifier, which is tacked on
  // at the end.
  SmallVector<unsigned, 7> IntParams = {0, 0, 0, 0, 0, 0};

  // Choose the dimension of the image--this corresponds to the Dim enum in
  // SPIR-V (first integer parameter of OpTypeImage).
  if (OpenCLName.starts_with("image2d"))
    IntParams[0] = 1; // 1D
  else if (OpenCLName.starts_with("image3d"))
    IntParams[0] = 2; // 2D
  else if (OpenCLName == "image1d_buffer")
    IntParams[0] = 5; // Buffer
  else
    assert(OpenCLName.starts_with("image1d") && "Unknown image type");

  // Set the other integer parameters of OpTypeImage if necessary. Note that the
  // OpenCL image types don't provide any information for the Sampled or
  // Image Format parameters.
  if (OpenCLName.contains("_depth"))
    IntParams[1] = 1;
  if (OpenCLName.contains("_array"))
    IntParams[2] = 1;
  if (OpenCLName.contains("_msaa"))
    IntParams[3] = 1;

  // Access qualifier
  IntParams.push_back(AccessQualifier);

  return llvm::TargetExtType::get(Ctx, BaseType, {llvm::Type::getVoidTy(Ctx)},
                                  IntParams);
}

llvm::Type *CommonSPIRTargetCodeGenInfo::getOpenCLType(CodeGenModule &CGM,
                                                       const Type *Ty) const {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  if (Ctx.supportsTypedPointers())
    return nullptr;
  if (auto *PipeTy = dyn_cast<PipeType>(Ty))
    return llvm::TargetExtType::get(Ctx, "spirv.Pipe", {},
                                    {!PipeTy->isReadOnly()});
  if (auto *BuiltinTy = dyn_cast<BuiltinType>(Ty)) {
    enum AccessQualifier : unsigned { AQ_ro = 0, AQ_wo = 1, AQ_rw = 2 };
    switch (BuiltinTy->getKind()) {
// clang-format off
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
    case BuiltinType::Id:                                                      \
      return getSPIRVImageType(Ctx, "spirv.Image", #ImgType, AQ_##Suffix);
#include "clang/Basic/OpenCLImageTypes.def"
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
    case BuiltinType::Sampled##Id:                                             \
      return getSPIRVImageType(Ctx, "spirv.SampledImage", #ImgType, AQ_##Suffix);
// clang-format on
#define IMAGE_WRITE_TYPE(Type, Id, Ext)
#define IMAGE_READ_WRITE_TYPE(Type, Id, Ext)
#include "clang/Basic/OpenCLImageTypes.def"
    case BuiltinType::OCLSampler:
      return llvm::TargetExtType::get(Ctx, "spirv.Sampler");
    case BuiltinType::OCLEvent:
      return llvm::TargetExtType::get(Ctx, "spirv.Event");
    case BuiltinType::OCLClkEvent:
      return llvm::TargetExtType::get(Ctx, "spirv.DeviceEvent");
    case BuiltinType::OCLQueue:
      return llvm::TargetExtType::get(Ctx, "spirv.Queue");
    case BuiltinType::OCLReserveID:
      return llvm::TargetExtType::get(Ctx, "spirv.ReserveId");
#define INTEL_SUBGROUP_AVC_TYPE(Name, Id)                                      \
    case BuiltinType::OCLIntelSubgroupAVC##Id:                                 \
      return llvm::TargetExtType::get(Ctx, "spirv.Avc" #Id "INTEL");
#include "clang/Basic/OpenCLExtensionTypes.def"
    default:
      return nullptr;
    }
  }

  return nullptr;
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createCommonSPIRTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<CommonSPIRTargetCodeGenInfo>(CGM.getTypes());
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createSPIRVTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<SPIRVTargetCodeGenInfo>(CGM.getTypes());
}
