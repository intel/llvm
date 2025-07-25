//===- NVPTX.cpp ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"
#include "clang/Basic/Cuda.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// NVPTX ABI Implementation
//===----------------------------------------------------------------------===//

namespace {

class NVPTXTargetCodeGenInfo;

class NVPTXABIInfo : public ABIInfo {
  NVPTXTargetCodeGenInfo &CGInfo;

public:
  NVPTXABIInfo(CodeGenTypes &CGT, NVPTXTargetCodeGenInfo &Info)
      : ABIInfo(CGT), CGInfo(Info) {}

  ABIArgInfo classifyReturnType(QualType RetTy) const;
  ABIArgInfo classifyArgumentType(QualType Ty) const;

  void computeInfo(CGFunctionInfo &FI) const override;
  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;
  bool isUnsupportedType(QualType T) const;
  ABIArgInfo coerceToIntArrayWithLimit(QualType Ty, unsigned MaxSize) const;
};

class NVPTXTargetCodeGenInfo : public TargetCodeGenInfo {
public:
  NVPTXTargetCodeGenInfo(CodeGenTypes &CGT)
      : TargetCodeGenInfo(std::make_unique<NVPTXABIInfo>(CGT, *this)) {}

  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &M) const override;
  bool shouldEmitStaticExternCAliases() const override;

  llvm::Constant *getNullPointer(const CodeGen::CodeGenModule &CGM,
                                 llvm::PointerType *T,
                                 QualType QT) const override;

  llvm::Type *getCUDADeviceBuiltinSurfaceDeviceType() const override {
    // On the device side, surface reference is represented as an object handle
    // in 64-bit integer.
    return llvm::Type::getInt64Ty(getABIInfo().getVMContext());
  }

  llvm::Type *getCUDADeviceBuiltinTextureDeviceType() const override {
    // On the device side, texture reference is represented as an object handle
    // in 64-bit integer.
    return llvm::Type::getInt64Ty(getABIInfo().getVMContext());
  }

  bool emitCUDADeviceBuiltinSurfaceDeviceCopy(CodeGenFunction &CGF, LValue Dst,
                                              LValue Src) const override {
    emitBuiltinSurfTexDeviceCopy(CGF, Dst, Src);
    return true;
  }

  bool emitCUDADeviceBuiltinTextureDeviceCopy(CodeGenFunction &CGF, LValue Dst,
                                              LValue Src) const override {
    emitBuiltinSurfTexDeviceCopy(CGF, Dst, Src);
    return true;
  }

  unsigned getDeviceKernelCallingConv() const override {
    return llvm::CallingConv::PTX_Kernel;
  }

  // Adds a NamedMDNode with GV, Name, and Operand as operands, and adds the
  // resulting MDNode to the nvvm.annotations MDNode.
  static void addNVVMMetadata(llvm::GlobalValue *GV, StringRef Name,
                              int Operand);

  static void
  addGridConstantNVVMMetadata(llvm::GlobalValue *GV,
                              const SmallVectorImpl<int> &GridConstantArgs);

  static void addNVVMMetadata(llvm::GlobalValue *GV, StringRef Name,
                              const std::vector<int> &Operands);

private:
  static void emitBuiltinSurfTexDeviceCopy(CodeGenFunction &CGF, LValue Dst,
                                           LValue Src) {
    llvm::Value *Handle = nullptr;
    llvm::Constant *C =
        llvm::dyn_cast<llvm::Constant>(Src.getAddress().emitRawPointer(CGF));
    // Lookup `addrspacecast` through the constant pointer if any.
    if (auto *ASC = llvm::dyn_cast_or_null<llvm::AddrSpaceCastOperator>(C))
      C = llvm::cast<llvm::Constant>(ASC->getPointerOperand());
    if (auto *GV = llvm::dyn_cast_or_null<llvm::GlobalVariable>(C)) {
      // Load the handle from the specific global variable using
      // `nvvm.texsurf.handle.internal` intrinsic.
      Handle = CGF.EmitRuntimeCall(
          CGF.CGM.getIntrinsic(llvm::Intrinsic::nvvm_texsurf_handle_internal,
                               {GV->getType()}),
          {GV}, "texsurf_handle");
    } else
      Handle = CGF.EmitLoadOfScalar(Src, SourceLocation());
    CGF.EmitStoreOfScalar(Handle, Dst);
  }
};

/// Checks if the type is unsupported directly by the current target.
bool NVPTXABIInfo::isUnsupportedType(QualType T) const {
  ASTContext &Context = getContext();
  if (!Context.getTargetInfo().hasFloat16Type() && T->isFloat16Type())
    return true;
  if (!Context.getTargetInfo().hasFloat128Type() &&
      (T->isFloat128Type() ||
       (T->isRealFloatingType() && Context.getTypeSize(T) == 128)))
    return true;
  if (const auto *EIT = T->getAs<BitIntType>())
    return EIT->getNumBits() >
           (Context.getTargetInfo().hasInt128Type() ? 128U : 64U);
  if (!Context.getTargetInfo().hasInt128Type() && T->isIntegerType() &&
      Context.getTypeSize(T) > 64U)
    return true;
  if (const auto *AT = T->getAsArrayTypeUnsafe())
    return isUnsupportedType(AT->getElementType());
  const auto *RT = T->getAs<RecordType>();
  if (!RT)
    return false;
  const RecordDecl *RD = RT->getDecl();

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD))
    for (const CXXBaseSpecifier &I : CXXRD->bases())
      if (isUnsupportedType(I.getType()))
        return true;

  for (const FieldDecl *I : RD->fields())
    if (isUnsupportedType(I->getType()))
      return true;
  return false;
}

/// Coerce the given type into an array with maximum allowed size of elements.
ABIArgInfo NVPTXABIInfo::coerceToIntArrayWithLimit(QualType Ty,
                                                   unsigned MaxSize) const {
  // Alignment and Size are measured in bits.
  const uint64_t Size = getContext().getTypeSize(Ty);
  const uint64_t Alignment = getContext().getTypeAlign(Ty);
  const unsigned Div = std::min<unsigned>(MaxSize, Alignment);
  llvm::Type *IntType = llvm::Type::getIntNTy(getVMContext(), Div);
  const uint64_t NumElements = (Size + Div - 1) / Div;
  return ABIArgInfo::getDirect(llvm::ArrayType::get(IntType, NumElements));
}

ABIArgInfo NVPTXABIInfo::classifyReturnType(QualType RetTy) const {
  if (RetTy->isVoidType())
    return ABIArgInfo::getIgnore();

  if (getContext().getLangOpts().OpenMP &&
      getContext().getLangOpts().OpenMPIsTargetDevice &&
      isUnsupportedType(RetTy))
    return coerceToIntArrayWithLimit(RetTy, 64);

  // note: this is different from default ABI
  if (!RetTy->isScalarType())
    return ABIArgInfo::getDirect();

  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = RetTy->getAs<EnumType>())
    RetTy = EnumTy->getDecl()->getIntegerType();

  return (isPromotableIntegerTypeForABI(RetTy) ? ABIArgInfo::getExtend(RetTy)
                                               : ABIArgInfo::getDirect());
}

ABIArgInfo NVPTXABIInfo::classifyArgumentType(QualType Ty) const {
  // Treat an enum type as its underlying type.
  if (const EnumType *EnumTy = Ty->getAs<EnumType>())
    Ty = EnumTy->getDecl()->getIntegerType();

  // Return aggregates type as indirect by value
  if (isAggregateTypeForABI(Ty)) {
    // Under CUDA device compilation, tex/surf builtin types are replaced with
    // object types and passed directly.
    if (getContext().getLangOpts().CUDAIsDevice) {
      if (Ty->isCUDADeviceBuiltinSurfaceType())
        return ABIArgInfo::getDirect(
            CGInfo.getCUDADeviceBuiltinSurfaceDeviceType());
      if (Ty->isCUDADeviceBuiltinTextureType())
        return ABIArgInfo::getDirect(
            CGInfo.getCUDADeviceBuiltinTextureDeviceType());
    }
    return getNaturalAlignIndirect(
        Ty, /* AddrSpace */ getDataLayout().getAllocaAddrSpace(),
        /* byval */ true);
  }

  if (const auto *EIT = Ty->getAs<BitIntType>()) {
    if ((EIT->getNumBits() > 128) ||
        (!getContext().getTargetInfo().hasInt128Type() &&
         EIT->getNumBits() > 64))
      return getNaturalAlignIndirect(
          Ty, /* AddrSpace */ getDataLayout().getAllocaAddrSpace(),
          /* byval */ true);
  }

  return (isPromotableIntegerTypeForABI(Ty) ? ABIArgInfo::getExtend(Ty)
                                            : ABIArgInfo::getDirect());
}

void NVPTXABIInfo::computeInfo(CGFunctionInfo &FI) const {
  if (!getCXXABI().classifyReturnType(FI))
    FI.getReturnInfo() = classifyReturnType(FI.getReturnType());

  for (auto &&[ArgumentsCount, I] : llvm::enumerate(FI.arguments()))
    I.info = ArgumentsCount < FI.getNumRequiredArgs()
                 ? classifyArgumentType(I.type)
                 : ABIArgInfo::getDirect();

  // Always honor user-specified calling convention.
  if (FI.getCallingConvention() != llvm::CallingConv::C)
    return;

  FI.setEffectiveCallingConvention(getRuntimeCC());
}

RValue NVPTXABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                               QualType Ty, AggValueSlot Slot) const {
  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, /*IsIndirect=*/false,
                          getContext().getTypeInfoInChars(Ty),
                          CharUnits::fromQuantity(1),
                          /*AllowHigherAlign=*/true, Slot);
}

// Get current OffloadArch and ignore any unknown values
// Copied from CGOpenMPRuntimeGPU
static OffloadArch getOffloadArch(CodeGenModule &CGM) {
  if (!CGM.getTarget().hasFeature("ptx"))
    return OffloadArch::UNKNOWN;
  for (const auto &Feature : CGM.getTarget().getTargetOpts().FeatureMap) {
    if (Feature.getValue()) {
      OffloadArch Arch = StringToOffloadArch(Feature.getKey());
      if (Arch != OffloadArch::UNKNOWN)
        return Arch;
    }
  }
  return OffloadArch::UNKNOWN;
}

static bool supportsGridConstant(OffloadArch Arch) {
  assert((Arch == OffloadArch::UNKNOWN || IsNVIDIAOffloadArch(Arch)) &&
         "Unexpected architecture");
  static_assert(OffloadArch::UNKNOWN < OffloadArch::SM_70);
  return Arch >= OffloadArch::SM_70;
}

void NVPTXTargetCodeGenInfo::setTargetAttributes(
    const Decl *D, llvm::GlobalValue *GV, CodeGen::CodeGenModule &M) const {
  if (GV->isDeclaration())
    return;
  const VarDecl *VD = dyn_cast_or_null<VarDecl>(D);
  if (VD) {
    if (M.getLangOpts().CUDA) {
      if (VD->getType()->isCUDADeviceBuiltinSurfaceType())
        addNVVMMetadata(GV, "surface", 1);
      else if (VD->getType()->isCUDADeviceBuiltinTextureType())
        addNVVMMetadata(GV, "texture", 1);
      return;
    }
  }

  const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D);
  if (!FD)
    return;

  llvm::Function *F = cast<llvm::Function>(GV);

  // Perform special handling in OpenCL mode
  if (M.getLangOpts().OpenCL || M.getLangOpts().SYCLIsDevice || M.getLangOpts().CUDA) {
    // Use OpenCL function attributes to check for kernel functions
    // Use function attributes to check for kernel functions
    // By default, all functions are device functions
    if (FD->hasAttr<DeviceKernelAttr>() || FD->hasAttr<CUDAGlobalAttr>()) {
      // OpenCL/CUDA kernel functions get kernel metadata
      // Create !{<func-ref>, metadata !"kernel", i32 1} node
      // And kernel functions are not subject to inlining
      F->addFnAttr(llvm::Attribute::NoInline);
      if (M.getLangOpts().SYCLIsDevice &&
          supportsGridConstant(getOffloadArch(M))) {
        // Add grid_constant annotations to all relevant kernel-function
        // parameters. We can guarantee that in SYCL, all by-val kernel
        // parameters are "grid_constant".
        std::vector<int> GridConstantParamIdxs;
        for (auto [Idx, Arg] : llvm::enumerate(F->args())) {
          if (Arg.getType()->isPointerTy() && Arg.hasByValAttr()) {
            // Note - the parameter indices are numbered from 1.
            GridConstantParamIdxs.push_back(Idx + 1);
          }
        }
        if (!GridConstantParamIdxs.empty())
          addNVVMMetadata(F, "grid_constant", GridConstantParamIdxs);
      }
      if (FD->hasAttr<CUDAGlobalAttr>()) {
        SmallVector<int, 10> GCI;
        for (auto IV : llvm::enumerate(FD->parameters()))
          if (IV.value()->hasAttr<CUDAGridConstantAttr>())
            // For some reason arg indices are 1-based in NVVM
            GCI.push_back(IV.index() + 1);
        // Create !{<func-ref>, metadata !"kernel", i32 1} node
        F->setCallingConv(llvm::CallingConv::PTX_Kernel);
        addGridConstantNVVMMetadata(F, GCI);
      }
      if (CUDALaunchBoundsAttr *Attr = FD->getAttr<CUDALaunchBoundsAttr>())
        M.handleCUDALaunchBoundsAttr(F, Attr);
    }
  }
  // Attach kernel metadata directly if compiling for NVPTX.
  if (FD->hasAttr<DeviceKernelAttr>()) {
    F->setCallingConv(llvm::CallingConv::PTX_Kernel);
  }
}

void NVPTXTargetCodeGenInfo::addNVVMMetadata(llvm::GlobalValue *GV,
                                             StringRef Name, int Operand) {
  llvm::Module *M = GV->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<llvm::Metadata *, 5> MDVals = {
      llvm::ConstantAsMetadata::get(GV), llvm::MDString::get(Ctx, Name),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Operand))};

  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}

void NVPTXTargetCodeGenInfo::addGridConstantNVVMMetadata(
    llvm::GlobalValue *GV, const SmallVectorImpl<int> &GridConstantArgs) {

  llvm::Module *M = GV->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  SmallVector<llvm::Metadata *, 5> MDVals = {llvm::ConstantAsMetadata::get(GV)};
  if (!GridConstantArgs.empty()) {
    SmallVector<llvm::Metadata *, 10> GCM;
    for (int I : GridConstantArgs)
      GCM.push_back(llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), I)));
    MDVals.append({llvm::MDString::get(Ctx, "grid_constant"),
                   llvm::MDNode::get(Ctx, GCM)});
  }

  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}

void NVPTXTargetCodeGenInfo::addNVVMMetadata(llvm::GlobalValue *GV,
                                             StringRef Name,
                                             const std::vector<int> &Operands) {
  llvm::Module *M = GV->getParent();
  llvm::LLVMContext &Ctx = M->getContext();

  // Get "nvvm.annotations" metadata node
  llvm::NamedMDNode *MD = M->getOrInsertNamedMetadata("nvvm.annotations");

  llvm::SmallVector<llvm::Metadata *, 8> MDOps;
  for (int Op : Operands) {
    MDOps.push_back(llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Op)));
  }
  auto *OpList = llvm::MDNode::get(Ctx, MDOps);

  llvm::Metadata *MDVals[] = {llvm::ConstantAsMetadata::get(GV),
                              llvm::MDString::get(Ctx, Name), OpList};
  // Append metadata to nvvm.annotations
  MD->addOperand(llvm::MDNode::get(Ctx, MDVals));
}

bool NVPTXTargetCodeGenInfo::shouldEmitStaticExternCAliases() const {
  return false;
}

llvm::Constant *
NVPTXTargetCodeGenInfo::getNullPointer(const CodeGen::CodeGenModule &CGM,
                                       llvm::PointerType *PT,
                                       QualType QT) const {
  auto &Ctx = CGM.getContext();
  if (PT->getAddressSpace() != Ctx.getTargetAddressSpace(LangAS::opencl_local))
    return llvm::ConstantPointerNull::get(PT);

  auto NPT = llvm::PointerType::get(
      PT->getContext(), Ctx.getTargetAddressSpace(LangAS::opencl_generic));
  return llvm::ConstantExpr::getAddrSpaceCast(
      llvm::ConstantPointerNull::get(NPT), PT);
}
} // namespace

void CodeGenModule::handleCUDALaunchBoundsAttr(llvm::Function *F,
                                               const CUDALaunchBoundsAttr *Attr,
                                               int32_t *MaxThreadsVal,
                                               int32_t *MinBlocksVal,
                                               int32_t *MaxClusterRankVal) {
  llvm::APSInt MaxThreads(32);
  MaxThreads = Attr->getMaxThreads()->EvaluateKnownConstInt(getContext());
  if (MaxThreads > 0) {
    if (MaxThreadsVal)
      *MaxThreadsVal = MaxThreads.getExtValue();
    if (F)
      F->addFnAttr("nvvm.maxntid", llvm::utostr(MaxThreads.getExtValue()));
  }

  // min and max blocks is an optional argument for CUDALaunchBoundsAttr. If it
  // was not specified in __launch_bounds__ or if the user specified a 0 value,
  // we don't have to add a PTX directive.
  if (Attr->getMinBlocks()) {
    llvm::APSInt MinBlocks(32);
    MinBlocks = Attr->getMinBlocks()->EvaluateKnownConstInt(getContext());
    if (MinBlocks > 0) {
      if (MinBlocksVal)
        *MinBlocksVal = MinBlocks.getExtValue();
      if (F)
        F->addFnAttr("nvvm.minctasm", llvm::utostr(MinBlocks.getExtValue()));
    }
  }
  if (Attr->getMaxBlocks()) {
    llvm::APSInt MaxBlocks(32);
    MaxBlocks = Attr->getMaxBlocks()->EvaluateKnownConstInt(getContext());
    if (MaxBlocks > 0) {
      if (MaxClusterRankVal)
        *MaxClusterRankVal = MaxBlocks.getExtValue();
      if (F)
        F->addFnAttr("nvvm.maxclusterrank",
                     llvm::utostr(MaxBlocks.getExtValue()));
    }
  }
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createNVPTXTargetCodeGenInfo(CodeGenModule &CGM) {
  return std::make_unique<NVPTXTargetCodeGenInfo>(CGM.getTypes());
}
