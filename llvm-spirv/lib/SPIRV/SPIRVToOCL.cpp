//===- SPIRVToOCL.cpp - Transform SPIR-V builtins to OCL builtins------===//
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
// This file implements common transform of SPIR-V builtins to OCL builtins.
//
// Some of the visit functions are translations to OCL2.0 builtins, but they
// are currently used also for OCL1.2, so theirs implementations are placed
// in this pass as a common functionality for both versions.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spvtocl"

#include "SPIRVToOCL.h"
#include "llvm/IR/Verifier.h"

namespace SPIRV {

static cl::opt<std::string>
    MangledAtomicTypeNamePrefix("spirv-atomic-prefix",
                                cl::desc("Mangled atomic type name prefix"),
                                cl::init("U7_Atomic"));

static cl::opt<std::string>
    OCLBuiltinsVersion("spirv-ocl-builtins-version",
                       cl::desc("Specify version of OCL builtins to translate "
                                "to (CL1.2, CL2.0, CL2.1)"));

char SPIRVToOCL::ID = 0;

void SPIRVToOCL::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto F = CI.getCalledFunction();
  if (!F)
    return;

  auto MangledName = F->getName();
  std::string DemangledName;
  Op OC = OpNop;
  if (!oclIsBuiltin(MangledName, &DemangledName) ||
      (OC = getSPIRVFuncOC(DemangledName)) == OpNop)
    return;
  LLVM_DEBUG(dbgs() << "DemangledName = " << DemangledName.c_str() << '\n'
                    << "OpCode = " << OC << '\n');

  if (OC == OpImageQuerySize || OC == OpImageQuerySizeLod) {
    visitCallSPRIVImageQuerySize(&CI);
    return;
  }
  if (OC == OpMemoryBarrier) {
    visitCallSPIRVMemoryBarrier(&CI);
    return;
  }
  if (OC == OpControlBarrier) {
    visitCallSPIRVControlBarrier(&CI);
  }
  if (isAtomicOpCode(OC)) {
    visitCallSPIRVAtomicBuiltin(&CI, OC);
    return;
  }
  if (isGroupOpCode(OC)) {
    visitCallSPIRVGroupBuiltin(&CI, OC);
    return;
  }
  if (isPipeOpCode(OC)) {
    visitCallSPIRVPipeBuiltin(&CI, OC);
    return;
  }
  if (OCLSPIRVBuiltinMap::rfind(OC))
    visitCallSPIRVBuiltin(&CI, OC);
}

void SPIRVToOCL::visitCastInst(CastInst &Cast) {
  if (!isa<ZExtInst>(Cast) && !isa<SExtInst>(Cast) && !isa<TruncInst>(Cast) &&
      !isa<FPTruncInst>(Cast) && !isa<FPExtInst>(Cast) &&
      !isa<FPToUIInst>(Cast) && !isa<FPToSIInst>(Cast) &&
      !isa<UIToFPInst>(Cast) && !isa<SIToFPInst>(Cast))
    return;

  Type const *SrcTy = Cast.getSrcTy();
  Type *DstVecTy = Cast.getDestTy();
  // Leave scalar casts as is. Skip boolean vector casts becase there
  // are no suitable OCL built-ins.
  if (!DstVecTy->isVectorTy() || SrcTy->getScalarSizeInBits() == 1 ||
      DstVecTy->getScalarSizeInBits() == 1)
    return;

  // Assemble built-in name -> convert_gentypeN
  std::string CastBuiltInName(kOCLBuiltinName::ConvertPrefix);
  // Check if this is 'floating point -> unsigned integer' cast
  CastBuiltInName += mapLLVMTypeToOCLType(DstVecTy, !isa<FPToUIInst>(Cast));

  // Replace LLVM conversion instruction with call to conversion built-in
  BuiltinFuncMangleInfo Mangle;
  // It does matter if the source is unsigned integer or not. SExt is for
  // signed source, ZExt and UIToFPInst are for unsigned source.
  if (isa<ZExtInst>(Cast) || isa<UIToFPInst>(Cast))
    Mangle.addUnsignedArg(0);

  AttributeList Attributes;
  CallInst *Call =
      addCallInst(M, CastBuiltInName, DstVecTy, Cast.getOperand(0), &Attributes,
                  &Cast, &Mangle, Cast.getName(), false);
  Cast.replaceAllUsesWith(Call);
  Cast.eraseFromParent();
}

void SPIRVToOCL::visitCallSPRIVImageQuerySize(CallInst *CI) {
  Function *Func = CI->getCalledFunction();
  // Get image type
  Type *ArgTy = Func->getFunctionType()->getParamType(0);
  assert(ArgTy->isPointerTy() &&
         "argument must be a pointer to opaque structure");
  StructType *ImgTy = cast<StructType>(ArgTy->getPointerElementType());
  assert(ImgTy->isOpaque() && "image type must be an opaque structure");
  StringRef ImgTyName = ImgTy->getName();
  assert(ImgTyName.startswith("opencl.image") && "not an OCL image type");

  unsigned ImgDim = 0;
  bool ImgArray = false;

  if (ImgTyName.startswith("opencl.image1d")) {
    ImgDim = 1;
  } else if (ImgTyName.startswith("opencl.image2d")) {
    ImgDim = 2;
  } else if (ImgTyName.startswith("opencl.image3d")) {
    ImgDim = 3;
  }
  assert(ImgDim != 0 && "unexpected image dimensionality");

  if (ImgTyName.count("_array_") != 0) {
    ImgArray = true;
  }

  AttributeList Attributes = CI->getCalledFunction()->getAttributes();
  BuiltinFuncMangleInfo Mangle;
  Type *Int32Ty = Type::getInt32Ty(*Ctx);
  Instruction *GetImageSize = nullptr;

  if (ImgDim == 1) {
    // OpImageQuerySize from non-arrayed 1d image is always translated
    // into get_image_width returning scalar argument
    GetImageSize = addCallInst(M, kOCLBuiltinName::GetImageWidth, Int32Ty,
                               CI->getArgOperand(0), &Attributes, CI, &Mangle,
                               CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from i32
    if (CI->getType()->getScalarType() != Int32Ty) {
      GetImageSize = CastInst::CreateIntegerCast(GetImageSize,
                                                 CI->getType()->getScalarType(),
                                                 false, CI->getName(), CI);
    }
  } else {
    assert((ImgDim == 2 || ImgDim == 3) && "invalid image type");
    assert(CI->getType()->isVectorTy() &&
           "this code can handle vector result type only");
    // get_image_dim returns int2 and int4 for 2d and 3d images respecitvely.
    const unsigned ImgDimRetEls = ImgDim == 2 ? 2 : 4;
    VectorType *RetTy = VectorType::get(Int32Ty, ImgDimRetEls);
    GetImageSize = addCallInst(M, kOCLBuiltinName::GetImageDim, RetTy,
                               CI->getArgOperand(0), &Attributes, CI, &Mangle,
                               CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from i32
    if (CI->getType()->getScalarType() != Int32Ty) {
      GetImageSize = CastInst::CreateIntegerCast(
          GetImageSize,
          VectorType::get(CI->getType()->getScalarType(),
                          GetImageSize->getType()->getVectorNumElements()),
          false, CI->getName(), CI);
    }
  }

  if (ImgArray || ImgDim == 3) {
    assert(
        CI->getType()->isVectorTy() &&
        "OpImageQuerySize[Lod] must return vector for arrayed and 3d images");
    const unsigned ImgQuerySizeRetEls = CI->getType()->getVectorNumElements();

    if (ImgDim == 1) {
      // get_image_width returns scalar result while OpImageQuerySize
      // for image1d_array_t returns <2 x i32> vector.
      assert(ImgQuerySizeRetEls == 2 &&
             "OpImageQuerySize[Lod] must return <2 x iN> vector type");
      GetImageSize = InsertElementInst::Create(
          UndefValue::get(CI->getType()), GetImageSize,
          ConstantInt::get(Int32Ty, 0), CI->getName(), CI);
    } else {
      // get_image_dim and OpImageQuerySize returns different vector
      // types for arrayed and 3d images.
      SmallVector<Constant *, 4> MaskEls;
      for (unsigned Idx = 0; Idx < ImgQuerySizeRetEls; ++Idx)
        MaskEls.push_back(ConstantInt::get(Int32Ty, Idx));
      Constant *Mask = ConstantVector::get(MaskEls);

      GetImageSize = new ShuffleVectorInst(
          GetImageSize, UndefValue::get(GetImageSize->getType()), Mask,
          CI->getName(), CI);
    }
  }

  if (ImgArray) {
    assert((ImgDim == 1 || ImgDim == 2) && "invalid image array type");
    // Insert get_image_array_size to the last position of the resulting vector.
    Type *SizeTy =
        Type::getIntNTy(*Ctx, M->getDataLayout().getPointerSizeInBits(0));
    Instruction *GetImageArraySize = addCallInst(
        M, kOCLBuiltinName::GetImageArraySize, SizeTy, CI->getArgOperand(0),
        &Attributes, CI, &Mangle, CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from size_t which is returned by get_image_array_size
    if (GetImageArraySize->getType() != CI->getType()->getScalarType()) {
      GetImageArraySize = CastInst::CreateIntegerCast(
          GetImageArraySize, CI->getType()->getScalarType(), false,
          CI->getName(), CI);
    }
    GetImageSize = InsertElementInst::Create(
        GetImageSize, GetImageArraySize,
        ConstantInt::get(Int32Ty, CI->getType()->getVectorNumElements() - 1),
        CI->getName(), CI);
  }

  assert(GetImageSize && "must not be null");
  CI->replaceAllUsesWith(GetImageSize);
  CI->eraseFromParent();
}

void SPIRVToOCL::visitCallSPIRVGroupBuiltin(CallInst *CI, Op OC) {
  auto DemangledName = OCLSPIRVBuiltinMap::rmap(OC);
  assert(DemangledName.find(kSPIRVName::GroupPrefix) == 0);

  std::string Prefix = getGroupBuiltinPrefix(CI);

  bool HasGroupOperation = hasGroupOperation(OC);
  if (!HasGroupOperation) {
    DemangledName = Prefix + DemangledName;
  } else {
    auto GO = getArgAs<spv::GroupOperation>(CI, 1);
    StringRef Op = DemangledName;
    Op = Op.drop_front(strlen(kSPIRVName::GroupPrefix));
    bool Unsigned = Op.front() == 'u';
    if (!Unsigned)
      Op = Op.drop_front(1);
    DemangledName = Prefix + kSPIRVName::GroupPrefix +
                    SPIRSPIRVGroupOperationMap::rmap(GO) + '_' + Op.str();
  }
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        Args.erase(Args.begin(), Args.begin() + (HasGroupOperation ? 2 : 1));
        if (OC == OpGroupBroadcast)
          expandVector(CI, Args, 1);
        return DemangledName;
      },
      &Attrs);
}

void SPIRVToOCL::visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC) {
  auto DemangledName = OCLSPIRVBuiltinMap::rmap(OC);
  bool HasScope = DemangledName.find(kSPIRVName::GroupPrefix) == 0;
  if (HasScope)
    DemangledName = getGroupBuiltinPrefix(CI) + DemangledName;

  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        if (HasScope)
          Args.erase(Args.begin(), Args.begin() + 1);

        if (!(OC == OpReadPipe || OC == OpWritePipe ||
              OC == OpReservedReadPipe || OC == OpReservedWritePipe ||
              OC == OpReadPipeBlockingINTEL || OC == OpWritePipeBlockingINTEL))
          return DemangledName;

        auto &P = Args[Args.size() - 3];
        auto T = P->getType();
        assert(isa<PointerType>(T));
        auto ET = T->getPointerElementType();
        if (!ET->isIntegerTy(8) ||
            T->getPointerAddressSpace() != SPIRAS_Generic) {
          auto NewTy = PointerType::getInt8PtrTy(*Ctx, SPIRAS_Generic);
          P = CastInst::CreatePointerBitCastOrAddrSpaceCast(P, NewTy, "", CI);
        }
        return DemangledName;
      },
      &Attrs);
}

void SPIRVToOCL::visitCallSPIRVBuiltin(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

void SPIRVToOCL::translateMangledAtomicTypeName() {
  for (auto &I : M->functions()) {
    if (!I.hasName())
      continue;
    std::string MangledName = I.getName();
    std::string DemangledName;
    if (!oclIsBuiltin(MangledName, &DemangledName) ||
        DemangledName.find(kOCLBuiltinName::AtomPrefix) != 0)
      continue;
    auto Loc = MangledName.find(kOCLBuiltinName::AtomPrefix);
    Loc = MangledName.find(kMangledName::AtomicPrefixInternal, Loc);
    MangledName.replace(Loc, strlen(kMangledName::AtomicPrefixInternal),
                        MangledAtomicTypeNamePrefix);
    I.setName(MangledName);
  }
}

std::string SPIRVToOCL::getGroupBuiltinPrefix(CallInst *CI) {
  std::string Prefix;
  auto ES = getArgAsScope(CI, 0);
  switch (ES) {
  case ScopeWorkgroup:
    Prefix = kOCLBuiltinName::WorkPrefix;
    break;
  case ScopeSubgroup:
    Prefix = kOCLBuiltinName::SubPrefix;
    break;
  default:
    llvm_unreachable("Invalid execution scope");
  }
  return Prefix;
}

} // namespace SPIRV

ModulePass *llvm::createSPIRVToOCL(Module &M) {
  if (OCLBuiltinsVersion.getNumOccurrences() > 0) {
    if (OCLBuiltinsVersion.getValue() == "CL1.2")
      return createSPIRVToOCL12();
    else if (OCLBuiltinsVersion.getValue() == "CL2.0" ||
             OCLBuiltinsVersion.getValue() == "CL2.1")
      return createSPIRVToOCL20();
    else {
      assert(0 && "Invalid spirv-ocl-builtins-version value");
      return nullptr;
    }
  }
  // Below part of code is here just temporarily (to not broke existing
  // projects based on translator), because ocl builtins versions shouldn't has
  // a dependency on OpSource spirv opcode. OpSource spec: "This has no semantic
  // impact and can safely be removed from a module." After some time it can be
  // removed, then only factor impacting version of ocl builtins will be
  // spirv-ocl-builtins-version command option.
  unsigned OCLVersion = getOCLVersion(&M);
  if (OCLVersion <= kOCLVer::CL12)
    return createSPIRVToOCL12();
  else if (OCLVersion >= kOCLVer::CL20)
    return createSPIRVToOCL20();
  else {
    assert(0 && "Invalid ocl version in llvm module");
    return nullptr;
  }
}
