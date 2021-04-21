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
#include "llvm/Support/CommandLine.h"

namespace SPIRV {

void SPIRVToOCLBase::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto F = CI.getCalledFunction();
  if (!F)
    return;

  auto MangledName = F->getName();
  StringRef DemangledName;
  Op OC = OpNop;
  if (!oclIsBuiltin(MangledName, DemangledName) ||
      (OC = getSPIRVFuncOC(DemangledName)) == OpNop)
    return;
  LLVM_DEBUG(dbgs() << "DemangledName = " << DemangledName.str() << '\n'
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
  if (isGroupOpCode(OC) || isGroupNonUniformOpcode(OC)) {
    visitCallSPIRVGroupBuiltin(&CI, OC);
    return;
  }
  if (isPipeOpCode(OC)) {
    visitCallSPIRVPipeBuiltin(&CI, OC);
    return;
  }
  if (isMediaBlockINTELOpcode(OC)) {
    visitCallSPIRVImageMediaBlockBuiltin(&CI, OC);
    return;
  }
  if (isCvtOpCode(OC)) {
    visitCallSPIRVCvtBuiltin(&CI, OC, DemangledName);
    return;
  }
  if (OC == OpGroupAsyncCopy) {
    visitCallAsyncWorkGroupCopy(&CI, OC);
    return;
  }
  if (OC == OpGroupWaitEvents) {
    visitCallGroupWaitEvents(&CI, OC);
    return;
  }
  if (OC == OpImageSampleExplicitLod) {
    visitCallSPIRVImageSampleExplicitLodBuiltIn(&CI, OC);
    return;
  }
  if (OCLSPIRVBuiltinMap::rfind(OC))
    visitCallSPIRVBuiltin(&CI, OC);
}

void SPIRVToOCLBase::visitCastInst(CastInst &Cast) {
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

void SPIRVToOCLBase::visitCallSPRIVImageQuerySize(CallInst *CI) {
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
    VectorType *RetTy = FixedVectorType::get(Int32Ty, ImgDimRetEls);
    GetImageSize = addCallInst(M, kOCLBuiltinName::GetImageDim, RetTy,
                               CI->getArgOperand(0), &Attributes, CI, &Mangle,
                               CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from i32
    if (CI->getType()->getScalarType() != Int32Ty) {
      GetImageSize = CastInst::CreateIntegerCast(
          GetImageSize,
          FixedVectorType::get(
              CI->getType()->getScalarType(),
              cast<FixedVectorType>(GetImageSize->getType())->getNumElements()),
          false, CI->getName(), CI);
    }
  }

  if (ImgArray || ImgDim == 3) {
    auto *VecTy = cast<FixedVectorType>(CI->getType());
    const unsigned ImgQuerySizeRetEls = VecTy->getNumElements();

    if (ImgDim == 1) {
      // get_image_width returns scalar result while OpImageQuerySize
      // for image1d_array_t returns <2 x i32> vector.
      assert(ImgQuerySizeRetEls == 2 &&
             "OpImageQuerySize[Lod] must return <2 x iN> vector type");
      GetImageSize = InsertElementInst::Create(
          UndefValue::get(VecTy), GetImageSize, ConstantInt::get(Int32Ty, 0),
          CI->getName(), CI);
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
    auto *VecTy = cast<FixedVectorType>(CI->getType());
    Type *SizeTy =
        Type::getIntNTy(*Ctx, M->getDataLayout().getPointerSizeInBits(0));
    Instruction *GetImageArraySize = addCallInst(
        M, kOCLBuiltinName::GetImageArraySize, SizeTy, CI->getArgOperand(0),
        &Attributes, CI, &Mangle, CI->getName(), false);
    // The width of integer type returning by OpImageQuerySize[Lod] may
    // differ from size_t which is returned by get_image_array_size
    if (GetImageArraySize->getType() != VecTy->getElementType()) {
      GetImageArraySize = CastInst::CreateIntegerCast(
          GetImageArraySize, VecTy->getElementType(), false, CI->getName(), CI);
    }
    GetImageSize = InsertElementInst::Create(
        GetImageSize, GetImageArraySize,
        ConstantInt::get(Int32Ty, VecTy->getNumElements() - 1), CI->getName(),
        CI);
  }

  assert(GetImageSize && "must not be null");
  CI->replaceAllUsesWith(GetImageSize);
  CI->eraseFromParent();
}

std::string SPIRVToOCLBase::getUniformArithmeticBuiltinName(CallInst *CI,
                                                            Op OC) {
  assert(isUniformArithmeticOpCode(OC) &&
         "Not intended to handle other than uniform arithmetic opcodes!");
  auto FuncName = OCLSPIRVBuiltinMap::rmap(OC);
  std::string Prefix = getGroupBuiltinPrefix(CI);
  std::string Op = FuncName;
  Op.erase(0, strlen(kSPIRVName::GroupPrefix));
  // unsigned prefix cannot be removed yet, as it is necessary to properly
  // mangle the function
  bool Unsigned = Op.front() == 'u';
  if (!Unsigned)
    Op = Op.erase(0, 1);

  std::string GroupOp;
  auto GO = getArgAs<spv::GroupOperation>(CI, 1);
  switch (GO) {
  case GroupOperationReduce:
    GroupOp = "reduce";
    break;
  case GroupOperationInclusiveScan:
    GroupOp = "scan_inclusive";
    break;
  case GroupOperationExclusiveScan:
    GroupOp = "scan_exclusive";
    break;
  default:
    llvm_unreachable("Unsupported group operation!");
    break;
  }
  return Prefix + kSPIRVName::GroupPrefix + GroupOp + "_" + Op;
}

std::string SPIRVToOCLBase::getNonUniformArithmeticBuiltinName(CallInst *CI,
                                                               Op OC) {
  assert(isNonUniformArithmeticOpCode(OC) &&
         "Not intended to handle other than non uniform arithmetic opcodes!");
  std::string Prefix = getGroupBuiltinPrefix(CI);
  assert((Prefix == kOCLBuiltinName::SubPrefix) &&
         "Workgroup scope is not supported for OpGroupNonUniform opcodes");
  auto FuncName = OCLSPIRVBuiltinMap::rmap(OC);
  std::string Op = FuncName;
  Op.erase(0, strlen(kSPIRVName::GroupNonUniformPrefix));

  if (!isGroupLogicalOpCode(OC)) {
    // unsigned prefix cannot be removed yet, as it is necessary to properly
    // mangle the function
    const char Sign = Op.front();
    bool Signed = (Sign == 'i' || Sign == 'f' || Sign == 's');
    if (Signed)
      Op = Op.erase(0, 1);
    else
      assert((Sign == 'u') && "Incorrect sign!");
  } else { // LogicalOpcode
    assert(
        (Op == "logical_iand" || Op == "logical_ior" || Op == "logical_ixor") &&
        "Incorrect logical operation");
    Op = Op.erase(8, 1);
  }

  std::string GroupOp;
  std::string GroupPrefix = kSPIRVName::GroupNonUniformPrefix;
  auto GO = getArgAs<spv::GroupOperation>(CI, 1);
  switch (GO) {
  case GroupOperationReduce:
    GroupOp = "reduce";
    break;
  case GroupOperationInclusiveScan:
    GroupOp = "scan_inclusive";
    break;
  case GroupOperationExclusiveScan:
    GroupOp = "scan_exclusive";
    break;
  case GroupOperationClusteredReduce:
    GroupOp = "clustered_reduce";
    // OpenCL clustered builtin has no non_uniform prefix, ex.
    // sub_group_reduce_clustered_logical_and
    GroupPrefix = kSPIRVName::GroupPrefix;
    break;
  default:
    llvm_unreachable("Unsupported group operation!");
    break;
  }

  return Prefix + GroupPrefix + GroupOp + "_" + Op;
}

std::string SPIRVToOCLBase::getBallotBuiltinName(CallInst *CI, Op OC) {
  assert((OC == OpGroupNonUniformBallotBitCount) &&
         "Not inteded to handle other opcodes than "
         "OpGroupNonUniformBallotBitCount!");
  std::string Prefix = getGroupBuiltinPrefix(CI);
  assert(
      (Prefix == kOCLBuiltinName::SubPrefix) &&
      "Workgroup scope is not supported for OpGroupNonUniformBallotBitCount");
  std::string GroupOp;
  auto GO = getArgAs<spv::GroupOperation>(CI, 1);
  switch (GO) {
  case GroupOperationReduce:
    GroupOp = "bit_count";
    break;
  case GroupOperationInclusiveScan:
    GroupOp = "inclusive_scan";
    break;
  case GroupOperationExclusiveScan:
    GroupOp = "exclusive_scan";
    break;
  default:
    llvm_unreachable("Unsupported group operation!");
    break;
  }
  return Prefix + kSPIRVName::GroupPrefix + "ballot_" + GroupOp;
}

std::string SPIRVToOCLBase::groupOCToOCLBuiltinName(CallInst *CI, Op OC) {
  auto FuncName = OCLSPIRVBuiltinMap::rmap(OC);
  assert(FuncName.find(kSPIRVName::GroupPrefix) == 0);

  if (!hasGroupOperation(OC)) {
    /// Transform OpenCL group builtin function names from group_
    /// to work_group_ and sub_group_.
    FuncName = getGroupBuiltinPrefix(CI) + FuncName;
  } else { // Opcodes with group operation parameter
    if (isUniformArithmeticOpCode(OC))
      FuncName = getUniformArithmeticBuiltinName(CI, OC);
    else if (isNonUniformArithmeticOpCode(OC))
      FuncName = getNonUniformArithmeticBuiltinName(CI, OC);
    else if (OC == OpGroupNonUniformBallotBitCount)
      FuncName = getBallotBuiltinName(CI, OC);
    else
      llvm_unreachable("Unsupported opcode!");
  }
  return FuncName;
}

/// Return true if the original boolean return type needs to be changed to i32
/// when mapping the SPIR-V op to an OpenCL builtin.
static bool needsInt32RetTy(Op OC) {
  return OC == OpGroupAny || OC == OpGroupAll || OC == OpGroupNonUniformAny ||
         OC == OpGroupNonUniformAll || OC == OpGroupNonUniformAllEqual ||
         OC == OpGroupNonUniformElect || OC == OpGroupNonUniformInverseBallot ||
         OC == OpGroupNonUniformBallotBitExtract || isGroupLogicalOpCode(OC);
}

void SPIRVToOCLBase::visitCallSPIRVGroupBuiltin(CallInst *CI, Op OC) {
  auto FuncName = groupOCToOCLBuiltinName(CI, OC);
  auto ModifyArguments = [=](CallInst *, std::vector<Value *> &Args,
                             llvm::Type *&RetTy) {
    Type *Int32Ty = Type::getInt32Ty(*Ctx);
    bool HasArg0ExtendedToi32 =
        OC == OpGroupAny || OC == OpGroupAll || OC == OpGroupNonUniformAny ||
        OC == OpGroupNonUniformAll || OC == OpGroupNonUniformBallot ||
        isGroupLogicalOpCode(OC);
    /// Remove Group Operation argument,
    /// as in OpenCL representation this is included in the function name
    Args.erase(Args.begin(), Args.begin() + (hasGroupOperation(OC) ? 2 : 1));

    // Handle function arguments
    if (OC == OpGroupBroadcast)
      expandVector(CI, Args, 1);
    else if (HasArg0ExtendedToi32)
      Args[0] = CastInst::CreateZExtOrBitCast(Args[0], Int32Ty, "", CI);

    // Handle function return type
    if (needsInt32RetTy(OC))
      RetTy = Int32Ty;

    return FuncName;
  };
  auto ModifyRetTy = [=](CallInst *CI) -> Instruction * {
    if (needsInt32RetTy(OC)) {
      // The OpenCL builtin returns a non-zero integer value. Convert to a
      // boolean value.
      Constant *Zero = ConstantInt::get(CI->getType(), 0);
      return new ICmpInst(CI->getNextNode(), CmpInst::ICMP_NE, CI, Zero);
    } else
      return CI;
  };

  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(M, CI, ModifyArguments, ModifyRetTy, &Attrs);
}

void SPIRVToOCLBase::visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC) {
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

void SPIRVToOCLBase::visitCallSPIRVImageMediaBlockBuiltin(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        // Moving the first argument to the end.
        std::rotate(Args.rbegin(), Args.rend() - 1, Args.rend());
        Type *RetType = CI->getType();
        if (OC == OpSubgroupImageMediaBlockWriteINTEL) {
          assert(Args.size() >= 4 && "Wrong media block write signature");
          RetType = Args.at(3)->getType(); // texel type
        }
        unsigned int BitWidth = RetType->getScalarSizeInBits();
        std::string FuncPostfix;
        if (BitWidth == 8)
          FuncPostfix = "_uc";
        else if (BitWidth == 16)
          FuncPostfix = "_us";
        else if (BitWidth == 32)
          FuncPostfix = "_ui";
        else
          assert(0 && "Unsupported texel type!");

        if (auto *VecTy = dyn_cast<FixedVectorType>(RetType)) {
          unsigned int NumEl = VecTy->getNumElements();
          assert((NumEl == 2 || NumEl == 4 || NumEl == 8 || NumEl == 16) &&
                 "Wrong function type!");
          FuncPostfix += std::to_string(NumEl);
        }

        return OCLSPIRVBuiltinMap::rmap(OC) + FuncPostfix;
      },
      &Attrs);
}

void SPIRVToOCLBase::visitCallSPIRVCvtBuiltin(CallInst *CI, Op OC,
                                              StringRef DemangledName) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *Call, std::vector<Value *> &Args) {
        std::string CastBuiltInName;
        if (isCvtFromUnsignedOpCode(OC))
          CastBuiltInName = "u";
        CastBuiltInName += kOCLBuiltinName::ConvertPrefix;
        Type *DstTy = Call->getType();
        CastBuiltInName +=
            mapLLVMTypeToOCLType(DstTy, !isCvtToUnsignedOpCode(OC));
        if (DemangledName.find("_sat") != StringRef::npos || isSatCvtOpCode(OC))
          CastBuiltInName += "_sat";
        Value *Src = Call->getOperand(0);
        assert(Src && "Invalid SPIRV convert builtin call");
        Type *SrcTy = Src->getType();
        auto Loc = DemangledName.find("_rt");
        if (Loc != StringRef::npos &&
            !(isa<IntegerType>(SrcTy) && isa<IntegerType>(DstTy)))
          CastBuiltInName += DemangledName.substr(Loc, 4).str();
        return CastBuiltInName;
      },
      &Attrs);
}

void SPIRVToOCLBase::visitCallAsyncWorkGroupCopy(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        // First argument of AsyncWorkGroupCopy instruction is Scope, OCL
        // built-in async_work_group_strided_copy doesn't have this argument
        Args.erase(Args.begin());
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

void SPIRVToOCLBase::visitCallGroupWaitEvents(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        // First argument of GroupWaitEvents instruction is Scope, OCL
        // built-in wait_group_events doesn't have this argument
        Args.erase(Args.begin());
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

static char getTypeSuffix(Type *T) {
  char Suffix;

  Type *ST = T->getScalarType();
  if (ST->isHalfTy())
    Suffix = 'h';
  else if (ST->isFloatTy())
    Suffix = 'f';
  else
    Suffix = 'i';

  return Suffix;
}

// TODO: Handle unsigned integer return type. May need spec change.
void SPIRVToOCLBase::visitCallSPIRVImageSampleExplicitLodBuiltIn(CallInst *CI,
                                                                 Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  StringRef ImageTypeName;
  bool IsDepthImage = false;
  if (isOCLImageType(
          (cast<CallInst>(CI->getOperand(0)))->getArgOperand(0)->getType(),
          &ImageTypeName))
    IsDepthImage = ImageTypeName.contains("_depth_");

  auto ModifyArguments = [=](CallInst *, std::vector<Value *> &Args,
                             llvm::Type *&RetTy) {
    CallInst *CallSampledImg = cast<CallInst>(Args[0]);
    auto Img = CallSampledImg->getArgOperand(0);
    assert(isOCLImageType(Img->getType()));
    auto Sampler = CallSampledImg->getArgOperand(1);
    Args[0] = Img;
    Args.insert(Args.begin() + 1, Sampler);
    if (Args.size() > 4) {
      ConstantInt *ImOp = dyn_cast<ConstantInt>(Args[3]);
      ConstantFP *LodVal = dyn_cast<ConstantFP>(Args[4]);
      // Drop "Image Operands" argument.
      Args.erase(Args.begin() + 3, Args.begin() + 4);
      // If the image operand is LOD and its value is zero, drop it too.
      if (ImOp && LodVal && LodVal->isNullValue() &&
          ImOp->getZExtValue() == ImageOperandsMask::ImageOperandsLodMask)
        Args.erase(Args.begin() + 3, Args.end());
    }
    if (CallSampledImg->hasOneUse()) {
      CallSampledImg->replaceAllUsesWith(
          UndefValue::get(CallSampledImg->getType()));
      CallSampledImg->dropAllReferences();
      CallSampledImg->eraseFromParent();
    }
    Type *T = CI->getType();
    if (auto VT = dyn_cast<VectorType>(T))
      T = VT->getElementType();
    RetTy = IsDepthImage ? T : CI->getType();
    return std::string(kOCLBuiltinName::SampledReadImage) + getTypeSuffix(T);
  };

  auto ModifyRetTy = [=](CallInst *NewCI) -> Instruction * {
    if (IsDepthImage) {
      auto Ins = InsertElementInst::Create(
          UndefValue::get(FixedVectorType::get(NewCI->getType(), 4)), NewCI,
          getSizet(M, 0));
      Ins->insertAfter(NewCI);
      return Ins;
    }
    return NewCI;
  };

  mutateCallInstOCL(M, CI, ModifyArguments, ModifyRetTy, &Attrs);
}

void SPIRVToOCLBase::visitCallSPIRVBuiltin(CallInst *CI, Op OC) {
  AttributeList Attrs = CI->getCalledFunction()->getAttributes();
  mutateCallInstOCL(
      M, CI,
      [=](CallInst *, std::vector<Value *> &Args) {
        return OCLSPIRVBuiltinMap::rmap(OC);
      },
      &Attrs);
}

std::string SPIRVToOCLBase::getGroupBuiltinPrefix(CallInst *CI) {
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

ModulePass *
llvm::createSPIRVBIsLoweringPass(Module &M,
                                 SPIRV::BIsRepresentation BIsRepresentation) {
  switch (BIsRepresentation) {
  case SPIRV::BIsRepresentation::OpenCL12:
    return createSPIRVToOCL12Legacy();
  case SPIRV::BIsRepresentation::OpenCL20:
    return createSPIRVToOCL20Legacy();
  case SPIRV::BIsRepresentation::SPIRVFriendlyIR:
    // nothing to do, already done
    return nullptr;
  }
  llvm_unreachable("Unsupported built-ins representation");
  return nullptr;
}
