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
#include "llvm/IR/TypedPointerType.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"

namespace SPIRV {

void SPIRVToOCLBase::visitCallInst(CallInst &CI) {
  LLVM_DEBUG(dbgs() << "[visistCallInst] " << CI << '\n');
  auto *F = CI.getCalledFunction();
  if (!F)
    return;

  OCLExtOpKind ExtOp;
  if (isSPIRVOCLExtInst(&CI, &ExtOp)) {
    switch (ExtOp) {
    case OpenCLLIB::Vloadn:
    case OpenCLLIB::Vloada_halfn:
    case OpenCLLIB::Vload_halfn:
      visitCallSPIRVVLoadn(&CI, ExtOp);
      break;
    case OpenCLLIB::Vstoren:
    case OpenCLLIB::Vstore_halfn:
    case OpenCLLIB::Vstorea_halfn:
    case OpenCLLIB::Vstore_half_r:
    case OpenCLLIB::Vstore_halfn_r:
    case OpenCLLIB::Vstorea_halfn_r:
      visitCallSPIRVVStore(&CI, ExtOp);
      break;
    case OpenCLLIB::Printf: {
      // TODO: Lower the printf instruction with the non-constant address space
      // format string to suitable for OpenCL representation
      auto *PT = dyn_cast<PointerType>(CI.getOperand(0)->getType());
      if (PT && PT->getAddressSpace() == SPIR::TypeAttributeEnum::ATTR_CONST)
        visitCallSPIRVPrintf(&CI, ExtOp);
      break;
    }
    default:
      visitCallSPIRVOCLExt(&CI, ExtOp);
      break;
    }
    return;
  }

  auto MangledName = F->getName();
  StringRef DemangledName;
  Op OC = OpNop;
  SPIRVBuiltinVariableKind BuiltinKind = SPIRVBuiltinVariableKind::BuiltInMax;
  if (!oclIsBuiltin(MangledName, DemangledName) ||
      ((OC = getSPIRVFuncOC(DemangledName)) == OpNop &&
       !getSPIRVBuiltin(DemangledName.str(), BuiltinKind)))
    return;
  LLVM_DEBUG(dbgs() << "DemangledName = " << DemangledName.str() << '\n'
                    << "OpCode = " << OC << '\n'
                    << "BuiltinKind = " << BuiltinKind << '\n');

  if (BuiltinKind != SPIRVBuiltinVariableKind::BuiltInMax) {
    if (static_cast<uint32_t>(BuiltinKind) >=
            internal::BuiltInSubDeviceIDINTEL &&
        static_cast<uint32_t>(BuiltinKind) <=
            internal::BuiltInGlobalHWThreadIDINTEL)
      return;

    visitCallSPIRVBuiltin(&CI, BuiltinKind);
    return;
  }

  if (OC == OpImageQuerySize || OC == OpImageQuerySizeLod) {
    visitCallSPIRVImageQuerySize(&CI);
    return;
  }
  if (OC == OpMemoryBarrier) {
    visitCallSPIRVMemoryBarrier(&CI);
    return;
  }
  if (OC == OpControlBarrier) {
    visitCallSPIRVControlBarrier(&CI);
  }
  if (isSplitBarrierINTELOpCode(OC)) {
    visitCallSPIRVSplitBarrierINTEL(&CI, OC);
    return;
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
  if (isIntelSubgroupOpCode(OC)) {
    visitCallSPIRVSubgroupINTELBuiltIn(&CI, OC);
    return;
  }
  if (isSubgroupAvcINTELEvaluateOpcode(OC)) {
    visitCallSPIRVAvcINTELEvaluateBuiltIn(&CI, OC);
    return;
  }
  if (isSubgroupAvcINTELInstructionOpCode(OC)) {
    visitCallSPIRVAvcINTELInstructionBuiltin(&CI, OC);
    return;
  }
  if (OC == OpBuildNDRange) {
    visitCallBuildNDRangeBuiltIn(&CI, OC, DemangledName);
    return;
  }
  if (OC == OpGenericCastToPtrExplicit) {
    visitCallGenericCastToPtrExplicitBuiltIn(&CI, OC);
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
  if (OC == OpImageWrite) {
    visitCallSPIRVImageWriteBuiltIn(&CI, OC);
    return;
  }
  if (OC == OpImageRead) {
    visitCallSPIRVImageReadBuiltIn(&CI, OC);
    return;
  }
  if (OC == OpImageQueryOrder || OC == OpImageQueryFormat) {
    visitCallSPIRVImageQueryBuiltIn(&CI, OC);
    return;
  }
  if (OC == OpEnqueueKernel) {
    visitCallSPIRVEnqueueKernel(&CI, OC);
    return;
  }
  if (OC == OpGenericPtrMemSemantics) {
    visitCallSPIRVGenericPtrMemSemantics(&CI);
    return;
  }
  // Check if OC is OpenCL relational builtin except bitselect and select.
  auto IsOclRelationalOp = [](Op OC) {
    return isUnaryPredicateOpCode(OC) || OC == OpOrdered || OC == OpUnordered ||
           OC == OpFOrdEqual || OC == OpFUnordNotEqual ||
           OC == OpFOrdGreaterThan || OC == OpFOrdGreaterThanEqual ||
           OC == OpFOrdLessThan || OC == OpFOrdLessThanEqual ||
           OC == OpFOrdNotEqual;
  };
  if (IsOclRelationalOp(OC)) {
    if (OC == OpAny || OC == OpAll)
      visitCallSPIRVAnyAll(&CI, OC);
    else
      visitCallSPIRVRelational(&CI, OC);
    return;
  }
  if (OC == internal::OpConvertFToBF16INTEL ||
      OC == internal::OpConvertBF16ToFINTEL) {
    visitCallSPIRVBFloat16Conversions(&CI, OC);
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

void SPIRVToOCLBase::visitCallSPIRVImageQuerySize(CallInst *CI) {
  // Get image type
  Type *ImgTy = getCallValueType(CI, 0);
  auto Desc = getImageDescriptor(ImgTy);
  unsigned ImgDim = getImageDimension(Desc.Dim);
  bool ImgArray = Desc.Arrayed;

  AttributeList Attributes = CI->getCalledFunction()->getAttributes();
  BuiltinFuncMangleInfo Mangle;
  Mangle.getTypeMangleInfo(0).PointerTy = ImgTy;
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

std::string SPIRVToOCLBase::getRotateBuiltinName(CallInst *CI, Op OC) {
  assert((OC == OpGroupNonUniformRotateKHR) &&
         "Not intended to handle other opcodes");
  std::string Prefix = getGroupBuiltinPrefix(CI);
  assert((Prefix == kOCLBuiltinName::SubPrefix) &&
         "Workgroup scope is not supported for OpGroupNonUniformRotateKHR");

  std::string OptionalClustered;
  if (CI->arg_size() == 4)
    OptionalClustered = "clustered_";
  return Prefix + kSPIRVName::GroupPrefix + OptionalClustered + "rotate";
}

std::string SPIRVToOCLBase::groupOCToOCLBuiltinName(CallInst *CI, Op OC) {
  if (OC == OpGroupNonUniformRotateKHR)
    return getRotateBuiltinName(CI, OC);

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
  auto Mutator = mutateCallInst(CI, FuncName);
  /// Remove Group Operation argument,
  /// as in OpenCL representation this is included in the function name
  Mutator.removeArgs(0, (hasGroupOperation(OC) ? 2 : 1));

  Type *Int32Ty = Type::getInt32Ty(*Ctx);
  bool HasArg0ExtendedToi32 =
      OC == OpGroupAny || OC == OpGroupAll || OC == OpGroupNonUniformAny ||
      OC == OpGroupNonUniformAll || OC == OpGroupNonUniformBallot ||
      isGroupLogicalOpCode(OC);

  // Handle function arguments
  if (OC == OpGroupBroadcast) {
    Value *VecArg = Mutator.getArg(1);
    if (auto *VT = dyn_cast<FixedVectorType>(VecArg->getType())) {
      unsigned NumElements = VT->getNumElements();
      for (unsigned I = 0; I < NumElements; I++)
        Mutator.insertArg(1 + I, Mutator.Builder.CreateExtractElement(
                                     VecArg, Mutator.Builder.getInt32(I)));
      Mutator.removeArg(1 + NumElements);
    }
  } else if (HasArg0ExtendedToi32)
    Mutator.mapArg(0, [](IRBuilder<> &Builder, Value *V) {
      return Builder.CreateZExt(V, Builder.getInt32Ty());
    });

  // Handle function return type
  if (needsInt32RetTy(OC))
    Mutator.changeReturnType(Int32Ty, [](IRBuilder<> &Builder, CallInst *CI) {
      // The OpenCL builtin returns a non-zero integer value. Convert to a
      // boolean value.
      return Builder.CreateICmpNE(CI, Builder.getInt32(0));
    });
}

void SPIRVToOCLBase::visitCallSPIRVPipeBuiltin(CallInst *CI, Op OC) {
  auto DemangledName = OCLSPIRVBuiltinMap::rmap(OC);
  bool HasScope = DemangledName.find(kSPIRVName::GroupPrefix) == 0;
  if (HasScope)
    DemangledName = getGroupBuiltinPrefix(CI) + DemangledName;

  assert(CI->getCalledFunction() && "Unexpected indirect call");
  auto Mutator = mutateCallInst(CI, DemangledName);
  if (HasScope)
    Mutator.removeArg(0);
  if (OC == OpReadPipe || OC == OpWritePipe || OC == OpReservedReadPipe ||
      OC == OpReservedWritePipe || OC == OpReadPipeBlockingINTEL ||
      OC == OpWritePipeBlockingINTEL) {
    Mutator.mapArg(Mutator.arg_size() - 3, [](IRBuilder<> &Builder, Value *P) {
      Type *T = P->getType();
      assert(isa<PointerType>(T));
      auto *NewTy = Builder.getPtrTy(SPIRAS_Generic);
      if (T != NewTy) {
        P = Builder.CreatePointerBitCastOrAddrSpaceCast(P, NewTy);
      }
      return std::make_pair(
          P, TypedPointerType::get(Builder.getInt8Ty(), SPIRAS_Generic));
    });
  }
}

void SPIRVToOCLBase::visitCallSPIRVImageMediaBlockBuiltin(CallInst *CI, Op OC) {
  Type *RetType = CI->getType();
  if (OC == OpSubgroupImageMediaBlockWriteINTEL) {
    assert(CI->arg_size() >= 5 && "Wrong media block write signature");
    RetType = CI->getArgOperand(4)->getType(); // texel type
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

  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC) + FuncPostfix)
      .moveArg(0, CI->arg_size() - 1);
}

void SPIRVToOCLBase::visitCallBuildNDRangeBuiltIn(CallInst *CI, Op OC,
                                                  StringRef DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  // __spirv_BuildNDRange_nD, drop __spirv_
  StringRef S = DemangledName;
  S = S.drop_front(strlen(kSPIRVName::Prefix));
  SmallVector<StringRef, 8> Split;
  // BuildNDRange_nD
  S.split(Split, kSPIRVPostfix::Divider,
          /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  assert(Split.size() >= 2 && "Invalid SPIRV function name");
  // Cut _nD and add it to function name.
  mutateCallInst(CI, std::string(kOCLBuiltinName::NDRangePrefix) +
                         Split[1].substr(0, 3).str())
      // OpenCL built-in has another order of parameters.
      .moveArg(2, 0);
}

void SPIRVToOCLBase::visitCallGenericCastToPtrExplicitBuiltIn(CallInst *CI,
                                                              Op OC) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  StringRef Name;
  auto AddrSpace =
      static_cast<SPIRAddressSpace>(CI->getType()->getPointerAddressSpace());
  switch (AddrSpace) {
  case SPIRAS_Global:
    Name = kOCLBuiltinName::ToGlobal;
    break;
  case SPIRAS_Local:
    Name = kOCLBuiltinName::ToLocal;
    break;
  case SPIRAS_Private:
    Name = kOCLBuiltinName::ToPrivate;
    break;
  default:
    llvm_unreachable("Invalid address space");
  }
  mutateCallInst(CI, Name.str())
      // The instruction has two arguments, whereas ocl built-in has only one
      // argument.
      .removeArg(1);
}

void SPIRVToOCLBase::visitCallSPIRVCvtBuiltin(CallInst *CI, Op OC,
                                              StringRef DemangledName) {
  std::string CastBuiltInName;
  if (isCvtFromUnsignedOpCode(OC))
    CastBuiltInName = "u";
  CastBuiltInName += kOCLBuiltinName::ConvertPrefix;
  Type *DstTy = CI->getType();
  CastBuiltInName += mapLLVMTypeToOCLType(DstTy, !isCvtToUnsignedOpCode(OC));
  if (DemangledName.find("_sat") != StringRef::npos || isSatCvtOpCode(OC))
    CastBuiltInName += "_sat";
  Value *Src = CI->getOperand(0);
  assert(Src && "Invalid SPIRV convert builtin call");
  Type *SrcTy = Src->getType();
  auto Loc = DemangledName.find("_rt");
  if (Loc != StringRef::npos &&
      !(isa<IntegerType>(SrcTy) && isa<IntegerType>(DstTy)))
    CastBuiltInName += DemangledName.substr(Loc, 4).str();
  mutateCallInst(CI, CastBuiltInName);
}

void SPIRVToOCLBase::visitCallAsyncWorkGroupCopy(CallInst *CI, Op OC) {
  // First argument of AsyncWorkGroupCopy instruction is Scope, OCL
  // built-in async_work_group_strided_copy doesn't have this argument
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC)).removeArg(0);
}

void SPIRVToOCLBase::visitCallGroupWaitEvents(CallInst *CI, Op OC) {
  // First argument of GroupWaitEvents instruction is Scope, OCL
  // built-in wait_group_events doesn't have this argument
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC)).removeArg(0);
}

static std::string getTypeSuffix(Type *T, bool IsSigned) {
  std::string Suffix;

  Type *ST = T->getScalarType();
  if (ST->isHalfTy())
    Suffix = "h";
  else if (ST->isFloatTy())
    Suffix = "f";
  else if (IsSigned)
    Suffix = "i";
  else
    Suffix = "ui";

  return Suffix;
}

BuiltinCallMutator
SPIRVToOCLBase::mutateCallImageOperands(CallInst *CI, StringRef NewFuncName,
                                        Type *T, unsigned ImOpArgIndex) {
  // Default to signed.
  bool IsSigned = true;
  uint64_t ImOpValue = 0;
  if (CI->arg_size() > ImOpArgIndex) {
    ConstantInt *ImOp = dyn_cast<ConstantInt>(CI->getArgOperand(ImOpArgIndex));
    if (ImOp)
      ImOpValue = ImOp->getZExtValue();
    unsigned SignZeroExtMasks = ImageOperandsMask::ImageOperandsSignExtendMask |
                                ImageOperandsMask::ImageOperandsZeroExtendMask;
    // If one of the SPIR-V 1.4 SignExtend/ZeroExtend operands is present, take
    // it into account and drop the mask.
    if (ImOpValue & SignZeroExtMasks) {
      if (ImOpValue & ImageOperandsMask::ImageOperandsZeroExtendMask)
        IsSigned = false;
      ImOpValue &= ~SignZeroExtMasks;
    }
  }

  auto Mutator =
      mutateCallInst(CI, NewFuncName.str() + getTypeSuffix(T, IsSigned));
  if (ImOpArgIndex < Mutator.arg_size()) {
    // Drop "Image Operands" argument.
    Mutator.removeArg(ImOpArgIndex);
    if (ImOpArgIndex < Mutator.arg_size()) {
      ConstantFP *LodVal = dyn_cast<ConstantFP>(Mutator.getArg(ImOpArgIndex));
      // If the image operand is LOD and its value is zero, drop it too.
      if (LodVal && LodVal->isNullValue() &&
          ImOpValue == ImageOperandsMask::ImageOperandsLodMask)
        Mutator.removeArgs(ImOpArgIndex, Mutator.arg_size() - ImOpArgIndex);
    }
  }
  return Mutator;
}

void SPIRVToOCLBase::visitCallSPIRVImageSampleExplicitLodBuiltIn(CallInst *CI,
                                                                 Op OC) {
  Type *T = CI->getType();
  if (auto *VT = dyn_cast<VectorType>(T))
    T = VT->getElementType();
  auto Mutator =
      mutateCallImageOperands(CI, kOCLBuiltinName::SampledReadImage, T, 2);

  CallInst *CallSampledImg = cast<CallInst>(CI->getArgOperand(0));
  auto Img = getCallValue(CallSampledImg, 0);
  auto Sampler = getCallValue(CallSampledImg, 1);
  bool IsDepthImage = false;
  Mutator.mapArg(0, [&](Value *SampledImg) {
    StringRef ImageTypeName;
    if (isOCLImageType(Img.second, &ImageTypeName))
      IsDepthImage = ImageTypeName.contains("_depth_");

    if (CallSampledImg->hasOneUse()) {
      CallSampledImg->replaceAllUsesWith(
          UndefValue::get(CallSampledImg->getType()));
      CallSampledImg->dropAllReferences();
      CallSampledImg->eraseFromParent();
    }
    return Img;
  });
  Mutator.insertArg(1, Sampler);
  if (IsDepthImage)
    Mutator.changeReturnType(T, [&](IRBuilder<> &Builder, CallInst *NewCI) {
      return Builder.CreateInsertElement(
          FixedVectorType::get(NewCI->getType(), 4), NewCI, uint64_t(0));
    });
}

void SPIRVToOCLBase::visitCallSPIRVImageWriteBuiltIn(CallInst *CI, Op OC) {
  auto Mutator = mutateCallImageOperands(CI, kOCLBuiltinName::WriteImage,
                                         CI->getArgOperand(2)->getType(), 3);
  if (Mutator.arg_size() > 3)
    Mutator.moveArg(3, 2);
}

void SPIRVToOCLBase::visitCallSPIRVImageReadBuiltIn(CallInst *CI, Op OC) {
  mutateCallImageOperands(CI, kOCLBuiltinName::ReadImage, CI->getType(), 2);
}

void SPIRVToOCLBase::visitCallSPIRVImageQueryBuiltIn(CallInst *CI, Op OC) {
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC))
      .changeReturnType(CI->getType(), [=](IRBuilder<> &Builder, CallInst *CI) {
        unsigned int Offset = 0;
        if (OC == OpImageQueryFormat)
          Offset = OCLImageChannelDataTypeOffset;
        else if (OC == OpImageQueryOrder)
          Offset = OCLImageChannelOrderOffset;
        else
          llvm_unreachable("Unsupported opcode");
        return Builder.CreateSub(CI, Builder.getInt32(Offset));
      });
}

void SPIRVToOCLBase::visitCallSPIRVSubgroupINTELBuiltIn(CallInst *CI, Op OC) {
  std::stringstream Name;
  Type *DataTy = nullptr;
  switch (OC) {
  case OpSubgroupBlockReadINTEL:
  case OpSubgroupImageBlockReadINTEL:
    Name << "intel_sub_group_block_read";
    DataTy = CI->getType();
    break;
  case OpSubgroupBlockWriteINTEL:
    Name << "intel_sub_group_block_write";
    DataTy = CI->getOperand(1)->getType();
    break;
  case OpSubgroupImageBlockWriteINTEL:
    Name << "intel_sub_group_block_write";
    DataTy = CI->getOperand(2)->getType();
    break;
  default:
    Name << OCLSPIRVBuiltinMap::rmap(OC);
    break;
  }
  if (DataTy) {
    unsigned VectorNumElements = 1;
    if (FixedVectorType *VT = dyn_cast<FixedVectorType>(DataTy))
      VectorNumElements = VT->getNumElements();
    unsigned ElementBitSize = DataTy->getScalarSizeInBits();
    Name << getIntelSubgroupBlockDataPostfix(ElementBitSize, VectorNumElements);
  }
  mutateCallInst(CI, Name.str());
}

void SPIRVToOCLBase::visitCallSPIRVAvcINTELEvaluateBuiltIn(CallInst *CI,
                                                           Op OC) {
  // There are three types of AVC Intel Evaluate opcodes:
  // 1. With multi reference images - does not use OpVmeImageINTEL opcode
  // for reference images
  // 2. With dual reference images - uses two OpVmeImageINTEL opcodes for
  // reference image
  // 3. With single reference image - uses one OpVmeImageINTEL opcode for
  // reference image
  StringRef FnName = CI->getCalledFunction()->getName();
  int NumImages = 0;
  if (FnName.contains("SingleReference"))
    NumImages = 2;
  else if (FnName.contains("DualReference"))
    NumImages = 3;
  else if (FnName.contains("MultiReference"))
    NumImages = 1;
  else if (FnName.contains("EvaluateIpe"))
    NumImages = 1;

  auto EraseVmeImageCall = [](CallInst *CI) {
    if (CI->hasOneUse()) {
      CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
      CI->dropAllReferences();
      CI->eraseFromParent();
    }
  };

  auto Mutator =
      mutateCallInst(CI, OCLSPIRVSubgroupAVCIntelBuiltinMap::rmap(OC));
  if (NumImages) {
    CallInst *SrcImage = cast<CallInst>(Mutator.getArg(0));
    if (NumImages == 1) {
      // Multi reference opcode - remove src image OpVmeImageINTEL opcode
      // and replace it with corresponding OpImage and OpSampler arguments
      size_t SamplerPos = Mutator.arg_size() - 1;
      Mutator.replaceArg(0, getCallValue(SrcImage, 0));
      Mutator.insertArg(SamplerPos, getCallValue(SrcImage, 1));
    } else {
      CallInst *FwdRefImage = cast<CallInst>(Mutator.getArg(1));
      CallInst *BwdRefImage =
          NumImages == 3 ? cast<CallInst>(Mutator.getArg(2)) : nullptr;
      // Single reference opcode - remove src and ref image
      // OpVmeImageINTEL opcodes and replace them with src and ref OpImage
      // opcodes and OpSampler
      Mutator.removeArgs(0, NumImages);
      // insert source OpImage and OpSampler
      Mutator.insertArg(0, getCallValue(SrcImage, 0));
      Mutator.insertArg(1, getCallValue(SrcImage, 1));
      // insert reference OpImage
      Mutator.insertArg(1, getCallValue(FwdRefImage, 0));
      EraseVmeImageCall(SrcImage);
      EraseVmeImageCall(FwdRefImage);
      if (BwdRefImage) {
        // Dual reference opcode - insert second reference OpImage argument
        Mutator.insertArg(2, getCallValue(BwdRefImage, 0));
        EraseVmeImageCall(BwdRefImage);
      }
    }
  } else
    llvm_unreachable("invalid avc instruction");
}

void SPIRVToOCLBase::visitCallSPIRVGenericPtrMemSemantics(CallInst *CI) {
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OpGenericPtrMemSemantics))
      .changeReturnType(CI->getType(),
                        [](IRBuilder<> &Builder, CallInst *NewCI) {
                          return Builder.CreateShl(NewCI, Builder.getInt32(8));
                        });
}

void SPIRVToOCLBase::visitCallSPIRVBFloat16Conversions(CallInst *CI, Op OC) {
  Type *ArgTy = CI->getOperand(0)->getType();
  std::string N =
      ArgTy->isVectorTy()
          ? std::to_string(cast<FixedVectorType>(ArgTy)->getNumElements())
          : "";
  std::string Name;
  switch (static_cast<uint32_t>(OC)) {
  case internal::OpConvertFToBF16INTEL:
    Name = "intel_convert_bfloat16" + N + "_as_ushort" + N;
    break;
  case internal::OpConvertBF16ToFINTEL:
    Name = "intel_convert_as_bfloat16" + N + "_float" + N;
    break;
  default:
    break; // do nothing
  }
  mutateCallInst(CI, Name);
}

void SPIRVToOCLBase::visitCallSPIRVBuiltin(CallInst *CI, Op OC) {
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC));
}

void SPIRVToOCLBase::visitCallSPIRVBuiltin(CallInst *CI,
                                           SPIRVBuiltinVariableKind Kind) {
  mutateCallInst(CI, SPIRSPIRVBuiltinVariableMap::rmap(Kind));
}

void SPIRVToOCLBase::visitCallSPIRVAvcINTELInstructionBuiltin(CallInst *CI,
                                                              Op OC) {
  mutateCallInst(CI, OCLSPIRVSubgroupAVCIntelBuiltinMap::rmap(OC));
}

void SPIRVToOCLBase::visitCallSPIRVOCLExt(CallInst *CI, OCLExtOpKind Kind) {
  mutateCallInst(CI, OCLExtOpMap::map(Kind));
}

void SPIRVToOCLBase::visitCallSPIRVVLoadn(CallInst *CI, OCLExtOpKind Kind) {
  std::string Name = OCLExtOpMap::map(Kind);
  unsigned LastArg = CI->arg_size() - 1;
  if (ConstantInt *C = dyn_cast<ConstantInt>(CI->getArgOperand(LastArg))) {
    uint64_t NumComponents = C->getZExtValue();
    std::stringstream SS;
    SS << NumComponents;
    Name.replace(Name.find("n"), 1, SS.str());
  }
  mutateCallInst(CI, Name).removeArg(LastArg);
}

void SPIRVToOCLBase::visitCallSPIRVVStore(CallInst *CI, OCLExtOpKind Kind) {
  std::string Name = OCLExtOpMap::map(Kind);
  bool DropLastArg = false;
  if (Kind == OpenCLLIB::Vstore_half_r || Kind == OpenCLLIB::Vstore_halfn_r ||
      Kind == OpenCLLIB::Vstorea_halfn_r) {
    auto *C = cast<ConstantInt>(CI->getArgOperand(CI->arg_size() - 1));
    auto RoundingMode = static_cast<SPIRVFPRoundingModeKind>(C->getZExtValue());
    Name.replace(Name.find("_r"), 2,
                 std::string("_") +
                     SPIRSPIRVFPRoundingModeMap::rmap(RoundingMode));
    DropLastArg = true;
  }

  if (Kind == OpenCLLIB::Vstore_halfn || Kind == OpenCLLIB::Vstore_halfn_r ||
      Kind == OpenCLLIB::Vstorea_halfn || Kind == OpenCLLIB::Vstorea_halfn_r ||
      Kind == OpenCLLIB::Vstoren) {
    if (auto *DataType =
            dyn_cast<VectorType>(CI->getArgOperand(0)->getType())) {
      uint64_t NumElements = DataType->getElementCount().getFixedValue();
      assert((NumElements == 2 || NumElements == 3 || NumElements == 4 ||
              NumElements == 8 || NumElements == 16) &&
             "Unsupported vector size for vstore instruction!");
      std::stringstream SS;
      SS << NumElements;
      Name.replace(Name.find("n"), 1, SS.str());
    }
  }

  auto Mutator = mutateCallInst(CI, Name);
  if (DropLastArg)
    Mutator.removeArg(Mutator.arg_size() - 1);
}

void SPIRVToOCLBase::visitCallSPIRVPrintf(CallInst *CI, OCLExtOpKind Kind) {
  CallInst *NewCI = cast<CallInst>(
      mutateCallInst(CI, OCLExtOpMap::map(OpenCLLIB::Printf)).getMutated());

  // Clang represents printf function without mangling
  std::string TargetName = "printf";
  if (Function *F = M->getFunction(TargetName))
    NewCI->setCalledFunction(F);
  else
    NewCI->getCalledFunction()->setName(TargetName);
}

void SPIRVToOCLBase::visitCallSPIRVAnyAll(CallInst *CI, Op OC) {
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC))
      .mapArg(0,
              [](IRBuilder<> &Builder, Value *V) {
                Type *NewArgTy = V->getType()->getWithNewBitWidth(8);
                return Builder.CreateSExt(V, NewArgTy);
              })
      .changeReturnType(Type::getInt32Ty(*Ctx),
                        [=](IRBuilder<> &Builder, CallInst *NewCI) {
                          return Builder.CreateTrunc(NewCI, CI->getType());
                        });
}

void SPIRVToOCLBase::visitCallSPIRVRelational(CallInst *CI, Op OC) {
  Type *IntTy = Type::getInt32Ty(*Ctx);
  Type *RetTy = IntTy;
  if (CI->getType()->isVectorTy()) {
    auto *OpElemTy =
        cast<FixedVectorType>(CI->getOperand(0)->getType())->getElementType();
    if (OpElemTy->isDoubleTy())
      IntTy = Type::getInt64Ty(*Ctx);
    if (OpElemTy->isHalfTy())
      IntTy = Type::getInt16Ty(*Ctx);
    RetTy = FixedVectorType::get(
        IntTy, cast<FixedVectorType>(CI->getType())->getNumElements());
  }
  mutateCallInst(CI, OCLSPIRVBuiltinMap::rmap(OC))
      .changeReturnType(RetTy, [=](IRBuilder<> &Builder, CallInst *NewCI) {
        return Builder.CreateTruncOrBitCast(NewCI, CI->getType());
      });
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

std::string
SPIRVToOCLBase::getOCLImageOpaqueType(SmallVector<std::string, 8> &Postfixes) {
  SmallVector<int, 7> Ops;
  for (unsigned I = 1; I < 8; ++I)
    Ops.push_back(atoi(Postfixes[I].c_str()));
  SPIRVTypeImageDescriptor Desc(static_cast<SPIRVImageDimKind>(Ops[0]), Ops[1],
                                Ops[2], Ops[3], Ops[4], Ops[5]);

  std::string OCLStructName =
      std::string(kSPR2TypeName::OCLPrefix) + rmap<std::string>(Desc);

  SPIRVAccessQualifierKind Acc = static_cast<SPIRVAccessQualifierKind>(Ops[6]);
  insertImageNameAccessQualifier(Acc, OCLStructName);
  return OCLStructName;
}

std::string
SPIRVToOCLBase::getOCLPipeOpaqueType(SmallVector<std::string, 8> &Postfixes) {
  assert(Postfixes.size() == 1);
  unsigned PipeAccess = atoi(Postfixes[0].c_str());
  assert((PipeAccess == AccessQualifierReadOnly ||
          PipeAccess == AccessQualifierWriteOnly) &&
         "Invalid access qualifier");
  return PipeAccess ? kSPR2TypeName::PipeWO : kSPR2TypeName::PipeRO;
}

void SPIRVToOCLBase::translateOpaqueTypes() {
  for (auto *S : M->getIdentifiedStructTypes()) {
    StringRef STName = S->getStructName();
    bool IsSPIRVOpaque =
        S->isOpaque() && STName.starts_with(kSPIRVTypeName::PrefixAndDelim);

    if (!IsSPIRVOpaque)
      continue;

    S->setName(translateOpaqueType(STName));
  }
}

std::string SPIRVToOCLBase::translateOpaqueType(StringRef STName) {
  if (!STName.starts_with(kSPIRVTypeName::PrefixAndDelim))
    return STName.str();

  SmallVector<std::string, 8> Postfixes;
  std::string DecodedST = decodeSPIRVTypeName(STName, Postfixes);

  if (!SPIRVOpaqueTypeOpCodeMap::find(DecodedST))
    return STName.str();

  Op OP = SPIRVOpaqueTypeOpCodeMap::map(DecodedST);
  std::string OCLOpaqueName;
  if (OP == OpTypeImage)
    OCLOpaqueName = getOCLImageOpaqueType(Postfixes);
  else if (OP == OpTypePipe)
    OCLOpaqueName = getOCLPipeOpaqueType(Postfixes);
  else if (isSubgroupAvcINTELTypeOpCode(OP))
    OCLOpaqueName = OCLSubgroupINTELTypeOpCodeMap::rmap(OP);
  else if (isOpaqueGenericTypeOpCode(OP))
    OCLOpaqueName = OCLOpaqueTypeOpCodeMap::rmap(OP);
  else
    return STName.str();

  return OCLOpaqueName;
}

void addSPIRVBIsLoweringPass(ModulePassManager &PassMgr,
                             SPIRV::BIsRepresentation BIsRep) {
  switch (BIsRep) {
  case SPIRV::BIsRepresentation::OpenCL12:
    PassMgr.addPass(SPIRVToOCL12Pass());
    break;
  case SPIRV::BIsRepresentation::OpenCL20:
    PassMgr.addPass(SPIRVToOCL20Pass());
    break;
  case SPIRV::BIsRepresentation::SPIRVFriendlyIR:
    // nothing to do, already done
    break;
  }
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
