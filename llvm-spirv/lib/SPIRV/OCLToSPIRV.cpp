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

#include "OCLToSPIRV.h"
#include "OCLTypeToSPIRV.h"
#include "SPIRVInternal.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <regex>
#include <set>

using namespace llvm;
using namespace PatternMatch;
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

static Type *getBlockStructType(Value *Parameter) {
  // In principle, this information should be passed to us from Clang via
  // an elementtype attribute. However, said attribute requires that the
  // function call be an intrinsic, which it is not. Instead, we rely on being
  // able to trace this to the declaration of a variable: OpenCL C specification
  // section 6.12.5 should guarantee that we can do this.
  Value *UnderlyingObject = Parameter->stripPointerCasts();
  Type *ParamType = nullptr;
  if (auto *GV = dyn_cast<GlobalValue>(UnderlyingObject))
    ParamType = GV->getValueType();
  else if (auto *Alloca = dyn_cast<AllocaInst>(UnderlyingObject))
    ParamType = Alloca->getAllocatedType();
  else
    llvm_unreachable("Blocks in OpenCL C must be traceable to allocation site");
  return ParamType;
}

/// Return one of the SPIR-V 1.4 SignExtend or ZeroExtend image operands
/// for a demangled function name, or 0 if the function does not return an
/// integer type (e.g. read_imagef).
static unsigned getImageSignZeroExt(StringRef DemangledName) {
  bool IsSigned = !DemangledName.ends_with("ui") && DemangledName.back() == 'i';
  bool IsUnsigned = DemangledName.ends_with("ui");

  if (IsSigned)
    return ImageOperandsMask::ImageOperandsSignExtendMask;
  if (IsUnsigned)
    return ImageOperandsMask::ImageOperandsZeroExtendMask;
  return 0;
}

bool OCLToSPIRVLegacy::runOnModule(Module &M) {
  setOCLTypeToSPIRV(&getAnalysis<OCLTypeToSPIRVLegacy>());
  return runOCLToSPIRV(M);
}

void OCLToSPIRVLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<OCLTypeToSPIRVLegacy>();
}

llvm::PreservedAnalyses OCLToSPIRVPass::run(llvm::Module &M,
                                            llvm::ModuleAnalysisManager &MAM) {
  setOCLTypeToSPIRV(&MAM.getResult<OCLTypeToSPIRVPass>(M));
  return runOCLToSPIRV(M) ? llvm::PreservedAnalyses::none()
                          : llvm::PreservedAnalyses::all();
}

/// Get vector width from OpenCL vload* function name.
SPIRVWord OCLToSPIRVBase::getVecLoadWidth(const std::string &DemangledName) {
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
void OCLToSPIRVBase::transVecLoadStoreName(std::string &DemangledName,
                                           const std::string &Stem,
                                           bool AlwaysN) {
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

char OCLToSPIRVLegacy::ID = 0;

bool OCLToSPIRVBase::runOCLToSPIRV(Module &Module) {
  initialize(Module);
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

  for (Instruction *I : ValuesToDelete)
    I->eraseFromParent();

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
  auto *F = CI.getCalledFunction();
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

    auto *PCI = &CI;
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
      visitCallReadImageWithSampler(&CI, MangledName, DemangledName);
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
      CI.getOperand(0)->getType()->isFloatingPointTy()) {
    visitCallDot(&CI);
    return;
  }
  if (DemangledName == kOCLBuiltinName::Dot ||
      DemangledName == kOCLBuiltinName::DotAccSat ||
      DemangledName.starts_with(kOCLBuiltinName::Dot4x8PackedPrefix) ||
      DemangledName.starts_with(kOCLBuiltinName::DotAccSat4x8PackedPrefix)) {
    if (CI.getOperand(0)->getType()->isVectorTy()) {
      auto *VT = (VectorType *)(CI.getOperand(0)->getType());
      if (!isa<llvm::IntegerType>(VT->getElementType())) {
        visitCallBuiltinSimple(&CI, MangledName, DemangledName);
        return;
      }
    }
    visitCallDot(&CI, MangledName, DemangledName);
    return;
  }
  if (DemangledName.starts_with(kOCLBuiltinName::ClockReadPrefix)) {
    visitCallClockRead(&CI, MangledName, DemangledName);
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
  if (DemangledName.find(kOCLBuiltinName::SplitBarrierINTELPrefix) == 0) {
    visitCallSplitBarrierINTEL(&CI, DemangledName);
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
  if (DemangledName == kOCLBuiltinName::ConvertBFloat16AsUShort ||
      DemangledName == kOCLBuiltinName::ConvertBFloat162AsUShort2 ||
      DemangledName == kOCLBuiltinName::ConvertBFloat163AsUShort3 ||
      DemangledName == kOCLBuiltinName::ConvertBFloat164AsUShort4 ||
      DemangledName == kOCLBuiltinName::ConvertBFloat168AsUShort8 ||
      DemangledName == kOCLBuiltinName::ConvertBFloat1616AsUShort16) {
    visitCallConvertBFloat16AsUshort(&CI, DemangledName);
    return;
  }
  if (DemangledName == kOCLBuiltinName::ConvertAsBFloat16Float ||
      DemangledName == kOCLBuiltinName::ConvertAsBFloat162Float2 ||
      DemangledName == kOCLBuiltinName::ConvertAsBFloat163Float3 ||
      DemangledName == kOCLBuiltinName::ConvertAsBFloat164Float4 ||
      DemangledName == kOCLBuiltinName::ConvertAsBFloat168Float8 ||
      DemangledName == kOCLBuiltinName::ConvertAsBFloat1616Float16) {
    visitCallConvertAsBFloat16Float(&CI, DemangledName);
    return;
  }
  visitCallBuiltinSimple(&CI, MangledName, DemangledName);
}

void OCLToSPIRVBase::visitCallNDRange(CallInst *CI, StringRef DemangledName) {
  assert(DemangledName.find(kOCLBuiltinName::NDRangePrefix) == 0);
  StringRef LenStr = DemangledName.substr(8, 1);
  auto Len = atoi(LenStr.data());
  assert(Len >= 1 && Len <= 3);
  // Translate ndrange_ND into differently named SPIR-V
  // decorated functions because they have array arugments
  // of different dimension which mangled the same way.
  std::string Postfix("_");
  Postfix += LenStr;
  Postfix += 'D';
  std::string FuncName = getSPIRVFuncName(OpBuildNDRange, Postfix);
  auto Mutator = mutateCallInst(CI, FuncName);

  // SPIR-V ndrange structure requires 3 members in the following order:
  //   global work offset
  //   global work size
  //   local work size
  // The arguments need to add missing members.
  for (size_t I = 1, E = CI->arg_size(); I != E; ++I)
    Mutator.mapArg(I, [=](Value *V) { return getScalarOrArray(V, Len, CI); });
  switch (CI->arg_size()) {
  case 2: {
    // Has global work size.
    auto *T = Mutator.getArg(1)->getType();
    auto *C = getScalarOrArrayConstantInt(CI, T, Len, 0);
    Mutator.appendArg(C);
    Mutator.appendArg(C);
    break;
  }
  case 3: {
    // Has global and local work size.
    auto *T = Mutator.getArg(1)->getType();
    Mutator.appendArg(getScalarOrArrayConstantInt(CI, T, Len, 0));
    break;
  }
  case 4: {
    // Move offset arg to the end
    Mutator.moveArg(1, CI->arg_size() - 1);
    break;
  }
  default:
    assert(0 && "Invalid number of arguments");
  }
}

void OCLToSPIRVBase::visitCallAsyncWorkGroupCopy(CallInst *CI,
                                                 StringRef DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  auto Mutator = mutateCallInst(CI, OpGroupAsyncCopy);
  if (DemangledName == OCLUtil::kOCLBuiltinName::AsyncWorkGroupCopy)
    Mutator.insertArg(3, addSizet(1));
  Mutator.insertArg(0, addInt32(ScopeWorkgroup));
}

CallInst *OCLToSPIRVBase::visitCallAtomicCmpXchg(CallInst *CI) {
  CallInst *NewCI = nullptr;
  {
    auto Mutator = mutateCallInst(CI, kOCLBuiltinName::AtomicCmpXchgStrong);
    Value *Expected = Mutator.getArg(1);
    Type *MemTy = Mutator.getArg(2)->getType();
    assert(MemTy->isIntegerTy() &&
           "In SPIR-V 1.0 arguments of OpAtomicCompareExchange must be "
           "an integer type scalars");
    Mutator.mapArg(1, [=](IRBuilder<> &Builder, Value *V) {
      return Builder.CreateLoad(MemTy, V, "exp");
    });
    Mutator.changeReturnType(
        MemTy, [Expected, &NewCI](IRBuilder<> &Builder, CallInst *NCI) {
          NewCI = NCI;
          Builder.CreateStore(NCI, Expected);
          return Builder.CreateICmpEQ(NCI, NCI->getArgOperand(1));
        });
  }
  return NewCI;
}

void OCLToSPIRVBase::visitCallAtomicInit(CallInst *CI) {
  auto *ST = new StoreInst(CI->getArgOperand(1), CI->getArgOperand(0),
                           CI->getIterator());
  ST->takeName(CI);
  CI->dropAllReferences();
  CI->eraseFromParent();
}

void OCLToSPIRVBase::visitCallAllAny(spv::Op OC, CallInst *CI) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");

  auto Args = getArguments(CI);
  assert(Args.size() == 1);

  auto *ArgTy = Args[0]->getType();
  auto *Zero = Constant::getNullValue(Args[0]->getType());

  auto *Cmp = CmpInst::Create(CmpInst::ICmp, CmpInst::ICMP_SLT, Args[0], Zero,
                              "cast", CI->getIterator());

  if (!isa<VectorType>(ArgTy)) {
    auto *Cast = CastInst::CreateZExtOrBitCast(
        Cmp, Type::getInt32Ty(*Ctx), "", Cmp->getNextNode()->getIterator());
    CI->replaceAllUsesWith(Cast);
    CI->eraseFromParent();
  } else {
    mutateCallInst(CI, OC).setArgs({Cmp}).changeReturnType(
        Type::getInt32Ty(*Ctx), [](IRBuilder<> &Builder, CallInst *CI) {
          return Builder.CreateZExtOrBitCast(CI, Builder.getInt32Ty());
        });
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
  mutateCallInst(CI, OpMemoryBarrier)
      .setArgs({addInt32(map<Scope>(std::get<2>(Lit))),
                addInt32(mapOCLMemSemanticToSPIRV(std::get<0>(Lit),
                                                  std::get<1>(Lit)))});
}

void OCLToSPIRVBase::visitCallAtomicLegacy(CallInst *CI, StringRef MangledName,
                                           StringRef DemangledName) {
  StringRef Stem = DemangledName;
  if (Stem.starts_with("atom_"))
    Stem = Stem.drop_front(strlen("atom_"));
  else if (Stem.starts_with("atomic_"))
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
  if (Stem.starts_with("compare_exchange"))
    PostOps.push_back(OCLLegacyAtomicMemOrder);
  PostOps.push_back(OCLLegacyAtomicMemScope);

  Info.PostProc = [=](BuiltinCallMutator &Mutator) {
    for (auto &I : PostOps) {
      Mutator.appendArg(addInt32(I));
    }
  };
  transAtomicBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallAtomicCpp11(CallInst *CI, StringRef MangledName,
                                          StringRef DemangledName) {
  StringRef Stem = DemangledName;
  if (Stem.starts_with("atomic_"))
    Stem = Stem.drop_front(strlen("atomic_"));
  else
    return;

  std::string NewStem(Stem);
  std::vector<int> PostOps;
  if (Stem.starts_with("store") || Stem.starts_with("load") ||
      Stem.starts_with("exchange") || Stem.starts_with("compare_exchange") ||
      Stem.starts_with("fetch") || Stem.starts_with("flag")) {
    if ((Stem.starts_with("fetch_min") || Stem.starts_with("fetch_max")) &&
        containsUnsignedAtomicType(MangledName))
      NewStem.insert(NewStem.begin() + strlen("fetch_"), 'u');

    if (!Stem.ends_with("_explicit")) {
      NewStem = NewStem + "_explicit";
      PostOps.push_back(OCLMO_seq_cst);
      if (Stem.starts_with("compare_exchange"))
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
  Info.PostProc = [=](BuiltinCallMutator &Mutator) {
    for (auto &I : PostOps) {
      Mutator.appendArg(addInt32(I));
    }
  };

  transAtomicBuiltin(CI, Info);
}

void OCLToSPIRVBase::transAtomicBuiltin(CallInst *CI,
                                        OCLBuiltinTransInfo &Info) {
  llvm::Type *AtomicBuiltinsReturnType = CI->getType();
  auto SPIRVFunctionName =
      getSPIRVFuncName(OCLSPIRVBuiltinMap::map(Info.UniqName));
  bool NeedsNegate = false;
  if (AtomicBuiltinsReturnType->isFloatingPointTy()) {
    // Translate FP-typed atomic builtins. Currently we only need to
    // translate atomic_fetch_[add, sub, max, min] and atomic_fetch_[add,
    // sub, max, min]_explicit to related float instructions.
    // Translate atomic_fetch_sub to OpAtomicFAddEXT with negative value
    // operand
    auto SPIRFunctionNameForFloatAtomics =
        llvm::StringSwitch<std::string>(SPIRVFunctionName)
            .Case("__spirv_AtomicIAdd", "__spirv_AtomicFAddEXT")
            .Case("__spirv_AtomicISub", "__spirv_AtomicFAddEXT")
            .Case("__spirv_AtomicSMax", "__spirv_AtomicFMaxEXT")
            .Case("__spirv_AtomicSMin", "__spirv_AtomicFMinEXT")
            .Default("others");
    if (SPIRVFunctionName == "__spirv_AtomicISub") {
      NeedsNegate = true;
    }
    if (SPIRFunctionNameForFloatAtomics != "others")
      SPIRVFunctionName = SPIRFunctionNameForFloatAtomics;
  }

  auto Mutator = mutateCallInst(CI, SPIRVFunctionName);
  Info.PostProc(Mutator);
  // Order of args in OCL20:
  // object, 0-2 other args, 1-2 order, scope
  const size_t NumOrder = getAtomicBuiltinNumMemoryOrderArgs(Info.UniqName);
  const size_t ArgsCount = Mutator.arg_size();
  const size_t ScopeIdx = ArgsCount - 1;
  const size_t OrderIdx = ScopeIdx - NumOrder;

  if (NeedsNegate) {
    Mutator.mapArg(1, [=](Value *V) {
      IRBuilder<> IRB(CI);
      return IRB.CreateFNeg(V);
    });
  }
  Mutator.mapArg(ScopeIdx, [=](Value *V) {
    return transOCLMemScopeIntoSPIRVScope(V, OCLMS_device, CI);
  });
  for (size_t I = 0; I < NumOrder; ++I) {
    Mutator.mapArg(OrderIdx + I, [=](Value *V) {
      return transOCLMemOrderIntoSPIRVMemorySemantics(V, OCLMO_seq_cst, CI);
    });
  }

  // Order of args in SPIR-V:
  // object, scope, 1-2 order, 0-2 other args
  for (size_t I = 0; I < NumOrder; ++I) {
    Mutator.moveArg(OrderIdx + I, I + 1);
  }
  Mutator.moveArg(ScopeIdx, 1);
  if (Info.UniqName.find("atomic_compare_exchange") == 0) {
    // For atomic_compare_exchange, the two "other args" are in the opposite
    // order from the SPIR-V order. Swap these two arguments.
    Mutator.moveArg(Mutator.arg_size() - 1, Mutator.arg_size() - 2);
  }
}

void OCLToSPIRVBase::visitCallBarrier(CallInst *CI) {
  auto Lit = getBarrierLiterals(CI);
  // Use sequential consistent memory order by default.
  // But if the flags argument is set to 0, we use
  // None(Relaxed) memory order.
  unsigned MemFenceFlag = std::get<0>(Lit);
  OCLMemOrderKind MemOrder = MemFenceFlag ? OCLMO_seq_cst : OCLMO_relaxed;
  mutateCallInst(CI, OpControlBarrier)
      .setArgs({// Execution scope
                addInt32(map<Scope>(std::get<2>(Lit))),
                // Memory scope
                addInt32(map<Scope>(std::get<1>(Lit))),
                // Memory semantics
                addInt32(mapOCLMemSemanticToSPIRV(MemFenceFlag, MemOrder))});
}

void OCLToSPIRVBase::visitCallConvert(CallInst *CI, StringRef MangledName,
                                      StringRef DemangledName) {
  // OpenCL Explicit Conversions (6.4.3) formed as below for scalars:
  // destType convert_destType<_sat><_roundingMode>(sourceType)
  // and for vector type:
  // destTypeN convert_destTypeN<_sat><_roundingMode>(sourceTypeN)
  // If the demangled name is not matching the suggested pattern and does not
  // meet allowed destination type restrictions - this is not an OpenCL builtin,
  // return from the function and translate such CallInst as a function call.
  if (eraseUselessConvert(CI, MangledName, DemangledName))
    return;
  Op OC = OpNop;
  auto *TargetTy = CI->getType();
  auto *SrcTy = CI->getArgOperand(0)->getType();
  if (auto *VecTy = dyn_cast<VectorType>(TargetTy))
    TargetTy = VecTy->getElementType();
  if (auto *VecTy = dyn_cast<VectorType>(SrcTy))
    SrcTy = VecTy->getElementType();
  auto IsTargetInt = isa<IntegerType>(TargetTy);

  // Validate conversion function name and vector size if present
  std::regex Expr(
      "convert_(float|double|half|u?char|u?short|u?int|u?long)(2|3|4|8|16)*"
      "(_sat)*(_rt[ezpn])*$");
  std::smatch DestTyMatch;
  std::string ConversionFunc(DemangledName.str());
  if (!std::regex_match(ConversionFunc, DestTyMatch, Expr))
    return;

  // The first sub_match is the whole string; the next
  // sub_matches are the parenthesized expressions.
  enum { TypeIdx = 1, VecSizeIdx = 2, SatIdx = 3, RoundingIdx = 4 };
  std::string DestTy = DestTyMatch[TypeIdx].str();
  std::string VecSize = DestTyMatch[VecSizeIdx].str();
  std::string Sat = DestTyMatch[SatIdx].str();
  std::string Rounding = DestTyMatch[RoundingIdx].str();

  bool TargetSigned = DestTy[0] != 'u';

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

  if (!Rounding.empty() && (isa<IntegerType>(SrcTy) && IsTargetInt))
    return;

  assert(CI->getCalledFunction() && "Unexpected indirect call");
  mutateCallInst(
      CI, getSPIRVFuncName(OC, "_R" + DestTy + VecSize + Sat + Rounding));
}

void OCLToSPIRVBase::visitCallGroupBuiltin(CallInst *CI,
                                           StringRef OrigDemangledName) {
  auto *F = CI->getCalledFunction();
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
          if (!FuncName.starts_with(S))
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
                                      3));    // assumes op is three characters
          (void)(GroupOp.consume_front("_")); // when op is two characters
          assert(!GroupOp.empty() && "Invalid OpenCL group builtin function");
          char OpTyC = 0;
          auto *OpTy = F->getReturnType();
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
  Info.PostProc = [=](BuiltinCallMutator &Mutator) {
    if (HasBoolArg) {
      Mutator.mapArg(0, [&](Value *V) {
        IRBuilder<> IRB(CI);
        return IRB.CreateICmpNE(V, IRB.getInt32(0));
      });
    }
    size_t E = Mutator.arg_size();
    if (DemangledName == "group_broadcast" && E > 2) {
      assert(E == 3 || E == 4);
      std::vector<Value *> Ops = getArguments(CI);
      makeVector(CI, Ops, std::make_pair(Ops.begin() + 1, Ops.end()));
      while (Mutator.arg_size() > 1)
        Mutator.removeArg(1);
      Mutator.appendArg(Ops.back());
    }
    for (unsigned I = 0; I < Consts.size(); I++)
      Mutator.insertArg(I, Consts[I]);
  };
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::transBuiltin(CallInst *CI, OCLBuiltinTransInfo &Info) {
  Op OC = OpNop;
  unsigned ExtOp = ~0U;
  SPIRVBuiltinVariableKind BVKind = BuiltInMax;
  if (StringRef(Info.UniqName).starts_with(kSPIRVName::Prefix))
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
  BuiltinCallMutator Mutator = mutateCallInst(CI, Info.UniqName + Info.Postfix);
  Info.PostProc(Mutator);
  if (Info.RetTy) {
    Type *OldRetTy = CI->getType();
    Mutator.changeReturnType(
        Info.RetTy, [OldRetTy, &Info](IRBuilder<> &Builder, CallInst *NewCI) {
          if (Info.RetTy->isIntegerTy() && OldRetTy->isIntegerTy()) {
            return Builder.CreateIntCast(NewCI, OldRetTy, false);
          }
          return Builder.CreatePointerBitCastOrAddrSpaceCast(NewCI, OldRetTy);
        });
  }
}

void OCLToSPIRVBase::visitCallReadImageMSAA(CallInst *CI,
                                            StringRef MangledName) {
  assert(MangledName.find("msaa") != StringRef::npos);
  mutateCallInst(
      CI, getSPIRVFuncName(OpImageRead, std::string(kSPIRVPostfix::ExtDivider) +
                                            getPostfixForReturnType(CI)))
      .insertArg(2, getInt32(M, ImageOperandsSampleMask));
}

void OCLToSPIRVBase::visitCallReadImageWithSampler(CallInst *CI,
                                                   StringRef MangledName,
                                                   StringRef DemangledName) {
  assert(MangledName.find(kMangledName::Sampler) != StringRef::npos);
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  Function *Func = CI->getCalledFunction();
  bool IsRetScalar = !CI->getType()->isVectorTy();
  Type *Ret = CI->getType();
  auto *ImageTy = OCLTypeToSPIRVPtr->getAdaptedArgumentType(Func, 0);
  if (!ImageTy)
    ImageTy = getCallValueType(CI, 0);

  auto Mutator = mutateCallInst(
      CI, getSPIRVFuncName(OpImageSampleExplicitLod,
                           std::string(kSPIRVPostfix::ExtDivider) +
                               getPostfixForReturnType(Ret)));
  Mutator.mapArg(0, [&](IRBuilder<> &Builder, Value *ImgArg, Type *ImgType) {
    auto *SampledImgTy = adjustImageType(ImageTy, kSPIRVTypeName::Image,
                                         kSPIRVTypeName::SampledImg);
    Value *SampledImgArgs[] = {CI->getArgOperand(0), CI->getArgOperand(1)};
    return addSPIRVCallPair(Builder, OpSampledImage, SampledImgTy,
                            SampledImgArgs, {ImgType, Mutator.getType(1)},
                            kSPIRVName::TempSampledImage);
  });
  Mutator.removeArg(1);
  unsigned ImgOpMask = getImageSignZeroExt(DemangledName);
  unsigned ImgOpMaskInsIndex = Mutator.arg_size();
  switch (Mutator.arg_size()) {
  case 2: // no lod
    ImgOpMask |= ImageOperandsMask::ImageOperandsLodMask;
    ImgOpMaskInsIndex = Mutator.arg_size();
    Mutator.appendArg(getFloat32(M, 0.f));
    break;
  case 3: // explicit lod
    ImgOpMask |= ImageOperandsMask::ImageOperandsLodMask;
    ImgOpMaskInsIndex = 2;
    break;
  case 4: // gradient
    ImgOpMask |= ImageOperandsMask::ImageOperandsGradMask;
    ImgOpMaskInsIndex = 2;
    break;
  default:
    assert(0 && "read_image* with unhandled number of args!");
  }
  Mutator.insertArg(ImgOpMaskInsIndex, getInt32(M, ImgOpMask));

  // SPIR-V instruction always returns 4-element vector
  if (IsRetScalar)
    Mutator.changeReturnType(FixedVectorType::get(Ret, 4),
                             [=](IRBuilder<> &Builder, CallInst *NewCI) {
                               return Builder.CreateExtractElement(
                                   NewCI, getSizet(M, 0));
                             });
}

void OCLToSPIRVBase::visitCallGetImageSize(CallInst *CI,
                                           StringRef DemangledName) {
  auto Desc = getImageDescriptor(getCallValueType(CI, 0));
  unsigned Dim = getImageDimension(Desc.Dim) + Desc.Arrayed;
  assert(Dim > 0 && "Invalid image dimension.");
  assert(CI->arg_size() == 1);
  Type *NewRet = CI->getType()->isIntegerTy(64) ? Type::getInt64Ty(*Ctx)
                                                : Type::getInt32Ty(*Ctx);
  if (Dim > 1)
    NewRet = FixedVectorType::get(NewRet, Dim);
  auto Mutator = mutateCallInst(CI, getSPIRVFuncName(Desc.Dim == DimBuffer
                                                         ? OpImageQuerySize
                                                         : OpImageQuerySizeLod,
                                                     CI->getType()));
  if (Desc.Dim != DimBuffer)
    Mutator.appendArg(getInt32(M, 0));
  Mutator.changeReturnType(
      NewRet, [&](IRBuilder<> &, CallInst *NCI) -> Value * {
        if (Dim == 1)
          return NCI;
        if (DemangledName == kOCLBuiltinName::GetImageDim) {
          if (Desc.Dim == Dim3D) {
            auto *ZeroVec = ConstantVector::getSplat(
                ElementCount::getFixed(3),
                Constant::getNullValue(
                    cast<VectorType>(NCI->getType())->getElementType()));
            Constant *Index[] = {getInt32(M, 0), getInt32(M, 1), getInt32(M, 2),
                                 getInt32(M, 3)};
            return new ShuffleVectorInst(NCI, ZeroVec,
                                         ConstantVector::get(Index), "",
                                         CI->getIterator());

          } else if (Desc.Dim == Dim2D && Desc.Arrayed) {
            Constant *Index[] = {getInt32(M, 0), getInt32(M, 1)};
            Constant *Mask = ConstantVector::get(Index);
            return new ShuffleVectorInst(NCI, UndefValue::get(NCI->getType()),
                                         Mask, NCI->getName(),
                                         CI->getIterator());
          }
          return NCI;
        }
        unsigned I = StringSwitch<unsigned>(DemangledName)
                         .Case(kOCLBuiltinName::GetImageWidth, 0)
                         .Case(kOCLBuiltinName::GetImageHeight, 1)
                         .Case(kOCLBuiltinName::GetImageDepth, 2)
                         .Case(kOCLBuiltinName::GetImageArraySize, Dim - 1);
        return ExtractElementInst::Create(NCI, getUInt32(M, I), "",
                                          NCI->getNextNode()->getIterator());
      });
}

/// Remove trivial conversion functions
bool OCLToSPIRVBase::eraseUselessConvert(CallInst *CI, StringRef MangledName,
                                         StringRef DemangledName) {
  auto *TargetTy = CI->getType();
  auto *SrcTy = CI->getArgOperand(0)->getType();
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
  if (DemangledName.find(kOCLBuiltinName::ReadImage) == 0) {
    Info.UniqName = kOCLBuiltinName::ReadImage;
    unsigned ImgOpMask = getImageSignZeroExt(DemangledName);
    if (ImgOpMask) {
      Module *Mod = M;
      Info.PostProc = [ImgOpMask, Mod](BuiltinCallMutator &Mutator) {
        Mutator.appendArg(getInt32(Mod, ImgOpMask));
      };
    }
  }

  if (DemangledName.find(kOCLBuiltinName::WriteImage) == 0) {
    Info.UniqName = kOCLBuiltinName::WriteImage;
    Info.PostProc = [&](BuiltinCallMutator &Mutator) {
      unsigned ImgOpMask = getImageSignZeroExt(DemangledName);
      unsigned ImgOpMaskInsIndex = Mutator.arg_size();
      if (Mutator.arg_size() == 4) // write with lod
      {
        ImgOpMask |= ImageOperandsMask::ImageOperandsLodMask;
        ImgOpMaskInsIndex = Mutator.arg_size() - 1;
        Mutator.moveArg(2, Mutator.arg_size() - 1);
      }
      if (ImgOpMask) {
        Mutator.insertArg(ImgOpMaskInsIndex, getInt32(M, ImgOpMask));
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
  auto *StorageClass = addInt32(SPIRSPIRVAddrSpaceMap::map(AddrSpace));
  Info.RetTy = getInt8PtrTy(cast<PointerType>(CI->getType()));
  Info.PostProc = [=](BuiltinCallMutator &Mutator) {
    Mutator
        .mapArg(Mutator.arg_size() - 1,
                [&](Value *V) {
                  return std::make_pair(
                      castToInt8Ptr(V, CI),
                      TypedPointerType::get(Type::getInt8Ty(V->getContext()),
                                            SPIRAS_Generic));
                })
        .appendArg(StorageClass);
  };
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallRelational(CallInst *CI,
                                         StringRef DemangledName) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  Op OC = OpNop;
  OCLSPIRVBuiltinMap::find(DemangledName.str(), &OC);
  // i1 or <i1 x N>, depending on whether it returns a vector type.
  Type *BoolTy = CI->getType()->getWithNewType(Type::getInt1Ty(*Ctx));
  mutateCallInst(CI, OC).changeReturnType(
      BoolTy, [=](IRBuilder<> &Builder, CallInst *NewCI) {
        Value *TrueOp = CI->getType()->isVectorTy()
                            ? Constant::getAllOnesValue(CI->getType())
                            : getInt32(M, 1);
        return Builder.CreateSelect(NewCI, TrueOp,
                                    Constant::getNullValue(CI->getType()));
      });
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
  Info.PostProc = [=](BuiltinCallMutator &Mutator) {
    for (auto *Value : Consts)
      Mutator.appendArg(Value);
  };
  transBuiltin(CI, Info);
}

void OCLToSPIRVBase::visitCallGetFence(CallInst *CI, StringRef DemangledName) {
  Op OC = OpNop;
  OCLSPIRVBuiltinMap::find(DemangledName.str(), &OC);
  mutateCallInst(CI, OC).changeReturnType(
      CI->getType(), [](IRBuilder<> &Builder, CallInst *NewCI) {
        return Builder.CreateLShr(NewCI, Builder.getInt32(8));
      });
}

void OCLToSPIRVBase::visitCallDot(CallInst *CI) {
  IRBuilder<> Builder(CI);
  Value *FMulVal = Builder.CreateFMul(CI->getOperand(0), CI->getOperand(1));
  CI->replaceAllUsesWith(FMulVal);
  CI->eraseFromParent();
}

void OCLToSPIRVBase::visitCallDot(CallInst *CI, StringRef MangledName,
                                  StringRef DemangledName) {
  // translation for dot function calls,
  // to differentiate between integer dot products

  bool IsFirstSigned, IsSecondSigned;
  bool IsDot = DemangledName == kOCLBuiltinName::Dot;
  bool IsAccSat = DemangledName.contains(kOCLBuiltinName::DotAccSat);
  bool IsPacked = CI->getOperand(0)->getType()->isIntegerTy();
  if (!IsPacked) {
    if (IsDot) {
      // dot(char4, char4) _Z3dotDv4_cS_
      // dot(char4, uchar4) _Z3dotDv4_cDv4_h
      // dot(uchar4, char4) _Z3dotDv4_hDv4_c
      // dot(uchar4, uchar4) _Z3dotDv4_hS_
      // or
      // dot(short2, short2) _Z3dotDv2_sS_
      // dot(short2, ushort2) _Z3dotDv2_sDv2_t
      // dot(ushort2, short2) _Z3dotDv2_tDv2_s
      // dot(ushort2, ushort2) _Z3dotDv2_tS_
      assert(MangledName.starts_with("_Z3dotDv"));
      if (MangledName[MangledName.size() - 1] == '_') {
        IsFirstSigned = ((MangledName[MangledName.size() - 3] == 'c') ||
                         (MangledName[MangledName.size() - 3] == 's'));
        IsSecondSigned = IsFirstSigned;
      } else {
        IsFirstSigned = ((MangledName[MangledName.size() - 6] == 'c') ||
                         (MangledName[MangledName.size() - 6] == 's'));
        IsSecondSigned = ((MangledName[MangledName.size() - 1] == 'c') ||
                          (MangledName[MangledName.size() - 1] == 's'));
      }
    } else {
      // dot_acc_sat(char4, char4, int) _Z11dot_acc_satDv4_cS_i
      // dot_acc_sat(char4, uchar4, int) _Z11dot_acc_satDv4_cDv4_hi
      // dot_acc_sat(uchar4, char4, int) _Z11dot_acc_satDv4_hDv4_ci
      // dot_acc_sat(uchar4, uchar4, uint) _Z11dot_acc_satDv4_hS_j
      // or
      // dot_acc_sat(short2, short2, int) _Z11dot_acc_satDv4_sS_i
      // dot_acc_sat(short2, ushort2, int) _Z11dot_acc_satDv4_sDv4_ti
      // dot_acc_sat(ushort2, short2, int) _Z11dot_acc_satDv4_tDv4_si
      // dot_acc_sat(ushort2, ushort2, uint) _Z11dot_acc_satDv4_tS_j
      assert(MangledName.starts_with("_Z11dot_acc_satDv"));
      IsFirstSigned = ((MangledName[19] == 'c') || (MangledName[19] == 's'));
      IsSecondSigned = (MangledName[20] == 'S'
                            ? IsFirstSigned
                            : ((MangledName[MangledName.size() - 2] == 'c') ||
                               (MangledName[MangledName.size() - 2] == 's')));
    }
  } else {
    // for packed format
    // dot_4x8packed_ss_int(uint, uint) _Z20dot_4x8packed_ss_intjj
    // dot_4x8packed_su_int(uint, uint) _Z20dot_4x8packed_su_intjj
    // dot_4x8packed_us_int(uint, uint) _Z20dot_4x8packed_us_intjj
    // dot_4x8packed_uu_uint(uint, uint) _Z21dot_4x8packed_uu_uintjj
    // or
    // dot_acc_sat_4x8packed_ss_int(uint, uint, int)
    // _Z28dot_acc_sat_4x8packed_ss_intjji
    // dot_acc_sat_4x8packed_su_int(uint, uint, int)
    // _Z28dot_acc_sat_4x8packed_su_intjji
    // dot_acc_sat_4x8packed_us_int(uint, uint, int)
    // _Z28dot_acc_sat_4x8packed_us_intjji
    // dot_acc_sat_4x8packed_uu_uint(uint, uint, uint)
    // _Z29dot_acc_sat_4x8packed_uu_uintjjj
    assert(MangledName.starts_with("_Z20dot_4x8packed") ||
           MangledName.starts_with("_Z21dot_4x8packed") ||
           MangledName.starts_with("_Z28dot_acc_sat_4x8packed") ||
           MangledName.starts_with("_Z29dot_acc_sat_4x8packed"));
    size_t SignIndex = IsAccSat
                           ? strlen(kOCLBuiltinName::DotAccSat4x8PackedPrefix)
                           : strlen(kOCLBuiltinName::Dot4x8PackedPrefix);
    IsFirstSigned = DemangledName[SignIndex] == 's';
    IsSecondSigned = DemangledName[SignIndex + 1] == 's';
  }
  Op OC;
  if (!IsAccSat) {
    OC =
        (IsFirstSigned != IsSecondSigned ? OpSUDot
                                         : ((IsFirstSigned) ? OpSDot : OpUDot));
  } else {
    OC = (IsFirstSigned != IsSecondSigned
              ? OpSUDotAccSat
              : ((IsFirstSigned) ? OpSDotAccSat : OpUDotAccSat));
  }
  auto Mutator = mutateCallInst(CI, OC);
  // If arguments are in order unsigned -> signed
  // then the translator should swap them,
  // so that the OpSUDotKHR can be used properly
  if (IsFirstSigned == false && IsSecondSigned == true) {
    Mutator.moveArg(1, 0);
  }
  if (IsPacked) {
    // As per SPIRV specification the dot OpCodes
    // which use scalar integers to represent
    // packed vectors need additional argument
    // specified - the Packed Vector Format
    Mutator.appendArg(
        getInt32(M, PackedVectorFormatPackedVectorFormat4x8BitKHR));
  }
}

void OCLToSPIRVBase::visitCallClockRead(CallInst *CI, StringRef MangledName,
                                        StringRef DemangledName) {
  // The builtin returns i64 or <2 x i32>, but both variants are mapped to the
  // same instruction; hence include the return type.
  std::string OpName = getSPIRVFuncName(OpReadClockKHR, CI->getType());

  // Scope is part of the OpenCL builtin name.
  Scope ScopeArg = StringSwitch<Scope>(DemangledName)
                       .EndsWith("device", ScopeDevice)
                       .EndsWith("work_group", ScopeWorkgroup)
                       .EndsWith("sub_group", ScopeSubgroup)
                       .Default(ScopeMax);

  auto Mutator = mutateCallInst(CI, OpName);
  Mutator.appendArg(getInt32(M, ScopeArg));
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

  assert(CI->arg_size() == VecPos.size() + ScalarPos.size() &&
         "Argument counts do not match up.");

  Type *VecTy = CI->getOperand(VecPos[0])->getType();
  auto VecElemCount = cast<VectorType>(VecTy)->getElementCount();
  auto Mutator = mutateCallInst(
      CI, getSPIRVExtFuncName(SPIRVEIS_OpenCL,
                              getExtOp(MangledName, DemangledName)));
  for (auto I : ScalarPos)
    Mutator.mapArg(I, [&](Value *V) {
      Instruction *Inst = InsertElementInst::Create(
          UndefValue::get(VecTy), V, getInt32(M, 0), "", CI->getIterator());
      return new ShuffleVectorInst(
          Inst, UndefValue::get(VecTy),
          ConstantVector::getSplat(VecElemCount, getInt32(M, 0)), "",
          CI->getIterator());
    });
}

namespace {
// Return true if any users of the CallInst use any of the constants
// introduced by the SPV_EXT_image_raw10_raw12 extension.
bool usesSpvExtImageRaw10Raw12Constants(const CallInst *CI) {
  const std::array ExtConstants{
      OCLImageChannelDataTypeOffset + ImageChannelDataTypeUnsignedIntRaw10EXT,
      OCLImageChannelDataTypeOffset + ImageChannelDataTypeUnsignedIntRaw12EXT};

  // The return values for `OpImageQueryFormat` added by the extension are
  // integer constants that may appear anywhere in LLVM IR.  Only detect some
  // common use patterns here.
  for (auto *U : CI->users()) {
    for (auto C : ExtConstants) {
      ICmpInst::Predicate Pred;
      if (match(U, m_c_ICmp(Pred, m_Value(), m_SpecificInt(C)))) {
        return true;
      }
      if (auto *Switch = dyn_cast<SwitchInst>(U)) {
        if (any_of(Switch->cases(), [C](const auto &Case) {
              return Case.getCaseValue()->equalsInt(C);
            })) {
          return true;
        }
      }
    }
  }
  return false;
}
} // anonymous namespace

void OCLToSPIRVBase::visitCallGetImageChannel(CallInst *CI,
                                              StringRef DemangledName,
                                              unsigned int Offset) {
  assert(CI->getCalledFunction() && "Unexpected indirect call");

  if (Offset == OCLImageChannelDataTypeOffset) {
    // See if any of the SPV_EXT_image_raw10_raw12 constants are used, and
    // add the extension if not already there.
    if (usesSpvExtImageRaw10Raw12Constants(CI)) {
      const char *ExtStr = "SPV_EXT_image_raw10_raw12";
      NamedMDNode *NMD = M->getOrInsertNamedMetadata(kSPIRVMD::Extension);
      if (none_of(NMD->operands(), [ExtStr](MDNode *N) {
            return N->getOperand(0).equalsStr(ExtStr);
          })) {
        MDString *MDS = MDString::get(*Ctx, ExtStr);
        NMD->addOperand(MDNode::get(*Ctx, MDS));
      }
    }
  }

  Op OC = OpNop;
  OCLSPIRVBuiltinMap::find(DemangledName.str(), &OC);
  mutateCallInst(CI, OC).changeReturnType(
      CI->getType(), [=](IRBuilder<> &Builder, CallInst *NewCI) {
        return Builder.CreateAdd(NewCI, Builder.getInt32(Offset));
      });
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
    Args.push_back(getInt32(M, 0)); // dummy num events
    Value *Null = Constant::getNullValue(PointerType::get(
        getSPIRVType(OpTypeDeviceEvent, true), SPIRAS_Generic));
    Args.push_back(Null); // dummy wait events
    Args.push_back(Null); // dummy ret event
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
  Type *ParamType = getBlockStructType(BlockLiteral);
  Args.push_back(getInt32(M, DL.getTypeStoreSize(ParamType)));
  Args.push_back(getInt32(M, DL.getPrefTypeAlign(ParamType).value()));

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
          "", CI->getIterator()));
  }

  StringRef NewName = "__spirv_EnqueueKernel__";
  FunctionType *FT = FunctionType::get(
      CI->getType(), getTypes(ArrayRef<Value *>(Args)), false /*isVarArg*/);
  Function *NewF =
      Function::Create(FT, GlobalValue::ExternalLinkage, NewName, M);
  NewF->setCallingConv(CallingConv::SPIR_FUNC);
  CallInst *NewCall = CallInst::Create(NewF, Args, "", CI->getIterator());
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
  ::mutateCallInst(
      M, CI,
      [=](CallInst *CI, std::vector<Value *> &Args) {
        Value *Param = *Args.rbegin();
        Type *ParamType = getBlockStructType(Param);
        // Last arg corresponds to SPIRV Param operand.
        // Insert Invoke in front of Param.
        // Add Param Size and Param Align at the end.
        Args[BlockFIdx] = BlockF;
        Args.push_back(getInt32(M, DL.getTypeStoreSize(ParamType)));
        Args.push_back(getInt32(M, DL.getPrefTypeAlign(ParamType).value()));

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
void OCLToSPIRVBase::processSubgroupBlockReadWriteINTEL(
    CallInst *CI, OCLBuiltinTransInfo &Info, const Type *DataTy) {
  unsigned VectorNumElements = 1;
  if (auto *VecTy = dyn_cast<FixedVectorType>(DataTy))
    VectorNumElements = VecTy->getNumElements();
  unsigned ElementBitSize = DataTy->getScalarSizeInBits();
  Info.Postfix = "_";
  Info.Postfix +=
      getIntelSubgroupBlockDataPostfix(ElementBitSize, VectorNumElements);
  assert(CI->getCalledFunction() && "Unexpected indirect call");
  mutateCallInst(CI, Info.UniqName + Info.Postfix);
}

// The intel_sub_group_block_read built-ins are overloaded to support both
// buffers and images, but need to be mapped to distinct SPIR-V instructions.
// Additionally, for block reads, need to distinguish between scalar block
// reads and vector block reads.
void OCLToSPIRVBase::visitSubgroupBlockReadINTEL(CallInst *CI) {
  OCLBuiltinTransInfo Info;
  if (isOCLImageType(getCallValueType(CI, 0)))
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupImageBlockReadINTEL);
  else
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupBlockReadINTEL);
  Type *DataTy = CI->getType();
  processSubgroupBlockReadWriteINTEL(CI, Info, DataTy);
}

// The intel_sub_group_block_write built-ins are similarly overloaded to support
// both buffers and images but need to be mapped to distinct SPIR-V
// instructions.
void OCLToSPIRVBase::visitSubgroupBlockWriteINTEL(CallInst *CI) {
  OCLBuiltinTransInfo Info;
  if (isOCLImageType(getCallValueType(CI, 0)))
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupImageBlockWriteINTEL);
  else
    Info.UniqName = getSPIRVFuncName(spv::OpSubgroupBlockWriteINTEL);
  assert(!CI->arg_empty() &&
         "Intel subgroup block write should have arguments");
  unsigned DataArg = CI->arg_size() - 1;
  Type *DataTy = CI->getArgOperand(DataArg)->getType();
  processSubgroupBlockReadWriteINTEL(CI, Info, DataTy);
}

void OCLToSPIRVBase::visitSubgroupImageMediaBlockINTEL(
    CallInst *CI, StringRef DemangledName) {
  spv::Op OpCode = DemangledName.rfind("read") != StringRef::npos
                       ? spv::OpSubgroupImageMediaBlockReadINTEL
                       : spv::OpSubgroupImageMediaBlockWriteINTEL;
  // Move the last argument to the beginning.
  mutateCallInst(CI, getSPIRVFuncName(OpCode, CI->getType()))
      .moveArg(CI->arg_size() - 1, 0);
}

static const char *getSubgroupAVCIntelOpKind(StringRef Name) {
  return StringSwitch<const char *>(Name.data())
      .StartsWith(kOCLSubgroupsAVCIntel::IMEPrefix, "ime")
      .StartsWith(kOCLSubgroupsAVCIntel::REFPrefix, "ref")
      .StartsWith(kOCLSubgroupsAVCIntel::SICPrefix, "sic");
}

static const char *getSubgroupAVCIntelTyKind(StringRef MangledName) {
  // We're looking for the type name of the last parameter, which will be at the
  // very end of the mangled name. Since we only care about the ending of the
  // name, we don't need to be any more clever than this.
  return MangledName.ends_with("_payload_t") ? "payload" : "result";
}

static Type *getSubgroupAVCIntelMCEType(Module *M, std::string &TName) {
  auto *Ty = StructType::getTypeByName(M->getContext(), TName);
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
    // _single_reference functions have 2 arguments, _dual_reference have 3
    // arguments.
    FName += (CI->arg_size() == 2) ? "_single_reference" : "_dual_reference";
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

  mutateCallInst(CI, OC);
}

// Handles Subgroup AVC Intel extension wrapper built-ins.
// 'IME', 'REF' and 'SIC' sets contain wrapper built-ins which don't have
// corresponded instructions in SPIRV and should be translated to a
// conterpart from 'MCE' with conversion for an argument and result (if needed).
void OCLToSPIRVBase::visitSubgroupAVCWrapperBuiltinCall(
    CallInst *CI, Op WrappedOC, StringRef DemangledName) {
  std::string Prefix = kOCLSubgroupsAVCIntel::Prefix;

  // Find 'to_mce' conversion function.
  // The operand required conversion is always the last one.
  const char *OpKind = getSubgroupAVCIntelOpKind(DemangledName);
  const char *TyKind =
      getSubgroupAVCIntelTyKind(CI->getCalledFunction()->getName());
  std::string MCETName =
      std::string(kOCLSubgroupsAVCIntel::TypePrefix) + "mce_" + TyKind + "_t";
  auto *MCESTy = getSubgroupAVCIntelMCEType(M, MCETName);
  auto *MCETy = TypedPointerType::get(MCESTy, SPIRAS_Private);
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

    mutateCallInst(CI, WrappedOC)
        .mapArg(CI->arg_size() - 1,
                [&](IRBuilder<> &Builder, Value *Arg, Type *ParamTy) {
                  // Create conversion function call for the last operand
                  return addSPIRVCallPair(Builder, ToMCEOC, MCETy, {Arg},
                                          {ParamTy});
                })
        .changeReturnType(MCETy, [&](IRBuilder<> &Builder, CallInst *NewCI) {
          // Create conversion function call for the return result
          return addSPIRVCall(Builder, FromMCEOC, CI->getType(), {NewCI},
                              {MCETy});
        });
  } else {
    // Wrapper built-ins which take the 'result_t' argument requires only one
    // conversion for the argument
    mutateCallInst(CI, WrappedOC)
        .mapArg(CI->arg_size() - 1, [&](IRBuilder<> &Builder, Value *Arg,
                                        Type *ParamTy) {
          // Create conversion function call for the last operand
          return addSPIRVCallPair(Builder, ToMCEOC, MCETy, {Arg}, {ParamTy});
        });
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

  SmallVector<Type *, 4> ParamTys;
  [[maybe_unused]] bool DidDemangle =
      getParameterTypes(CI->getCalledFunction(), ParamTys);
  assert(DidDemangle && "Expected SPIR-V builtins to be properly mangled");
  auto *TyIt = std::find_if(ParamTys.begin(), ParamTys.end(), isSamplerTy);
  assert(TyIt != ParamTys.end() && "Invalid Subgroup AVC Intel built-in call");
  unsigned SamplerIndex = TyIt - ParamTys.begin();
  Value *SamplerVal = CI->getOperand(SamplerIndex);
  Type *SamplerTy = ParamTys[SamplerIndex];

  SmallVector<Type *, 4> AdaptedTys;
  for (unsigned I = 0; I < CI->arg_size(); I++)
    AdaptedTys.push_back(
        OCLTypeToSPIRVPtr->getAdaptedArgumentType(CI->getCalledFunction(), I));
  auto *AdaptedIter = AdaptedTys.begin();

  mutateCallInst(CI, OC)
      .mapArgs([&](IRBuilder<> &Builder, Value *Arg, Type *ArgTy) {
        if (!isOCLImageType(ArgTy))
          return BuiltinCallMutator::ValueTypePair(Arg, ArgTy);

        auto *ImageTy = *AdaptedIter++;
        if (!ImageTy)
          ImageTy = ArgTy;
        auto *SampledImgTy = adjustImageType(ImageTy, kSPIRVTypeName::Image,
                                             kSPIRVTypeName::VmeImageINTEL);

        Value *SampledImgArgs[] = {Arg, SamplerVal};
        return addSPIRVCallPair(Builder, OpVmeImageINTEL, SampledImgTy,
                                SampledImgArgs, {ArgTy, SamplerTy},
                                kSPIRVName::TempSampledImage);
      })
      .removeArg(SamplerIndex);
}

void OCLToSPIRVBase::visitCallSplitBarrierINTEL(CallInst *CI,
                                                StringRef DemangledName) {
  auto Lit = getBarrierLiterals(CI);
  Op OpCode =
      StringSwitch<Op>(DemangledName)
          .Case("intel_work_group_barrier_arrive", OpControlBarrierArriveINTEL)
          .Case("intel_work_group_barrier_wait", OpControlBarrierWaitINTEL)
          .Default(OpNop);

  // Map memory semantics as follows:
  // OpControlBarrierArriveINTEL -> Release,
  // OpControlBarrierWaitINTEL -> Acquire
  unsigned MemFenceFlag = std::get<0>(Lit);
  OCLMemOrderKind MemOrder =
      OpCode == OpControlBarrierArriveINTEL ? OCLMO_release : OCLMO_acquire;
  mutateCallInst(CI, OpCode)
      .removeArgs(0, CI->arg_size())
      // Execution scope
      .appendArg(addInt32(map<Scope>(std::get<2>(Lit))))
      // Memory scope
      .appendArg(addInt32(map<Scope>(std::get<1>(Lit))))
      // Memory semantics
      .appendArg(addInt32(mapOCLMemSemanticToSPIRV(MemFenceFlag, MemOrder)));
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

void OCLToSPIRVBase::visitCallConvertBFloat16AsUshort(CallInst *CI,
                                                      StringRef DemangledName) {
  Type *RetTy = CI->getType();
  Type *ArgTy = CI->getOperand(0)->getType();
  if (DemangledName == kOCLBuiltinName::ConvertBFloat16AsUShort) {
    if (!RetTy->isIntegerTy(16U) || !ArgTy->isFloatTy())
      report_fatal_error(
          "OpConvertBFloat16AsUShort must be of i16 and take float");
  } else {
    FixedVectorType *RetTyVec = cast<FixedVectorType>(RetTy);
    FixedVectorType *ArgTyVec = cast<FixedVectorType>(ArgTy);
    if (!RetTyVec || !RetTyVec->getElementType()->isIntegerTy(16U) ||
        !ArgTyVec || !ArgTyVec->getElementType()->isFloatTy())
      report_fatal_error("OpConvertBFloat16NAsUShortN must be of <N x i16> and "
                         "take <N x float>");
    unsigned RetTyVecSize = RetTyVec->getNumElements();
    unsigned ArgTyVecSize = ArgTyVec->getNumElements();
    if (DemangledName == kOCLBuiltinName::ConvertBFloat162AsUShort2) {
      if (RetTyVecSize != 2 || ArgTyVecSize != 2)
        report_fatal_error("ConvertBFloat162AsUShort2 must be of <2 x i16> and "
                           "take <2 x float>");
    } else if (DemangledName == kOCLBuiltinName::ConvertBFloat163AsUShort3) {
      if (RetTyVecSize != 3 || ArgTyVecSize != 3)
        report_fatal_error("ConvertBFloat163AsUShort3 must be of <3 x i16> and "
                           "take <3 x float>");
    } else if (DemangledName == kOCLBuiltinName::ConvertBFloat164AsUShort4) {
      if (RetTyVecSize != 4 || ArgTyVecSize != 4)
        report_fatal_error("ConvertBFloat164AsUShort4 must be of <4 x i16> and "
                           "take <4 x float>");
    } else if (DemangledName == kOCLBuiltinName::ConvertBFloat168AsUShort8) {
      if (RetTyVecSize != 8 || ArgTyVecSize != 8)
        report_fatal_error("ConvertBFloat168AsUShort8 must be of <8 x i16> and "
                           "take <8 x float>");
    } else if (DemangledName == kOCLBuiltinName::ConvertBFloat1616AsUShort16) {
      if (RetTyVecSize != 16 || ArgTyVecSize != 16)
        report_fatal_error("ConvertBFloat1616AsUShort16 must be of <16 x i16> "
                           "and take <16 x float>");
    }
  }

  mutateCallInst(CI, internal::OpConvertFToBF16INTEL);
}

void OCLToSPIRVBase::visitCallConvertAsBFloat16Float(CallInst *CI,
                                                     StringRef DemangledName) {
  Type *RetTy = CI->getType();
  Type *ArgTy = CI->getOperand(0)->getType();
  if (DemangledName == kOCLBuiltinName::ConvertAsBFloat16Float) {
    if (!RetTy->isFloatTy() || !ArgTy->isIntegerTy(16U))
      report_fatal_error(
          "OpConvertAsBFloat16Float must be of float and take i16");
  } else {
    FixedVectorType *RetTyVec = cast<FixedVectorType>(RetTy);
    FixedVectorType *ArgTyVec = cast<FixedVectorType>(ArgTy);
    if (!RetTyVec || !RetTyVec->getElementType()->isFloatTy() || !ArgTyVec ||
        !ArgTyVec->getElementType()->isIntegerTy(16U))
      report_fatal_error("OpConvertAsBFloat16NFloatN must be of <N x float> "
                         "and take <N x i16>");
    unsigned RetTyVecSize = RetTyVec->getNumElements();
    unsigned ArgTyVecSize = ArgTyVec->getNumElements();
    if (DemangledName == kOCLBuiltinName::ConvertAsBFloat162Float2) {
      if (RetTyVecSize != 2 || ArgTyVecSize != 2)
        report_fatal_error("ConvertAsBFloat162Float2 must be of <2 x float> "
                           "and take <2 x i16>");
    } else if (DemangledName == kOCLBuiltinName::ConvertAsBFloat163Float3) {
      if (RetTyVecSize != 3 || ArgTyVecSize != 3)
        report_fatal_error("ConvertAsBFloat163Float3 must be of <3 x float> "
                           "and take <3 x i16>");
    } else if (DemangledName == kOCLBuiltinName::ConvertAsBFloat164Float4) {
      if (RetTyVecSize != 4 || ArgTyVecSize != 4)
        report_fatal_error("ConvertAsBFloat164Float4 must be of <4 x float> "
                           "and take <4 x i16>");
    } else if (DemangledName == kOCLBuiltinName::ConvertAsBFloat168Float8) {
      if (RetTyVecSize != 8 || ArgTyVecSize != 8)
        report_fatal_error("ConvertAsBFloat168Float8 must be of <8 x float> "
                           "and take <8 x i16>");
    } else if (DemangledName == kOCLBuiltinName::ConvertAsBFloat1616Float16) {
      if (RetTyVecSize != 16 || ArgTyVecSize != 16)
        report_fatal_error("ConvertAsBFloat1616Float16 must be of <16 x float> "
                           "and take <16 x i16>");
    }
  }

  mutateCallInst(CI, internal::OpConvertBF16ToFINTEL);
}
} // namespace SPIRV

INITIALIZE_PASS_BEGIN(OCLToSPIRVLegacy, "ocl-to-spv",
                      "Transform OCL 2.0 to SPIR-V", false, false)
INITIALIZE_PASS_DEPENDENCY(OCLTypeToSPIRVLegacy)
INITIALIZE_PASS_END(OCLToSPIRVLegacy, "ocl-to-spv",
                    "Transform OCL 2.0 to SPIR-V", false, false)

ModulePass *llvm::createOCLToSPIRVLegacy() { return new OCLToSPIRVLegacy(); }
