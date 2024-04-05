//===- SPIRVBuiltinHelper.cpp - Helpers for managing calls to builtins ----===//
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
// This file implements helper functions for adding calls to OpenCL or SPIR-V
// builtin functions, or for rewriting calls to one into calls to the other.
//
//===----------------------------------------------------------------------===//

#include "SPIRVBuiltinHelper.h"

#include "OCLUtil.h"
#include "SPIRVInternal.h"

using namespace llvm;
using namespace SPIRV;

static std::unique_ptr<BuiltinFuncMangleInfo> makeMangler(CallBase *CB,
                                                          ManglingRules Rules) {
  switch (Rules) {
  case ManglingRules::None:
    return nullptr;
  case ManglingRules::SPIRV:
    return std::make_unique<BuiltinFuncMangleInfo>();
  case ManglingRules::OpenCL:
    return OCLUtil::makeMangler(*CB->getCalledFunction());
  }
  llvm_unreachable("Unknown mangling rules to make a name mangler");
}

BuiltinCallMutator::BuiltinCallMutator(
    CallInst *CI, std::string FuncName, ManglingRules Rules,
    std::function<std::string(StringRef)> NameMapFn)
    : CI(CI), FuncName(FuncName),
      Attrs(CI->getCalledFunction()->getAttributes()),
      CallAttrs(CI->getAttributes()), ReturnTy(CI->getType()), Args(CI->args()),
      Rules(Rules), Builder(CI) {
  bool DidDemangle = getParameterTypes(CI->getCalledFunction(), PointerTypes,
                                       std::move(NameMapFn));
  if (!DidDemangle) {
    // TODO: PipeBlocking.ll causes demangling failures.
    // assert(isNonMangledOCLBuiltin(CI->getCalledFunction()->getName()) &&
    //    "SPIR-V builtin functions should be mangled");
    for (Value *Arg : Args)
      PointerTypes.push_back(Arg->getType());
  }
}

BuiltinCallMutator::BuiltinCallMutator(BuiltinCallMutator &&Other)
    : CI(Other.CI), FuncName(std::move(Other.FuncName)),
      MutateRet(std::move(Other.MutateRet)), Attrs(Other.Attrs),
      CallAttrs(Other.CallAttrs), ReturnTy(Other.ReturnTy),
      Args(std::move(Other.Args)), PointerTypes(std::move(Other.PointerTypes)),
      Rules(std::move(Other.Rules)), Builder(CI) {
  // Clear the other's CI instance so that it knows not to construct the actual
  // call.
  Other.CI = nullptr;
}

Value *BuiltinCallMutator::doConversion() {
  assert(CI && "Need to have a call instruction to do the conversion");
  auto Mangler = makeMangler(CI, Rules);
  for (unsigned I = 0; I < Args.size(); I++) {
    Mangler->getTypeMangleInfo(I).PointerTy =
        dyn_cast<TypedPointerType>(PointerTypes[I]);
  }
  assert(Attrs.getNumAttrSets() <= Args.size() + 2 && "Too many attributes?");

  // Sanitize the return type, in case it's a TypedPointerType.
  if (auto *TPT = dyn_cast<TypedPointerType>(ReturnTy))
    ReturnTy = PointerType::get(TPT->getElementType(), TPT->getAddressSpace());

  CallInst *NewCall =
      Builder.Insert(addCallInst(CI->getModule(), FuncName, ReturnTy, Args,
                                 &Attrs, nullptr, Mangler.get()));
  NewCall->copyMetadata(*CI);
  NewCall->setAttributes(CallAttrs);
  NewCall->setTailCall(CI->isTailCall());
  if (CI->hasFnAttr("fpbuiltin-max-error")) {
    auto Attr = CI->getFnAttr("fpbuiltin-max-error");
    NewCall->addFnAttr(Attr);
  }
  Value *Result = MutateRet ? MutateRet(Builder, NewCall) : NewCall;
  Result->takeName(CI);
  if (!CI->getType()->isVoidTy())
    CI->replaceAllUsesWith(Result);
  CI->dropAllReferences();
  CI->eraseFromParent();
  CI = nullptr;
  return Result;
}

BuiltinCallMutator &BuiltinCallMutator::setArgs(ArrayRef<Value *> NewArgs) {
  // Retain only the function attributes, not any parameter attributes.
  Attrs = AttributeList::get(CI->getContext(), Attrs.getFnAttrs(),
                             Attrs.getRetAttrs(), {});
  CallAttrs = AttributeList::get(CI->getContext(), CallAttrs.getFnAttrs(),
                                 CallAttrs.getRetAttrs(), {});
  Args.clear();
  PointerTypes.clear();
  for (Value *Arg : NewArgs) {
    assert(!Arg->getType()->isPointerTy() &&
           "Cannot use this signature with pointer types");
    Args.push_back(Arg);
    PointerTypes.push_back(Arg->getType());
  }
  return *this;
}

// This is a helper method to handle splicing of the attribute lists, as
// llvm::AttributeList doesn't have any helper methods for this sort of design.
// (It's designed to be manually built-up, not adjusted to add/remove
// arguments on the fly).
static void moveAttributes(LLVMContext &Ctx, AttributeList &Attrs,
                           unsigned Start, unsigned Len, unsigned Dest) {
  SmallVector<std::pair<unsigned, AttributeSet>, 6> NewAttrs;
  for (unsigned Index : Attrs.indexes()) {
    AttributeSet AttrSet = Attrs.getAttributes(Index);
    if (!AttrSet.hasAttributes())
      continue;

    // If the attribute is a parameter index, check to see how its index should
    // be adjusted.
    if (Index > AttributeList::FirstArgIndex) {
      unsigned ParamIndex = Index - AttributeList::FirstArgIndex;
      if (ParamIndex >= Start && ParamIndex < Start + Len)
        // A parameter in this range needs to have its index adjusted to its
        // destination location.
        Index += Dest - Start;
      else if (ParamIndex >= Dest && ParamIndex < Dest + Len)
        // This parameter will be overwritten by one of the moved parameters, so
        // omit it entirely.
        continue;
    }

    // The array is usually going to be sorted, but because of the above
    // adjustment, we might end up out of order. This logic ensures that the
    // array always remains in sorted order.
    std::pair<unsigned, AttributeSet> ToInsert(Index, AttrSet);
    NewAttrs.insert(llvm::lower_bound(NewAttrs, ToInsert, llvm::less_first()),
                    ToInsert);
  }
  Attrs = AttributeList::get(Ctx, NewAttrs);
}

BuiltinCallMutator &BuiltinCallMutator::insertArg(unsigned Index,
                                                  ValueTypePair Arg) {
  Args.insert(Args.begin() + Index, Arg.first);
  PointerTypes.insert(PointerTypes.begin() + Index, Arg.second);
  moveAttributes(CI->getContext(), Attrs, Index, Args.size() - Index,
                 Index + 1);
  moveAttributes(CI->getContext(), CallAttrs, Index, Args.size() - Index,
                 Index + 1);
  return *this;
}

BuiltinCallMutator &BuiltinCallMutator::replaceArg(unsigned Index,
                                                   ValueTypePair Arg) {
  Args[Index] = Arg.first;
  PointerTypes[Index] = Arg.second;
  Attrs = Attrs.removeParamAttributes(CI->getContext(), Index);
  CallAttrs = CallAttrs.removeParamAttributes(CI->getContext(), Index);
  return *this;
}

BuiltinCallMutator &BuiltinCallMutator::removeArg(unsigned Index) {
  // If the argument being dropped is the last one, there is nothing to move, so
  // just remove the attributes.
  auto &Ctx = CI->getContext();
  if (Index == Args.size() - 1) {
    Attrs = Attrs.removeParamAttributes(Ctx, Index);
    CallAttrs = CallAttrs.removeParamAttributes(Ctx, Index);
  } else {
    moveAttributes(Ctx, Attrs, Index + 1, Args.size() - Index - 1, Index);
    moveAttributes(Ctx, CallAttrs, Index + 1, Args.size() - Index - 1, Index);
  }
  Args.erase(Args.begin() + Index);
  PointerTypes.erase(PointerTypes.begin() + Index);
  return *this;
}

BuiltinCallMutator &
BuiltinCallMutator::changeReturnType(Type *NewReturnTy,
                                     MutateRetFuncTy MutateFunc) {
  ReturnTy = NewReturnTy;
  MutateRet = std::move(MutateFunc);
  return *this;
}

BuiltinCallMutator BuiltinCallHelper::mutateCallInst(CallInst *CI,
                                                     spv::Op Opcode) {
  return mutateCallInst(CI, getSPIRVFuncName(Opcode));
}

BuiltinCallMutator BuiltinCallHelper::mutateCallInst(CallInst *CI,
                                                     std::string FuncName) {
  assert(CI->getCalledFunction() && "Can only mutate direct function calls.");
  return BuiltinCallMutator(CI, std::move(FuncName), Rules, NameMapFn);
}

Value *BuiltinCallHelper::addSPIRVCall(IRBuilder<> &Builder, spv::Op Opcode,
                                       Type *ReturnTy, ArrayRef<Value *> Args,
                                       ArrayRef<Type *> ArgTys,
                                       const Twine &Name) {
  // Sanitize the return type, in case it's a TypedPointerType.
  if (auto *TPT = dyn_cast<TypedPointerType>(ReturnTy))
    ReturnTy = PointerType::get(TPT->getElementType(), TPT->getAddressSpace());

  // Copy the types into the mangling info.
  BuiltinFuncMangleInfo BtnInfo;
  for (unsigned I = 0; I < ArgTys.size(); I++) {
    if (Args[I]->getType()->isPointerTy())
      BtnInfo.getTypeMangleInfo(I).PointerTy = ArgTys[I];
  }

  // Create the function and the call.
  auto *F = getOrCreateFunction(M, ReturnTy, getTypes(Args),
                                getSPIRVFuncName(Opcode), &BtnInfo);
  return Builder.CreateCall(F, Args, ReturnTy->isVoidTy() ? "" : Name);
}

Type *BuiltinCallHelper::adjustImageType(Type *T, StringRef OldImageKind,
                                         StringRef NewImageKind) {
  if (auto *TypedPtrTy = dyn_cast<TypedPointerType>(T)) {
    Type *StructTy = TypedPtrTy->getElementType();
    // Adapt opencl.* struct type names to spirv.* struct type names.
    if (isOCLImageType(T)) {
      if (OldImageKind != kSPIRVTypeName::Image)
        report_fatal_error("Type was not an image type");
      auto ImageTypeName = StructTy->getStructName();
      auto Desc =
          map<SPIRVTypeImageDescriptor>(getImageBaseTypeName(ImageTypeName));
      spv::AccessQualifier Acc = AccessQualifierReadOnly;
      if (hasAccessQualifiedName(ImageTypeName))
        Acc = getAccessQualifier(ImageTypeName);
      auto NewImageType = SPIRVOpaqueTypeOpCodeMap::map(NewImageKind.str());
      return getSPIRVType(NewImageType, Type::getVoidTy(M->getContext()), Desc,
                          Acc);
    }

    // Change type name (e.g., spirv.Image -> spirv.SampledImg) if necessary.
    StringRef Postfixes;
    if (isSPIRVStructType(StructTy, OldImageKind, &Postfixes))
      StructTy = getOrCreateOpaqueStructType(
          M, getSPIRVTypeName(NewImageKind, Postfixes));
    else {
      report_fatal_error("Type did not have expected image kind");
    }
    return TypedPointerType::get(StructTy, TypedPtrTy->getAddressSpace());
  }

  if (auto *TargetTy = dyn_cast<TargetExtType>(T)) {
    StringRef Name = TargetTy->getName();
    if (!Name.consume_front(kSPIRVTypeName::PrefixAndDelim) ||
        Name != OldImageKind)
      report_fatal_error("Type did not have expected image kind");
    return TargetExtType::get(
        TargetTy->getContext(),
        (Twine(kSPIRVTypeName::PrefixAndDelim) + NewImageKind).str(),
        TargetTy->type_params(), TargetTy->int_params());
  }

  report_fatal_error("Expected type to be a SPIRV image type");
}

Type *BuiltinCallHelper::getSPIRVType(spv::Op TypeOpcode, bool UseRealType) {
  return getSPIRVType(TypeOpcode, "", {}, UseRealType);
}

Type *BuiltinCallHelper::getSPIRVType(spv::Op TypeOpcode,
                                      spv::AccessQualifier Access,
                                      bool UseRealType) {
  return getSPIRVType(TypeOpcode, "", {(unsigned)Access}, UseRealType);
}

Type *BuiltinCallHelper::getSPIRVType(
    spv::Op TypeOpcode, Type *InnerType, SPIRVTypeImageDescriptor Desc,
    std::optional<spv::AccessQualifier> Access, bool UseRealType) {
  return getSPIRVType(TypeOpcode, convertTypeToPostfix(InnerType),
                      {(unsigned)Desc.Dim, (unsigned)Desc.Depth,
                       (unsigned)Desc.Arrayed, (unsigned)Desc.MS,
                       (unsigned)Desc.Sampled, (unsigned)Desc.Format,
                       (unsigned)Access.value_or(AccessQualifierReadOnly)},
                      UseRealType);
}

Type *BuiltinCallHelper::getSPIRVType(spv::Op TypeOpcode,
                                      StringRef InnerTypeName,
                                      ArrayRef<unsigned> Parameters,
                                      bool UseRealType) {
  if (UseTargetTypes) {
    std::string BaseName = (Twine(kSPIRVTypeName::PrefixAndDelim) +
                            SPIRVOpaqueTypeOpCodeMap::rmap(TypeOpcode))
                               .str();
    SmallVector<Type *, 1> TypeParams;
    if (!InnerTypeName.empty()) {
      TypeParams.push_back(getLLVMTypeForSPIRVImageSampledTypePostfix(
          InnerTypeName, M->getContext()));
    }
    return TargetExtType::get(M->getContext(), BaseName, TypeParams,
                              Parameters);
  }

  std::string FullName;
  {
    raw_string_ostream OS(FullName);
    OS << kSPIRVTypeName::PrefixAndDelim
       << SPIRVOpaqueTypeOpCodeMap::rmap(TypeOpcode);
    if (!InnerTypeName.empty() || !Parameters.empty())
      OS << kSPIRVTypeName::Delimiter;
    if (!InnerTypeName.empty())
      OS << kSPIRVTypeName::PostfixDelim << InnerTypeName;
    for (unsigned IntParam : Parameters)
      OS << kSPIRVTypeName::PostfixDelim << IntParam;
  }
  auto *STy = StructType::getTypeByName(M->getContext(), FullName);
  if (!STy)
    STy = StructType::create(M->getContext(), FullName);

  unsigned AddrSpace = getOCLOpaqueTypeAddrSpace(TypeOpcode);
  return UseRealType ? (Type *)PointerType::get(STy, AddrSpace)
                     : TypedPointerType::get(STy, AddrSpace);
}

void BuiltinCallHelper::initialize(llvm::Module &M) {
  this->M = &M;
  // We want to use pointers-to-opaque-structs for the special types if:
  // * We are translating from SPIR-V to LLVM IR (which means we are using
  //   OpenCL mangling rules)
  // * There are %opencl.* or %spirv.* struct type names already present.
  UseTargetTypes = Rules != ManglingRules::OpenCL;
  for (StructType *Ty : M.getIdentifiedStructTypes()) {
    if (!Ty->isOpaque() || !Ty->hasName())
      continue;
    StringRef Name = Ty->getName();
    if (Name.starts_with("opencl.") || Name.starts_with("spirv.")) {
      UseTargetTypes = false;
    }
  }
}

BuiltinCallMutator::ValueTypePair
BuiltinCallHelper::getCallValue(CallInst *CI, unsigned ArgNo) {
  Function *CalledFunc = CI->getCalledFunction();
  assert(CalledFunc && "Unexpected indirect call");
  if (CalledFunc != CachedFunc) {
    CachedFunc = CalledFunc;
    [[maybe_unused]] bool DidDemangle =
        getParameterTypes(CalledFunc, CachedParameterTypes, NameMapFn);
    assert(DidDemangle && "Expected SPIR-V builtins to be properly mangled");
  }

  Value *ParamValue = CI->getArgOperand(ArgNo);
  Type *ParamType = CachedParameterTypes[ArgNo];
  return {ParamValue, ParamType};
}
