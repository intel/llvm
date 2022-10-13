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
      Attrs(CI->getCalledFunction()->getAttributes()), ReturnTy(CI->getType()),
      Args(CI->args()), Rules(Rules), Builder(CI) {
  getParameterTypes(CI->getCalledFunction(), PointerTypes,
                    std::move(NameMapFn));
  PointerTypes.resize(Args.size(), nullptr);
}

BuiltinCallMutator::BuiltinCallMutator(BuiltinCallMutator &&Other)
    : CI(Other.CI), FuncName(std::move(Other.FuncName)),
      MutateRet(std::move(Other.MutateRet)), Attrs(Other.Attrs),
      ReturnTy(Other.ReturnTy), Args(std::move(Other.Args)),
      PointerTypes(std::move(Other.PointerTypes)),
      Rules(std::move(Other.Rules)), Builder(CI) {
  // Clear the other's CI instance so that it knows not to construct the actual
  // call.
  Other.CI = nullptr;
}

Value *BuiltinCallMutator::doConversion() {
  assert(CI && "Need to have a call instruction to do the conversion");
  auto Mangler = makeMangler(CI, Rules);
  for (unsigned I = 0; I < Args.size(); I++) {
    Mangler->getTypeMangleInfo(I).PointerTy = PointerTypes[I];
  }
  assert(Attrs.getNumAttrSets() <= Args.size() + 2 && "Too many attributes?");
  CallInst *NewCall =
      Builder.Insert(addCallInst(CI->getModule(), FuncName, ReturnTy, Args,
                                 &Attrs, nullptr, Mangler.get()));
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
  Args.clear();
  PointerTypes.clear();
  for (Value *Arg : NewArgs) {
    assert(!Arg->getType()->isPointerTy() &&
           "Cannot use this signature with pointer types");
    Args.push_back(Arg);
    PointerTypes.emplace_back();
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

// Convert a ValueTypePair to a TypedPointerType for storing in the PointerTypes
// array.
static TypedPointerType *toTPT(BuiltinCallMutator::ValueTypePair Pair) {
  if (!Pair.second)
    return nullptr;
  unsigned AS = 0;
  if (auto *TPT = dyn_cast<TypedPointerType>(Pair.first->getType()))
    AS = TPT->getAddressSpace();
  else if (isa<PointerType>(Pair.first->getType()))
    AS = Pair.first->getType()->getPointerAddressSpace();
  return TypedPointerType::get(Pair.second, AS);
}

BuiltinCallMutator &BuiltinCallMutator::insertArg(unsigned Index,
                                                  ValueTypePair Arg) {
  Args.insert(Args.begin() + Index, Arg.first);
  PointerTypes.insert(PointerTypes.begin() + Index, toTPT(Arg));
  moveAttributes(CI->getContext(), Attrs, Index, Args.size() - Index,
                 Index + 1);
  return *this;
}

BuiltinCallMutator &BuiltinCallMutator::replaceArg(unsigned Index,
                                                   ValueTypePair Arg) {
  Args[Index] = Arg.first;
  PointerTypes[Index] = toTPT(Arg);
  Attrs = Attrs.removeParamAttributes(CI->getContext(), Index);
  return *this;
}

BuiltinCallMutator &BuiltinCallMutator::removeArg(unsigned Index) {
  // If the argument being dropped is the last one, there is nothing to move, so
  // just remove the attributes.
  if (Index == Args.size() - 1)
    Attrs = Attrs.removeParamAttributes(CI->getContext(), Index);
  else
    moveAttributes(CI->getContext(), Attrs, Index + 1, Args.size() - Index - 1,
                   Index);
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
  return BuiltinCallMutator(CI, std::move(FuncName), Rules, NameMapFn);
}
