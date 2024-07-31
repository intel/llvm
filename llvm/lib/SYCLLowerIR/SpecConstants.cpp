//===----- SpecConstants.cpp - SYCL Specialization Constants Pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// See comments in the header.
//===----------------------------------------------------------------------===//

#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/SYCLLowerIR/Support.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/TargetParser/Triple.h"

#include <vector>

#define DEBUG_TYPE "SpecConst"

using namespace llvm;

namespace {

// __sycl* intrinsic names are Itanium ABI-mangled; this is common prefix for
// all mangled names of __sycl_getSpecConstantValue intrinsics, which differ by
// the template type parameter and the specialization constant value type.
constexpr char SYCL_GET_SCALAR_2020_SPEC_CONST_VAL[] =
    "_Z37__sycl_getScalar2020SpecConstantValue";
constexpr char SYCL_GET_COMPOSITE_2020_SPEC_CONST_VAL[] =
    "_Z40__sycl_getComposite2020SpecConstantValue";

// Unmangled base name of all __spirv_SpecConstant intrinsics which differ by
// the value type.
constexpr char SPIRV_GET_SPEC_CONST_VAL[] = "__spirv_SpecConstant";
// Unmangled base name of all __spirv_SpecConstantComposite intrinsics which
// differ by the value type.
constexpr char SPIRV_GET_SPEC_CONST_COMPOSITE[] =
    "__spirv_SpecConstantComposite";

// Name of the metadata which holds a list of all specialization constants (with
// associated information) encountered in the module
constexpr char SPEC_CONST_MD_STRING[] = "sycl.specialization-constants";
// Name of the metadata which holds a default value list of all specialization
// constants encountered in the module
constexpr char SPEC_CONST_DEFAULT_VAL_MD_STRING[] =
    "sycl.specialization-constants-default-values";

/// Spec. Constant ID is a pair of Id and a flag whether this Id belongs to an
/// undefined value. Undefined values ('undef' in the IR) are used to get the
/// required alignment and should be handled in a special manner as padding.
struct ID {
  unsigned ID;
  bool Undef;
};

StringRef getStringLiteralArg(const CallInst *CI, unsigned ArgNo,
                              SmallVectorImpl<Instruction *> &DelInsts) {
  Value *V = CI->getArgOperand(ArgNo)->stripPointerCasts();

  if (auto *L = dyn_cast<LoadInst>(V)) {
    // Must be a
    // vvvvvvvvvvvvvvvvvvvv
    // @.str = private unnamed_addr constant[10 x i8] c"SpecConst\00", align 1
    // ...
    // %TName = alloca i8 addrspace(4)*, align 8
    // %TName.ascast = addrspacecast i8 addrspace(4)** %TName to
    //                               i8 addrspace(4)* addrspace(4)*
    // ...
    // store i8 addrspace(4)* getelementptr inbounds ([19 x i8], [19 x i8]
    //    addrspace(4)* addrspacecast ([19 x i8] addrspace(1)* @str to [19 x i8]
    //    addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspace(4)*
    //    %TName.ascast, align 8
    // %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %TName.ascast,
    //    align 8
    // %call = call spir_func zeroext
    //   i1 @_Z27__sycl_getSpecConstantValueIbET_PKc(i8 addrspace(4)* %0)
    // ^^^^^^^^^^^^^^^^^^^^
    // or (optimized version)
    // vvvvvvvvvvvvvvvvvvvv
    // @.str = private unnamed_addr constant[10 x i8] c"SpecConst\00", align 1
    // ...
    // %TName = alloca i8 addrspace(4)*, align 8
    // ...
    // store i8 addrspace(4)* addrspacecast(
    //    i8* getelementptr inbounds([10 x i8], [10 x i8] * @.str, i32 0, i32 0)
    //    to i8 addrspace(4)*), i8 addrspace(4)** %TName, align 8, !tbaa !10
    // %1 = load i8 addrspace(4)*, i8 addrspace(4)** %TName, align 8, !tbaa !10
    // %call = call spir_func zeroext
    //   i1 @_Z33__sycl_getScalarSpecConstantValueIbET_PKc(i8 addrspace(4)* %1)
    // ^^^^^^^^^^^^^^^^^^^^
    // sequence, w/o any intervening stores and calls between the store and load
    // so that %1 is trivially known to be the address of the @.str literal.

    Value *TmpPtr = L->getPointerOperand();
    AssertRelease((isa<AddrSpaceCastInst>(TmpPtr) &&
                   isa<AllocaInst>(cast<AddrSpaceCastInst>(TmpPtr)
                                       ->getPointerOperand()
                                       ->stripPointerCasts())) ||
                      isa<AllocaInst>(TmpPtr),
                  "unexpected instruction type");

    // find the store of the literal address into TmpPtr
    StoreInst *Store = nullptr;

    for (User *U : TmpPtr->users()) {
      if (StoreInst *St = dyn_cast<StoreInst>(U)) {
        AssertRelease(!Store, "single store expected");
        Store = St;
#ifndef NDEBUG
        break;
#endif // NDEBUG
      }
    }
    AssertRelease(Store, "unexpected spec const IR pattern 0");
    DelInsts.push_back(Store);
#ifndef NDEBUG
    // verify there are no intervening stores/calls
    AssertRelease(L->getParent() == Store->getParent(), "same BB expected");

    for (const Instruction *I = Store->getNextNode(); I; I = I->getNextNode()) {
      if (I == L) {
        DelInsts.push_back(L);
        L = nullptr; // mark as met
        break;
      }
      AssertRelease(!I->mayHaveSideEffects(),
                    "unexpected spec const IR pattern 1");
    }
    AssertRelease(!L, "load not met after the store");
#endif // NDEBUG
    AssertRelease(Store, "store not met");
    V = Store->getValueOperand()->stripPointerCasts();
  }
  const Constant *Init = cast<GlobalVariable>(V)->getInitializer();
  StringRef Res = cast<ConstantDataArray>(Init)->getAsString();
  if (Res.size() > 0 && Res[Res.size() - 1] == '\0')
    Res = Res.substr(0, Res.size() - 1);
  return Res;
}

std::string mangleType(const Type *T) {
  if (T->isFloatTy())
    return "f";
  if (T->isDoubleTy())
    return "d";
  if (T->isHalfTy())
    return "Dh";
  if (T->isIntegerTy()) {
    switch (T->getIntegerBitWidth()) {
    case 1:
      return "b";
    case 8:
      return "a";
    case 16:
      return "s";
    case 32:
      return "i";
    case 64:
      return "x";
    default:
      llvm_unreachable("unsupported spec const integer type");
    }
  }
  // Mangling, which is generated below is not fully conformant with C++ ABI
  // rules
  // (https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangle.unqualified-name)
  // But it should be more or less okay, because these declarations only
  // exists in the module between invocations of sycl-post-link and llvm-spirv,
  // llvm-spirv doesn't care about the mangling and the only intent here is to
  // make sure that we won't encounter redefinition error when we proceed two
  // spec constants with different types.
  if (T->isStructTy())
    return T->getStructName().str();
  if (T->isArrayTy())
    return "A" + std::to_string(T->getArrayNumElements()) + "_" +
           mangleType(T->getArrayElementType());

  if (auto *VecTy = dyn_cast<FixedVectorType>(T))
    return "Dv" + std::to_string(VecTy->getNumElements()) + "_" +
           mangleType(VecTy->getElementType());
  llvm_unreachable("unsupported spec const type");
  return "";
}

// This is a very basic mangler which can mangle non-templated and non-member
// functions with primitive types in the signature.
// FIXME: generated mangling is not always complies with C++ ABI rules and might
// not be demanglable. Consider fixing this.
std::string mangleFuncItanium(StringRef BaseName, const FunctionType *FT) {
  std::string Res =
      (Twine("_Z") + Twine(BaseName.size()) + Twine(BaseName)).str();
  for (unsigned I = 0; I < FT->getNumParams(); ++I)
    Res += mangleType(FT->getParamType(I));
  if (FT->getReturnType()->isArrayTy() || FT->getReturnType()->isStructTy() ||
      FT->getReturnType()->isVectorTy()) {
    // It is possible that we need to generate several calls to
    // __spirv_SpecConstantComposite, accepting the same argument types, but
    // returning different types. Therefore, we incorporate the return type into
    // the mangling name as well to distinguish between those functions
    Res += "_R" + mangleType(FT->getReturnType());
  }
  return Res;
}

MDNode *generateSpecConstDefaultValueMetadata(Value *Default) {
  LLVMContext &Ctx = Default->getContext();
  return MDNode::get(Ctx, ConstantAsMetadata::get(cast<Constant>(Default)));
}

/// Recursively iterates over a composite type in order to collect information
/// about its scalar elements.
void collectCompositeElementsInfoRecursive(
    const Module &M, Type *Ty, const ID *&IDIter, unsigned &Offset,
    std::vector<SpecConstantDescriptor> &Result) {
  if (IDIter->Undef) {
    ++IDIter;
    // Skip undef IDs, they are not reported to runtime.
    return;
  }
  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    for (size_t I = 0; I < ArrTy->getNumElements(); ++I) {
      // TODO: this is a spot for potential optimization: for arrays we could
      // just make a single recursive call here and use it to populate Result
      // in a loop.
      collectCompositeElementsInfoRecursive(M, ArrTy->getElementType(), IDIter,
                                            Offset, Result);
    }
    return;
  }
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = M.getDataLayout().getStructLayout(StructTy);
    const unsigned BaseOffset = Offset;
    unsigned LocalOffset = Offset;
    for (size_t I = 0, E = StructTy->getNumElements(); I < E; ++I) {
      auto *ElTy = StructTy->getElementType(I);
      // When handling elements of a structure, we do not use manually
      // calculated offsets (which are sum of sizes of all previously
      // encountered elements), but instead rely on data provided for us by
      // DataLayout, because the structure can be unpacked, i.e. padded in
      // order to ensure particular alignment of its elements.
      LocalOffset = Offset + SL->getElementOffset(I);
      collectCompositeElementsInfoRecursive(M, ElTy, IDIter, LocalOffset,
                                            Result);
    }

    // Add a special descriptor if the struct has padding at the end.
    const unsigned PostStructPadding =
        BaseOffset + SL->getSizeInBytes() - LocalOffset;
    if (PostStructPadding > 0) {
      SpecConstantDescriptor Desc;
      // ID of padding descriptors is the max value possible. This value is a
      // magic value for the runtime and will just be skipped. Even if there
      // are many specialization constants and every constant has padding of
      // a different length, everything will work regardless rewriting
      // the descriptions with Desc.ID equals to the max value: they will just
      // be ignored at all.
      Desc.ID = std::numeric_limits<unsigned>::max();
      Desc.Offset = LocalOffset;
      Desc.Size = PostStructPadding;
      Result.push_back(Desc);
    }

    // Update "global" offset according to the total size of a handled struct
    // type.
    Offset += SL->getSizeInBytes();
    return;
  }
  if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    for (size_t I = 0; I < VecTy->getNumElements(); ++I) {
      // TODO: this is a spot for potential optimization: for vectors we could
      // just make a single recursive call here and use it to populate Result
      // in a loop.
      collectCompositeElementsInfoRecursive(M, VecTy->getElementType(), IDIter,
                                            Offset, Result);
    }
    return;
  }

  // Assume that we encountered some scalar element
  SpecConstantDescriptor Desc;
  Desc.ID = IDIter->ID;
  Desc.Offset = Offset;
  Desc.Size = M.getDataLayout().getTypeStoreSize(Ty);
  Result.push_back(Desc);

  // Move current ID and offset
  ++IDIter;
  Offset += Desc.Size;
}

/// Recursively iterates over a composite type in order to collect information
/// about default values of its scalar elements.
/// TODO: processing of composite spec constants here is similar to
/// collectCompositeElementsInfoRecursive. Possible place for improvement -
/// factor out the common code, e.g. using visitor pattern.
void collectCompositeElementsDefaultValuesRecursive(
    const Module &M, Constant *C, unsigned &Offset,
    std::vector<char> &DefaultValues) {
  if (isa<ConstantAggregateZero>(C) || isa<UndefValue>(C)) {
    // This code is generic for both arrays and structs
    size_t NumBytes = M.getDataLayout().getTypeStoreSize(C->getType());
    std::fill_n(std::back_inserter(DefaultValues), NumBytes, 0);
    Offset += NumBytes;
    // Print tuple {Offset, Size, DefaultValue}.
    LLVM_DEBUG(dbgs() << "{" << Offset - NumBytes << ", " << NumBytes << ", "
                      << 0 << "}\n");
    return;
  }

  if (auto *DataSeqC = dyn_cast<ConstantDataSequential>(C)) {
    // This code is generic for both vectors and arrays of scalars
    for (size_t I = 0; I < DataSeqC->getNumElements(); ++I) {
      Constant *El = cast<Constant>(DataSeqC->getElementAsConstant(I));
      collectCompositeElementsDefaultValuesRecursive(M, El, Offset,
                                                     DefaultValues);
    }
    return;
  }

  if (auto *ArrayC = dyn_cast<ConstantArray>(C)) {
    // This branch handles arrays of composite types (structs, arrays, etc.)
    assert(!C->isZeroValue() && "C must not be a zeroinitializer");
    for (size_t I = 0; I < ArrayC->getType()->getNumElements(); ++I) {
      collectCompositeElementsDefaultValuesRecursive(M, ArrayC->getOperand(I),
                                                     Offset, DefaultValues);
    }
    return;
  }

  if (auto *StructC = dyn_cast<ConstantStruct>(C)) {
    assert(!C->isZeroValue() && "C must not be a zeroinitializer");
    auto *StructTy = StructC->getType();
    const StructLayout *SL = M.getDataLayout().getStructLayout(StructTy);
    const size_t BaseDefaultValueOffset = DefaultValues.size();
    for (size_t I = 0, E = StructTy->getNumElements(); I < E; ++I) {
      // When handling elements of a structure, we do not use manually
      // calculated offsets (which are sum of sizes of all previously
      // encountered elements), but instead rely on data provided for us by
      // DataLayout, because the structure can be unpacked, i.e. padded in
      // order to ensure particular alignment of its elements.
      unsigned LocalOffset = Offset + SL->getElementOffset(I);

      // If there was some alignment, fill the data between values with zeros.
      while (LocalOffset != DefaultValues.size())
        DefaultValues.push_back(0);

      collectCompositeElementsDefaultValuesRecursive(
          M, StructC->getOperand(I), LocalOffset, DefaultValues);
    }
    const size_t SLSize = SL->getSizeInBytes();

    // Additional padding may be needed at the end of the struct if size does
    // not match the number of bytes inserted.
    if (DefaultValues.size() < BaseDefaultValueOffset + SLSize)
      DefaultValues.resize(BaseDefaultValueOffset + SLSize);

    // Update "global" offset according to the total size of a handled struct
    // type.
    Offset += SLSize;
    return;
  }

  // Assume that we encountered some scalar element
  size_t NumBytes = M.getDataLayout().getTypeStoreSize(C->getType());
  if (auto *IntConst = dyn_cast<ConstantInt>(C)) {
    auto Val = IntConst->getValue().getZExtValue();
    std::copy_n(reinterpret_cast<char *>(&Val), NumBytes,
                std::back_inserter(DefaultValues));
    // Print tuple {Offset, Size, DefaultValue}.
    LLVM_DEBUG(dbgs() << "{" << Offset << ", " << NumBytes << ", " << Val
                      << "}\n");
  } else if (auto *FPConst = dyn_cast<ConstantFP>(C)) {
    auto Val = FPConst->getValue();

    if (NumBytes == 2) {
      auto IVal = Val.bitcastToAPInt();
      assert(IVal.getBitWidth() == 16);
      auto Storage = static_cast<uint16_t>(IVal.getZExtValue());
      std::copy_n(reinterpret_cast<char *>(&Storage), NumBytes,
                  std::back_inserter(DefaultValues));
      // Print tuple {Offset, Size, DefaultValue}.
      LLVM_DEBUG(dbgs() << "{" << Offset << ", " << NumBytes << ", " << IVal
                        << "}\n");
    } else if (NumBytes == 4) {
      float V = Val.convertToFloat();
      std::copy_n(reinterpret_cast<char *>(&V), NumBytes,
                  std::back_inserter(DefaultValues));
      // Print tuple {Offset, Size, DefaultValue}.
      LLVM_DEBUG(dbgs() << "{" << Offset << ", " << NumBytes << ", " << V
                        << "}\n");
    } else if (NumBytes == 8) {
      double V = Val.convertToDouble();
      std::copy_n(reinterpret_cast<char *>(&V), NumBytes,
                  std::back_inserter(DefaultValues));
      // Print tuple {Offset, Size, DefaultValue}.
      LLVM_DEBUG(dbgs() << "{" << Offset << ", " << NumBytes << ", " << V
                        << "}\n");
    } else {
      llvm_unreachable("Unexpected constant floating point type");
    }
  } else {
    llvm_unreachable("Unexpected constant scalar type");
  }
  Offset += NumBytes;
}

MDNode *generateSpecConstantMetadata(const Module &M, StringRef SymbolicID,
                                     Type *SCTy, ArrayRef<ID> IDs,
                                     bool IsNativeSpecConstant) {
  SmallVector<Metadata *, 16> MDOps;
  LLVMContext &Ctx = M.getContext();
  auto *Int32Ty = Type::getInt32Ty(Ctx);

  // First element is always Symbolic ID
  MDOps.push_back(MDString::get(Ctx, SymbolicID));

  if (IsNativeSpecConstant) {
    std::vector<SpecConstantDescriptor> Result;
    Result.reserve(IDs.size());
    unsigned Offset = 0;
    const ID *IDPtr = IDs.data();

    // Not all IDs are turned into metadata, because some of them may
    // represent padding within structures. Additionally, there could
    // be emitted multiple extra special ID describing post-struct
    // padding to align spec constants for runtime.
    collectCompositeElementsInfoRecursive(M, SCTy, IDPtr, Offset, Result);

    for (unsigned I = 0; I < Result.size(); ++I) {
      MDOps.push_back(ConstantAsMetadata::get(
          Constant::getIntegerValue(Int32Ty, APInt(32, Result[I].ID))));
      MDOps.push_back(ConstantAsMetadata::get(
          Constant::getIntegerValue(Int32Ty, APInt(32, Result[I].Offset))));
      MDOps.push_back(ConstantAsMetadata::get(
          Constant::getIntegerValue(Int32Ty, APInt(32, Result[I].Size))));
    }
  } else {
    assert(IDs.size() == 1 &&
           "There must be a single ID for emulated spec constant");
    MDOps.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(32, IDs[0].ID))));
    // Second element is always zero here
    MDOps.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(32, 0))));

    unsigned Size = M.getDataLayout().getTypeStoreSize(SCTy);

    MDOps.push_back(ConstantAsMetadata::get(
        Constant::getIntegerValue(Int32Ty, APInt(32, Size))));
  }

  return MDNode::get(Ctx, MDOps);
}

Instruction *emitCall(Type *RetTy, StringRef BaseFunctionName,
                      ArrayRef<Value *> Args, Instruction *InsertBefore) {
  SmallVector<Type *, 8> ArgTys(Args.size());
  for (unsigned I = 0; I < Args.size(); ++I) {
    ArgTys[I] = Args[I]->getType();
  }
  auto *FT = FunctionType::get(RetTy, ArgTys, false /*isVarArg*/);
  std::string FunctionName = mangleFuncItanium(BaseFunctionName, FT);
  Module *M = InsertBefore->getFunction()->getParent();

  if (RetTy->isIntegerTy(1)) {
    assert(ArgTys.size() == 2 && "Expected a scalar spec constant");
    // There is a problem with bool data type: depending on how it is used in
    // source code, clang can emit it as either i1 or i8. It might lead to a
    // situation where we need to emit call to
    // i1 __spirv_SpecConstantia(i32, i8) function for bool spec constant and
    // call to i8 __spirv_SpecConstantia(i32, i8) for char spec constants.
    // Those two calls are only differ by return type and generating them both
    // will result in something like:
    // call i8 bitcast (i1 (i32, i8)* @_Z20__spirv_SpecConstantia to i8 (i32,
    // i8)*)(i32 47, i8 20) and it will confuse the SPIR-V translator.
    //
    // In order to avoid that, we detect all situations when we need to emit
    // i1 __spirv_SpecConstantia(i32, i8) and instead emit a call to
    // i8 __spirv_SpecConstantia(i32, i8) followed by a trunc instruction to
    // make types consistent with the rest of LLVM IR.
    if (ArgTys[1]->isIntegerTy(8)) {
      LLVMContext &Ctx = RetTy->getContext();
      auto *NewRetTy = Type::getInt8Ty(Ctx);
      auto *NewFT = FunctionType::get(NewRetTy, ArgTys, false /*isVarArg*/);
      auto NewFC = M->getOrInsertFunction(FunctionName, NewFT);

      auto *Call =
          CallInst::Create(NewFT, NewFC.getCallee(), Args, "", InsertBefore);
      return CastInst::CreateTruncOrBitCast(Call, RetTy, "tobool",
                                            InsertBefore);
    }
  }

  // There is one more example where call bitcast construct might appear: it
  // would be user-defined data types, which are named differently, but their
  // content is the same:
  // %struct.A = { float, i32, i8, [3 x i8] }
  // %struct.B = { float, i32, i8. [3 x i8] }
  // If we have spec constants using both those types, we will end up with
  // something like:
  // %struct.A (float, i32, i8, [3 x i8])* bitcast (%struct.B (float, i32, i8,
  // [3 x i8])* @_Z29__spirv_SpecConstantCompositefiaAa to %struct.A (float,
  // i32, i8, [3 x i8])*) Such call of bitcast doesn't seem to confuse the
  // translator, but still doesn't look clean in LLVM IR.
  // FIXME: is it possible to avoid call bitcast construct for composite
  // types? Is it necessary?

  FunctionCallee FC = M->getOrInsertFunction(FunctionName, FT);
  return CallInst::Create(FT, FC.getCallee(), Args, "", InsertBefore);
}

Instruction *emitSpecConstant(unsigned NumericID, Type *Ty,
                              Instruction *InsertBefore,
                              Constant *DefaultValue) {
  Function *F = InsertBefore->getFunction();
  // Generate arguments needed by the SPIRV version of the intrinsic
  // - integer constant ID:
  Value *ID = ConstantInt::get(Type::getInt32Ty(F->getContext()), NumericID);
  // ... Now replace the call with SPIRV intrinsic version.
  assert(DefaultValue &&
         "default value of spec constant is expected to be known");
  Value *Args[] = {ID, DefaultValue};
  return emitCall(Ty, SPIRV_GET_SPEC_CONST_VAL, Args, InsertBefore);
}

Instruction *emitSpecConstantComposite(Type *Ty, ArrayRef<Value *> Elements,
                                       Instruction *InsertBefore) {
  return emitCall(Ty, SPIRV_GET_SPEC_CONST_COMPOSITE, Elements, InsertBefore);
}

// Select corresponding element of the default value.  For a
// struct, we getting the corresponding default value is a little
// tricky.  There are potentially distinct two types: the type of
// the default value, which comes from the initializer of the
// global spec constant value, and the return type of the call to
// getComposite2020SpecConstValue. The return type can be a
// version of the default value type, with padding fields
// potentially inserted at the top level and within nested
// structs.

// Examples: (RT = Return Type, DVT = Default Value Type)
// RT: { i8, [3 x i8], i32 }, DVT = { i8, i32 }
// RT: { { i32, i8, [3 x i8] }, i32 } DVT = { { i32, i8 }, i32 }

// For a given element of the default value type we are
// trying to initialize, we will initialize that element with
// the element of the default value type that has the same offset
// as the element we are trying to initialize. If no such element
// exists, we used undef as the initializer.
Constant *getElemDefaultValue(Type *Ty, Type *ElTy, Constant *DefaultValue,
                              size_t ElemIndex, const DataLayout &DL) {
  if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    auto *DefaultValueType = cast<StructType>(DefaultValue->getType());
    const auto &DefaultValueTypeSL = DL.getStructLayout(DefaultValueType);
    // The struct has padding, so we have to adjust ElemIndex
    if (DefaultValueTypeSL->hasPadding()) {
      const auto &ReturnTypeSL = DL.getStructLayout(StructTy);
      ArrayRef<TypeSize> DefaultValueOffsets =
          DefaultValueTypeSL->getMemberOffsets();
      TypeSize CurrentIterationOffset =
          ReturnTypeSL->getElementOffset(ElemIndex);
      const auto It =
          std::find(DefaultValueOffsets.begin(), DefaultValueOffsets.end(),
                    CurrentIterationOffset);

      // The element we are looking at is a padding field
      if (It == DefaultValueOffsets.end())
        return UndefValue::get(ElTy);
      // Select the index with the same offset
      ElemIndex = It - DefaultValueOffsets.begin();
    }
  }
  return DefaultValue->getAggregateElement(ElemIndex);
}

/// For specified specialization constant type emits LLVM IR which is required
/// in order to correctly handle it later during LLVM IR -> SPIR-V translation.
///
/// @param Ty [in] Specialization constant type to handle.
/// @param InsertBefore [in] Location in the module where new instructions
/// should be inserted.
/// @param IDs [in,out] List of IDs which are assigned for scalar specialization
/// constants. If \c IsNewSpecConstant is true, this vector is expected to
/// contain a single element with ID of the first spec constant - the rest of
/// generated spec constants will have their IDs generated by incrementing that
/// first ID. If  \c IsNewSpecConstant is false, this vector is expected to
/// contain enough elements to assign ID to each scalar element encountered in
/// the specified composite type.
/// @param [in,out] Index Index of scalar element within a composite type
///
/// @returns Instruction* representing specialization constant in LLVM IR, which
/// is in SPIR-V friendly LLVM IR form.
/// For scalar types it results in a single __spirv_SpecConstant call.
/// For composite types it results in a number of __spirv_SpecConstant calls
/// for each scalar member of the composite plus in a number of
/// __spirvSpecConstantComposite calls for each composite member of the
/// composite (plus for the top-level composite). Also enumerates all
/// encountered scalars and assigns them IDs (or re-uses existing ones).
Instruction *emitSpecConstantRecursiveImpl(Type *Ty, Instruction *InsertBefore,
                                           SmallVectorImpl<ID> &IDs,
                                           unsigned &Index,
                                           Constant *DefaultValue) {
  const Module &M = *InsertBefore->getModule();
  if (!Ty->isArrayTy() && !Ty->isStructTy() && !Ty->isVectorTy()) { // Scalar
    if (Index >= IDs.size()) {
      // If it is a new specialization constant, we need to generate IDs for
      // scalar elements, starting with the second one.
      assert(!isa<UndefValue>(DefaultValue) &&
             "All scalar values should be defined");
      IDs.push_back({IDs.back().ID + 1, false});
    }
    return emitSpecConstant(IDs[Index++].ID, Ty, InsertBefore, DefaultValue);
  }

  SmallVector<Value *, 8> Elements;
  auto HandleUndef = [&](Constant *Def) {
    if (Index >= IDs.size()) {
      // If it is a new specialization constant, we need to generate IDs for
      // the whole undef value.
      IDs.push_back({IDs.back().ID + 1, true});
    }
    Elements.push_back(Def);
    Index++;
  };
  auto LoopIteration = [&](Type *ElTy, unsigned LocalIndex) {
    const auto ElemDefaultValue = getElemDefaultValue(
        Ty, ElTy, DefaultValue, LocalIndex, M.getDataLayout());

    // If the default value is a composite and has the value 'undef', we should
    // not generate a bunch of __spirv_SpecConstant for its elements but
    // pass it into __spirv_SpecConstantComposite as is.
    if (isa<UndefValue>(ElemDefaultValue))
      HandleUndef(ElemDefaultValue);
    else
      Elements.push_back(emitSpecConstantRecursiveImpl(
          ElTy, InsertBefore, IDs, Index, ElemDefaultValue));
  };

  if (auto *ArrTy = dyn_cast<ArrayType>(Ty)) {
    for (size_t I = 0; I < ArrTy->getNumElements(); ++I)
      LoopIteration(ArrTy->getElementType(), I);
  } else if (auto *StructTy = dyn_cast<StructType>(Ty)) {
    size_t I = 0;
    for (Type *ElTy : StructTy->elements())
      LoopIteration(ElTy, I++);
  } else if (auto *VecTy = dyn_cast<FixedVectorType>(Ty)) {
    for (size_t I = 0; I < VecTy->getNumElements(); ++I)
      LoopIteration(VecTy->getElementType(), I);
  } else {
    llvm_unreachable("Unexpected spec constant type");
  }

  return emitSpecConstantComposite(Ty, Elements, InsertBefore);
}

/// Wrapper intended to hide IsFirstElement argument from the caller
Instruction *emitSpecConstantRecursive(Type *Ty, Instruction *InsertBefore,
                                       SmallVectorImpl<ID> &IDs,
                                       Constant *DefaultValue) {
  unsigned Index = 0;
  return emitSpecConstantRecursiveImpl(Ty, InsertBefore, IDs, Index,
                                       DefaultValue);
}

/// Function creates load instruction from the given Buffer by the given Offset.
/// Function returns the value of load instruction.
Value *createLoadFromBuffer(CallInst *InsertBefore, Value *Buffer,
                            size_t Offset, Type *SCType) {
  LLVMContext &C = InsertBefore->getContext();
  Type *Int8Ty = Type::getInt8Ty(C);
  Type *Int32Ty = Type::getInt32Ty(C);
  GetElementPtrInst *GEP = GetElementPtrInst::Create(
      Int8Ty, Buffer, {ConstantInt::get(Int32Ty, Offset, false)}, "gep",
      InsertBefore);

  Instruction *BitCast = nullptr;
  if (SCType->isIntegerTy(1)) // No bitcast to i1 before load
    BitCast = GEP;
  else
    BitCast =
        new BitCastInst(GEP, PointerType::get(SCType, GEP->getAddressSpace()),
                        "bc", InsertBefore);

  // When we encounter i1 spec constant, we still load the whole byte
  Value *Load = new LoadInst(SCType->isIntegerTy(1) ? Int8Ty : SCType, BitCast,
                             "load", InsertBefore);
  if (SCType->isIntegerTy(1)) // trunc back to i1 if necessary
    Load = CastInst::CreateIntegerCast(Load, SCType, /* IsSigned */ false,
                                       "tobool", InsertBefore);

  return Load;
}

/// Function tries to dig out the initializer from the given CallInst to
/// SpecConst function. ArgIndex is the expected index of the function operand
/// leading to the initializer.
///
/// Examples:
/// 1)
///   %"spec_id" = type { i32 }
///   @value = internal addrspace(1) constant %"spec_id" { i32 123 }, align 4
///   call spir_func i32 @sycl_getScalar2020SpecConst(%1, @value, %2)
///
/// 2)
///   %"spec_id" = type { %A }
///   %A = type { i32 }
///   @value = constant %"spec_id" { %A { i32 1 } }, align 4
///   call spir_func void @getCompositeSpecConst(%1, %2, @value, %3)
Constant *getSpecConstInitializerFromCI(CallInst *CI, unsigned ArgIndex) {
  auto *GV =
      cast<GlobalVariable>(CI->getArgOperand(ArgIndex)->stripPointerCasts());

  // Go through global variable if the argument was not null.
  assert(GV->hasInitializer() && "GV is expected to have initializer");
  Constant *Initializer = GV->getInitializer();
  assert((isa<ConstantAggregate>(Initializer) || Initializer->isZeroValue()) &&
         "expected specialization_id instance");
  // specialization_id structure contains a single field which is the
  // default value of corresponding specialization constant.
  return Initializer->getAggregateElement(0u);
}

/// Function replaces last Metadata node in the given vector with new
/// node which contains given Padding.
void updatePaddingInLastMDNode(LLVMContext &Ctx,
                               MapVector<StringRef, MDNode *> &SCMetadata,
                               unsigned Padding) {
  // The spec constant map can't be empty as the first offset is 0
  // and so it can't be misaligned.
  assert(!SCMetadata.empty() && "Cannot add padding to first spec constant");

  // To communicate the padding to the runtime, update the metadata
  // node of the previous spec constant to append a padding node. It
  // can't be added in front of the current spec constant, as doing
  // so would require the spec constant node to have a non-zero
  // CompositeOffset which breaks accessing it in the runtime.
  auto Last = SCMetadata.back();

  // Emulated spec constants don't use composite so should
  // always be formatted as (SymID, ID, Offset, Size), except when
  // they include padding, but since padding is added at insertion
  // of the next element, the last element of the map can never be
  // padded.
  assert(Last.second->getNumOperands() == 4 &&
         "Incorrect emulated spec constant format");

  Type *Int32Ty = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 16> MDOps;

  // Copy the existing metadata.
  MDOps.push_back(Last.second->getOperand(0));
  MDOps.push_back(Last.second->getOperand(1));
  MDOps.push_back(Last.second->getOperand(2));
  auto &SizeOp = Last.second->getOperand(3);
  MDOps.push_back(SizeOp);

  // Extract the size of the previous node to use as CompositeOffset
  // for the padding node.
  auto PrevSize = mdconst::extract<ConstantInt>(SizeOp)->getValue();

  // The max value is a magic value used for padding that the
  // runtime knows to skip.
  MDOps.push_back(ConstantAsMetadata::get(Constant::getIntegerValue(
      Int32Ty, APInt(32, std::numeric_limits<unsigned>::max()))));
  MDOps.push_back(
      ConstantAsMetadata::get(Constant::getIntegerValue(Int32Ty, PrevSize)));
  MDOps.push_back(ConstantAsMetadata::get(
      Constant::getIntegerValue(Int32Ty, APInt(32, Padding))));

  // Replace the last metadata node with the node including the padding.
  SCMetadata[Last.first] = MDNode::get(Ctx, MDOps);
}

/// Function creates 'store' instruction from the given Value @V into
/// the given Value @Dst.
/// Note: Types of values Dst and V might differ because of padding bytes
/// inserted by Clang FE.
/// For example:
/// Type of specialization constant might be <{ i32, i8, [ 3 x i8 ] }>, where
/// the last component are padding bytes.
/// specialization id in this case could be { i32, i8 } { i32 1, i8 1 }.
/// As you can see, padding bytes are absent. In order to mitigate this we
/// perform bitcast from specialization id type to specialization constant
/// type.
void createStoreInstructionIntoSpecConstValue(Value *Dst, Value *V,
                                              CallInst *InsertBefore) {
  Type *PointerType =
      PointerType::get(V->getType(), Dst->getType()->getPointerAddressSpace());
  IRBuilder B(InsertBefore);
  Value *Bitcast = B.CreateBitCast(Dst, PointerType);
  B.CreateStore(V, Bitcast);
}

} // namespace

PreservedAnalyses SpecConstantsPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  ID NextID = {0, false};
  unsigned NextOffset = 0;
  StringMap<SmallVector<ID, 1>> IDMap;
  StringMap<unsigned> OffsetMap;
  MapVector<StringRef, MDNode *> SCMetadata;
  SmallVector<MDNode *, 4> DefaultsMetadata;

  // Iterate through all declarations of instances of function template
  // template <typename T> T __sycl_get*SpecConstantValue(const char *ID)
  // intrinsic to find its calls and lower them depending on the HandlingMode.
  bool IRModified = false;
  LLVMContext &Ctx = M.getContext();
  bool IsSPIREmulated =
      Triple(M.getTargetTriple()).isSPIR() && Mode == HandlingMode::emulation;
  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    const bool IsSYCLAlloca = F.getIntrinsicID() == Intrinsic::sycl_alloca;

    // 'llvm.sycl.alloca' is not supported in emulation mode on SPIR-V targets.
    if (IsSPIREmulated && IsSYCLAlloca)
      continue;

    if (!F.getName().starts_with(SYCL_GET_SCALAR_2020_SPEC_CONST_VAL) &&
        !F.getName().starts_with(SYCL_GET_COMPOSITE_2020_SPEC_CONST_VAL) &&
        !IsSYCLAlloca)
      continue;

    SmallVector<CallInst *, 32> SCIntrCalls;
    for (auto *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U))
        SCIntrCalls.push_back(CI);
    }

    IRModified = IRModified || (SCIntrCalls.size() > 0);

    for (auto *CI : SCIntrCalls) {
      // 1. Find the Symbolic ID (string literal) passed as the actual argument
      // to the intrinsic - this should always be possible, as only string
      // literals are passed to it in the SYCL RT source code, and application
      // code can't use this intrinsic directly.

      SmallVector<Instruction *, 3> DelInsts;
      DelInsts.push_back(CI);
      Function *Callee = CI->getCalledFunction();
      assert(Callee && "Failed to get spec constant call");

      // Structs are returned via 'sret' arguments if they are larger than 64b
      bool HasSretParameter = Callee->hasStructRetAttr();
      assert(!(HasSretParameter && IsSYCLAlloca) &&
             "'llvm.sycl.alloca' returns a pointer");
      // Skip 'sret' parameter.
      unsigned NameArgNo = HasSretParameter ? 1 : 0;

      StringRef SymID = getStringLiteralArg(CI, NameArgNo, DelInsts);
      Value *Replacement = nullptr;

      Constant *DefaultValue = getSpecConstInitializerFromCI(CI, NameArgNo + 1);
      Type *SCTy;
      if (HasSretParameter) {
        // Specialization constant type is given by the 'sret' parameter.
        SCTy = Callee->getParamStructRetType(0);
      } else if (IsSYCLAlloca) {
        // 'llvm.sycl.alloca' returns a pointer, so we need to take the
        // specialization constant type from the default value. At this stage,
        // we will have lost the original scalar representation of the type, so
        // we have to take the in-memory representation. This is only relevant
        // when a 'bool' ('i1' scalar representation and 'i8' in-memory
        // representation) specialization constant is used as size. In that
        // case, for a value of 'true' (the only legal value), the default value
        // will be 1 ('i8'), thus keeping the original semantics.
        SCTy = DefaultValue->getType();
      } else {
        // Specialization constant type is the same as the one returned by the
        // function in the general case.
        SCTy = CI->getType();
      }

      bool IsNewSpecConstant = false;
      unsigned Padding = 0;
      if (Mode == HandlingMode::native) {
        // 2. Spec constant value will be set at run time - then add the literal
        // to a "spec const string literal ID" -> "vector of integer IDs" map,
        // making the integer IDs unique if this is a new literal
        auto Ins = IDMap.insert(std::make_pair(SymID, SmallVector<ID, 1>{}));
        IsNewSpecConstant = Ins.second;
        auto &IDs = Ins.first->second;
        if (IsNewSpecConstant) {
          // For any spec constant type there will be always at least one ID
          // generated.
          IDs.push_back(NextID);
        }

        //  3. Transform to spirv intrinsic _Z*__spirv_SpecConstant* or
        //  _Z*__spirv_SpecConstantComposite
        Replacement = emitSpecConstantRecursive(SCTy, CI, IDs, DefaultValue);
        if (IsNewSpecConstant) {
          // emitSpecConstantRecursive might emit more than one spec constant
          // (because of composite types) and therefore, we need to adjust
          // NextID according to the actual amount of emitted spec constants.
          NextID.ID += IDs.size();

          // Generate necessary metadata which later will be pulled by
          // sycl-post-link and transformed into device image properties
          SCMetadata[SymID] = generateSpecConstantMetadata(
              M, SymID, SCTy, IDs, /* is native spec constant */ true);
        }
      } else if (Mode == HandlingMode::emulation) {
        // 2a. Spec constant will be passed as kernel argument;

        // Replace it with a load from the pointer to the specialization
        // constant value.
        // A pointer to a single RT-buffer with all the values of
        // specialization constants is passed as a 3rd argument of intrinsic.
        Value *RTBuffer =
            HasSretParameter ? CI->getArgOperand(3) : CI->getArgOperand(2);

        // Add the string literal to a "spec const string literal ID" ->
        // "offset" map, uniquing the integer offsets if this is new
        // literal.
        auto Ins = OffsetMap.insert(std::make_pair(SymID, NextOffset));
        IsNewSpecConstant = Ins.second;
        unsigned CurrentOffset = Ins.first->second;
        if (IsNewSpecConstant) {
          unsigned Size = M.getDataLayout().getTypeStoreSize(SCTy);
          uint64_t Align = M.getDataLayout().getABITypeAlign(SCTy).value();

          // Ensure correct alignment
          if (CurrentOffset % Align != 0) {
            // Compute necessary padding to correctly align the constant.
            Padding = Align - CurrentOffset % Align;

            // Update offsets.
            NextOffset += Padding;
            CurrentOffset += Padding;
            OffsetMap[SymID] = NextOffset;

            assert(CurrentOffset % Align == 0 && "Alignment calculation error");
            updatePaddingInLastMDNode(Ctx, SCMetadata, Padding);
          }

          SCMetadata[SymID] = generateSpecConstantMetadata(
              M, SymID, SCTy, NextID, /* is native spec constant */ false);

          ++NextID.ID;
          NextOffset += Size;
        }

        Replacement = createLoadFromBuffer(CI, RTBuffer, CurrentOffset, SCTy);
      } else if (Mode == HandlingMode::default_values) {
        if (SCTy->isIntegerTy(1)) {
          assert(DefaultValue->getType()->isIntegerTy(8) &&
                 "For bool spec constant default value is expected to be i8");
          Replacement =
              new TruncInst(DefaultValue, Type::getInt1Ty(Ctx), "bool", CI);
        } else
          Replacement = DefaultValue;
      }

      if (IsNewSpecConstant) {
        if (Padding != 0) {
          // Initialize the padding with null data
          auto PadTy = ArrayType::get(Type::getInt8Ty(Ctx), Padding);
          DefaultsMetadata.push_back(MDNode::get(
              Ctx,
              ConstantAsMetadata::get(llvm::Constant::getNullValue(PadTy))));
        }
        DefaultsMetadata.push_back(
            generateSpecConstDefaultValueMetadata(DefaultValue));
      }

      if (IsSYCLAlloca) {
        // In case this is a 'sycl.llvm.alloca' intrinsic, use the emitted
        // specialization constant as the allocation size.
        auto *Intr = cast<SYCLAllocaInst>(CI);
        // For emulation mode, use the default value for now. This code should
        // never be run, as the runtime should throw a 'kernel_not_supported'
        // exception.
        Value *ArraySize =
            Mode == HandlingMode::emulation ? DefaultValue : Replacement;
        assert(ArraySize->getType()->isIntegerTy() && "Expecting integer type");
        Replacement =
            new AllocaInst(Intr->getAllocatedType(), Intr->getAddressSpace(),
                           ArraySize, Intr->getAlign(), "alloca", CI);
      }

      if (HasSretParameter)
        createStoreInstructionIntoSpecConstValue(CI->getArgOperand(0),
                                                 Replacement, CI);
      else
        CI->replaceAllUsesWith(Replacement);

      for (auto *I : DelInsts) {
        I->removeFromParent();
        I->deleteValue();
      }
    }
  }

  // Emit metadata about encountered specializaiton constants. This metadata
  // is later queried by sycl-post-link in order to be converted into device
  // image properties.
  // Generated metadata looks like:
  // !sycl.specialization-constants = !{!1, !2, ... for each spec constant}
  // !1 = !{!"SymbolicID1", i32 1, i32 0, i32 4, i32 2, i32 4, i32 8}
  // !2 = !{!"SymbolicID2", i32 3, i32 0, i32 4}
  // The format is [Symbolic ID, list of triplets: numeric ID, offset, size]
  // For more infor about meaning of those triplets see comments about
  // SpecConstantDescriptor structure in SpecConstants.h
  NamedMDNode *MD = M.getOrInsertNamedMetadata(SPEC_CONST_MD_STRING);
  for (const auto &P : SCMetadata)
    MD->addOperand(P.second);

  // Emit default values metadata
  NamedMDNode *MDDefaults =
      M.getOrInsertNamedMetadata(SPEC_CONST_DEFAULT_VAL_MD_STRING);
  for (const auto &P : DefaultsMetadata)
    MDDefaults->addOperand(P);

  return IRModified ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

bool SpecConstantsPass::collectSpecConstantMetadata(const Module &M,
                                                    SpecIDMapTy &IDMap) {
  NamedMDNode *MD = M.getNamedMetadata(SPEC_CONST_MD_STRING);
  if (!MD)
    return false;

  auto ExtractIntegerFromMDNodeOperand = [=](const MDNode *N,
                                             unsigned OpNo) -> unsigned {
    Constant *C =
        cast<ConstantAsMetadata>(N->getOperand(OpNo).get())->getValue();
    return static_cast<unsigned>(C->getUniqueInteger().getZExtValue());
  };

  // Print MD name only if there are any operands.
  if (MD->getNumOperands() > 0)
    LLVM_DEBUG(dbgs() << MD->getName() << "\n");

  for (const auto *Node : MD->operands()) {
    StringRef ID = cast<MDString>(Node->getOperand(0).get())->getString();
    assert((Node->getNumOperands() - 1) % 3 == 0 &&
           "Unexpected amount of operands");
    std::vector<SpecConstantDescriptor> Descs((Node->getNumOperands() - 1) / 3);
    for (unsigned NI = 1, I = 0; NI < Node->getNumOperands(); NI += 3, ++I) {
      Descs[I].ID = ExtractIntegerFromMDNodeOperand(Node, NI + 0);
      Descs[I].Offset = ExtractIntegerFromMDNodeOperand(Node, NI + 1);
      Descs[I].Size = ExtractIntegerFromMDNodeOperand(Node, NI + 2);
      // Print Node ID along with tuple {ID, Offset, Size}.
      LLVM_DEBUG(dbgs() << ID << "={" << Descs[I].ID << ", " << Descs[I].Offset
                        << ", " << Descs[I].Size << "}\n");
    }

    IDMap[ID] = Descs;
  }

  return true;
}

bool SpecConstantsPass::collectSpecConstantDefaultValuesMetadata(
    const Module &M, std::vector<char> &DefaultValues) {
  NamedMDNode *N = M.getNamedMetadata(SPEC_CONST_DEFAULT_VAL_MD_STRING);
  if (!N)
    return false;

  // Print N name only if there are any operands.
  if (N->getNumOperands() > 0)
    LLVM_DEBUG(dbgs() << N->getName() << "\n");

  unsigned Offset = 0;
  for (const auto *Node : N->operands()) {
    auto *Constant = cast<ConstantAsMetadata>(Node->getOperand(0))->getValue();
    collectCompositeElementsDefaultValuesRecursive(M, Constant, Offset,
                                                   DefaultValues);
  }

  return true;
}

bool llvm::checkModuleContainsSpecConsts(const Module &M) {
  for (const Function &F : M.functions()) {
    if (F.getName().starts_with(SYCL_GET_SCALAR_2020_SPEC_CONST_VAL) ||
        F.getName().starts_with(SYCL_GET_COMPOSITE_2020_SPEC_CONST_VAL) ||
        F.getIntrinsicID() == llvm::Intrinsic::sycl_alloca)
      return true;
  }

  return false;
}
