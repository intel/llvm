// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicsRISCV.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Target/TargetMachine.h>
#include <multi_llvm/multi_llvm.h>
#include <multi_llvm/vector_type_helper.h>

#include "transform/packetization_helpers.h"
#include "vecz/vecz_target_info.h"

using namespace vecz;
using namespace llvm;

namespace vecz {

class TargetInfoRISCV final : public TargetInfo {
 public:
  TargetInfoRISCV(TargetMachine *tm) : TargetInfo(tm) {}

  ~TargetInfoRISCV() = default;

  bool canPacketize(const llvm::Value *Val, ElementCount Width) const override;

  // These functions should only be overriden in LLVM >= 13.
  llvm::Value *createScalableExtractElement(
      llvm::IRBuilder<> &B, vecz::VectorizationContext &Ctx,
      llvm::Instruction *extract, llvm::Type *narrowTy, llvm::Value *src,
      llvm::Value *index, llvm::Value *evl) const override;

  llvm::Value *createOuterScalableBroadcast(
      llvm::IRBuilder<> &builder, llvm::Value *vector, llvm::Value *VL,
      ElementCount factor) const override {
    return createScalableBroadcast(builder, vector, VL, factor,
                                   /* URem */ true);
  }

  llvm::Value *createInnerScalableBroadcast(
      llvm::IRBuilder<> &builder, llvm::Value *vector, llvm::Value *VL,
      ElementCount factor) const override {
    return createScalableBroadcast(builder, vector, VL, factor,
                                   /* URem */ false);
  }

  llvm::Value *createScalableInsertElement(llvm::IRBuilder<> &builder,
                                           vecz::VectorizationContext &Ctx,
                                           llvm::Instruction *insert,
                                           llvm::Value *elt, llvm::Value *into,
                                           llvm::Value *index,
                                           llvm::Value *evl) const override;
  bool isVPVectorLegal(const llvm::Function &F, llvm::Type *Ty) const override;

  llvm::Value *createVectorShuffle(llvm::IRBuilder<> &builder, llvm::Value *src,
                                   llvm::Value *mask,
                                   llvm::Value *evl) const override;

  llvm::Value *createVectorSlideUp(llvm::IRBuilder<> &builder, llvm::Value *src,
                                   llvm::Value *insert,
                                   llvm::Value *evl) const override;

 private:
  bool isOperationLegal(llvm::Intrinsic::ID ID,
                        llvm::ArrayRef<llvm::Type *> Tys) const;

  /// @brief Maximum vector type size in bits for VP intrinsics.
  static constexpr unsigned MaxLegalVectorTypeBits = 8 * 64;

  /// @return Whether the minimum size of a given vector type is less than 64
  /// bytes and the length is a power of 2.
  bool isVectorTypeLegal(llvm::Type *Ty) const;

  llvm::Value *createScalableBroadcast(llvm::IRBuilder<> &builder,
                                       llvm::Value *vector, llvm::Value *VL,
                                       ElementCount factor, bool URem) const;

  Value *createVPKernelWidth(IRBuilder<> &, Value *, unsigned,
                             ElementCount) const override;
};

// LLVM 14 introduced vp intrinsics legalization.
bool TargetInfoRISCV::isVPVectorLegal(const llvm::Function &F,
                                      llvm::Type *Ty) const {
  (void)F;
  return isVectorTypeLegal(Ty);
}

// Should be target-dependent. Take RISCV legal types for now.
// FIXME: LLVM 14 adds better support for legalization of vp intrinsics, but
// not RISCV ones like vrgather_vv. See CA-4071.
bool TargetInfoRISCV::isVectorTypeLegal(Type *Ty) const {
  assert(Ty->isVectorTy() && "Expecting a vector type.");
  (void)Ty;
  // FIXME: VP boolean logical operators (and,or,xor) are not matched in the
  // LLVM 13 RVV backend: we must backport https://reviews.llvm.org/D115546
  // before we can enable this for Int1Ty as well.
  bool isLegal = isLegalVPElementType(multi_llvm::getVectorElementType(Ty));
  if (isLegal) {
    const uint32_t MinSize =
        multi_llvm::getVectorElementCount(Ty).getKnownMinValue();
    isLegal = isPowerOf2_32(MinSize) &&
              MinSize * Ty->getScalarSizeInBits() <= MaxLegalVectorTypeBits;
  }
  return isLegal;
}

std::unique_ptr<TargetInfo> createTargetInfoRISCV(TargetMachine *tm) {
  return std::make_unique<TargetInfoRISCV>(tm);
}

}  // namespace vecz

bool TargetInfoRISCV::canPacketize(const llvm::Value *Val,
                                   ElementCount Width) const {
  // If we're not scalable, assume the backend will sort everything out.
  if (!Width.isScalable()) {
    return true;
  }
  // Do a relatively simple check that instructions aren't defining any types
  // that can't be legalized when turned into scalable vectors.
  if (!llvm::isa<llvm::Instruction>(Val)) {
    return true;
  }
  const auto *I = llvm::cast<llvm::Instruction>(Val);

  const auto IsIllegalIntBitwidth = [](const llvm::Type *Ty) {
    if (!Ty->isIntOrIntVectorTy()) {
      return false;
    }
    auto ScalarBitWidth =
        llvm::cast<IntegerType>(Ty->getScalarType())->getBitWidth();
    return ScalarBitWidth > 64;
  };

  if (IsIllegalIntBitwidth(I->getType())) {
    return false;
  }
  for (auto *O : I->operand_values()) {
    if (IsIllegalIntBitwidth(O->getType())) {
      return false;
    }
  }
  return true;
}

/// @return Whether RISCV intrinsic @a ID is legal for types @a Tys.
///
/// This function does not check whether the intrinsic is being called
/// with the right argument types, it just tests that all the types
/// used to call the intrinsic (and its return type) are
/// isVectorTypeLegal().
///
/// @param[in] ID The intrinsic ID
/// @param[in] Tys A subset of the overloaded types of the intrinsic required to
/// check whether it's legal.
bool TargetInfoRISCV::isOperationLegal(llvm::Intrinsic::ID ID,
                                       llvm::ArrayRef<llvm::Type *> Tys) const {
  switch (ID) {
    case Intrinsic::RISCVIntrinsics::riscv_vrgather_vv:
    case Intrinsic::RISCVIntrinsics::riscv_vrgather_vv_mask:
      // riscv_vrgather_vv[_mask](RetTy, _IdxTy)
      // We only need to check the return type here, as it should be greater or
      // equal to the index type.
      assert(Tys.size() == 1 &&
             "Only the return type is needed to check vrgather_vv intrinsics");
      return isVectorTypeLegal(Tys.front());
    case Intrinsic::RISCVIntrinsics::riscv_vrgatherei16_vv:
    case Intrinsic::RISCVIntrinsics::riscv_vrgatherei16_vv_mask: {
      constexpr unsigned MaxVectorSize = MaxLegalVectorTypeBits / 16;
      // riscv_vrgatherei16_vv[_mask](RetTy, _IdxTy)
      // Case similar to that of riscv_vrgather_vv[_mask], but we also need to
      // check that the vector size is no greater than MaxLegalVectorTypeSize /
      // 16, as the index type will always be i16.
      assert(
          Tys.size() == 1 &&
          "Only the return type is needed to check vrgatherei16_vv intrinsics");
      auto *const RetTy = Tys.front();
      return isVectorTypeLegal(RetTy) &&
             multi_llvm::getVectorElementCount(RetTy).getKnownMinValue() <=
                 MaxVectorSize;
    }
    default:
      break;
  }
  llvm_unreachable("Don't know how to check whether this intrinsic is legal.");
}

namespace {
static unsigned getRISCVBits(const TargetMachine *TM) {
  const auto &Triple = TM->getTargetTriple();
  return Triple.isArch32Bit() ? 32 : 64;
}

/// @brief Get VL to be used as a parameter of a RISCV intrinsic.
///
/// The type of this value will depend on the architecture (RISCV32 or
/// RISCV64).
///
/// @return A pair containig the VL value and its type.
///
/// @param[in] B Builder to use when creating the VL value.
/// @param[in] VL Original VL. If non-nullptr, this value (zero-extended for
/// RISCV64) will be returned.
/// @param[in] wideTy Type of the vectors which will be used in the intrinsics.
/// If no VL is provided and `<vscale x N x Ty>` is used here, `<call
/// llvm.vscale> * N` will be returned.
/// @param[in] TM Target machine.
/// @param[in] N name of the instruction to generate. "xlen" by default.
llvm::Value *getIntrinsicVL(llvm::IRBuilderBase &B, llvm::Value *VL,
                            llvm::Type *wideTy, llvm::TargetMachine *TM,
                            const Twine &N = "xlen") {
  const unsigned XLenTyWidth = getRISCVBits(TM);
  Type *XLen = B.getIntNTy(XLenTyWidth);

  if (VL) {
    // Our incoming VP VL type is always i32, so zero-extend to 64 bits if
    // required.
    return XLenTyWidth == 32 ? VL : B.CreateZExt(VL, XLen, N);
  }

  // Else create a 'default' VL which covers the entire scalable vector.
  return B.CreateVScale(
      B.getIntN(XLenTyWidth,
                cast<VectorType>(wideTy)->getElementCount().getKnownMinValue()),
      N);
}

/// @brief Returns a pair with the `vrgather` intrinsic variation to use and the
/// bitwidth of the `vs1` parameter to this intrinsic.
///
/// @param[in] vs2Ty Type of the source vector.
/// @param[in] isMasked Whether the intrinsic should be masked.
std::pair<llvm::Intrinsic::RISCVIntrinsics, unsigned> getGatherIntrinsic(
    llvm::Type *vs2Ty, bool isMasked = false) {
  assert(!vs2Ty->isPtrOrPtrVectorTy() &&
         "Cannot get gather intrinsic for a vector of pointers");

  Intrinsic::RISCVIntrinsics Opc;
  auto *vecTy = multi_llvm::getVectorElementType(vs2Ty);
  unsigned vs1Width;
  if (vecTy->isIntegerTy() && vecTy->getIntegerBitWidth() == 8) {
    Opc = isMasked ? Intrinsic::RISCVIntrinsics::riscv_vrgatherei16_vv_mask
                   : Intrinsic::RISCVIntrinsics::riscv_vrgatherei16_vv;

    vs1Width = 16;
  } else {
    Opc = isMasked ? Intrinsic::RISCVIntrinsics::riscv_vrgather_vv_mask
                   : Intrinsic::RISCVIntrinsics::riscv_vrgather_vv;

    vs1Width = vecTy->getScalarSizeInBits();
  }
  return std::make_pair(Opc, vs1Width);
}

/// @brief Returns the `v?slide1up.v?` intrinsic variation to use.
///
/// @param[in] vs2Ty Type of the source vector.
llvm::Intrinsic::RISCVIntrinsics getSlideUpIntrinsic(llvm::Type *vs2Ty) {
  assert(!vs2Ty->isPtrOrPtrVectorTy() &&
         "Cannot get gather intrinsic for a vector of pointers");

  Intrinsic::RISCVIntrinsics Opc;
  auto *vecTy = multi_llvm::getVectorElementType(vs2Ty);
  if (vecTy->isFloatingPointTy()) {
    Opc = Intrinsic::RISCVIntrinsics::riscv_vfslide1up;
  } else {
    Opc = Intrinsic::RISCVIntrinsics::riscv_vslide1up;
  }
  return Opc;
}

}  // namespace

llvm::Value *TargetInfoRISCV::createScalableExtractElement(
    llvm::IRBuilder<> &B, vecz::VectorizationContext &Ctx,
    llvm::Instruction *origExtract, llvm::Type *narrowTy, llvm::Value *src,
    llvm::Value *index, llvm::Value *VL) const {
  // In RISCV, we can use vrgather_vv and vrgatherei16_vv to avoid going through
  // memory when creating this operation.
  //   vrgather: vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];
  // or,
  //   vrgather: res[i] = (idxs[i] >= VLMAX) ? 0 : src[idxs[i]];
  // An example: extractelement <A,B,C,D>, I - vectorized by <vscale x 1> - we
  // receive here as packetized arguments:
  //   src:  <A,B,C,D, E,F,G,H,    ...> (  <vscale x 4 x ty>    )
  //   idxs: <I,       J,       K, ...> (  <vscale x 1 x idxty> )
  // We want to construct operands such that we have:
  //   srcs: as before
  //   idxs: <I+0,J+4,K+8,...>           (  <vscale x 4 x idxty> )
  // So that vrgather extracts the Ith element from the first 4 elements, the
  // Jth element from the second 4, etc.
  auto *srcTy = cast<ScalableVectorType>(src->getType());

  Intrinsic::ID intrinsicID;
  unsigned intrIdxBitWidth;
  std::tie(intrinsicID, intrIdxBitWidth) = getGatherIntrinsic(srcTy);

  const auto srcEC = multi_llvm::getVectorElementCount(srcTy);
  const auto resEC = multi_llvm::getVectorElementCount(narrowTy);

  auto *const indexEltTy = B.getIntNTy(intrIdxBitWidth);
  Type *const indexVecTy = VectorType::get(indexEltTy, resEC);

  // We cannot use this optimization if the types are not legal in the target
  // machine.
  if (!isOperationLegal(intrinsicID, {srcTy})) {
    return TargetInfo::createScalableExtractElement(B, Ctx, origExtract,
                                                    narrowTy, src, index, VL);
  }

  auto *const avl = getIntrinsicVL(B, VL, narrowTy, getTargetMachine());

  auto *indexTy = index->getType();
  const bool isIdxVector = indexTy->isVectorTy();
  const unsigned idxBitWidth = indexTy->getScalarSizeInBits();

  // The intrinsic may demand a larger index type than we currently have;
  // extend up to the right type.
  if (idxBitWidth != intrIdxBitWidth) {
    index = B.CreateZExtOrTrunc(index, isIdxVector ? indexVecTy : indexEltTy);
  }

  // If the index is uniform, it may not be a vector. We need one for the
  // intrinsic, so splat it here.
  if (!isIdxVector) {
    index = B.CreateVectorSplat(resEC, index);
  }

  // Construct the indices such that each packetized index (still indexing into
  // the original vector of 4 elements) is spread out such that each index
  // indexes into its own 4-element slice: e.g., <I+0, J+4, K+8, ...>.
  auto *indices = getGatherIndicesVector(
      B, index, indexVecTy,
      multi_llvm::getVectorNumElements(origExtract->getOperand(0)->getType()),
      "vs1");

  auto *const zero = B.getInt64(0);

  // Our indices are still in the narrower vectorized type (e.g., <vscale x 1 x
  // idxTy>), but the vrgather intrinsics need equally-sized vector types. So
  // insert the indices into a wide dummy vector (e.g., <vscale x 4 x idxTy>),
  // perform the vrgather, and extract the subvector back out again.
  auto *const intrIndexTy = VectorType::get(indexEltTy, srcEC);
  indices = B.CreateInsertVector(intrIndexTy, PoisonValue::get(intrIndexTy),
                                 indices, zero);

  SmallVector<Value *, 4> ops;
  // Add the a pass-through operand - we set it to undef.
  ops.push_back(UndefValue::get(srcTy));
  ops.push_back(src);
  ops.push_back(indices);
  ops.push_back(avl);

  auto *const gather =
      B.CreateIntrinsic(intrinsicID, {srcTy, avl->getType()}, ops);

  return B.CreateExtractVector(narrowTy, gather, zero);
}

llvm::Value *TargetInfoRISCV::createScalableBroadcast(llvm::IRBuilder<> &B,
                                                      llvm::Value *vector,
                                                      llvm::Value *VL,
                                                      ElementCount factor,
                                                      bool URem) const {
  // Using rvv instruction:
  // vrgather.vv vd, vs2, vs1, vm s.t.
  // vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]

  auto *vectorTy = vector->getType();
  auto *const origElTy = multi_llvm::getVectorElementType(vectorTy);

  // We first check we are not broadcasting a vector of pointers,
  // unsupported by the intrinsic.
  const bool isVectorOfPointers = origElTy->isPtrOrPtrVectorTy();
  if (isVectorOfPointers) {
    vectorTy = VectorType::get(B.getIntNTy(getRISCVBits(getTargetMachine())),
                               multi_llvm::getVectorElementCount(vectorTy));
  }

  auto *const wideTy = ScalableVectorType::get(
      multi_llvm::getVectorElementType(vectorTy),
      factor.getKnownMinValue() *
          multi_llvm::getVectorElementCount(vectorTy).getKnownMinValue());

  Intrinsic::RISCVIntrinsics intrinsicID;
  unsigned vs1Width;
  std::tie(intrinsicID, vs1Width) = getGatherIntrinsic(wideTy);
  auto *const vs1ElTy = B.getIntNTy(vs1Width);

  // We cannot use this optimization if the types are not legal in the target
  // machine.
  if (!isOperationLegal(intrinsicID, {wideTy})) {
    return URem
               ? TargetInfo::createOuterScalableBroadcast(B, vector, VL, factor)
               : TargetInfo::createInnerScalableBroadcast(B, vector, VL,
                                                          factor);
  }

  // Cast the vector of pointers to a vector of integers if needed.
  if (isVectorOfPointers) {
    vector = B.CreatePtrToInt(vector, vectorTy);
  }

  // We grow the fixed vector to consume an entire RVV register.
  auto *const vs2 = B.CreateInsertVector(wideTy, PoisonValue::get(wideTy),
                                         vector, B.getInt64(0), "vs2");

  auto *const vs1 = createBroadcastIndexVector(
      B, VectorType::get(vs1ElTy, wideTy), factor, URem, "vs1");

  auto *const avl = getIntrinsicVL(B, VL, wideTy, getTargetMachine());

  SmallVector<Value *, 4> ops;
  // Add the pass-through operand - we set it to undef.
  ops.push_back(UndefValue::get(vs2->getType()));
  ops.push_back(vs2);
  ops.push_back(vs1);
  ops.push_back(avl);

  Value *gather =
      B.CreateIntrinsic(intrinsicID, {vs2->getType(), avl->getType()}, ops);

  // If we had to cast the vector before, we do the reverse operation
  // on the result.
  if (isVectorOfPointers) {
    gather = B.CreateIntToPtr(gather, VectorType::get(origElTy, wideTy));
  }

  return gather;
}

static CallInst *createRISCVMaskedIntrinsic(IRBuilder<> &B, Intrinsic::ID ID,
                                            ArrayRef<Type *> Types,
                                            ArrayRef<Value *> Args,
                                            unsigned TailPolicy,
                                            Instruction *FMFSource = nullptr,
                                            const Twine &Name = "") {
  SmallVector<Value *> InArgs(Args.begin(), Args.end());
  InArgs.push_back(
      B.getIntN(Args.back()->getType()->getIntegerBitWidth(), TailPolicy));
  return B.CreateIntrinsic(ID, Types, InArgs, FMFSource, Name);
}

llvm::Value *TargetInfoRISCV::createScalableInsertElement(
    llvm::IRBuilder<> &B, vecz::VectorizationContext &Ctx,
    llvm::Instruction *origInsert, llvm::Value *elt, llvm::Value *into,
    llvm::Value *index, llvm::Value *VL) const {
  // In RISCV, we can use vrgather_vv and vrgatherei16_vv to avoid going through
  // memory when creating this operation.
  //   vrgather: vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]];
  // or,
  //   vrgather: res[i] = (idxs[i] >= VLMAX) ? 0 : src[idxs[i]];
  // An example: insertelement <A,B,C,D>, X, I - vectorized by <vscale x 1> -
  // we receive here as packetized arguments:
  //   into: <A,B,C,D, E,F,G,H,    ...> (  <vscale x 4 x ty>    )
  //   elt:  <X,       Y,       Z, ...> (  <vscale x 1 x ty>    )
  //   idxs: <I,       J,       K, ...> (  <vscale x 1 x idxty> )
  // We want to construct operands such that we have:
  //   into: as before
  //   elt:  <X,X,X,X, Y,Y,Y,Y, Z,Z,Z,Z, ... >  ( <vscale x 4 x ty>    )
  //   mask: true where the elts indices are to be inserted according to the
  //         indices, e.g.,
  //         <0,1,0,0, 0,0,0,1,   1,0,0,0, ...  ( <vscale x 4 x i1>    )
  //   idxs: <0,I,0,0, 0,0,0,J+4, K+8,...>      ( <vscale x 4 x idxty> )
  // So that vrgather inserts X into the Ith element of the first 4 elements, Y
  // into the Jth element of the second 4, etc:
  //   res:  <u,X,u,u, u,u,u,Y, Z,u,u,u, ... >
  // If instead we use a masked vrgather with the same mask as before and with
  // a merge operand of 'into', we expect the blended operation to be correct:
  //   res:  <A,X,C,D, E,F,G,Y, Z,I,J,K, ... >
  auto *const eltTy = elt->getType();
  auto *const intoTy = into->getType();

  Intrinsic::ID intrinsicID;
  unsigned intrIdxBitWidth;
  std::tie(intrinsicID, intrIdxBitWidth) =
      getGatherIntrinsic(intoTy, /*isMasked*/ true);

  const auto eltEC = multi_llvm::getVectorElementCount(eltTy);
  const auto intoEC = multi_llvm::getVectorElementCount(intoTy);
  const auto fixedAmt =
      multi_llvm::getVectorElementCount(origInsert->getType());
  assert(!fixedAmt.isScalable() && "Scalable pre-packetized value?");

  auto *indexEltTy = B.getIntNTy(intrIdxBitWidth);
  Type *const indexVecTy = VectorType::get(indexEltTy, eltEC);

  // We cannot use this optimization if the types are not legal in the target
  // machine.
  if (!isOperationLegal(intrinsicID, {intoTy})) {
    return TargetInfo::createScalableInsertElement(B, Ctx, origInsert, elt,
                                                   into, index, VL);
  }

  auto *const avl = getIntrinsicVL(B, VL, intoTy, getTargetMachine());

  auto *const indexTy = index->getType();
  const unsigned idxBitWidth = indexTy->getScalarSizeInBits();
  const bool indexIsVector = indexTy->isVectorTy();

  // The intrinsic may demand a larger index type than we currently have;
  // extend up to the right type.
  if (idxBitWidth != intrIdxBitWidth) {
    index = B.CreateZExtOrTrunc(index, indexIsVector ? indexVecTy : indexEltTy);
  }

  // If the index is uniform, it may not be a vector. We need one for the
  // intrinsic, so splat it here.
  if (!indexIsVector) {
    index = B.CreateVectorSplat(intoEC, index);
  } else {
    index = createInnerScalableBroadcast(B, index, VL, fixedAmt);
  }

  auto *const zero = B.getInt64(0);

  auto *const intrEltTy =
      VectorType::get(multi_llvm::getVectorElementType(elt->getType()), intoEC);
  elt = B.CreateInsertVector(intrEltTy, PoisonValue::get(intrEltTy), elt, zero,
                             "vs2");

  auto *steps = B.CreateStepVector(VectorType::get(indexEltTy, intoEC));

  // Create our inner indices, e.g.: <0,1,2,3, 0,1,2,3, 0,1,2,3, ... >
  auto *const innerIndices = B.CreateURem(
      steps,
      ConstantVector::getSplat(
          intoEC, ConstantInt::get(indexEltTy, fixedAmt.getFixedValue())));

  // Create our outer indices, e.g., <0,0,0,0,1,1,1,1,2,2,2,2,...>
  auto *const outerIndices = B.CreateUDiv(
      steps,
      ConstantVector::getSplat(
          intoEC, ConstantInt::get(indexEltTy, fixedAmt.getFixedValue())));

  // Now compare the insert indices with the inner index vector: only one per
  // N-element slice will be 'on', depending on the exact indices, e.g., if we
  // originally have:
  //    <1,3,0, ...>
  // we have prepared it when constructing the indices:
  //    <1,1,1,1, 3,3,3,3, 0,0,0,0, ...>
  // == <0,1,2,3, 0,1,2,3, 0,1,2,3, ...>
  // -> <0,1,0,0, 0,0,0,1, 1,0,0,0, ...>
  auto *const mask = B.CreateICmpEQ(index, innerIndices, "vm");

  return createRISCVMaskedIntrinsic(B, intrinsicID, {intoTy, avl->getType()},
                                    {into, elt, outerIndices, mask, avl},
                                    /*TailUndisturbed*/ 1);
}

llvm::Value *TargetInfoRISCV::createVectorShuffle(llvm::IRBuilder<> &B,
                                                  llvm::Value *src,
                                                  llvm::Value *mask,
                                                  llvm::Value *VL) const {
  // In RISCV, we can use vrgather_vv and vrgatherei16_vv to avoid going through
  // memory when creating this operation.
  assert(isa<VectorType>(src->getType()) &&
         "TargetInfoRISCV::createVectorShuffle: source must have vector type");
  assert(isa<VectorType>(mask->getType()) &&
         "TargetInfoRISCV::createVectorShuffle: mask must have vector type");

  auto *const srcTy = cast<VectorType>(src->getType());
  if (isa<Constant>(mask)) {
    // Special case if the mask happens to be a constant.
    return B.CreateShuffleVector(src, UndefValue::get(srcTy), mask);
  }

  if (isa<FixedVectorType>(srcTy)) {
    // The gather intrinsics don't work with fixed vectors.
    return TargetInfo::createVectorShuffle(B, src, mask, VL);
  }

  auto *const maskTy = cast<VectorType>(mask->getType());
  const auto srcEC = multi_llvm::getVectorElementCount(srcTy);
  const auto resEC = multi_llvm::getVectorElementCount(maskTy);

  auto *const resTy = VectorType::get(srcTy->getElementType(), resEC);

  // We can't create the intrinsics with a scalar size smaller than 8 bits, so
  // extend it to i8, perform the shuffle, and truncate the result back.
  if (srcTy->getScalarSizeInBits() < 8) {
    auto *const fix = B.CreateZExt(src, VectorType::get(B.getInt8Ty(), srcEC));
    auto *const res = createVectorShuffle(B, fix, mask, VL);
    return B.CreateTrunc(res, resTy);
  }

  Intrinsic::ID intrinsicID;
  unsigned intrIdxBitWidth;
  std::tie(intrinsicID, intrIdxBitWidth) = getGatherIntrinsic(srcTy);

  auto *const indexEltTy = B.getIntNTy(intrIdxBitWidth);
  auto *const indexVecTy = VectorType::get(indexEltTy, resEC);

  // We cannot use this optimization if the types are not legal in the target
  // machine.
  if (!isOperationLegal(intrinsicID, {srcTy})) {
    return TargetInfo::createVectorShuffle(B, src, mask, VL);
  }

  // The intrinsic may demand a larger index type than we currently have;
  // extend up to the right type.
  if (indexVecTy != maskTy) {
    mask = B.CreateZExtOrTrunc(mask, indexVecTy);
  }

  auto *const zero = B.getInt64(0);

  const bool same = (resEC == srcEC);
  const bool narrow = !same && (srcEC.isScalable() || !resEC.isScalable()) &&
                      resEC.getKnownMinValue() <= srcEC.getKnownMinValue();
  const bool widen = !same && (resEC.isScalable() || !srcEC.isScalable()) &&
                     srcEC.getKnownMinValue() <= resEC.getKnownMinValue();

  assert((srcTy == resTy || narrow || widen) &&
         "TargetInfoRISCV::createVectorShuffle: "
         "unexpected combination of source and mask vector types");

  auto *gatherTy = resTy;
  if (narrow) {
    // The vrgather intrinsics need equally-sized vector types. So
    // insert the indices into a wide dummy vector (e.g., <vscale x 4 x idxTy>),
    // perform the vrgather, and extract the subvector back out again.
    auto *const wideMaskTy = VectorType::get(indexEltTy, srcEC);
    mask = B.CreateInsertVector(wideMaskTy, PoisonValue::get(wideMaskTy), mask,
                                zero);
    gatherTy = srcTy;
  } else if (widen) {
    // The result is wider than the source, so insert the source vector into a
    // wider vector first.
    src = B.CreateInsertVector(resTy, PoisonValue::get(resTy), src, zero);
  }

  auto *const avl = getIntrinsicVL(B, VL, gatherTy, getTargetMachine());

  SmallVector<Value *, 4> ops;
  // Add the pass-through operand - we set it to undef.
  ops.push_back(UndefValue::get(gatherTy));
  ops.push_back(src);
  ops.push_back(mask);
  ops.push_back(avl);

  auto *const gather =
      B.CreateIntrinsic(intrinsicID, {gatherTy, avl->getType()}, ops);

  if (narrow) {
    return B.CreateExtractVector(resTy, gather, zero);
  }
  return gather;
}

llvm::Value *TargetInfoRISCV::createVectorSlideUp(llvm::IRBuilder<> &B,
                                                  llvm::Value *src,
                                                  llvm::Value *insert,
                                                  llvm::Value *VL) const {
  auto *const srcTy = dyn_cast<VectorType>(src->getType());
  assert(srcTy &&
         "TargetInfo::createVectorShuffle: source must have vector type");

  if (isa<FixedVectorType>(srcTy)) {
    // The slide1up intrinsics don't work with fixed vectors.
    return TargetInfo::createVectorSlideUp(B, src, insert, VL);
  }

  const auto intrinsicID = getSlideUpIntrinsic(srcTy);

  auto *const avl = getIntrinsicVL(B, VL, srcTy, getTargetMachine());

  SmallVector<Value *, 4> ops;
  // Add the pass-through operand - we set it to undef.
  ops.push_back(UndefValue::get(srcTy));
  ops.push_back(src);
  ops.push_back(insert);
  ops.push_back(avl);

  return B.CreateIntrinsic(intrinsicID,
                           {srcTy, insert->getType(), avl->getType()}, ops);
}

// This enum was copy/pasted from the RISCV backend
enum VLMUL : uint8_t {
  LMUL_1 = 0,
  LMUL_2,
  LMUL_4,
  LMUL_8,
  LMUL_RESERVED,
  LMUL_F8,
  LMUL_F4,
  LMUL_F2
};

Value *TargetInfoRISCV::createVPKernelWidth(IRBuilder<> &B,
                                            Value *RemainingIters,
                                            unsigned WidestEltTy,
                                            ElementCount VF) const {
  // The widest element type can only be one of the supported legal RVV vector
  // element types.
  if (WidestEltTy < 8 || WidestEltTy > 64 || !isPowerOf2_32(WidestEltTy)) {
    return nullptr;
  }
  const auto KnownMin = VF.getKnownMinValue();
  // The vectorization factor must be scalable and a legal vsetvli amount: no
  // greater than the maximum vector length for each element width:
  // nx64vi8,nx32vi16,nx16vi32,nxv8i64
  if (!VF.isScalable() || !isPowerOf2_32(KnownMin) ||
      KnownMin > MaxLegalVectorTypeBits / WidestEltTy) {
    return nullptr;
  }

  unsigned LMUL = 0;
  const unsigned MaxLegalElementWidth = 64;

  if ((WidestEltTy * KnownMin) / MaxLegalElementWidth) {
    // Non-fractional LMULs
    LMUL = Log2_64((WidestEltTy * KnownMin) / MaxLegalElementWidth);
  } else {
    // Fractional LMULs
    const auto Fraction = MaxLegalElementWidth / (WidestEltTy * KnownMin);
    if (Fraction == 2) {
      LMUL = LMUL_F2;
    } else if (Fraction == 4) {
      LMUL = LMUL_F4;
    } else if (Fraction == 8) {
      LMUL = LMUL_F4;
    } else {
      return nullptr;
    }
  }

  auto *const VLMul = B.getInt64(LMUL);
  auto *const VSEW = B.getInt64(Log2_64(WidestEltTy) - 3);

  auto *const I32Ty = Type::getInt32Ty(B.getContext());
  auto *const I64Ty = Type::getInt64Ty(B.getContext());

  auto *const VL =
#if LLVM_VERSION_GREATER_EQUAL(17, 0)
      B.CreateIntrinsic(Intrinsic::RISCVIntrinsics::riscv_vsetvli, {I64Ty},
#else
      B.CreateIntrinsic(Intrinsic::RISCVIntrinsics::riscv_vsetvli_opt, {I64Ty},
#endif
                        {RemainingIters, VSEW, VLMul});

  return B.CreateTrunc(VL, I32Ty);
}
