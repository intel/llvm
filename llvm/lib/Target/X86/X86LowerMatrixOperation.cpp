//===- Target/X86/X86LowerMatrixOperation.cpp - -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file defines the pass which transforms the following matrix
/// operations function call into amx's intrinsics:
/// llvm.experimental.matrix.store => llvm.x86.tilestored64.internal
/// llvm.experimental.matrix.load => llvm.x86.tileloadd64.internal
/// llvm.experimental.matrix.mad => llvm.x86.tdpbssd(or tdpbf16ps).internal
///
/// Example:
/// Calculate the formatted size according to the real size, layout&type(i8 or
/// i16).
/// %val = call void @llvm.experimental.matrix.store.v4i8.p4i8(<8 x i8> %src,
/// i32* ptr, i64 %stride, i1 false, i32 4, i32 2, metadata !"matrix.rowmajor",
/// metadata !"matrix.rowmajor", metadata !"scope.subgroup").
/// =>
/// %amxsrc = call x86_amx @llvm.x86.cast.vector.to.tile.v4i8(<4 x i8> %src)
/// %val = call x86_amx @llvm.x86.tilestored64.internal(i32 1, i32 8, i8* ptr,
/// i64 stride,x86_amx %amxsrc)
//
//===----------------------------------------------------------------------===//
//

#include "X86.h"
#include "X86TargetMachine.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "x86-lower-matrix-operation"

namespace {

class X86LowerMatrixOperationPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid..

  X86LowerMatrixOperationPass() : FunctionPass(ID) {
    initializeX86LowerMatrixOperationPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    bool MadeChange = false;
    // Put matrix operations into worklist
    SmallVector<Instruction *, 8> Worklist;
    for (BasicBlock *BB : depth_first(&F)) {
      for (BasicBlock::iterator BBI = BB->begin(), BBIE = BB->end();
           BBI != BBIE; ++BBI) {
        IntrinsicInst *II = dyn_cast<IntrinsicInst>(&*BBI);
        if (II) {
          switch (II->getIntrinsicID()) {
          default:
            break;
          case Intrinsic::experimental_matrix_load:
          case Intrinsic::experimental_matrix_store:
          case Intrinsic::experimental_matrix_mad:
            Worklist.push_back(&*BBI);
            break;
          }
        }
      }
    }
    // Process matrix operations int the worklist
    for (auto It : Worklist) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(&*It);
      assert(II && "Not a valiad intrinsic");
      MadeChange |= ProcessMatrixOperation(II);
    }
    return MadeChange;
  }

private:
  bool ProcessMatrixOperation(IntrinsicInst *II);
  bool ProcessMatrixLoad(IntrinsicInst *II);
  bool ProcessMatrixStore(IntrinsicInst *II);
  bool ProcessMatrixMad(IntrinsicInst *II);
};

} // end anonymous namespace

char X86LowerMatrixOperationPass::ID = 0;

INITIALIZE_PASS_BEGIN(X86LowerMatrixOperationPass, DEBUG_TYPE,
                      "X86 transform matrix operations to amx intrinsics pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(X86LowerMatrixOperationPass, DEBUG_TYPE,
                    "X86 transform matrix operations to amx intrinsics pass",
                    false, false)

FunctionPass *llvm::createX86LowerMatrixOperationPass() {
  return new X86LowerMatrixOperationPass();
}

bool X86LowerMatrixOperationPass::ProcessMatrixOperation(IntrinsicInst *II) {
  bool MadeChange = false;
  switch (II->getIntrinsicID()) {
  default:
    break;
  case Intrinsic::experimental_matrix_load:
    MadeChange |= ProcessMatrixLoad(II);
    break;
  case Intrinsic::experimental_matrix_store:
    MadeChange |= ProcessMatrixStore(II);
    break;
  case Intrinsic::experimental_matrix_mad:
    MadeChange |= ProcessMatrixMad(II);
    break;
  }
  return MadeChange;
}

bool X86LowerMatrixOperationPass::ProcessMatrixLoad(IntrinsicInst *II) {
  // Calculate the formatted size according to the real size, layout&type(i8 or
  // i16).
  // %res = call <8 x i8> @llvm.experimental.matrix.load.v8i8.p4i8(
  //   i32* addressspace(4) ptr, i64 stride, i1 false, i32 4, i32 2,
  //   metadata !"matrix.packed_b", metadata !"scope.subgroup")
  // =>
  // %val = call x86_amx @llvm.x86.tileloadd64.internal(i32 1, i32 8, i8* ptr,
  // i64 stride).
  // %res = call x86_amx @llvm.x86.cast.tile.to.vector.v4i8(<4 x i8> %val)
  IRBuilder<> Builder(II);
  int64_t MRows = cast<ConstantInt>(II->getOperand(3))->getSExtValue();
  int64_t MCols = cast<ConstantInt>(II->getOperand(4))->getSExtValue();
  FixedVectorType *MatrixType = cast<FixedVectorType>(II->getType());
  Type *MatrixElemType = MatrixType->getElementType();
  int64_t Factor = 1;
  int64_t SizeFactor = 1;
  if (MatrixElemType->isIntegerTy(16))
    SizeFactor = 2;
  else if (MatrixElemType->isFloatTy() || MatrixElemType->isIntegerTy(32))
    SizeFactor = 4;
  else if (MatrixElemType->isIntegerTy(8))
    SizeFactor = 1;
  else
    assert(false && "Unsupported Type!");
  Metadata *MDLayout = cast<MetadataAsValue>(II->getOperand(5))->getMetadata();
  // If it is packed_b/packed_a, the type can only be int8/bf16.
  // If it is row_major, the type can be int8/bf16/float/int32, Factor can only
  // be 1.
  if (cast<MDString>(MDLayout)->getString().equals("matrix.packed.b") &&
      MatrixElemType->isIntegerTy(8))
    Factor = 4;
  else if (cast<MDString>(MDLayout)->getString().equals("matrix.packed.b") &&
           MatrixElemType->isIntegerTy(16))
    Factor = 2;
  else if (cast<MDString>(MDLayout)->getString().equals("matrix.packed.a") ||
           cast<MDString>(MDLayout)->getString().equals("matrix.rowmajor"))
    Factor = 1;
  else
    assert(false && "Unsupported Layout!");
  // Handle cases where it is vxi8 and packedb.
  assert(MRows >= Factor && MRows % Factor == 0 &&
         "Invalid Matrix Rows Value!");
  Value *Rows = Builder.getInt16(MRows / Factor);
  Value *Cols = Builder.getInt16(MCols * Factor * SizeFactor);
  Value *Ptr = II->getOperand(0)->getType()->getPointerAddressSpace() == 0
                   ? Builder.CreateBitCast(
                         II->getOperand(0),
                         llvm::Type::getInt8PtrTy(Builder.getContext()))
                   : Builder.CreateAddrSpaceCast(
                         II->getOperand(0),
                         llvm::Type::getInt8PtrTy(Builder.getContext()));
  // Create the argument list
  std::array<Value *, 4> Args{
      Rows, Cols, Ptr,
      Builder.CreateMul(II->getOperand(1), Builder.getInt64(SizeFactor))};
  Value *NewInst =
      Builder.CreateIntrinsic(Intrinsic::x86_tileloadd64_internal, None, Args);
  II->replaceAllUsesWith(Builder.CreateIntrinsic(
      Intrinsic::x86_cast_tile_to_vector, {MatrixType}, {NewInst}));
  II->eraseFromParent();
  return true;
}

bool X86LowerMatrixOperationPass::ProcessMatrixStore(IntrinsicInst *II) {
  // Calculate the formatted size according to the real size, layout&type(i8 or
  // i16).
  // %val = call void @llvm.experimental.matrix.store.v4i8.p4i8(<8 x i8> %src,
  // i32* ptr, i64 %stride, i1 false, i32 4, i32 2, metadata !"matrix.rowmajor",
  // metadata !"scope.subgroup").
  // =>
  // %amxsrc = call x86_amx @llvm.x86.cast.vector.to.tile.v4i8(<4 x i8> %src)
  // %val = call x86_amx @llvm.x86.tilestored64.internal(i32 1, i32 8, i8* ptr,
  // i64 stride, x86_amx %amxsrc).
  IRBuilder<> Builder(II);
  int64_t MRows = cast<ConstantInt>(II->getOperand(4))->getSExtValue();
  int64_t MCols = cast<ConstantInt>(II->getOperand(5))->getSExtValue();
  FixedVectorType *MatrixType =
      cast<FixedVectorType>(II->getOperand(0)->getType());
  Type *MatrixElemType = MatrixType->getElementType();
  int64_t Factor = 1;
  int64_t SizeFactor = 1;
  // FIXME: SizeFactor = MatrixElemType->getScalarSizeInBits()/8?
  if (MatrixElemType->isIntegerTy(16))
    SizeFactor = 2;
  else if (MatrixElemType->isFloatTy() || MatrixElemType->isIntegerTy(32))
    SizeFactor = 4;
  else if (MatrixElemType->isIntegerTy(8))
    SizeFactor = 1;
  else
    assert(false && "Unsupported Type!");
  Metadata *MDLayout = cast<MetadataAsValue>(II->getOperand(6))->getMetadata();
  // If it is wordpackedb/wordpackeda, the type can only be int8/bf16.
  // If it is row_major, the type can be int8/bf16/float/int32.
  if (cast<MDString>(MDLayout)->getString().equals("matrix.packed.b") &&
      MatrixElemType->isIntegerTy(8))
    Factor = 4;
  else if (cast<MDString>(MDLayout)->getString().equals("matrix.packed.b") &&
           MatrixElemType->isIntegerTy(16))
    Factor = 2;
  else if (cast<MDString>(MDLayout)->getString().equals("matrix.packed.a") ||
           cast<MDString>(MDLayout)->getString().equals("matrix.rowmajor"))
    Factor = 1;
  else
    assert(false && "Unsupported Layout!");
  assert(MRows >= Factor && MRows % Factor == 0 &&
         "Invalid Matrix Rows Value!");
  Value *Rows = Builder.getInt16(MRows / Factor);
  Value *Cols = Builder.getInt16(MCols * Factor * SizeFactor);
  Value *Ptr = II->getOperand(1)->getType()->getPointerAddressSpace() == 0
                   ? Builder.CreateBitCast(
                         II->getOperand(1),
                         llvm::Type::getInt8PtrTy(Builder.getContext()))
                   : Builder.CreateAddrSpaceCast(
                         II->getOperand(1),
                         llvm::Type::getInt8PtrTy(Builder.getContext()));
  // Create the argument list
  std::array<Value *, 5> Args{
      Rows, Cols, Ptr,
      Builder.CreateMul(II->getOperand(2), Builder.getInt64(SizeFactor)),
      Builder.CreateIntrinsic(Intrinsic::x86_cast_vector_to_tile,
                              {II->getOperand(0)->getType()},
                              {II->getOperand(0)})};
  Value *NewInst =
      Builder.CreateIntrinsic(Intrinsic::x86_tilestored64_internal, None, Args);
  II->replaceAllUsesWith(NewInst);
  II->eraseFromParent();
  return true;
}

bool X86LowerMatrixOperationPass::ProcessMatrixMad(IntrinsicInst *II) {
  // Transform %mad = call <4 x i8>
  // @llvm.experimental.matrix.mad.v4i8.v8i8.v8i8( <8 x i8> %A, metadata
  // !"matrix.rowmajor", <8 x i8> %B, metadata !"matrix.wordpacked", <4 x i32>
  // %C, metadata !"matrix.rowmajor", i32 2(A.rows), i32 4(B.rows), i32
  // 2(C.cols), metadata !"scope.subgroup").
  // into:
  // %a = call x86_amx @llvm.x86.cast.vector.to.tile.v4i8(<4 x i8> %A).
  // %b = call x86_amx @llvm.x86.cast.vector.to.tile.v4i8(<4 x i8> %B).
  // %c = call x86_amx @llvm.x86.cast.vector.to.tile.v4i8(<4 x i32> %C).
  // %d = call x86_amx @llvm.x86.tdpbssd.internal(i16 2(C.rows), i16 2*4(C.cols
  // in int8), i16 4(A.cols), x86_amx %c, x86_amx %a, x86_amx %b).
  // A.cols = B.rows, C.rows = A.rows, C.cols in int8 = B.rows * 4.
  IRBuilder<> Builder(II);
  FixedVectorType *MatrixType = cast<FixedVectorType>(II->getType());
  Type *MatrixElemType = MatrixType->getElementType();
  Intrinsic::ID IID = MatrixElemType->isFloatTy()
                          ? Intrinsic::x86_tdpbf16ps_internal
                          : Intrinsic::x86_tdpbssd_internal;
  Value *M =
      Builder.getInt16(cast<ConstantInt>(II->getOperand(6))->getSExtValue());
  Value *K =
      Builder.getInt16(cast<ConstantInt>(II->getOperand(7))->getSExtValue() *
                       (MatrixElemType->isFloatTy() ? 2 : 1));
  Value *N = Builder.getInt16(
      cast<ConstantInt>(II->getOperand(8))->getSExtValue() * 4);
  // M=A.rows, N=C.cols*4, K=B.rows,C,A,B
  std::array<Value *, 6> Args{
      M,
      N,
      K,
      Builder.CreateIntrinsic(Intrinsic::x86_cast_vector_to_tile,
                              {II->getOperand(4)->getType()},
                              {II->getOperand(4)}),
      Builder.CreateIntrinsic(Intrinsic::x86_cast_vector_to_tile,
                              {II->getOperand(0)->getType()},
                              {II->getOperand(0)}),
      Builder.CreateIntrinsic(Intrinsic::x86_cast_vector_to_tile,
                              {II->getOperand(2)->getType()},
                              {II->getOperand(2)})};
  Value *NewInst = Builder.CreateIntrinsic(IID, None, Args);
  II->replaceAllUsesWith(Builder.CreateIntrinsic(
      Intrinsic::x86_cast_tile_to_vector, {MatrixType}, {NewInst}));
  II->eraseFromParent();
  return true;
}
