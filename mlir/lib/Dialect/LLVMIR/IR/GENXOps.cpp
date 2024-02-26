//===- GENXOps.cpp - MLIR GENX operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations in the GENX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// genx.conv.fptofp
//===----------------------------------------------------------------------===//

LogicalResult GENX::FpToFpOp::verify() {
  unsigned srcTySizeInBits = getArg().getType().getWidth();
  unsigned resTySizeInBits = getRes().getType().getWidth();
  if (srcTySizeInBits == resTySizeInBits)
    return this->emitOpError(
        "expecting first argument and result size to be different");
  if (!getRoundingMode() && srcTySizeInBits >= resTySizeInBits)
    return this->emitOpError("expecting rounding mode for truncation");
  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.dpas
//===----------------------------------------------------------------------===//

LogicalResult GENX::MatrixDPASOp::verify() {
  if (getRc() != 1 && getRc() != 2 && getRc() != 4 && getRc() != 8)
    return this->emitOpError("expecting repeat count to be 1, 2, 4, or 8");

  GENX::PrecisionType precision = getPa();
  if (getPa() != getPb())
    return this->emitOpError(
        "expecting precision of matrix A and B to be the same");

  VectorType ATy = getA().getType();
  VectorType BTy = getB().getType();
  VectorType CTy = getC().getType();
  VectorType DTy = getD().getType();
  if (CTy != DTy)
    return this->emitOpError(
        "1st operand (C) and result (D) should have the same type");

  if (CTy.getNumElements() != getRc() || DTy.getNumElements() != getRc())
    return this->emitOpError("the dimension for 1st operand (C) and "
                             "result (D) should match repeat count");

  Type AElemTy = ATy.getElementType();
  Type BElemTy = BTy.getElementType();
  Type CElemTy = CTy.getElementType();

  // ATy is required to be vector<RC x i16> as hard coded by IGC.
  if (ATy.getNumElements() * AElemTy.getIntOrFloatBitWidth() != getRc() * 16)
    return this->emitOpError(
        "2nd operand (A) bit-size should be repeat count times 16");

  // BTy is required to be vector<SD x i32> as hard coded by IGC.
  constexpr unsigned SD = 8;
  if (BTy.getNumElements() * BElemTy.getIntOrFloatBitWidth() != SD * 32)
    return this->emitOpError(
        "3rd operand (B) bit-size should be systolic depth (8) times 32");

  if (precision == GENX::PrecisionType::U8 ||
      precision == GENX::PrecisionType::S8) {
    if (!CElemTy.isInteger(32))
      return this->emitOpError("the element type for 1st operand (C) and "
                               "the result should be i32");
  } else if (!CElemTy.isF32())
    return this->emitOpError("the element type for 1st operand (C) and the "
                             "result should be f32");

  switch (precision) {
  case GENX::PrecisionType::TF32:
    if (!AElemTy.isa<Float32Type>() && !AElemTy.isInteger(32))
      return this->emitOpError("A and B operand element type should be f32 or "
                               "i32 when precision type is tf32");
    break;
  case GENX::PrecisionType::BF16:
    if (!AElemTy.isa<BFloat16Type>() && !AElemTy.isInteger(16))
      return this->emitOpError("A and B operand element type should be bf16 or "
                               "i16 when precision type is bf16");
    break;
  case GENX::PrecisionType::FP16:
    if (!AElemTy.isa<Float16Type>() && !AElemTy.isInteger(16))
      return this->emitOpError("A and B operand element type should be f16 or "
                               "i16 when precision type is f16");
    break;
  case GENX::PrecisionType::U8:
    if (!(AElemTy.isInteger(8) && !AElemTy.cast<IntegerType>().isSigned()) &&
        !AElemTy.isInteger(16))
      return this->emitOpError("A and B operand element type should be u8, i8, "
                               "or i16 when precision type is u8");
    break;
  case GENX::PrecisionType::S8:
    if (!(AElemTy.isInteger(8) && !AElemTy.cast<IntegerType>().isUnsigned()) &&
        !AElemTy.isInteger(16))
      return this->emitOpError("A and B operand element type should be s8, i8, "
                               "or i16 when precision type is s8");
    break;
  default:
    return this->emitOpError(
        "expecting precision type to be tf32, bf16, fp16, u8, or s8");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.2Dblockload
//===----------------------------------------------------------------------===//

static std::optional<int> getConstantInt(Value v) {
  Operation *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;

  if (!op->hasTrait<OpTrait::ConstantLike>())
    return std::nullopt;

  llvm::SmallVector<OpFoldResult> folded;
  if (failed(op->fold({}, folded)) || folded.size() != 1)
    return std::nullopt;

  if (!folded.front().is<Attribute>() ||
      !isa<IntegerAttr>(folded.front().get<Attribute>()))
    return std::nullopt;

  return cast<IntegerAttr>(folded.front().get<Attribute>()).getInt();
}

template <typename Op>
static LogicalResult verifyInput(Op op) {
  if (op.getElemSizeInBits() != 8 && op.getElemSizeInBits() != 16 &&
      op.getElemSizeInBits() != 32)
    return op->emitOpError("expecting 'elem_size_in_bits' to be 8, 16, or 32");

  if (op.getTranspose() && op.getVnniTransform())
    return op->emitOpError(
        "transpose and vnni transform are mutually exclusive");

  std::optional<int> width = getConstantInt(op.getBaseWidth());
  std::optional<int> pitch = getConstantInt(op.getBasePitch());
  if (pitch && width && *pitch < *width)
    return op->emitOpError(
        "4th operand (base pitch) should be >= 2nd operand (base width)");

  uint32_t TileWidth = op.getTileWidth();
  uint32_t TileHeight = op.getTileHeight();
  switch (op.getElemSizeInBits()) {
  case 32:
    if (TileWidth != 8)
      return op->emitOpError("tile_width for 32 bit elements should be equal "
                             "to systolic depth, i.e., 8 elements");
    if (TileHeight != 8)
      return op->emitOpError("tile_height for 32 bit elements should be 8");
    break;

  case 16:
    if (TileWidth != 16)
      return op->emitOpError("tile_width for 16 bit elements should be equal "
                             "to systolic depth times 2, i.e., 16 elements");
    if (TileHeight != 16)
      return op->emitOpError("tile_height for 16 bit elements should be 16");
    break;

  case 8:
    if (TileWidth != 32)
      return op->emitOpError("tile_width for 8 bit elements should be equal "
                             "to systolic depth times 4, i.e., 32 elements");
    if (TileHeight != 32)
      return op->emitOpError("tile_height for 8 bit elements should be 32");
    break;

  default:
    return op->emitOpError("element size should be 8, 16 or 32 bits");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// genx.matrix.2Dblockload
//===----------------------------------------------------------------------===//

LogicalResult GENX::Matrix2DBlockLoadOp::verify() { return verifyInput(*this); }

//===----------------------------------------------------------------------===//
// genx.matrix.2Dblockstore
//===----------------------------------------------------------------------===//

LogicalResult GENX::Matrix2DBlockStoreOp::verify() {
  return verifyInput(*this);
}
