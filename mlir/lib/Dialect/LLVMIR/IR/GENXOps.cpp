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

//  if (CTy.getNumElements() != getRc() || DTy.getNumElements() != getRc())
//    return this->emitOpError("the dimension for 1st operand (C) and "
//                             "result (D) should match repeat count");

  Type AElemTy = ATy.getElementType();
  Type BElemTy = BTy.getElementType();
  Type CElemTy = CTy.getElementType();
  if (AElemTy != BElemTy)
    return this->emitOpError(
        "element type of 2nd (A) and 3rd (B) operands must match");

//  // ATy is required to be vector<RC x i16> as hard coded by IGC.
//  if (ATy.getNumElements() * AElemTy.getIntOrFloatBitWidth() != getRc() * 16)
//    return this->emitOpError(
//        "2nd operand (A) bit-size should be repeat count times 16");
//
//  // BTy is required to be vector<SD x i32> as hard coded by IGC.
//  constexpr unsigned SD = 8;
//  if (BTy.getNumElements() * BElemTy.getIntOrFloatBitWidth() != SD * 32)
//    return this->emitOpError(
//        "3rd operand (B) bit-size should be systolic depth (8) times 32");

  return TypeSwitch<Type, LogicalResult>(AElemTy)
      .Case<Float32Type>([&](auto ty) -> LogicalResult {
        if (precision != GENX::PrecisionType::TF32)
          return this->emitOpError("precision should be TF32 when 2nd (A) or "
                                   "3rd (B) operand element type is f32");
        if (!CElemTy.isF32())
          return this->emitOpError("the element type for 1st operand (C) and "
                                   "the result should be f32");
        return success();
      })
      .Case<BFloat16Type>([&](auto ty) -> LogicalResult {
        if (precision != GENX::PrecisionType::BF16)
          return this->emitOpError(
              "precision should be BF16 when 2nd (A) or 3rd (B) operand "
              "element type is bf16");
        if (!CElemTy.isF32())
          return this->emitOpError(
              "the element type for 1st operand (C) and the "
              "result should be f32");
        return success();
      })
      .Case<Float16Type>([&](auto ty) -> LogicalResult {
        if (precision != GENX::PrecisionType::FP16)
          return this->emitOpError("precision should be FP16 when 2nd (A) or "
                                   "3rd (B) operand element type is f16");
        if (!CElemTy.isF32())
          return this->emitOpError(
              "the element type for 1st operand (C) and the "
              "result should be f32");
        return success();
      })
      .Case<IntegerType>([&](auto ty) -> LogicalResult {
//        if (!ty.isInteger(8))
//          return this->emitOpError(
//              "expecting 2nd (A) or 3rd (B) operand element type to be f32, "
//              "bf16, f16, or i8");

        if (precision == GENX::PrecisionType::U8) {
          if (ty.isSigned())
            return this->emitOpError(
                "precision should be S8 when 2nd (A) or 3rd (B) operand "
                "element type is signed i8");
        } else if (precision == GENX::PrecisionType::S8) {
          if (ty.isUnsigned())
            return this->emitOpError(
                "precision should be U8 when 2nd (A) or 3rd (B) operand "
                "element type is unsigned i8");
        } /*else
          return this->emitOpError("precision should be U8 or S8 when 2nd (A) "
                                   "or 3rd (B) operand element type is i8");*/

//        if (!CElemTy.isInteger(32))
//          return this->emitOpError("the element type for 1st operand (C) and "
//                                   "the result should be i32");

        return success();
      })
      .Default([&](mlir::Type) -> LogicalResult {
        return this->emitOpError("expecting 2nd (A) or 3rd (B) operand element "
                                 "type to be f32, bf16, f16, or i8");
      });
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
    return op->emitOpError(
        "expecting 'elem_size_in_bits' to be 8, 16, or 32");

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

LogicalResult GENX::Matrix2DBlockLoadOp::verify() {
  return verifyInput(*this);
}

//===----------------------------------------------------------------------===//
// genx.matrix.2Dblockstore
//===----------------------------------------------------------------------===//

LogicalResult GENX::Matrix2DBlockStoreOp::verify() {
  return verifyInput(*this);
}
