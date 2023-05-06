//===- MatrixTest.cpp - Tests for AccessMatrix ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::polygeist;

static void loadDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<affine::AffineDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
}

static OpBuilder getBuilder(MLIRContext &ctx) {
  loadDialects(ctx);
  OpBuilder builder(&ctx);
  return builder;
}

TEST(MatrixTest, RowAndColumnSize) {
  MemoryAccessMatrix matrix(2, 3);
  EXPECT_EQ(matrix.getNumRows(), 2u);
  EXPECT_EQ(matrix.getNumColumns(), 3u);
}

TEST(MatrixTest, Init) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);

  // clang-format off
  MemoryAccessMatrix matrix({{zero, zero}, 
                             {zero, zero}});
  // clang-format on
  EXPECT_EQ(matrix.getNumRows(), 2u);
  EXPECT_EQ(matrix.getNumColumns(), 2u);
}

TEST(MatrixTest, ReadWrite) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value ten = builder.create<arith::ConstantIntOp>(loc, 10, 32);

  MemoryAccessMatrix matrix(2, 3);
  matrix(0, 0) = ten;
  EXPECT_EQ(matrix(0, 0), ten);
}

TEST(MatrixTest, SwapRows) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  MemoryAccessMatrix matrix(5, 5);
  for (size_t row = 0; row < 5; ++row)
    for (size_t col = 0; col < 5; ++col)
      matrix(row, col) = row == 2 ? one : zero;

  matrix.swapRows(2, 0);
  for (size_t row = 0; row < 5; ++row)
    for (size_t col = 0; col < 5; ++col)
      EXPECT_EQ(matrix(row, col), row == 0 ? one : zero);
}

TEST(MatrixTest, SwapColumns) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  MemoryAccessMatrix matrix(5, 5);
  for (size_t row = 0; row < 5; ++row)
    for (size_t col = 0; col < 5; ++col)
      matrix(row, col) = col == 3 ? one : zero;

  matrix.swapColumns(3, 1);

  for (size_t row = 0; row < 5; ++row)
    for (size_t col = 0; col < 5; ++col)
      EXPECT_EQ(matrix(row, col), col == 1 ? one : zero);
}

TEST(MatrixTest, SetGetFillRow) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  MemoryAccessMatrix matrix(2, 5);
  matrix.fillRow(0, zero);
  matrix.fillRow(1, one);

  SmallVector<Value> row = matrix.getRow(1);
  EXPECT_EQ(row.size(), (size_t)5);

  matrix.setRow(0, row);

  for (size_t row = 0; row < 2; ++row)
    for (size_t col = 0; col < 5; ++col)
      EXPECT_EQ(matrix(row, col), one);
}

TEST(MatrixTest, AppendRow) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  MemoryAccessMatrix matrix(1, 5);
  matrix.fillRow(0, one);
  SmallVector<Value> elems = matrix.getRow(0);
  matrix.appendRow(elems);
  EXPECT_EQ(matrix.getNumRows(), 2u);

  for (size_t row = 0; row < 2; ++row)
    for (size_t col = 0; col < 5; ++col)
      EXPECT_EQ(matrix(row, col), one);
}

TEST(MatrixTest, GetSetColumn) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  MemoryAccessMatrix matrix(2, 2);
  matrix.fillRow(0, zero);
  matrix.fillRow(1, one);

  SmallVector<Value> column = matrix.getColumn(1);
  EXPECT_EQ(column[0], zero);
  EXPECT_EQ(column[1], one);

  column[0] = one;
  matrix.setColumn(1, column);

  for (size_t row = 0; row < 2; ++row)
    EXPECT_EQ(matrix(row, 1), one);
}

TEST(MatrixTest, SubMatrix) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);

  MemoryAccessMatrix matrix(3, 3);
  matrix.fillRow(0, zero);
  matrix.fillRow(1, one);
  matrix.fillRow(2, two);

  {
    MemoryAccessMatrix subMatrix = matrix.getRows({0, 2});
    EXPECT_EQ(subMatrix.getNumRows(), 2u);
    EXPECT_EQ(subMatrix.getNumColumns(), 3u);
    for (size_t col = 0; col < 3; ++col) {
      EXPECT_EQ(subMatrix(0, col), zero);
      EXPECT_EQ(subMatrix(1, col), two);
    }
  }

  {
    matrix.fillColumn(0, one);
    matrix.fillColumn(2, two);
    MemoryAccessMatrix subMatrix = matrix.getColumns({2, 0});
    EXPECT_EQ(subMatrix.getNumRows(), 3u);
    EXPECT_EQ(subMatrix.getNumColumns(), 2u);
    for (size_t row = 0; row < 3; ++row) {
      EXPECT_EQ(subMatrix(row, 0), one);
      EXPECT_EQ(subMatrix(row, 1), two);
    }
  }

  {
    MemoryAccessMatrix subMatrix = matrix.getSubMatrix({0, 1}, {0, 2});
    EXPECT_EQ(subMatrix.getNumRows(), 2u);
    EXPECT_EQ(subMatrix.getNumColumns(), 2u);
    for (size_t row = 0; row < 2; ++row) {
      EXPECT_EQ(subMatrix(row, 0), one);
      EXPECT_EQ(subMatrix(row, 1), two);
    }
  }
}

TEST(MatrixTest, Shapes) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();

  auto funcTy = FunctionType::get(builder.getContext(), {}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "test", funcTy);
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  builder.create<func::ReturnOp>(loc);

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(funcOp)))
    assert(false);

  {
    // Create the zero matrix.
    MemoryAccessMatrix matrix(3, 3);
    matrix.fill(zero);

    EXPECT_THAT(matrix.isSquare(), true);
    EXPECT_THAT(matrix.isZero(solver), true);
    EXPECT_THAT(matrix.isIdentity(solver), false);
  }

  {
    // Create the identity matrix.
    MemoryAccessMatrix matrix(3, 3);
    matrix.fill(zero);
    for (size_t row = 0; row < 3; ++row)
      matrix(row, row) = one;

    EXPECT_THAT(matrix.isZero(solver), false);
    EXPECT_THAT(matrix.isDiagonal(solver), true);
    EXPECT_THAT(matrix.isIdentity(solver), true);
    EXPECT_THAT(matrix.isLowerTriangular(solver), false);
    EXPECT_THAT(matrix.isUpperTriangular(solver), false);
  }

  {
    // clang-format off
    MemoryAccessMatrix matrix(
        {{one,  zero, zero}, 
         {zero, one,  zero}, 
         {zero, zero, two}});
    // clang-format on

    EXPECT_THAT(matrix.isDiagonal(solver), true);
    EXPECT_THAT(matrix.isIdentity(solver), false);
  }

  {
    // clang-format off
    MemoryAccessMatrix matrix(
      {{one, zero, zero}, 
       {two, two,  zero}, 
       {two, two,  one}});
    // clang-format on

    EXPECT_THAT(matrix.isZero(solver), false);
    EXPECT_THAT(matrix.isDiagonal(solver), false);
    EXPECT_THAT(matrix.isIdentity(solver), false);
    EXPECT_THAT(matrix.isLowerTriangular(solver), true);
    EXPECT_THAT(matrix.isUpperTriangular(solver), false);
  }

  {
    // clang-format off
    MemoryAccessMatrix matrix(
      {{one,  two,  two}, 
       {zero, two,  two}, 
       {zero, zero, one}});
    // clang-format on

    EXPECT_THAT(matrix.isZero(solver), false);
    EXPECT_THAT(matrix.isDiagonal(solver), false);
    EXPECT_THAT(matrix.isIdentity(solver), false);
    EXPECT_THAT(matrix.isLowerTriangular(solver), false);
    EXPECT_THAT(matrix.isUpperTriangular(solver), true);
  }
}

TEST(MatrixTest, PatternClassification) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();

  auto funcTy = FunctionType::get(builder.getContext(), {}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "test", funcTy);
  builder.setInsertionPointToStart(funcOp.addEntryBlock());

  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value negativeOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value negativeTwo = builder.create<arith::ConstantIntOp>(loc, -2, 32);
  Value two = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  auto bufferTy = MemRefType::get({}, builder.getI32Type());
  Value memref = builder.create<memref::AllocaOp>(loc, bufferTy);
  auto loadOp = builder.create<affine::AffineLoadOp>(loc, memref);
  builder.create<func::ReturnOp>(loc);

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::IntegerRangeAnalysis>();
  if (failed(solver.initializeAndRun(funcOp)))
    assert(false);

  // clang-format off
  MemoryAccessMatrix identityMatrix(
      {{one,  zero, zero}, 
       {zero, one,  zero}, 
       {zero, zero, one }});
  // clang-format on
  EXPECT_THAT(identityMatrix.isIdentity(solver), true);

  OffsetVector zeroOffsets(3);
  zeroOffsets.fill(zero);
  EXPECT_THAT(zeroOffsets.isZero(solver), true);

  // Test linear access pattern.
  {
    MemoryAccessMatrix linearAccess(identityMatrix);
    EXPECT_THAT(linearAccess.hasLinearAccessPattern(solver), true);

    OffsetVector offsets(zeroOffsets);
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(linearAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::Linear);
  }

  // Test reverse linear access pattern.
  {
    // clang-format off
    MemoryAccessMatrix reverseAccess(
        {{one,  zero, zero}, 
         {zero, one,  zero}, 
         {zero, zero, negativeOne}});
    // clang-format on
    EXPECT_THAT(reverseAccess.hasReverseLinearAccessPattern(solver), true);

    OffsetVector offsets({zero, zero, two});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(reverseAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::ReverseLinear);
  }

  // Test liner overlapped access pattern.
  {
    // clang-format off
    MemoryAccessMatrix linearOverlappedAccess(
        {{one, zero, zero}, 
         {one, one,  zero}, 
         {one, one,  one }});
    // clang-format on
    EXPECT_THAT(linearOverlappedAccess.hasLinearOverlappedAccessPattern(solver),
                true);

    OffsetVector offsets(zeroOffsets);
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(linearOverlappedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::LinearOverlapped);
  }

  // Test strided access pattern.
  {
    // clang-format off
    MemoryAccessMatrix stridedAccess(
        {{one,  zero, zero}, 
         {zero, one,  zero}, 
         {zero, zero, two }});
    // clang-format on
    EXPECT_THAT(stridedAccess.hasStridedAccessPattern(solver), true);

    OffsetVector offsets(zeroOffsets);
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(stridedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::Strided);
  }

  // Test strided overlapped access pattern.
  {
    // clang-format off
    MemoryAccessMatrix stridedOverlappedAccess(
        {{one, zero, zero}, 
         {one, one,  zero}, 
         {one, one,  two }});
    // clang-format on
    EXPECT_THAT(
        stridedOverlappedAccess.hasStridedOverlappedAccessPattern(solver),
        true);

    OffsetVector offsets(zeroOffsets);
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(stridedOverlappedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::StridedOverlapped);
  }

  // Test linear shifted access pattern.
  {
    MemoryAccessMatrix linearAccess(identityMatrix);
    EXPECT_THAT(linearAccess.hasLinearAccessPattern(solver), true);

    OffsetVector offsets({zero, zero, two});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(linearAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::LinearShifted);
  }

  // Test reverse linear shifted access pattern.
  {
    // clang-format off
    MemoryAccessMatrix reverseAccess(
        {{one,  zero, zero}, 
         {zero, one,  zero}, 
         {zero, zero, negativeOne}});
    // clang-format on
    EXPECT_THAT(reverseAccess.hasReverseLinearAccessPattern(solver), true);

    OffsetVector offsets({zero, zero, one});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(reverseAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::ReverseLinearShifted);
  }

  // Test reverse linear overlapped access pattern.
  {
    // clang-format off
    MemoryAccessMatrix reverseLinearOverlappedAccess(
        {{one, zero, zero}, 
         {one, one,  zero}, 
         {one, one,  negativeOne}});
    // clang-format on
    EXPECT_THAT(
        reverseLinearOverlappedAccess.hasReverseLinearOverlappedAccessPattern(
            solver),
        true);

    OffsetVector offsets({zero, zero, one});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(reverseLinearOverlappedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::ReverseLinearOverlapped);
  }

  // Test reverse strided access pattern.
  {
    // clang-format off
    MemoryAccessMatrix reverseStridedAccess(
        {{one,  zero, zero},
         {zero, one,  zero},
         {zero, zero, negativeTwo}});
    // clang-format on
    EXPECT_THAT(reverseStridedAccess.hasReverseStridedAccessPattern(solver),
                true);

    OffsetVector offsets({zero, zero, two});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(reverseStridedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::ReverseStrided);
  }

  // Test strided shifted access pattern.
  {
    // clang-format off
    MemoryAccessMatrix stridedShiftedAccess(
        {{one,  zero, zero},
         {zero, one,  zero},
         {zero, zero, two}});
    // clang-format on
    EXPECT_THAT(stridedShiftedAccess.hasStridedAccessPattern(solver), true);

    OffsetVector offsets({zero, zero, one});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(stridedShiftedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::StridedShifted);
  }

  // Test reverse strided shifted access pattern.
  {
    // clang-format off
    MemoryAccessMatrix reverseStridedShiftedAccess(
        {{one,  zero, zero},
         {zero, one,  zero},
         {zero, zero, negativeTwo}});
    // clang-format on
    EXPECT_THAT(
        reverseStridedShiftedAccess.hasReverseStridedAccessPattern(solver),
        true);

    OffsetVector offsets({zero, zero, one});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(reverseStridedShiftedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::ReverseStridedShifted);
  }

  // Test reverse strided overlapped access pattern.
  {
    // clang-format off
    MemoryAccessMatrix reverseStridedOverlappedAccess(
        {{one, zero, zero},
         {one, one,  zero},
         {one, one,  negativeTwo}});
    // clang-format on
    EXPECT_THAT(
        reverseStridedOverlappedAccess.hasReverseStridedOverlappedAccessPattern(
            solver),
        true);

    OffsetVector offsets({zero, zero, one});
    MemoryAccess<affine::AffineLoadOp> memoryAccess(
        loadOp, std::move(reverseStridedOverlappedAccess), std::move(offsets));
    EXPECT_THAT(memoryAccess.classifyMemoryAccess(solver),
                MemoryAccessPattern::ReverseStridedOverlapped);
  }
}
