//===- MatrixTest.cpp - Tests for AccessMatrix ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/Analysis/MemoryAccessAnalysis.h"
#include "mlir/IR/BuiltinTypes.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::sycl;

static void loadDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<arith::ArithDialect>();
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
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      matrix(row, col) = row == 2 ? one : zero;

  matrix.swapRows(2, 0);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      EXPECT_EQ(matrix(row, col), row == 0 ? one : zero);
}

TEST(MatrixTest, SwapColumns) {
  MLIRContext ctx;
  OpBuilder builder = getBuilder(ctx);
  Location loc = builder.getUnknownLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

  MemoryAccessMatrix matrix(5, 5);
  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
      matrix(row, col) = col == 3 ? one : zero;

  matrix.swapColumns(3, 1);

  for (unsigned row = 0; row < 5; ++row)
    for (unsigned col = 0; col < 5; ++col)
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

  for (unsigned row = 0; row < 2; ++row)
    for (unsigned col = 0; col < 5; ++col)
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

  for (unsigned row = 0; row < 2; ++row)
    for (unsigned col = 0; col < 5; ++col)
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

  for (unsigned row = 0; row < 2; ++row)
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
    for (unsigned row = 0; row < 2; ++row)
      for (unsigned col = 0; col < 3; ++col)
        EXPECT_EQ(subMatrix(row, col), row == 0 ? zero : two);
  }

  {
    MemoryAccessMatrix subMatrix = matrix.getColumns({0, 2});
    EXPECT_EQ(subMatrix.getNumRows(), 3u);
    EXPECT_EQ(subMatrix.getNumColumns(), 2u);
    for (unsigned row = 0; row < 3; ++row)
      for (unsigned col = 0; col < 2; ++col)
        EXPECT_EQ(subMatrix(row, col), row == 0 ? zero : row == 1 ? one : two);
  }

  {
    MemoryAccessMatrix subMatrix = matrix.getSubMatrix({0, 1}, {0, 1});
    EXPECT_EQ(subMatrix.getNumRows(), 2u);
    EXPECT_EQ(subMatrix.getNumColumns(), 2u);
    for (unsigned row = 0; row < 2; ++row)
      for (unsigned col = 0; col < 2; ++col)
        EXPECT_EQ(subMatrix(row, col), row == 0 ? zero : one);
  }
}
