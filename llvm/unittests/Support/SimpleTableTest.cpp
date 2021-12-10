//===- llvm/unittest/Support/SimpleTableTest.cpp -- Simple table tests ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SimpleTable.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::util;

namespace {

TEST(SimpleTable, IO) {
  auto Content = "[Code|Symbols|Properties]\n"
                 "a_0.bc|a_0.sym|a_0.props\n"
                 "a_1.bc|a_1.sym|a_1.props\n";
  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  // Parse a property set registry
  auto Table = SimpleTable::read(MemBuf.get());

  if (!Table)
    FAIL() << "SimpleTable::read failed\n";

  std::string Serialized;
  {
    llvm::raw_string_ostream OS(Serialized);
    // Serialize
    Table->get()->write(OS);
  }
  // Check that the original and the serialized version are equal
  ASSERT_EQ(Serialized, Content);
}

TEST(SimpleTable, Operations) {
  auto Content = "[Code|Symbols|Properties]\n"
                 "a_0.bc|a_0.sym|a_0.props\n"
                 "a_1.bc|a_1.sym|a_1.props\n";

  auto ReplaceCodeWith = "a_0.spv\n"
                         "a_1.spv\n";

  auto ReplaceSinglePropertyWith = "a_2.props";

  auto MemBuf = MemoryBuffer::getMemBuffer(Content);
  auto MemBufRepl = MemoryBuffer::getMemBuffer(ReplaceCodeWith);
  // Create tables from the strings above
  auto Table = SimpleTable::read(MemBuf.get());
  auto TableRepl = SimpleTable::read(MemBufRepl.get());

  if (!Table || !TableRepl)
    FAIL() << "SimpleTable::read failed\n";

  // Perform operations
  // -- Replace
  if (Error Err = Table->get()->replaceColumn("Code", *TableRepl->get(), ""))
    FAIL() << "SimpleTable::replaceColumn failed: " << Err << "\n";

  // -- Update cell
  if (Error Err = Table->get()->updateCellValue("Properties", 1,
                                                ReplaceSinglePropertyWith))
    FAIL() << "SimpleTable::updateCellValue failed: " << Err << "\n";

  // -- Add
  SmallVector<StringRef, 2> NewCol = {"a_0.mnf", "a_1.mnf"};
  if (Error Err = Table->get()->addColumn("Manifest", NewCol))
    FAIL() << "SimpleTable::addColumn failed: " << Err << "\n";

  // -- Peel
  if (Error Err = Table->get()->peelColumns({"Code", "Properties", "Manifest"}))
    FAIL() << "SimpleTable::peelColumns failed: " << Err << "\n";

  // Check the result
  std::string Result;
  {
    llvm::raw_string_ostream OS(Result);
    // Serialize
    Table->get()->write(OS);
  }
  auto Expected = "[Code|Properties|Manifest]\n"
                  "a_0.spv|a_0.props|a_0.mnf\n"
                  "a_1.spv|a_2.props|a_1.mnf\n";
  ASSERT_EQ(Result, Expected);
}

} // namespace
