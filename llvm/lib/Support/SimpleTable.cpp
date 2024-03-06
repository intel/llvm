//==-- SimpleTable.cpp -- tabular data simple transforms and I/O -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SimpleTable.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"

#include <algorithm>
#include <set>
#include <string>

using namespace llvm;

static Error makeError(const Twine &Msg) {
  return createStringError(errc::invalid_argument, Msg);
}

namespace llvm {
namespace util {

StringRef SimpleTable::Row::getCell(StringRef ColName) const {
  int I = Parent->getColumnId(ColName);
  assert(I >= 0 && "column name not found");
  return Cells[I];
}

StringRef SimpleTable::Row::getCell(StringRef ColName,
                                    StringRef DefaultVal) const {
  int I = Parent->getColumnId(ColName);
  return (I >= 0) ? Cells[I] : DefaultVal;
}

Expected<SimpleTable::UPtrTy>
SimpleTable::create(ArrayRef<StringRef> ColNames) {
  auto Res = std::make_unique<SimpleTable>();

  for (auto &N : ColNames)
    if (Error Err = Res->addColumnName(N))
      return std::move(Err);
  return std::move(Res);
}

Expected<SimpleTable::UPtrTy> SimpleTable::create(int NColumns) {
  auto Res = std::make_unique<SimpleTable>();

  for (int I = 0; I < NColumns; I++)
    if (Error Err = Res->addColumnName(Twine(I).str()))
      return std::move(Err);
  return std::move(Res);
}

int SimpleTable::getColumnId(StringRef ColName) const {
  auto It = ColumnName2Num.find(ColName);
  return (It != ColumnName2Num.end()) ? It->second : -1;
}

Error SimpleTable::addColumnName(StringRef ColName) {
  if (ColumnName2Num.find(ColName) != ColumnName2Num.end())
    return makeError("column already exists " + ColName);
  ColumnNames.emplace_back(ColName.str());
  ColumnName2Num[ColumnNames.back()] = static_cast<int>(ColumnNames.size()) - 1;
  ColumnNum2Name.push_back(std::prev(ColumnNames.end()));
  return Error::success();
}

Error SimpleTable::addColumn(const Twine &Title, ArrayRef<std::string> Cells) {
  const auto N = Cells.size();
  if (!Rows.empty() && (Rows.size() != N))
    return makeError("column size mismatch for " + Title);
  if (Error Err = addColumnName(Title.str()))
    return Err;
  if (Rows.empty()) {
    Rows.resize(Cells.size());
    for (auto &R : Rows)
      R.setParent(this);
  }
  int I = 0;

  for (auto &R : Rows)
    R.Cells.push_back(Cells[I++]);
  return Error::success();
}

Error SimpleTable::addColumn(const Twine &Title, ArrayRef<StringRef> Cells) {
  std::vector<std::string> CellsVec(Cells.begin(), Cells.end());
  return addColumn(Title, CellsVec);
}

Error SimpleTable::replaceColumn(StringRef Name, const SimpleTable &Src,
                                 StringRef SrcName) {
  if (Rows.size() != Src.rows().size())
    return makeError("column length mismatch for '" + Name + "' and '" +
                     SrcName + "'");
  if ((getNumColumns() == 0) && (Src.getNumColumns() == 0))
    return makeError("empty table");
  int Cdst = getNumColumns() > 1 ? getColumnId(Name) : 0;
  int Csrc = Src.getNumColumns() > 1 ? Src.getColumnId(SrcName) : 0;
  if (Cdst < 0)
    return makeError("Column not found: " + Name);
  if (Csrc < 0)
    return makeError("Column not found: " + SrcName);
  for (unsigned R = 0; R < Rows.size(); ++R)
    Rows[R][Cdst] = Src[R][Csrc];
  return Error::success();
}

Error SimpleTable::updateCellValue(StringRef ColName, int Row,
                                   StringRef NewValue) {
  if (getNumColumns() == 0)
    return makeError("empty table");
  if (Row > getNumRows() || Row < 0)
    return makeError("row index out of bounds");
  int Col = getColumnId(ColName);
  if (Col < 0)
    return makeError("Column not found: " + ColName);
  Rows[Row][Col] = NewValue.str();
  return Error::success();
}

Error SimpleTable::renameColumn(StringRef OldName, StringRef NewName) {
  int I = getColumnId(OldName);

  if (I < 0)
    return makeError("column not found: " + OldName);
  *ColumnNum2Name[I] = NewName.str();
  ColumnName2Num.erase(OldName);
  ColumnName2Num[StringRef(*ColumnNum2Name[I])] = I;
  return Error::success();
}

void SimpleTable::rebuildName2NumMapping() {
  int Ind = 0;
  ColumnNum2Name.resize(ColumnNames.size());

  for (auto It = ColumnNames.begin(); It != ColumnNames.end(); It++, ++Ind) {
    ColumnNum2Name[Ind] = It;
    ColumnName2Num[*It] = Ind;
  }
}

Error SimpleTable::peelColumns(ArrayRef<StringRef> ColNames) {
  std::set<StringRef> Names(ColNames.begin(), ColNames.end());

  if (Names.size() != ColNames.size())
    return makeError("duplicated column names found");

  // go backwards not to affect prior column numbers
  for (int Col = getNumColumns() - 1; Col >= 0; --Col) {
    std::list<std::string>::iterator Iter = ColumnNum2Name[Col];
    // see if current column is among those which will stay
    if (Names.erase(StringRef(*Iter)) > 0)
      continue; // yes
    // no - remove from titles (ColumnNum2Name will be updated in rebuild below)
    ColumnName2Num.erase(*Iter);
    ColumnNames.erase(Iter);
    // ... and from data
    for (int Row = 0; Row < getNumRows(); ++Row)
      Rows[Row].Cells.erase(Rows[Row].Cells.begin() + Col);
  }
  if (Names.size() > 0)
    return makeError("column not found " + *Names.begin());
  rebuildName2NumMapping();
  return Error::success();
}

void SimpleTable::linearize(std::vector<std::string> &Res) const {
  for (const auto &R : Rows)
    for (const auto &C : R.Cells)
      Res.push_back(C);
}

static constexpr char COL_TITLE_LINE_OPEN[] = "[";
static constexpr char COL_TITLE_LINE_CLOSE[] = "]";
static constexpr char ROW_SEP[] = "\n";

void SimpleTable::write(raw_ostream &Out, bool WriteTitles, char ColSep) const {
  if (WriteTitles) {
    Out << COL_TITLE_LINE_OPEN;

    for (unsigned I = 0; I < ColumnNames.size(); ++I) {
      if (I != 0)
        Out << ColSep;
      Out << *ColumnNum2Name[I];
    }
    Out << COL_TITLE_LINE_CLOSE << ROW_SEP;
  }
  const unsigned N = ColumnNames.size();

  for (unsigned I = 0; I < Rows.size(); ++I) {
    const auto &R = Rows[I];

    for (unsigned J = 0; J < N; ++J) {
      if (J != 0)
        Out << ColSep;
      Out << R.Cells[J];
    }
    Out << ROW_SEP;
  }
}

Expected<SimpleTable::UPtrTy> SimpleTable::read(MemoryBuffer *Buf,
                                                char ColSep) {
  line_iterator LI(*Buf);

  if (LI.is_at_end() || LI->empty()) // empty table
    return std::make_unique<SimpleTable>();
  UPtrTy Res;

  if (LI->starts_with(COL_TITLE_LINE_OPEN)) {
    if (!LI->ends_with(COL_TITLE_LINE_CLOSE))
      return createStringError(errc::invalid_argument, "malformed title line");
    // column titles present
    StringRef L = LI->substr(1, LI->size() - 2); // trim '[' and ']'
    SmallVector<StringRef, 4> Titles;
    L.split(Titles, ColSep);
    auto Table = SimpleTable::create(Titles);
    if (!Table)
      return Table.takeError();
    Res = std::move(Table.get());
    LI++;
  }
  // parse rows
  while (!LI.is_at_end()) {
    SmallVector<StringRef, 4> Vals;
    LI->split(Vals, ColSep);

    if (!Res) {
      auto Table = SimpleTable::create(Vals.size());
      if (!Table)
        return Table.takeError();
      Res = std::move(Table.get());
    }
    if (static_cast<int>(Vals.size()) != Res->getNumColumns())
      return createStringError(errc::invalid_argument,
                               "row size mismatch at line " +
                                   Twine(LI.line_number()));
    Res->addRow(Vals);
    LI++;
  }
  return std::move(Res);
}

Expected<SimpleTable::UPtrTy> SimpleTable::read(const Twine &FileName,
                                                char ColSep) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> MemBuf =
      MemoryBuffer::getFileAsStream(FileName);
  if (!MemBuf || !MemBuf->get())
    return createFileError(Twine("can't read ") + FileName, MemBuf.getError());
  return read(MemBuf->get(), ColSep);
}

Error SimpleTable::merge(const SimpleTable &Other) {
  if (getNumColumns() != Other.getNumColumns())
    return makeError("different number of columns");
  if (ColumnNames != Other.ColumnNames)
    return makeError("different column names");
  Rows.insert(Rows.end(), Other.Rows.begin(), Other.Rows.end());
  return Error::success();
}

} // namespace util
} // namespace llvm
