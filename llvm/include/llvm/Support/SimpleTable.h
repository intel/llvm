//==-- SimpleTable.h -- tabular data simple transforms and I/O -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Defines a simple model for a tabular data container with simple operations
// over rows and columns. Columns are named and referenced by name.
// Major use case is to model dynamically-sized "2D" sets of output files by
// tools like post-link and being able to manipulate columns - for example
// replace a column listing files with bitcode with a column of .spv files.
//
// TODO May make sense to make the interface SQL-like in future if evolves.
// TODO Use YAML as serialization format.
// TODO Today cells are strings, but can be extended to other commonly used
//      types such as integers.
//
// Example of a table:
// [Code|Symbols|Properties]
// a_0.bc|a_0.sym|a_0.props
// a_1.bc|a_1.sym|a_1.props
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIMPLETABLE_H
#define LLVM_SUPPORT_SIMPLETABLE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <list>
#include <map>
#include <memory>
#include <string>

namespace llvm {
namespace util {

// The tabular data abstraction.
// TODO Supports only cells of string type only for now.
class SimpleTable {
public:
  using UPtrTy = std::unique_ptr<SimpleTable>;

  // A single row in the table. Basically a vector of string data cells.
  class Row {
  public:
    Row() = default;

    Row(SimpleTable *Parent, int NCols) : Parent(Parent) {
      Cells.resize(NCols);
    }
    StringRef getCell(StringRef ColName) const;
    StringRef getCell(StringRef ColName, StringRef DefaultVal) const;

    void setParent(SimpleTable *P) {
      assert(Parent == nullptr && "parent already set");
      Parent = P;
    }

  private:
    friend class SimpleTable;

    Row(SimpleTable *Parent) : Parent(Parent) {}

    Row(SimpleTable *Parent, ArrayRef<StringRef> R) : Row(Parent) {
      for (auto Cell : R)
        Cells.emplace_back(Cell.str());
    }

    std::string &operator[](int I) { return Cells[I]; }

    const std::string &operator[](int I) const { return Cells[I]; }

  private:
    std::vector<std::string> Cells;
    SimpleTable *Parent;
  };

public:
  SimpleTable() = default;
  static Expected<UPtrTy> create(ArrayRef<StringRef> ColNames);
  static Expected<UPtrTy> create(int NColumns);
  int getNumColumns() const { return static_cast<int>(ColumnNames.size()); }
  int getNumRows() const { return static_cast<int>(rows().size()); }

  // Add a column with given title and assign cells to given values. The table
  // must be empty or the number of the input cells must match column size.
  Error addColumn(const Twine &Title, ArrayRef<StringRef> Cells);
  Error addColumn(const Twine &Title, ArrayRef<std::string> Cells);

  // Replaces a column in this table with another column of the same size from
  // another table. Columns are identified by their names. If source column name
  // is empty, it is assumed to match the source's name.
  Error replaceColumn(StringRef Name, const SimpleTable &Src,
                      StringRef SrcName = "");

  // Replaces the value in a cell at a given column and row with the new value.
  Error updateCellValue(StringRef ColName, int Row, StringRef NewValue);

  // Renames a column.
  Error renameColumn(StringRef OldName, StringRef NewName);

  // Removes all columns except those with given names.
  Error peelColumns(ArrayRef<StringRef> ColNames);

  // Iterates all cells top-down lef-right and adds their values to given
  // container.
  void linearize(std::vector<std::string> &Res) const;

  // Serialized the table to a stream.
  void write(raw_ostream &Out, bool WriteTitles = true,
             char ColSep = '|') const;

  // De-serializes a table from a stream.
  static Expected<UPtrTy> read(MemoryBuffer *Buf, char ColSep = '|');

  // De-serializes a table from a file.
  static Expected<UPtrTy> read(const Twine &FileName, char ColSep = '|');

  const SmallVectorImpl<Row> &rows() const { return Rows; }

  void addRow(ArrayRef<StringRef> R) {
    assert((R.size() == ColumnNames.size()) && "column number mismatch");
    Rows.emplace_back(Row(this, R));
  }

  int getColumnId(StringRef ColName) const;

  Row &operator[](int I) { return Rows[I]; }
  const Row &operator[](int I) const { return Rows[I]; }

private:
  Error addColumnName(StringRef ColName);
  void rebuildName2NumMapping();

  std::map<StringRef, int> ColumnName2Num;
  // Use list as the holder of string objects as modification never invalidate
  // element addresses and iterators, unlike vector.
  std::list<std::string> ColumnNames;
  SmallVector<std::list<std::string>::iterator, 4> ColumnNum2Name;
  SmallVector<Row, 4> Rows;
};

} // namespace util
} // namespace llvm

#endif // LLVM_SUPPORT_SIMPLETABLE_H
