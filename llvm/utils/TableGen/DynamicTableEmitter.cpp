//===- DynamicTableEmitter.cpp - Generate efficiently dynamic tables -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a generic array initialized by specified fields,
// together with companion index tables and lookup functions (binary search,
// currently).
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "dynamic-table-emitter"

namespace {

int getAsInt(Init *B) {
  return cast<IntInit>(
             B->convertInitializerTo(IntRecTy::get(B->getRecordKeeper())))
      ->getValue();
}

struct GenericField {
  std::string Name;
  RecTy *RecType = nullptr;
  bool IsCode = false;
  bool IsList = false;

  GenericField(StringRef Name) : Name(std::string(Name)) {}
};

struct DynamicTable {
  std::string Name;
  ArrayRef<SMLoc> Locs; // Source locations from the Record instance.
  std::string PreprocessorGuard;
  std::string CppTypeName;
  SmallVector<GenericField, 2> Fields;
  std::vector<Record *> Entries;

  const GenericField *getFieldByName(StringRef Name) const {
    for (const auto &Field : Fields) {
      if (Name == Field.Name)
        return &Field;
    }
    return nullptr;
  }
};

class DynamicTableEmitter {
  RecordKeeper &Records;
  std::set<std::string> PreprocessorGuards;

public:
  DynamicTableEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS);

private:
  std::string primaryRepresentation(SMLoc Loc, const GenericField &Field,
                                    Init *I) {
    if (StringInit *SI = dyn_cast<StringInit>(I)) {
      if (Field.IsCode || SI->hasCodeFormat())
        return std::string(SI->getValue());

      return SI->getAsString();
    } else if (BitsInit *BI = dyn_cast<BitsInit>(I))
      return "0x" + utohexstr(getAsInt(BI));
    else if (BitInit *BI = dyn_cast<BitInit>(I))
      return BI->getValue() ? "true" : "false";
    else if (Field.IsList) {
      if (auto LI = dyn_cast<ListInit>(I)) {
        std::stringstream res;
        auto values = LI->getValues();
        bool NeedQuotes = (Field.Name == "aspects");
        for (const auto &[idx, val] : enumerate(values)) {
          if (idx > 0)
            res << ", ";

          if (NeedQuotes)
            res << "\"" << val->getAsString() << "\"";
          else
            res << val->getAsString();
        }
        return res.str();
      }
      PrintFatalError(Loc,
                      Twine("Entry for field '") + Field.Name + "' is null");
    }
    PrintFatalError(Loc, Twine("invalid field type for field '") + Field.Name + 
                             "'; expected: bit, bits, string, or code");
  }

  void emitDynamicTable(const DynamicTable &Table, raw_ostream &OS);
  void emitIfdef(StringRef Guard, raw_ostream &OS);

  bool parseFieldType(GenericField &Field, Init *II);
  void collectTableEntries(DynamicTable &Table,
                           const std::vector<Record *> &Items);
};

} // End anonymous namespace.

void DynamicTableEmitter::emitIfdef(StringRef Guard, raw_ostream &OS) {
  OS << "#ifdef " << Guard << "\n";
  PreprocessorGuards.insert(std::string(Guard));
}

void DynamicTableEmitter::emitDynamicTable(const DynamicTable &Table,
                                           raw_ostream &OS) {
  emitIfdef((Twine("GET_") + Table.PreprocessorGuard + "_IMPL").str(), OS);

  // The primary data table contains all the fields defined for this map.
  OS << "std::map<std::string, " << Table.CppTypeName << "> " << Table.Name
     << " = {\n";
  for (unsigned i = 0; i < Table.Entries.size(); ++i) {
    Record *Entry = Table.Entries[i];
    OS << "  { ";

    ListSeparator LS;
    bool first = true;
    bool second = false;
    for (const auto &Field : Table.Fields) {
      std::string val = primaryRepresentation(Table.Locs[0], Field,
                                              Entry->getValueInit(Field.Name));
      if (first) {
        first = false;
        second = true;
        OS << LS << val << ", { ";
      } else {
        if (!second) {
          OS << LS;
        }
        if (Field.IsList) {
          val.insert(0, 1, '{');
          val.push_back('}');
        }
        OS << val;
        second = false;
      }
    }

    OS << " }}, // " << i << "\n";
  }
  OS << " };\n";

  OS << "#endif\n\n";
}

bool DynamicTableEmitter::parseFieldType(GenericField &Field, Init *TypeOf) {
  if (auto Type = dyn_cast<StringInit>(TypeOf)) {
    if (Type->getValue() == "code") {
      Field.IsCode = true;
      return true;
    } else if (Type->getValue().starts_with("list")) {
      // Nested lists are not allowed, make sure there are none
      auto occurrences = Type->getValue().count("list");
      if (occurrences > 1) {
        PrintFatalError(Twine("Nested lists are not allowed: ") +
                        Type->getValue().str());
      }
      Field.IsList = true;
      return true;
    }
  }

  return false;
}

void DynamicTableEmitter::collectTableEntries(
    DynamicTable &Table, const std::vector<Record *> &Items) {
  if (Items.empty())
    PrintFatalError(Table.Locs,
                    Twine("Table '") + Table.Name + "' has no entries");

  for (auto *EntryRec : Items) {
    for (auto &Field : Table.Fields) {
      auto TI = dyn_cast<TypedInit>(EntryRec->getValueInit(Field.Name));
      if (!TI || !TI->isComplete()) {
        PrintFatalError(EntryRec, Twine("Record '") + EntryRec->getName() +
                                      "' for table '" + Table.Name +
                                      "' is missing field '" + Field.Name +
                                      "'");
      }
      if (!Field.RecType) {
        Field.RecType = TI->getType();
      } else {
        RecTy *Ty = resolveTypes(Field.RecType, TI->getType());
        if (!Ty)
          PrintFatalError(EntryRec->getValue(Field.Name), 
                          Twine("Field '") + Field.Name + "' of table '" +
                          Table.Name + "' entry has incompatible type: " +
                          TI->getType()->getAsString() + " vs. " +
                          Field.RecType->getAsString());
        Field.RecType = Ty;
      }
    }

    Table.Entries.push_back(EntryRec); // Add record to table's record list.
  }
}

void DynamicTableEmitter::run(raw_ostream &OS) {
  // Emit tables in a deterministic order to avoid needless rebuilds.
  SmallVector<std::unique_ptr<DynamicTable>, 4> Tables;
  DenseMap<Record *, DynamicTable *> TableMap;

  // Collect all definitions first.
  for (auto *TableRec : Records.getAllDerivedDefinitions("DynamicTable")) {
    auto Table = std::make_unique<DynamicTable>();
    Table->Name = std::string(TableRec->getName());
    Table->Locs = TableRec->getLoc();
    Table->PreprocessorGuard = std::string(TableRec->getName());
    Table->CppTypeName = std::string(TableRec->getValueAsString("CppTypeName"));

    std::vector<StringRef> Fields = TableRec->getValueAsListOfStrings("Fields");
    for (const auto &FieldName : Fields) {
      Table->Fields.emplace_back(FieldName); // Construct a GenericField.

      if (auto TypeOfRecordVal = TableRec->getValue(("TypeOf_" + FieldName).str())) {
        if (!parseFieldType(Table->Fields.back(), TypeOfRecordVal->getValue())) {
          PrintError(TypeOfRecordVal, 
                     Twine("Table '") + Table->Name +
                         "' has invalid 'TypeOf_" + FieldName +
                         "': " + TypeOfRecordVal->getValue()->getAsString());
          PrintFatalNote("The 'TypeOf_xxx' field must be a string naming a "
                         "GenericEnum record, or \"code\"");
        }
      }
    }

    StringRef FilterClass = TableRec->getValueAsString("FilterClass");
    if (!Records.getClass(FilterClass))
      PrintFatalError(TableRec->getValue("FilterClass"), 
                      Twine("Table FilterClass '") +
                          FilterClass + "' does not exist");

    collectTableEntries(*Table, Records.getAllDerivedDefinitions(FilterClass));

    TableMap.insert(std::make_pair(TableRec, Table.get()));
    Tables.emplace_back(std::move(Table));
  }

  // Emit everything.
  for (const auto &Table : Tables)
    emitDynamicTable(*Table, OS);

  // Put all #undefs last, to allow multiple sections guarded by the same
  // define.
  for (const auto &Guard : PreprocessorGuards)
    OS << "#undef " << Guard << "\n";
}

static TableGen::Emitter::OptClass<DynamicTableEmitter>
    X("gen-dynamic-tables", "Generate generic binary-dynamic table");
