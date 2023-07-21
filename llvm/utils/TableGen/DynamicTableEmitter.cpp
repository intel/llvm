//===========- DynamicTableEmitter.cpp - Generate dynamic tables -============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a key-value map that can be dynamically extended
// at runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <set>
#include <sstream>

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
    }
    if (BitsInit *BI = dyn_cast<BitsInit>(I))
      return "0x" + utohexstr(getAsInt(BI));
    if (BitInit *BI = dyn_cast<BitInit>(I))
      return BI->getValue() ? "true" : "false";
    if (Field.IsList) {
      if (auto LI = dyn_cast<ListInit>(I)) {
        std::stringstream Result;
        // Open list
        Result << "{";
        auto Values = LI->getValues();
        bool IsAspect = (Field.Name == "aspects");
        ListSeparator LS;
        for (const auto &[Idx, Val] : enumerate(Values)) {
          // ListSeparator only provides the StringRef() operator.
          StringRef Separator = LS;
          Result << Separator.str();

          if (!IsAspect)
            Result << Val->getAsString();
          else {
            auto Record = LI->getElementAsRecord(Idx);
            Result << "\"" << Record->getValueAsString("Name").str() << "\"";
          }
        }
        // Close list
        Result << "}";
        return Result.str();
      }
      PrintFatalError(Loc,
                      Twine("Entry for field '") + Field.Name + "' is null");
    }
    PrintFatalError(Loc, Twine("invalid field type for field '") + Field.Name +
                             "'; expected: bit, bits, string, or code");
  }

  void emitDynamicTable(const DynamicTable &Table, raw_ostream &OS);
  void emitIfdef(Twine Guard, raw_ostream &OS);

  bool parseFieldType(GenericField &Field, Init *II);
  void collectTableEntries(DynamicTable &Table,
                           const std::vector<Record *> &Items);
};

} // End anonymous namespace.

void DynamicTableEmitter::emitIfdef(Twine Guard, raw_ostream &OS) {
  OS << "#ifdef " << Guard.str() << "\n";
  PreprocessorGuards.insert(Guard.str());
}

void DynamicTableEmitter::emitDynamicTable(const DynamicTable &Table,
                                           raw_ostream &OS) {
  emitIfdef((Twine("GET_") + Table.PreprocessorGuard + "_IMPL"), OS);

  // The primary data table contains all the fields defined for this map.
  OS << "std::map<std::string, " << Table.CppTypeName << "> " << Table.Name
     << " = {\n";
  // Iterate over the key-value pairs the dynamic table will contain.
  for (unsigned I = 0; I < Table.Entries.size(); ++I) {
    Record *Entry = Table.Entries[I];
    // Open key-value pair
    OS << "  { ";

    ListSeparator MapElemSeparator;
    ListSeparator TargetInfoElemSeparator;
    // Iterate over the different fields of each entry. First field is the key,
    // the rest of fields are part of the value.
    for (const auto &[Idx, Field] : enumerate(Table.Fields)) {
      bool IsKey = (Idx == 0);
      std::string Val = primaryRepresentation(Table.Locs[0], Field,
                                              Entry->getValueInit(Field.Name));
      if (!IsKey) {
        OS << TargetInfoElemSeparator << Val;
      } else {
        // Emit key and open the TargetInfo object.
        OS << MapElemSeparator << Val << MapElemSeparator << "{";
      }
    }
    // Close TargetInfo object and key-value pair.
    OS << " }}, // " << I << "\n";
  }
  // Close map.
  OS << " };\n";

  OS << "#endif\n\n";
}

bool DynamicTableEmitter::parseFieldType(GenericField &Field, Init *TypeOf) {
  if (auto Type = dyn_cast<StringInit>(TypeOf)) {
    if (Type->getValue() == "code") {
      Field.IsCode = true;
      return true;
    }
    if (Type->getValue().starts_with("list")) {
      // Nested lists are not allowed, make sure there are none
      auto Occurrences = Type->getValue().count("list");
      if (Occurrences > 1) {
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

      if (auto TypeOfRecordVal =
              TableRec->getValue(("TypeOf_" + FieldName).str())) {
        if (!parseFieldType(Table->Fields.back(),
                            TypeOfRecordVal->getValue())) {
          PrintError(TypeOfRecordVal,
                     Twine("Table '") + Table->Name + "' has invalid 'TypeOf_" +
                         FieldName +
                         "': " + TypeOfRecordVal->getValue()->getAsString());
          PrintFatalNote("The 'TypeOf_xxx' field must be a string naming a "
                         "GenericEnum record, or \"code\"");
        }
      }
    }

    StringRef FilterClass = TableRec->getValueAsString("FilterClass");
    if (!Records.getClass(FilterClass))
      PrintFatalError(TableRec->getValue("FilterClass"),
                      Twine("Table FilterClass '") + FilterClass +
                          "' does not exist");

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
