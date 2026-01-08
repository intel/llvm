//==----------- syclbin-dump.cpp -------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "syclbin-dump" utility lists the contents of a SYCLBIN file in a
// human-readable format.
//

#include "llvm/Object/OffloadBinary.h"
#include "llvm/Object/SYCLBIN.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
#include <regex>
#include <string>

using namespace llvm;

thread_local size_t CurrentIndentationLevel = 0;

class ScopedIndent {
public:
  ScopedIndent(size_t Indents = 2) : Incremented(Indents) {
    CurrentIndentationLevel += Incremented;
  }

  ScopedIndent(const ScopedIndent &Other) = default;
  ScopedIndent(ScopedIndent &&Other) = default;

  ~ScopedIndent() { CurrentIndentationLevel -= Incremented; }

  ScopedIndent &operator=(const ScopedIndent &Other) = delete;
  ScopedIndent &operator=(ScopedIndent &&Other) = delete;

  std::string str() const { return std::string(CurrentIndentationLevel, ' '); }

private:
  friend raw_ostream &operator<<(raw_ostream &, const ScopedIndent &);

  const size_t Incremented;
};

raw_ostream &operator<<(raw_ostream &OS, const ScopedIndent &) {
  return OS.indent(CurrentIndentationLevel);
}

std::string PropertyValueToString(const llvm::util::PropertyValue &PropVal) {
  switch (PropVal.getType()) {
  case llvm::util::PropertyValue::UINT32:
    return std::to_string(PropVal.asUint32());
  case llvm::util::PropertyValue::BYTE_ARRAY:
    return std::string{PropVal.data(), PropVal.getByteArraySize()};
  case llvm::util::PropertyValue::NONE:
    break;
  }
  return "!UNKNOWN PROPERTY VALUE TYPE!";
}

void PrintProperties(raw_ostream &OS,
                     llvm::util::PropertySetRegistry &Properties) {
  for (auto &PropertySet : Properties) {
    ScopedIndent Ind;
    OS << Ind << PropertySet.first << ":\n";
    for (auto &PropertyValue : PropertySet.second) {
      ScopedIndent Ind;
      std::string PropValStr = PropertyValueToString(PropertyValue.second);
      // If there is a newline in the value, start at next line and do
      // proper indentation.
      std::regex NewlineRegex{"\r\n|\r|\n"};
      if (std::smatch Match;
          std::regex_search(PropValStr, Match, NewlineRegex)) {
        ScopedIndent Ind;
        PropValStr = "\n" + Ind.str() + PropValStr;
        // Add indentation to newlines in the returned string.
        PropValStr =
            std::regex_replace(PropValStr, NewlineRegex, "\n" + Ind.str());
      }
      OS << Ind << PropertyValue.first << ": " << PropValStr << "\n";
    }
  }
}

int main(int argc, char **argv) {
  cl::opt<std::string> TargetSYCLBIN(
      cl::Positional, cl::desc("<target syclbin>"), cl::Required);

  cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                      cl::value_desc("filename"),
                                      cl::init("-"));

  cl::ParseCommandLineOptions(argc, argv);

  std::string TargetFilename{TargetSYCLBIN};

  auto FileMemBufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(
      TargetFilename, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (std::error_code EC = FileMemBufferOrError.getError()) {
    errs() << "Failed to open or read file " << TargetFilename << ": "
           << EC.message() << "\n";
    return 1;
  }

  std::error_code OSErr{};
  raw_fd_ostream OS(OutputFilename, OSErr, sys::fs::CD_CreateAlways,
                    sys::fs::FA_Write, sys::fs::OF_None);
  if (OSErr) {
    errs() << "Failed to open output file " << OutputFilename << " : "
           << OSErr.message() << "\n";
    return 1;
  }

  Expected<std::unique_ptr<llvm::object::SYCLBIN>> SYCLBINPtrOrErr =
      llvm::object::SYCLBIN::read(**FileMemBufferOrError);

  // If direct SYCLBIN parsing failed, try parsing as OffloadBinary wrapper.
  if (!SYCLBINPtrOrErr) {
    consumeError(SYCLBINPtrOrErr.takeError());
    auto OffloadBinaryVecOrError =
        llvm::object::OffloadBinary::create(**FileMemBufferOrError);
    if (!OffloadBinaryVecOrError) {
      errs() << "Failed to parse SYCLBIN file: "
             << OffloadBinaryVecOrError.takeError() << "\n";
      std::abort();
    }

    SYCLBINPtrOrErr = llvm::object::SYCLBIN::read(
        MemoryBufferRef(OffloadBinaryVecOrError->front()->getImage(), ""));
    if (!SYCLBINPtrOrErr) {
      errs() << "Failed to parse SYCLBIN file: " << SYCLBINPtrOrErr.takeError()
             << "\n";
      std::abort();
    }
  }

  std::unique_ptr<llvm::object::SYCLBIN> ParsedSYCLBIN =
      std::move(*SYCLBINPtrOrErr);

  OS << "Global metadata:\n";
  PrintProperties(OS, *(ParsedSYCLBIN->GlobalMetadata));

  for (const auto &OBPtr : ParsedSYCLBIN->getOffloadBinaries()) {
    OS << "Abstract Module ID: "
       << OBPtr->getString("syclbin_abstract_module_id") << "\n";
    OS << "Image Kind: "
       << llvm::object::getImageKindName(OBPtr->getImageKind()) << "\n";
    OS << "Triple: " << OBPtr->getString("triple") << "\n";
    OS << "Arch: " << OBPtr->getString("Arch") << "\n";

    OS << "Metadata:\n";
    PrintProperties(OS, *ParsedSYCLBIN->Metadata[OBPtr.get()]);

    OS << "Raw bytes: <Binary blob of " << OBPtr->getImage().size()
       << " bytes>\n";
  }

  return 0;
}
