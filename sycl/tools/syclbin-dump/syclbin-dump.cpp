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

std::string_view StateToString(llvm::object::SYCLBIN::BundleState State) {
  switch (State) {
  case llvm::object::SYCLBIN::BundleState::Input:
    return "input";
  case llvm::object::SYCLBIN::BundleState::Object:
    return "object";
  case llvm::object::SYCLBIN::BundleState::Executable:
    return "executable";
  default:
    return "UNKNOWN";
  }
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
      // proper indentantion.
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
  if (!FileMemBufferOrError) {
    errs() << "Failed to open or read file " << TargetFilename << "\n";
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

  std::unique_ptr<llvm::object::OffloadBinary> ParsedOffloadBinary;
  MemoryBufferRef SYCLBINImageBuffer = [&]() {
    // If we failed to load as an offload binary, it may still be a SYCLBIN at
    // an outer level.
    if (llvm::object::OffloadBinary::create(**FileMemBufferOrError)
            .moveInto(ParsedOffloadBinary))
      return MemoryBufferRef(**FileMemBufferOrError);
    else
      return MemoryBufferRef(ParsedOffloadBinary->getImage(), "");
  }();

  std::unique_ptr<llvm::object::SYCLBIN> ParsedSYCLBIN;
  if (llvm::object::SYCLBIN::read(SYCLBINImageBuffer).moveInto(ParsedSYCLBIN)) {
    errs() << "Failed to parse SYCLBIN file.\n";
    return 1;
  }

  OS << "Version: " << ParsedSYCLBIN->Version << "\n";
  OS << "Global metadata:\n";
  PrintProperties(OS, *(ParsedSYCLBIN->GlobalMetadata));
  OS << "Number of Abstract Modules: " << ParsedSYCLBIN->AbstractModules.size()
     << "\n";

  for (size_t I = 0; I < ParsedSYCLBIN->AbstractModules.size(); ++I) {
    const llvm::object::SYCLBIN::AbstractModule &AM =
        ParsedSYCLBIN->AbstractModules[I];

    OS << "Abstract Module " << I << ":\n";

    ScopedIndent Ind;

    // Metadata.
    OS << Ind << "Metadata:\n";
    PrintProperties(OS, *AM.Metadata);

    // IR Modules.
    OS << Ind << "Number of IR Modules: " << AM.IRModules.size() << "\n";
    for (size_t J = 0; J < AM.IRModules.size(); ++J) {
      const llvm::object::SYCLBIN::IRModule &IRM = AM.IRModules[J];
      OS << Ind << "IR module " << J << ":\n";
      {
        ScopedIndent Ind;
        OS << Ind << "Metadata:\n";
        PrintProperties(OS, *IRM.Metadata);
        OS << Ind << "Raw IR bytes: <Binary blob of " << IRM.RawIRBytes.size()
           << " bytes>\n";
      }
    }

    // Native device code images.
    OS << Ind << "Number of Native Device Code Images: "
       << AM.NativeDeviceCodeImages.size() << "\n";
    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const llvm::object::SYCLBIN::NativeDeviceCodeImage &NDCI =
          AM.NativeDeviceCodeImages[J];
      OS << Ind << "Native device code image " << J << ":\n";
      {
        ScopedIndent Ind;
        OS << Ind << "Metadata:\n";
        PrintProperties(OS, *NDCI.Metadata);
        OS << Ind << "Raw native device code image bytes: <Binary blob of "
           << NDCI.RawDeviceCodeImageBytes.size() << " bytes>\n";
      }
    }
  }

  return 0;
}
