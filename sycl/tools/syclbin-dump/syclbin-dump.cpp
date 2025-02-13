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

#include <fstream>
#include <iostream>
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

  ~ScopedIndent() {
    CurrentIndentationLevel -= Incremented;
  }

  std::string str() const { return std::string(CurrentIndentationLevel, ' '); }

private:
  friend std::ostream &operator<<(std::ostream &OS,
                                  const ScopedIndent &IH);

  const size_t Incremented;

};

std::ostream &operator<<(std::ostream &OS, const ScopedIndent &IH) {
  return (OS << IH.str());
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

std::string_view IRTypeToString(llvm::object::SYCLBIN::IRType IRType) {
  switch (IRType) {
  case llvm::object::SYCLBIN::IRType::SPIRV:
    return "SPIR-V";
  case llvm::object::SYCLBIN::IRType::PTX:
    return "PTX";
  case llvm::object::SYCLBIN::IRType::AMDGCN:
    return "AMDGCN";
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

int main(int argc, char **argv) {
  cl::opt<std::string> TargetSYCLBIN(
      cl::Positional, cl::desc("<target syclbin>"), cl::Required);

  cl::ParseCommandLineOptions(argc, argv);

  std::string TargetFilename{TargetSYCLBIN};

  auto FileMemBufferOrError = llvm::MemoryBuffer::getFileOrSTDIN(
      TargetFilename, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (!FileMemBufferOrError) {
    std::cerr << "Failed to open or read file: " << TargetFilename << std::endl;
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
    std::cerr << "Failed to parse SYCLBIN file." << std::endl;
    return 1;
  }

  std::cout << "Version:                    " << ParsedSYCLBIN->Header.Version
            << "\n";
  std::cout << "State:                      "
            << StateToString(ParsedSYCLBIN->Metadata.State) << "\n";
  std::cout << "Number of Abstract Modules: "
            << ParsedSYCLBIN->AbstractModules.size() << "\n";

  for (size_t I = 0; I < ParsedSYCLBIN->AbstractModules.size(); ++I) {
    const llvm::object::SYCLBIN::AbstractModule &AM =
        ParsedSYCLBIN->AbstractModules[I];

    std::cout << "Abstract Module " << I << ":\n";

    ScopedIndent Ind;

    // Metadata.
    std::cout << Ind << "Metadata:\n";
    {
      ScopedIndent Ind;

      std::cout << Ind << "Kernel names:\n";
      for (const llvm::SmallString<0> &KernelName : AM.KernelNames) {
        ScopedIndent Ind;
        std::cout << Ind << static_cast<std::string>(KernelName) << "\n";
      }

      std::cout << Ind << "Properties:\n";
      for (auto PropertySet : *AM.Properties) {
        ScopedIndent Ind;
        std::cout << Ind << PropertySet.first.c_str() << ":\n";
        for (auto PropertyValue : PropertySet.second) {
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
          std::cout << Ind << PropertyValue.first.c_str() << ":" << PropValStr
                    << "\n";
        }
      }

      // IR Modules.
      std::cout << Ind << "Number of IR Modules: " << AM.IRModules.size()
                << "\n";
      for (size_t J = 0; J < AM.IRModules.size(); ++J) {
        const llvm::object::SYCLBIN::IRModule &IRM = AM.IRModules[J];
        std::cout << Ind << "IR module " << J << ":\n";
        {
          ScopedIndent Ind;
          std::cout << Ind << "IR type: " << IRTypeToString(IRM.Type) << "\n";
          std::cout << Ind << "Raw IR bytes: <Binary blob of "
                    << IRM.RawIRBytes.size() << " bytes>\n";
        }
      }

      // Native device code images.
      std::cout << Ind << "Number of Native Device Code Images: "
                << AM.NativeDeviceCodeImages.size() << "\n";
      for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
        const llvm::object::SYCLBIN::NativeDeviceCodeImage &NDCI =
            AM.NativeDeviceCodeImages[J];
        std::cout << Ind << "Native device code image " << J << ":\n";
        {
          ScopedIndent Ind;
          std::cout << Ind << "Architecture: "
                    << static_cast<std::string>(NDCI.ArchString) << "\n";
          std::cout << Ind
                    << "Raw native device code image bytes: <Binary blob of "
                    << NDCI.RawDeviceCodeImageBytes.size() << " bytes>\n";
        }
      }
    }
  }

  std::cout << std::flush;
  return 0;
}
