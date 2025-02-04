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
#include <string>

using namespace llvm;

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

int main(int argc, char **argv, char *env[]) {
  cl::opt<std::string> TargetSYCLBIN(
      cl::Positional, cl::desc("<target syclbin>"), cl::Required);

  cl::ParseCommandLineOptions(argc, argv);

  std::string TargetFilename{TargetSYCLBIN};

  auto FileMemBufferOrError =
      llvm::MemoryBuffer::getFileAsStream(TargetFilename);
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
            << StateToString(ParsedSYCLBIN->Header.State) << "\n";
  std::cout << "Number of Abstract Modules: "
            << ParsedSYCLBIN->AbstractModules.size() << "\n";

  for (size_t I = 0; I < ParsedSYCLBIN->AbstractModules.size(); ++I) {
    const llvm::object::SYCLBIN::AbstractModule &AM =
        ParsedSYCLBIN->AbstractModules[I];

    std::cout << "Abstract Module " << I << ":\n";

    // Metadata.
    std::cout << "  Metadata:\n";
    std::cout << "    Kernel names:\n";
    for (const llvm::SmallString<0> &KernelName : AM.KernelNames)
      std::cout << "      " << static_cast<std::string>(KernelName) << "\n";
    std::cout << "    Imported symbols:\n";
    for (const llvm::SmallString<0> &ImportedSymbol : AM.ImportedSymbols)
      std::cout << "      " << static_cast<std::string>(ImportedSymbol) << "\n";
    std::cout << "    Exported symbols:\n";
    for (const llvm::SmallString<0> &ExportedSymbol : AM.ExportedSymbols)
      std::cout << "      " << static_cast<std::string>(ExportedSymbol) << "\n";
    std::cout << "    Properties: <Binary blob of "
              << AM.Properties->getPropSets().size() << " bytes>\n";

    // IR Modules.
    std::cout << "  Number of IR Modules: " << AM.IRModules.size() << "\n";
    for (size_t J = 0; J < AM.IRModules.size(); ++J) {
      const llvm::object::SYCLBIN::IRModule &IRM = AM.IRModules[J];
      std::cout << "  IR module " << J << ":\n";
      std::cout << "    IR type: " << IRTypeToString(IRM.Type) << "\n";
      std::cout << "    Raw IR bytes: <Binary blob of " << IRM.RawIRBytes.size()
                << " bytes>\n";
    }

    // Native device code images.
    std::cout << "  Number of Native Device Code Images: "
              << AM.NativeDeviceCodeImages.size() << "\n";
    for (size_t J = 0; J < AM.NativeDeviceCodeImages.size(); ++J) {
      const llvm::object::SYCLBIN::NativeDeviceCodeImage &NDCI =
          AM.NativeDeviceCodeImages[J];
      std::cout << "  Native device code image " << J << ":\n";
      std::cout << "    Architecture: "
                << static_cast<std::string>(NDCI.ArchString) << "\n";
      std::cout << "    Raw native device code image bytes: <Binary blob of "
                << NDCI.RawDeviceCodeImageBytes.size() << " bytes>\n";
    }
  }

  std::cout << std::flush;
  return 0;
}
