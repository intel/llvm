//==-------------------------- SYCLKernelInfo.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLKernelInfo.h"

#include "KernelIO.h"
#include "metadata/MDParsing.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace jit_compiler;

static llvm::cl::opt<std::string>
    ModuleInfoFilePath("sycl-info-path",
                       llvm::cl::desc("Path to the SYCL module info YAML file"),
                       llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::AnalysisKey SYCLModuleInfoAnalysis::Key;

// Keep this in sync with the enum definition in
// `sycl-fusion/common/include/Kernel.h`.
static constexpr unsigned NumParameterKinds = 6;
static constexpr std::array<llvm::StringLiteral, NumParameterKinds>
    ParameterKindStrings = {
        StringLiteral{"Accessor"},
        StringLiteral{"StdLayout"},
        StringLiteral{"Sampler"},
        StringLiteral{"Pointer"},
        StringLiteral{"SpecConstantBuffer"},
        StringLiteral{"Stream"},
};
static constexpr llvm::StringLiteral InvalidParameterKindString{"Invalid"};

void SYCLModuleInfoAnalysis::loadModuleInfoFromFile() {
  DiagnosticPrinterRawOStream Printer{llvm::errs()};
  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrError =
      MemoryBuffer::getFile(ModuleInfoFilePath);
  if (std::error_code EC = FileOrError.getError()) {
    Printer << "Could not open file " << ModuleInfoFilePath << " due to error "
            << EC.message() << "\n";
    return;
  }
  llvm::yaml::Input In{FileOrError->get()->getMemBufferRef()};
  ModuleInfo = std::make_unique<SYCLModuleInfo>();
  In >> *ModuleInfo;
  if (In.error()) {
    Printer << "Error parsing YAML from " << ModuleInfoFilePath << ": "
            << In.error().message() << "\n";
    return;
  }
}

void SYCLModuleInfoAnalysis::loadModuleInfoFromMetadata(Module &M) {
  DiagnosticPrinterRawOStream Printer{llvm::errs()};
  ModuleInfo = std::make_unique<SYCLModuleInfo>();
  auto *MIMD = M.getNamedMetadata(ModuleInfoMDKey);
  if (!MIMD) {
    Printer << "No !" << ModuleInfoMDKey << " metadata found in module\n";
    return;
  }

  for (auto *KIMD : MIMD->operands()) {
    assert(KIMD->getNumOperands() >= 3 && "Incomplete kernel info entry");
    const auto *It = KIMD->op_begin(), *End = KIMD->op_end();

    // Operand 0: Kernel name
    auto Name = cast<MDString>(*It)->getString().str();
    SYCLKernelInfo KernelInfo{std::move(Name)};
    ++It;

    // Operand 1: Argument kinds
    auto *ArgsKindsMD = cast<MDNode>(*It);
    llvm::transform(
        ArgsKindsMD->operands(), std::back_inserter(KernelInfo.Args.Kinds),
        [](const auto &Op) {
          auto KindStr = cast<MDString>(Op)->getString();
          auto It = std::find_if(
              ParameterKindStrings.begin(), ParameterKindStrings.end(),
              [&KindStr](auto &SL) { return KindStr == SL; });
          if (It == ParameterKindStrings.end())
            return ParameterKind::Invalid;
          auto Idx = std::distance(ParameterKindStrings.begin(), It);
          return static_cast<ParameterKind>(Idx);
        });
    ++It;

    // Operand 2: Argument usage mask
    auto *ArgsUsageMaskMD = cast<MDNode>(*It);
    llvm::transform(
        ArgsUsageMaskMD->operands(),
        std::back_inserter(KernelInfo.Args.UsageMask), [](const auto &Op) {
          auto UIntOrErr = metadataToUInt<std::underlying_type_t<ArgUsage>>(Op);
          if (UIntOrErr.takeError()) {
            llvm_unreachable("Invalid kernel info metadata");
          }
          return *UIntOrErr;
        });
    ++It;

    // Operands 3..n: Attributes
    for (; It != End; ++It) {
      auto *AIMD = cast<MDNode>(*It);
      assert(AIMD->getNumOperands() > 1);
      const auto *AttrIt = AIMD->op_begin(), *AttrEnd = AIMD->op_end();

      // Operand 0: Attribute name
      auto Name = cast<MDString>(*AttrIt)->getString().str();
      ++AttrIt;

      // Operands 1..m: String values
      auto &KernelAttr = KernelInfo.Attributes.emplace_back(std::move(Name));
      for (; AttrIt != AttrEnd; ++AttrIt) {
        auto Value = cast<MDString>(*AttrIt)->getString().str();
        KernelAttr.Values.emplace_back(std::move(Value));
      }
    }

    ModuleInfo->addKernel(KernelInfo);
  }

  // Metadata is no longer needed.
  MIMD->eraseFromParent();
}

SYCLModuleInfoAnalysis::Result
SYCLModuleInfoAnalysis::run(Module &M, ModuleAnalysisManager &) {
  if (!ModuleInfo) {
    if (!ModuleInfoFilePath.empty()) {
      loadModuleInfoFromFile();
    } else {
      loadModuleInfoFromMetadata(M);
    }
  }
  return {ModuleInfo.get()};
}

PreservedAnalyses SYCLModuleInfoPrinter::run(Module &Mod,
                                             ModuleAnalysisManager &MAM) {
  jit_compiler::SYCLModuleInfo *ModuleInfo =
      MAM.getResult<SYCLModuleInfoAnalysis>(Mod).ModuleInfo;
  if (!ModuleInfo) {
    DiagnosticPrinterRawOStream Printer{llvm::errs()};
    Printer << "Error: No module info available\n";
    return PreservedAnalyses::all();
  }

  formatted_raw_ostream Out{llvm::outs()};
  constexpr auto Indent = 2, Pad = 26;
  for (const auto &KernelInfo : ModuleInfo->kernels()) {
    Out << "KernelName:";
    Out.PadToColumn(Pad);
    Out << KernelInfo.Name << '\n';

    Out.indent(Indent) << "Args:\n";
    Out.indent(Indent * 2) << "Kinds:";
    Out.PadToColumn(Pad);
    llvm::interleaveComma(KernelInfo.Args.Kinds, Out, [&Out](auto Kind) {
      auto KindInt = static_cast<unsigned>(Kind);
      Out << (KindInt < NumParameterKinds ? ParameterKindStrings[KindInt]
                                          : InvalidParameterKindString);
    });
    Out << '\n';

    Out.indent(Indent * 2) << "Mask:";
    Out.PadToColumn(Pad);
    llvm::interleaveComma(KernelInfo.Args.UsageMask, Out, [&Out](auto Mask) {
      Out << static_cast<unsigned>(Mask);
    });
    Out << '\n';

    if (KernelInfo.Attributes.empty()) {
      continue;
    }

    Out.indent(Indent) << "Attributes:\n";
    for (const auto &AttrInfo : KernelInfo.Attributes) {
      Out.indent(Indent * 2) << AttrInfo.AttributeName << ':';
      Out.PadToColumn(Pad);
      llvm::interleaveComma(AttrInfo.Values, Out);
      Out << '\n';
    }
  }

  return PreservedAnalyses::all();
}
