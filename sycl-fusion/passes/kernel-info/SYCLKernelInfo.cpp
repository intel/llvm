//==-------------------------- SYCLKernelInfo.cpp --------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCLKernelInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace jit_compiler;

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

template <typename T> static T getUInt(const MDOperand &Operand) {
  auto *ConstantMD = cast<ConstantAsMetadata>(Operand);
  auto *ConstInt = cast<ConstantInt>(ConstantMD->getValue());
  return static_cast<T>(ConstInt->getZExtValue());
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
    ++It;

    // Operand 1: Argument kinds
    auto *ArgsKindsMD = cast<MDNode>(*It);
    ++It;

    // Operand 2: Argument usage mask
    auto *ArgsUsageMaskMD = cast<MDNode>(*It);
    ++It;

    assert(ArgsKindsMD->getNumOperands() == ArgsUsageMaskMD->getNumOperands());
    SYCLKernelInfo KernelInfo{Name, ArgsKindsMD->getNumOperands()};

    llvm::transform(
        ArgsKindsMD->operands(), KernelInfo.Args.Kinds.begin(),
        [](const auto &Op) {
          auto KindStr = cast<MDString>(Op)->getString();
          auto It = std::find_if(
              ParameterKindStrings.begin(), ParameterKindStrings.end(),
              [&KindStr](auto &SL) { return KindStr == SL; });
          if (It == ParameterKindStrings.end()) {
            return ParameterKind::Invalid;
          }
          auto Idx = std::distance(ParameterKindStrings.begin(), It);
          return static_cast<ParameterKind>(Idx);
        });

    llvm::transform(ArgsUsageMaskMD->operands(),
                    KernelInfo.Args.UsageMask.begin(), getUInt<ArgUsageUT>);

    // Operands 3..n: Attributes
    KernelInfo.Attributes = jit_compiler::SYCLAttributeList{
        static_cast<size_t>(std::distance(It, End))};
    std::transform(It, End, KernelInfo.Attributes.begin(), [](const auto &Op) {
      auto *AIMD = cast<MDNode>(Op);
      assert(AIMD->getNumOperands() == 4);
      const auto *AttrIt = AIMD->op_begin(), *AttrEnd = AIMD->op_end();

      // Operand 0: Attribute name
      auto Name = cast<MDString>(*AttrIt)->getString().str();
      auto Kind = SYCLKernelAttribute::parseKind(Name.c_str());
      assert(Kind != SYCLKernelAttribute::AttrKind::Invalid);
      ++AttrIt;

      // Operands 1..3: Values
      Indices Values;
      std::transform(AttrIt, AttrEnd, Values.begin(), getUInt<size_t>);

      return SYCLKernelAttribute{Kind, Values};
    });

    ModuleInfo->addKernel(KernelInfo);
  }

  // Metadata is no longer needed.
  MIMD->eraseFromParent();
}

SYCLModuleInfoAnalysis::Result
SYCLModuleInfoAnalysis::run(Module &M, ModuleAnalysisManager &) {
  if (!ModuleInfo) {
    loadModuleInfoFromMetadata(M);
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
      Out.indent(Indent * 2) << AttrInfo.getName() << ':';
      Out.PadToColumn(Pad);
      llvm::interleaveComma(AttrInfo.Values, Out);
      Out << '\n';
    }
  }

  return PreservedAnalyses::all();
}
