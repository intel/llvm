//===------ SPIRVWriterPass.h - SPIRV writing pass --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides a SPIRV writing pass.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_SPIRVWRITERPASS_H
#define SPIRV_SPIRVWRITERPASS_H

#include "LLVMSPIRVOpts.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
class ModulePass;
class PreservedAnalyses;

/// \brief Create and return a pass that writes the module to the specified
/// ostream. Note that this pass is designed for use with the legacy pass
/// manager.
ModulePass *createSPIRVWriterPass(std::ostream &Str);

/// \brief Create and return a pass that writes the module to the specified
/// ostream. Note that this pass is designed for use with the legacy pass
/// manager.
ModulePass *createSPIRVWriterPass(std::ostream &Str,
                                  const SPIRV::TranslatorOpts &Opts);

/// \brief Pass for writing a module of IR out to a SPIRV file.
///
/// Note that this is intended for use with the new pass manager. To construct
/// a pass for the legacy pass manager, use the function above.
class SPIRVWriterPass {
  std::ostream &OS;
  SPIRV::TranslatorOpts Opts;

public:
  /// \brief Construct a SPIRV writer pass around a particular output stream.
  explicit SPIRVWriterPass(std::ostream &OS) : OS(OS) {
    Opts.enableAllExtensions();
  }
  SPIRVWriterPass(std::ostream &OS, const SPIRV::TranslatorOpts &Opts)
      : OS(OS), Opts(Opts) {}

  /// \brief Run the SPIRV writer pass, and output the module to the selected
  /// output stream.
  PreservedAnalyses run(Module &M);

  static StringRef name() { return "SPIRVWriterPass"; }
};

} // namespace llvm

#endif // SPIRV_SPIRVWRITERPASS_H
