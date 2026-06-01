//===- SPIRVLegalizeNonStandardIntegers.h - Legalize int types -*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements legalization of non-standard integer types (i24, i48,
// etc.) for targets that don't support the arbitrary_precision_integers
// extension. Non-standard integers are widened to the next power-of-2.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_SPIRVLEGALIZENONSTANDARDINTEGERS_H
#define SPIRV_SPIRVLEGALIZENONSTANDARDINTEGERS_H

#include "LLVMSPIRVOpts.h"
#include "llvm/IR/PassManager.h"

namespace SPIRV {

class SPIRVLegalizeNonStandardIntegersPass
    : public llvm::PassInfoMixin<SPIRVLegalizeNonStandardIntegersPass> {
  SPIRV::TranslatorOpts Opts;

public:
  SPIRVLegalizeNonStandardIntegersPass(const SPIRV::TranslatorOpts &Opts)
      : Opts(Opts) {}

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

} // namespace SPIRV

#endif // SPIRV_SPIRVLEGALIZENONSTANDARDINTEGERS_H
