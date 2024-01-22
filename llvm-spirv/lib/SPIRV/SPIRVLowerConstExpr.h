//===- SPIRVLowerConstExpr.h - Lower constant expression --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file SPIRVLowerConstExpr.h
///
/// This file declares SPIRVLowerConstExprPass that lowers constant expression.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LOWERCONSTEXPR_H
#define SPIRV_LOWERCONSTEXPR_H

#include "llvm/IR/PassManager.h"

namespace SPIRV {

class SPIRVLowerConstExprBase {
public:
  SPIRVLowerConstExprBase() : M(nullptr), Ctx(nullptr) {}

  bool runLowerConstExpr(llvm::Module &M);
  bool visit(llvm::Module *M);

private:
  llvm::Module *M;
  llvm::LLVMContext *Ctx;
};

class SPIRVLowerConstExprPass
    : public llvm::PassInfoMixin<SPIRVLowerConstExprPass>,
      public SPIRVLowerConstExprBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runLowerConstExpr(M) ? llvm::PreservedAnalyses::none()
                                : llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // namespace SPIRV

#endif // SPIRV_LOWERCONSTEXPR_H
