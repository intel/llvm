//===- SPIRVLowerSaddWithOverflow.cpp - Lower llvm.sadd.with.overflow -----===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2022 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of The Khronos Group, nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering of llvm.sadd.with.overflow.* into basic LLVM
// operations. Probably, in the future this pass can be generalized for other
// function calls
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spv-lower-llvm_sadd_with_overflow"

#include "SPIRVLowerSaddWithOverflow.h"
#include "LLVMSaddWithOverflow.h"

#include "LLVMSPIRVLib.h"
#include "SPIRVError.h"
#include "libSPIRV/SPIRVDebug.h"

#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {

void SPIRVLowerSaddWithOverflowBase::visitIntrinsicInst(CallInst &I) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);
  if (!II || II->getIntrinsicID() != Intrinsic::sadd_with_overflow)
    return;

  Function *IntrinsicFunc = I.getCalledFunction();
  assert(IntrinsicFunc && "Missing function");
  StringRef IntrinsicName = IntrinsicFunc->getName();
  std::string FuncName = "llvm_sadd_with_overflow_i";
  if (IntrinsicName.ends_with(".i16"))
    FuncName += "16";
  else if (IntrinsicName.ends_with(".i32"))
    FuncName += "32";
  else if (IntrinsicName.ends_with(".i64"))
    FuncName += "64";
  else {
    assert(false &&
           "Unsupported overloading of llvm.sadd.with.overflow intrinsic");
    return;
  }

  // Redirect @llvm.sadd.with.overflow.* call to the function we have in
  // the loaded module @llvm_sadd_with_overflow_*
  Function *F = Mod->getFunction(FuncName);
  if (F) { // This function is already linked in.
    I.setCalledFunction(F);
    return;
  }
  FunctionCallee FC = Mod->getOrInsertFunction(FuncName, I.getFunctionType());
  I.setCalledFunction(FC);

  // Read LLVM IR with the intrinsic's implementation
  SMDiagnostic Err;
  auto MB = MemoryBuffer::getMemBuffer(LLVMSaddWithOverflow);
  auto SaddWithOverflowModule =
      parseIR(MB->getMemBufferRef(), Err, *Context,
              ParserCallbacks([&](StringRef, StringRef) {
                return Mod->getDataLayoutStr();
              }));
  if (!SaddWithOverflowModule) {
    std::string ErrMsg;
    raw_string_ostream ErrStream(ErrMsg);
    Err.print("", ErrStream);
    SPIRVErrorLog EL;
    EL.checkError(false, SPIRVEC_InvalidLlvmModule, ErrMsg);
    return;
  }

  // Link in the intrinsic's implementation.
  if (!Linker::linkModules(*Mod, std::move(SaddWithOverflowModule),
                           Linker::LinkOnlyNeeded))
    TheModuleIsModified = true;
}

bool SPIRVLowerSaddWithOverflowBase::runLowerSaddWithOverflow(Module &M) {
  Context = &M.getContext();
  Mod = &M;
  visit(M);

  verifyRegularizationPass(M, "SPIRVLowerSaddWithOverflow");
  return TheModuleIsModified;
}

llvm::PreservedAnalyses
SPIRVLowerSaddWithOverflowPass::run(llvm::Module &M,
                                    llvm::ModuleAnalysisManager &MAM) {
  return runLowerSaddWithOverflow(M) ? llvm::PreservedAnalyses::none()
                                     : llvm::PreservedAnalyses::all();
}

SPIRVLowerSaddWithOverflowLegacy::SPIRVLowerSaddWithOverflowLegacy()
    : ModulePass(ID) {
  initializeSPIRVLowerSaddWithOverflowLegacyPass(
      *PassRegistry::getPassRegistry());
}

bool SPIRVLowerSaddWithOverflowLegacy::runOnModule(Module &M) {
  return runLowerSaddWithOverflow(M);
}

char SPIRVLowerSaddWithOverflowLegacy::ID = 0;

} // namespace SPIRV

INITIALIZE_PASS(SPIRVLowerSaddWithOverflowLegacy,
                "spv-lower-llvm_sadd_with_overflow",
                "Lower llvm.sadd.with.overflow.* intrinsics", false, false)

ModulePass *llvm::createSPIRVLowerSaddWithOverflowLegacy() {
  return new SPIRVLowerSaddWithOverflowLegacy();
}
