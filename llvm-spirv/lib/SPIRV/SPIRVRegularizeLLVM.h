//=- SPIRVRegularizeLLVM.h - LLVM Module regularization pass -*- C++ -*-=//
//
//                     The LLVM/SPIR-V Translator
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

#ifndef SPIRV_SPIRVREGULARIZELLVM_H
#define SPIRV_SPIRVREGULARIZELLVM_H

#include "SPIRVInternal.h"

#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace SPIRV {

class SPIRVRegularizeLLVMBase {
public:
  SPIRVRegularizeLLVMBase() : M(nullptr), Ctx(nullptr) {}

  bool runRegularizeLLVM(llvm::Module &M);
  // Lower functions
  bool regularize();

  /// Some LLVM intrinsics that have no SPIR-V counterpart may be wrapped in
  /// @spirv.llvm_intrinsic_* function. During reverse translation from SPIR-V
  /// to LLVM IR we can detect this @spirv.llvm_intrinsic_* function and
  /// replace it with @llvm.intrinsic.* back.
  void lowerIntrinsicToFunction(llvm::IntrinsicInst *Intrinsic);

  /// No SPIR-V counterpart for @llvm.fshl.*(@llvm.fshr.*) intrinsic. It will be
  /// lowered to a newly generated @spirv.llvm_fshl_*(@spirv.llvm_fshr_*)
  /// function.
  ///
  /// Conceptually, FSHL (FSHR):
  /// 1. concatenates the ints, the first one being the more significant;
  /// 2. performs a left (right) shift-rotate on the resulting doubled-sized
  /// int;
  /// 3. returns the most (least) significant bits of the shift-rotate result,
  ///    the number of bits being equal to the size of the original integers.
  /// If FSHL (FSHR) operates on a vector type instead, the same operations are
  /// performed for each set of corresponding vector elements.
  ///
  /// The actual implementation algorithm will be slightly different for
  /// simplification purposes.
  void lowerFunnelShift(llvm::IntrinsicInst *FSHIntrinsic);

  void lowerUMulWithOverflow(llvm::IntrinsicInst *UMulIntrinsic);
  void buildUMulWithOverflowFunc(llvm::Function *UMulFunc);

  // For some cases Clang emits VectorExtractDynamic as:
  // void @_Z28__spirv_VectorExtractDynamic(<Ty>* sret(<Ty>), jointMatrix, idx);
  // Instead of:
  // <Ty> @_Z28__spirv_VectorExtractDynamic(JointMatrix, Idx);
  // And VectorInsertDynamic as:
  // @_Z27__spirv_VectorInsertDynamic(jointMatrix, <Ty>* byval(<Ty>), idx);
  // Instead of:
  // @_Z27__spirv_VectorInsertDynamic(jointMatrix, <Ty>, idx)
  // Need to add additional GEP, store and load instructions and mutate called
  // function to avoid translation failures
  void expandSYCLTypeUsing(llvm::Module *M);
  void expandVEDWithSYCLTypeSRetArg(llvm::Function *F);
  void expandVIDWithSYCLTypeByValComp(llvm::Function *F);

  // According to the specification, the operands of a shift instruction must be
  // a scalar/vector of integer. When LLVM-IR contains a shift instruction with
  // i1 operands, they are treated as a bool. We need to extend them to i32 to
  // comply with the specification. For example: "%shift = lshr i1 0, 1";
  // The bit instruction should be changed to the extended version
  // "%shift = lshr i32 0, 1" so the args are treated as int operands.
  Value *extendBitInstBoolArg(llvm::Instruction *OldInst);

  static std::string lowerLLVMIntrinsicName(llvm::IntrinsicInst *II);
  static char ID;

private:
  llvm::Module *M;
  llvm::LLVMContext *Ctx;
};

class SPIRVRegularizeLLVMPass
    : public llvm::PassInfoMixin<SPIRVRegularizeLLVMPass>,
      public SPIRVRegularizeLLVMBase {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM) {
    return runRegularizeLLVM(M) ? llvm::PreservedAnalyses::none()
                                : llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

class SPIRVRegularizeLLVMLegacy : public llvm::ModulePass,
                                  public SPIRVRegularizeLLVMBase {
public:
  SPIRVRegularizeLLVMLegacy() : ModulePass(ID) {
    initializeSPIRVRegularizeLLVMLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(llvm::Module &M) override;

  static char ID;
};

} // namespace SPIRV

#endif // SPIRV_SPIRVREGULARIZELLVM_H
