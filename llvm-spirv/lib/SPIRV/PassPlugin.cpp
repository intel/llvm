//===- PassPlugin.cpp - Register SPIRV passes as plugin -------------------===//
//
//                     The LLVM/SPIR-V Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2024 The Khronos Group Inc.
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
// This file implements pass plugin to register llvm-spirv passes for opt tool.
//
//===----------------------------------------------------------------------===//

#include "OCLToSPIRV.h"
#include "PreprocessMetadata.h"
#include "SPIRVLowerBitCastToNonStandardType.h"
#include "SPIRVLowerBool.h"
#include "SPIRVLowerConstExpr.h"
#include "SPIRVLowerMemmove.h"
#include "SPIRVLowerOCLBlocks.h"
#include "SPIRVLowerSaddWithOverflow.h"
#include "SPIRVRegularizeLLVM.h"
#include "SPIRVToOCL.h"
#include "SPIRVWriter.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {

PassPluginLibraryInfo getSPIRVPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION, "SPIRV", LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerAnalysisRegistrationCallback([](ModuleAnalysisManager &AM) {
          AM.registerPass([] { return OCLTypeToSPIRVPass(); });
        });
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &PM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name.equals("spirv-lower-bitcast")) {
                PM.addPass(
                    SPIRVLowerBitCastToNonStandardTypePass(TranslatorOpts{}));
                return true;
              }
              return false;
            });
        PB.registerPipelineParsingCallback(
            [](StringRef Name, ModulePassManager &PM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name.equals("ocl-to-spirv")) {
                PM.addPass(OCLToSPIRVPass());
                return true;
              }
              if (Name.equals("llvm-to-spirv")) {
                SPIRV::TranslatorOpts DefaultOpts;
                DefaultOpts.enableAllExtensions();
                SPIRVModule *BM = SPIRVModule::createSPIRVModule(DefaultOpts);
                PM.addPass(LLVMToSPIRVPass(BM));
                return true;
              }
              if (Name.equals("process-metadata")) {
                PM.addPass(PreprocessMetadataPass());
                return true;
              }
              if (Name.equals("spirv-lower-bool")) {
                PM.addPass(SPIRVLowerBoolPass());
                return true;
              }
              if (Name.equals("spirv-lower-constexpr")) {
                PM.addPass(SPIRVLowerConstExprPass());
                return true;
              }
              if (Name.equals("spirv-lower-memmove")) {
                PM.addPass(SPIRVLowerMemmovePass());
                return true;
              }
              if (Name.equals("spirv-lower-ocl-blocks")) {
                PM.addPass(SPIRVLowerOCLBlocksPass());
                return true;
              }
              if (Name.equals("spirv-lower-sadd-with-overflow")) {
                PM.addPass(SPIRVLowerSaddWithOverflowPass());
                return true;
              }
              if (Name.equals("spirv-regularize-llvm")) {
                PM.addPass(SPIRVRegularizeLLVMPass());
                return true;
              }
              if (Name.equals("spirv-to-ocl12")) {
                PM.addPass(SPIRVToOCL12Pass());
                return true;
              }
              if (Name.equals("spirv-to-ocl20")) {
                PM.addPass(SPIRVToOCL20Pass());
                return true;
              }
              return false;
            });
      }};
}

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getSPIRVPluginInfo();
}
