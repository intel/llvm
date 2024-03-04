//===- SPIRVLowerLLVMIntrinsic.cpp - Lower llvm-intrinsics -----===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2024 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===----------------------------------------------------------------------===//
//
// This file implements lowering of:
//   llvm.sadd.with.overflow.*
//   llvm.bitreverse.*
// into basic LLVM operations.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "spv-lower-llvm_intrinsic"

#include "SPIRVLowerLLVMIntrinsic.h"
#include "LLVMBitreverse.h"
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

namespace {

typedef struct {
  // Extension that supports the LLVM Intrinsic.
  // Thus, emulation is not needed if extension is enabled.
  const ExtensionID SupportingExtension;
  const char *SPIRVFuncName;
  const char *ModuleText;
} LLVMIntrinsicMapValueType;

#define NO_SUPPORTING_EXTENSION ExtensionID::Last

// clang-format off
const std::map<const StringRef, const LLVMIntrinsicMapValueType> LLVMIntrinsicMapEntries= {
//  LLVM Intrinsic Name             Supporting Extension                   Emulation Name                 Module with
//                                                                                                        emulation function
  { "llvm.bitreverse.i8",          {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_i8",          LLVMBitreverseScalari8}},
  { "llvm.bitreverse.i16",         {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_i16",         LLVMBitreverseScalari16}},
  { "llvm.bitreverse.i32",         {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_i32",         LLVMBitreverseScalari32}},
  { "llvm.bitreverse.i64",         {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_i64",         LLVMBitreverseScalari64}},

  { "llvm.bitreverse.v2i8",        {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v2i8",        LLVMBitreverseV2i8}},
  { "llvm.bitreverse.v2i16",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v2i16",       LLVMBitreverseV2i16}},
  { "llvm.bitreverse.v2i32",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v2i32",       LLVMBitreverseV2i32}},
  { "llvm.bitreverse.v2i64",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v2i64",       LLVMBitreverseV2i64}},

  { "llvm.bitreverse.v3i8",        {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v3i8",        LLVMBitreverseV3i8}},
  { "llvm.bitreverse.v3i16",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v3i16",       LLVMBitreverseV3i16}},
  { "llvm.bitreverse.v3i32",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v3i32",       LLVMBitreverseV3i32}},
  { "llvm.bitreverse.v3i64",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v3i64",       LLVMBitreverseV3i64}},

  { "llvm.bitreverse.v4i8",        {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v4i8",        LLVMBitreverseV4i8}},
  { "llvm.bitreverse.v4i16",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v4i16",       LLVMBitreverseV4i16}},
  { "llvm.bitreverse.v4i32",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v4i32",       LLVMBitreverseV4i32}},
  { "llvm.bitreverse.v4i64",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v4i64",       LLVMBitreverseV4i64}},

  { "llvm.bitreverse.v8i8",        {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v8i8",        LLVMBitreverseV8i8}},
  { "llvm.bitreverse.v8i16",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v8i16",       LLVMBitreverseV8i16}},
  { "llvm.bitreverse.v8i32",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v8i32",       LLVMBitreverseV8i32}},
  { "llvm.bitreverse.v8i64",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v8i64",       LLVMBitreverseV8i64}},

  { "llvm.bitreverse.v16i8",       {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v16i8",       LLVMBitreverseV16i8}},
  { "llvm.bitreverse.v16i16",      {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v16i16",      LLVMBitreverseV16i16}},
  { "llvm.bitreverse.v16i32",      {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v16i32",      LLVMBitreverseV16i32}},
  { "llvm.bitreverse.v16i64",      {ExtensionID::SPV_KHR_bit_instructions, "llvm_bitreverse_v16i64",      LLVMBitreverseV16i64}},

  { "llvm.sadd.with.overflow.i16", {NO_SUPPORTING_EXTENSION,               "llvm_sadd_with_overflow_i16", LLVMSaddWithOverflow}},
  { "llvm.sadd.with.overflow.i32", {NO_SUPPORTING_EXTENSION,               "llvm_sadd_with_overflow_i32", LLVMSaddWithOverflow}},
  { "llvm.sadd.with.overflow.i64", {NO_SUPPORTING_EXTENSION,               "llvm_sadd_with_overflow_i64", LLVMSaddWithOverflow}},
};
// clang-format on

} // namespace

void SPIRVLowerLLVMIntrinsicBase::visitIntrinsicInst(CallInst &I) {
  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I);

  if (!II)
    return;

  Function *IntrinsicFunc = I.getCalledFunction();
  assert(IntrinsicFunc && "Missing function");
  StringRef IntrinsicName = IntrinsicFunc->getName();

  const LLVMIntrinsicMapValueType *MapEntry{nullptr};
  auto It = LLVMIntrinsicMapEntries.find(IntrinsicName);
  if (It != LLVMIntrinsicMapEntries.end())
    MapEntry = &It->second;
  if (!MapEntry || Opts.isAllowedToUseExtension(MapEntry->SupportingExtension))
    return;

  // Redirect @llvm.* call to the function we have in
  // the loaded module in ModuleText
  Function *F = Mod->getFunction(MapEntry->SPIRVFuncName);
  if (F) { // This function is already linked in.
    I.setCalledFunction(F);
    return;
  }
  FunctionCallee FC =
      Mod->getOrInsertFunction(MapEntry->SPIRVFuncName, I.getFunctionType());
  I.setCalledFunction(FC);

  // Read LLVM IR with the intrinsic's implementation
  SMDiagnostic Err;
  auto MB = MemoryBuffer::getMemBuffer(MapEntry->ModuleText);
  auto EmulationModule = parseIR(MB->getMemBufferRef(), Err, *Context,
                                 ParserCallbacks([&](StringRef, StringRef) {
                                   return Mod->getDataLayoutStr();
                                 }));
  if (!EmulationModule) {
    std::string ErrMsg;
    raw_string_ostream ErrStream(ErrMsg);
    Err.print("", ErrStream);
    SPIRVErrorLog EL;
    EL.checkError(false, SPIRVEC_InvalidLlvmModule, ErrMsg);
    return;
  }

  // Link in the intrinsic's implementation.
  if (!Linker::linkModules(*Mod, std::move(EmulationModule),
                           Linker::LinkOnlyNeeded))
    TheModuleIsModified = true;
}

bool SPIRVLowerLLVMIntrinsicBase::runLowerLLVMIntrinsic(Module &M) {
  Context = &M.getContext();
  Mod = &M;
  visit(M);

  verifyRegularizationPass(M, "SPIRVLowerLLVMIntrinsic");
  return TheModuleIsModified;
}

SPIRVLowerLLVMIntrinsicPass::SPIRVLowerLLVMIntrinsicPass(
    const SPIRV::TranslatorOpts &Opts)
    : SPIRVLowerLLVMIntrinsicBase(Opts) {}

llvm::PreservedAnalyses
SPIRVLowerLLVMIntrinsicPass::run(llvm::Module &M,
                                 llvm::ModuleAnalysisManager &MAM) {
  return runLowerLLVMIntrinsic(M) ? llvm::PreservedAnalyses::none()
                                  : llvm::PreservedAnalyses::all();
}

SPIRVLowerLLVMIntrinsicLegacy::SPIRVLowerLLVMIntrinsicLegacy(
    const SPIRV::TranslatorOpts &Opts)
    : ModulePass(ID), SPIRVLowerLLVMIntrinsicBase(Opts) {
  initializeSPIRVLowerLLVMIntrinsicLegacyPass(*PassRegistry::getPassRegistry());
}

bool SPIRVLowerLLVMIntrinsicLegacy::runOnModule(Module &M) {
  return runLowerLLVMIntrinsic(M);
}

char SPIRVLowerLLVMIntrinsicLegacy::ID = 0;

} // namespace SPIRV

// INITIALIZE_PASS defines static functions but clang-tidy enforces
// anonymous namespace
// NOLINTNEXTLINE
INITIALIZE_PASS(SPIRVLowerLLVMIntrinsicLegacy, "spv-lower-llvm-intrinsic",
                "Lower llvm intrinsics", false, false)

ModulePass *
llvm::createSPIRVLowerLLVMIntrinsicLegacy(const SPIRV::TranslatorOpts &Opts) {
  return new SPIRVLowerLLVMIntrinsicLegacy(Opts);
}
