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

#define DEBUG_TYPE "spv-lower-llvm_intrinsic"

using namespace llvm;
using namespace SPIRV;

namespace SPIRV {

namespace {

typedef struct {
  // Extension that is required for an emulation
  const ExtensionID RequiredExtension;
  // Extension that supports the LLVM Intrinsic.
  // Thus, emulation is not needed if extension is enabled.
  const ExtensionID ForbiddenExtension;

  // A mapping is only applied if the RequiredExtension and the
  // ForbiddenExtension tests are both met. Thus, llvm.bitreverse.i2
  // will not be lowered even if its RequiredExtension,
  // SPV_INTEL_arbitrary_precision_integers, is enabled if its
  // ForbiddenExtension, SPV_KHR_bit_instructions, is also enabled.

  const char *ModuleText;
} LLVMIntrinsicMapValueType;

// clang-format off
#define NO_REQUIRED_EXTENSION  ExtensionID::Last
#define NO_FORBIDDEN_EXTENSION ExtensionID::Last

const std::map<const StringRef, const LLVMIntrinsicMapValueType> LLVMIntrinsicMapEntries = {
//  LLVM Intrinsic Name             Required Extension                                   Forbidden Extension                    Module with
//                                                                                                                              emulation function
  { "llvm.bitreverse.i2",          {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei2}},
  { "llvm.bitreverse.i4",          {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei4}},
  { "llvm.bitreverse.i8",          {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei8}},
  { "llvm.bitreverse.i16",         {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei16}},
  { "llvm.bitreverse.i32",         {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei32}},
  { "llvm.bitreverse.i64",         {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversei64}},

  { "llvm.bitreverse.v2i2",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev2i2}},
  { "llvm.bitreverse.v2i4",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev2i4}},
  { "llvm.bitreverse.v2i8",        {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev2i8}},
  { "llvm.bitreverse.v2i16",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev2i16}},
  { "llvm.bitreverse.v2i32",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev2i32}},
  { "llvm.bitreverse.v2i64",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev2i64}},

  { "llvm.bitreverse.v3i2",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev3i2}},
  { "llvm.bitreverse.v3i4",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev3i4}},
  { "llvm.bitreverse.v3i8",        {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev3i8}},
  { "llvm.bitreverse.v3i16",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev3i16}},
  { "llvm.bitreverse.v3i32",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev3i32}},
  { "llvm.bitreverse.v3i64",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev3i64}},

  { "llvm.bitreverse.v4i2",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev4i2}},
  { "llvm.bitreverse.v4i4",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev4i4}},
  { "llvm.bitreverse.v4i8",        {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev4i8}},
  { "llvm.bitreverse.v4i16",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev4i16}},
  { "llvm.bitreverse.v4i32",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev4i32}},
  { "llvm.bitreverse.v4i64",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev4i64}},

  { "llvm.bitreverse.v8i2",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev8i2}},
  { "llvm.bitreverse.v8i4",        {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev8i4}},
  { "llvm.bitreverse.v8i8",        {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev8i8}},
  { "llvm.bitreverse.v8i16",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev8i16}},
  { "llvm.bitreverse.v8i32",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev8i32}},
  { "llvm.bitreverse.v8i64",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev8i64}},

  { "llvm.bitreverse.v16i2",       {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev16i2}},
  { "llvm.bitreverse.v16i4",       {ExtensionID::SPV_INTEL_arbitrary_precision_integers, ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev16i4}},
  { "llvm.bitreverse.v16i8",       {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev16i8}},
  { "llvm.bitreverse.v16i16",      {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev16i16}},
  { "llvm.bitreverse.v16i32",      {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev16i32}},
  { "llvm.bitreverse.v16i64",      {NO_REQUIRED_EXTENSION,                               ExtensionID::SPV_KHR_bit_instructions, LLVMBitreversev16i64}},

  { "llvm.sadd.with.overflow.i16", {NO_REQUIRED_EXTENSION,                               NO_FORBIDDEN_EXTENSION,                LLVMSaddWithOverflow}},
  { "llvm.sadd.with.overflow.i32", {NO_REQUIRED_EXTENSION,                               NO_FORBIDDEN_EXTENSION,                LLVMSaddWithOverflow}},
  { "llvm.sadd.with.overflow.i64", {NO_REQUIRED_EXTENSION,                               NO_FORBIDDEN_EXTENSION,                LLVMSaddWithOverflow}},
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
  if (!MapEntry ||
      !(MapEntry->RequiredExtension == NO_REQUIRED_EXTENSION ||
        Opts.isAllowedToUseExtension(MapEntry->RequiredExtension)) ||
      Opts.isAllowedToUseExtension(MapEntry->ForbiddenExtension))
    return;

  // Redirect @llvm.* call to the function we have in
  // the loaded module in ModuleText
  std::string SPIRVFuncName = IntrinsicName.str();
  std::replace(SPIRVFuncName.begin(), SPIRVFuncName.end(), '.', '_');
  Function *F = Mod->getFunction(SPIRVFuncName);
  if (F) { // This function is already linked in.
    I.setCalledFunction(F);
    return;
  }
  FunctionCallee FC =
      Mod->getOrInsertFunction(SPIRVFuncName, I.getFunctionType());
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
