// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// @file
///
/// @brief Hold global state and objects used for managing a pass pipeline.

#ifndef COMPILER_UTILS_PASS_MACHINERY_H_INCLUDED
#define COMPILER_UTILS_PASS_MACHINERY_H_INCLUDED

#include <llvm/ADT/StringMap.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/StandardInstrumentations.h>

namespace llvm {
class TargetMachine;
}

namespace compiler {
namespace utils {

/// @brief Mirror's LLVM's DebugLogging options in its `opt` tool. Clang has
/// a boolean on/off version.
enum class DebugLogging { None, Normal, Verbose, Quiet };

/// @brief A class that manages the lifetime and initialization of all
/// components required to set up a new-style LLVM pass manager.
class PassMachinery {
 public:
  PassMachinery(llvm::LLVMContext &Ctx, llvm::TargetMachine *TM,
                bool VerifyEach = false,
                DebugLogging debugLogLevel = DebugLogging::None);

  virtual ~PassMachinery();

  /// @brief Initializes the PassBuilder and calls registerPasses.
  void initializeStart(
      llvm::PipelineTuningOptions PTO = llvm::PipelineTuningOptions());

  /// @brief Cross-registers analysis managers, adds callbacks and
  /// instrumentation support. Calls addClassToPassNames and
  /// registerPassCallbacks.
  void initializeFinish();

  /// @brief Calls buildDefaultAAPipeline and registerLLVMAnalyses.
  virtual void registerPasses();

  /// @brief Helper method to register the standard LLVM AA pipeline.
  ///
  /// Registers:
  /// * llvm::PassBuilder::buildDefaultAAPipeline
  void buildDefaultAAPipeline();

  /// @brief Helper method to register the standard LLVM analyses.
  ///
  /// Calls:
  /// * llvm::PassBuilder::registerModuleAnalyses
  /// * llvm::PassBuilder::registerCGSCCAnalyses
  /// * llvm::PassBuilder::registerFunctionAnalyses
  /// * llvm::PassBuilder::registerLoopAnalyses
  void registerLLVMAnalyses();

  /// @brief Method to allow customization of class-to-pass-names for
  /// instrumentation purposes. By default, none are set up by
  /// PassMachinery::initialize.
  virtual void addClassToPassNames() {}

  /// @brief Method to allow customization of pass callbacks via
  /// llvm::PassBuilder. of customization of class-to-pass-names for By
  /// default, no callbacks are set up by PassMachinery::initialize.
  virtual void registerPassCallbacks() {}

  /// @brief print pass names in style of opt --print-passes
  /// @note This should print parameters too
  virtual void printPassNames(llvm::raw_ostream &) {}

  llvm::ModuleAnalysisManager &getMAM() { return MAM; }
  const llvm::ModuleAnalysisManager &getMAM() const { return MAM; }

  llvm::FunctionAnalysisManager &getFAM() { return FAM; }
  const llvm::FunctionAnalysisManager &getFAM() const { return FAM; }

  llvm::PassBuilder &getPB() { return PB; }
  const llvm::PassBuilder &getPB() const { return PB; }

  llvm::TargetMachine *getTM() { return TM; }
  const llvm::TargetMachine *getTM() const { return TM; }

 protected:
  /// @brief TargetMachine to be used for passes. May be nullptr.
  llvm::TargetMachine *TM;
  // Note: the order here is important! They must be destructed in this order.
  /// @brief Holds state for Loop analyses.
  llvm::LoopAnalysisManager LAM;
  /// @brief Holds state for Function analyses.
  llvm::FunctionAnalysisManager FAM;
  /// @brief Holds state for CGSCC analyses.
  llvm::CGSCCAnalysisManager CGAM;
  /// @brief Holds state for Module analyses.
  llvm::ModuleAnalysisManager MAM;
  /// @brief Manages the state for any instrumentation callbacks.
  std::unique_ptr<llvm::StandardInstrumentations> SI;
  /// @brief Provides an interface to register callbacks.
  llvm::PassInstrumentationCallbacks PIC;
  /// @brief Helper to build and parse pass pipelines.
  llvm::PassBuilder PB;
};

/// Helper functions for pass printing.

/// @brief Helper function for printing a pass name, to be used by
/// printPassNames.
/// @param PassName Name of pass from a debug/parsing perspective.
/// @param OS stream to write to.
/// @note This is a direct copy from PassBuilder.cpp.
void printPassName(llvm::StringRef PassName, llvm::raw_ostream &OS);

/// @brief Helper function for printing a pass name with parameters, to be.
/// @param PassName Name of pass from a debug/parsing perspective.
/// @param Params Textual representation of the parameters.
/// @param OS stream to write to.
/// @note This is a direct copy from PassBuilder.cpp.
void printPassName(llvm::StringRef PassName, llvm::StringRef Params,
                   llvm::raw_ostream &OS);

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_PASS_MACHINERY_H_INCLUDED
