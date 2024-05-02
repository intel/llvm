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

#ifndef COMPILER_UTILS_SUB_GROUP_ANALYSIS_H_INCLUDED
#define COMPILER_UTILS_SUB_GROUP_ANALYSIS_H_INCLUDED

#include <compiler/utils/builtin_info.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/PassManager.h>

#include <map>
#include <set>

namespace compiler {
namespace utils {

/// @brief Provides module-level information about the sub-group usage of each
/// function contained within.
///
/// The results for each function are cached in a map. Declarations are not
/// processed. Thus an external function declaration that uses sub-group
/// builtins will be missed.
///
/// Internal mux sub-group 'setter' functions are not counted. This is because
/// they only used internally by the oneAPI Construction Kit as scaffolding for
/// the sub-group support that the user can observe.
///
/// Each function contains the set of mux sub-group builtins it (transitively)
/// calls.
class GlobalSubgroupInfo {
  struct SubgroupInfo {
    std::set<BuiltinID> UsedSubgroupBuiltins;
  };

  using FunctionMapTy =
      std::map<const llvm::Function *, std::unique_ptr<SubgroupInfo>>;

  FunctionMapTy FunctionMap;

  compiler::utils::BuiltinInfo &BI;

 public:
  GlobalSubgroupInfo(llvm::Module &M, BuiltinInfo &);

  compiler::utils::BuiltinInfo &getBuiltinInfo() { return BI; }

  using iterator = FunctionMapTy::iterator;
  using const_iterator = FunctionMapTy::const_iterator;

  /// @brief Returns the SubgroupInfo for the provided function.
  ///
  /// The function must already exist in the map.
  inline const SubgroupInfo *operator[](const llvm::Function *F) const {
    const const_iterator I = FunctionMap.find(F);
    assert(I != FunctionMap.end() && "Function not in sub-group info!");
    return I->second.get();
  }

  bool usesSubgroups(const llvm::Function &F) const;

  /// @brief Returns true if the provided function is a mux sub-group
  /// collective builtin or sub-group barrier.
  std::optional<compiler::utils::Builtin> isMuxSubgroupBuiltin(
      const llvm::Function *F) const;
};

/// @brief Computes and returns the GlobalSubgroupInfo for a Module.
class SubgroupAnalysis : public llvm::AnalysisInfoMixin<SubgroupAnalysis> {
  friend AnalysisInfoMixin<SubgroupAnalysis>;

 public:
  using Result = GlobalSubgroupInfo;

  explicit SubgroupAnalysis() {}

  /// @brief Retrieve the GlobalSubgroupInfo for the module.
  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &);

  /// @brief Return the name of the pass.
  static llvm::StringRef name() { return "Sub-group analysis"; }

 private:
  /// @brief Unique pass identifier.
  static llvm::AnalysisKey Key;
};

/// @brief Helper pass to print out the contents of the SubgroupAnalysis
/// analysis.
class SubgroupAnalysisPrinterPass
    : public llvm::PassInfoMixin<SubgroupAnalysisPrinterPass> {
  llvm::raw_ostream &OS;

 public:
  explicit SubgroupAnalysisPrinterPass(llvm::raw_ostream &OS) : OS(OS) {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_SUB_GROUP_ANALYSIS_H_INCLUDED
