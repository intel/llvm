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
/// @brief OpenCL's BuiltinInfo implementation.

#ifndef COMPILER_UTILS_CL_BUILTIN_INFO_H_INCLUDED
#define COMPILER_UTILS_CL_BUILTIN_INFO_H_INCLUDED

#include <compiler/utils/builtin_info.h>
#include <compiler/utils/mangling.h>

namespace compiler {
namespace utils {
/// @addtogroup utils
/// @{

/// @brief Convenience function for constructing a CLBuiltinInfo as a unique_ptr
/// @param[in] builtins the Builtin module
/// @return a std::unique_ptr to a new CLBuiltinInfo
std::unique_ptr<BILangInfoConcept> createCLBuiltinInfo(llvm::Module *builtins);

/// @brief Builtin loader base class.
class CLBuiltinLoader {
 protected:
  CLBuiltinLoader() = default;

 public:
  virtual ~CLBuiltinLoader() = default;

  /// @brief Load a builtin function.
  /// @param[in] BuiltinName Name of the builtin function to materialize.
  /// @param[in] DestM Optional module in which to load the builtin function.
  /// @param[in] Flags Materialization flags to use.
  /// @return Pointer to the materialized builtin function on success.
  /// If a module is passed, the returned builtin function must live in
  /// that module.
  virtual llvm::Function *materializeBuiltin(llvm::StringRef BuiltinName,
                                             llvm::Module *DestM,
                                             BuiltinMatFlags Flags);

  /// @brief Expose any builtins Module
  virtual llvm::Module *getBuiltinsModule() { return nullptr; }
};

/// @brief Simple Builtin loader wrapping a given builtins module.
class SimpleCLBuiltinLoader final : public CLBuiltinLoader {
 public:
  SimpleCLBuiltinLoader(llvm::Module *builtins) : BuiltinModule(builtins) {}

  ~SimpleCLBuiltinLoader() = default;

  /// @brief Expose any builtins Module
  virtual llvm::Module *getBuiltinsModule() override { return BuiltinModule; }

 private:
  /// @brief Loaded builtins module.
  llvm::Module *BuiltinModule;
};

///  @brief A class that encapsulates information and transformations concerning
/// compiler OpenCL builtin functions.
class CLBuiltinInfo : public BILangInfoConcept {
 public:
  /// @brief Constructs a CLBuiltinInfo from a given Builtins module
  CLBuiltinInfo(llvm::Module *Builtins);

  /// @brief Constructs a CLBuiltinInfo with a user-provided loader
  CLBuiltinInfo(std::unique_ptr<CLBuiltinLoader> L) : Loader(std::move(L)) {}

  ~CLBuiltinInfo();

  llvm::Module *getBuiltinsModule() override;

  /// @see BuiltinInfo::isBuiltinUniform
  BuiltinUniformity isBuiltinUniform(const Builtin &B, const llvm::CallInst *CI,
                                     unsigned SimdDimIdx) const override;

  /// @see BuiltinInfo::analyzeBuiltin
  Builtin analyzeBuiltin(const llvm::Function &F) const override;
  /// @see BuiltinInfo::getVectorEquivalent
  llvm::Function *getVectorEquivalent(const Builtin &B, unsigned Width,
                                      llvm::Module *M = nullptr) override;
  /// @see BuiltinInfo::getScalarEquivalent
  llvm::Function *getScalarEquivalent(const Builtin &B,
                                      llvm::Module *M) override;
  /// @see BuiltinInfo::emitBuiltinInline
  llvm::Value *emitBuiltinInline(llvm::Function *Builtin, llvm::IRBuilder<> &B,
                                 llvm::ArrayRef<llvm::Value *> Args) override;

  /// @see BuiltinInfo::lowerBuiltinToMuxBuiltin
  llvm::Instruction *lowerBuiltinToMuxBuiltin(llvm::CallInst &,
                                              BIMuxInfoConcept &) override;
  /// @see BuiltinInfo::getPrintfBuiltin
  BuiltinID getPrintfBuiltin() const override;

 private:
  BuiltinID identifyBuiltin(const llvm::Function &) const;

  llvm::Function *materializeBuiltin(
      llvm::StringRef BuiltinName, llvm::Module *DestM = nullptr,
      BuiltinMatFlags Flags = eBuiltinMatDefault);

  llvm::Instruction *lowerGroupBuiltinToMuxBuiltin(llvm::CallInst &CI,
                                                   BuiltinID ID,
                                                   BIMuxInfoConcept &BIMuxImpl);
  llvm::Instruction *lowerAsyncBuiltinToMuxBuiltin(llvm::CallInst &CI,
                                                   BuiltinID ID,
                                                   BIMuxInfoConcept &BIMuxImpl);

  llvm::Value *emitBuiltinInline(BuiltinID ID, llvm::IRBuilder<> &B,
                                 llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineAsLLVMBinaryIntrinsic(llvm::IRBuilder<> &B,
                                                      llvm::Value *LHS,
                                                      llvm::Value *RHS,
                                                      llvm::Intrinsic::ID ID);
  // 6.2 Conversions & Type Casting
  llvm::Value *emitBuiltinInlineAs(llvm::Function *F, llvm::IRBuilder<> &B,
                                   llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineConvert(llvm::Function *F, BuiltinID ID,
                                        llvm::IRBuilder<> &B,
                                        llvm::ArrayRef<llvm::Value *> Args);

  // 6.11.5 Geometric Built-in Functions
  llvm::Value *emitBuiltinInlineGeometrics(BuiltinID builtinID,
                                           llvm::IRBuilder<> &B,
                                           llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineDot(llvm::IRBuilder<> &B,
                                    llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineCross(llvm::IRBuilder<> &B,
                                      llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineLength(llvm::IRBuilder<> &B,
                                       llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineNormalize(llvm::IRBuilder<> &B,
                                          llvm::ArrayRef<llvm::Value *> Args);

  // 6.11.6 Relational Built-in Functions
  llvm::Value *emitBuiltinInlineRelationalsWithTwoArguments(
      BuiltinID BuiltinID, llvm::IRBuilder<> &B,
      llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineRelationalsWithOneArgument(BuiltinID BuiltinID,
                                                           llvm::IRBuilder<> &B,
                                                           llvm::Value *Arg);
  llvm::Value *emitBuiltinInlineAll(llvm::IRBuilder<> &B,
                                    llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineAny(llvm::IRBuilder<> &B,
                                    llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineSelect(llvm::Function *F, llvm::IRBuilder<> &B,
                                       llvm::ArrayRef<llvm::Value *> Args);

  // 6.11.7 Vector Data Load/Store Functions
  llvm::Value *emitBuiltinInlineVLoad(llvm::Function *F, unsigned Width,
                                      llvm::IRBuilder<> &B,
                                      llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineVStore(llvm::Function *F, unsigned Width,
                                       llvm::IRBuilder<> &B,
                                       llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineVLoadHalf(llvm::Function *F,
                                          llvm::IRBuilder<> &B,
                                          llvm::ArrayRef<llvm::Value *> Args);
  llvm::Value *emitBuiltinInlineVStoreHalf(llvm::Function *F,
                                           llvm::StringRef Mode,
                                           llvm::IRBuilder<> &B,
                                           llvm::ArrayRef<llvm::Value *> Args);

  // 6.11.12 Miscellaneous Vector Functions
  llvm::Value *emitBuiltinInlineShuffle(BuiltinID BuiltinID,
                                        llvm::IRBuilder<> &B,
                                        llvm::ArrayRef<llvm::Value *> Args);

  llvm::Value *emitBuiltinInlinePrintf(BuiltinID BuiltinID,
                                       llvm::IRBuilder<> &B,
                                       llvm::ArrayRef<llvm::Value *> Args);

  /// @brief Return the name of the builtin with the given identifier.
  /// @param[in] ID Identifier of the builtin to return the name.
  /// @return Name of the builtin.
  llvm::StringRef getBuiltinName(BuiltinID ID) const;

  /// @brief Declare the specified OpenCL builtin in the given module.
  /// @param[in] M Module in which declare the builtin.
  /// @param[in] ID Builtin identifier.
  /// @param[in] RetTy Return type for the builtin.
  /// @param[in] ArgTys List of argument types.
  /// @param[in] ArgQuals List of argument qualifiers.
  /// @param[in] Suffix Optional builtin name suffix.
  /// @return Builtin function declaration.
  llvm::Function *declareBuiltin(llvm::Module *M, BuiltinID ID,
                                 llvm::Type *RetTy,
                                 llvm::ArrayRef<llvm::Type *> ArgTys,
                                 llvm::ArrayRef<TypeQualifiers> ArgQuals,
                                 llvm::Twine Suffix = "");

  /// @brief BuiltinLoader used to load builtins.
  std::unique_ptr<CLBuiltinLoader> Loader;
};

/// @}
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_CL_BUILTIN_INFO_H_INCLUDED
