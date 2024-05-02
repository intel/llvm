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
/// LLVM pass utility functions.

#ifndef COMPILER_UTILS_PASS_FUNCTIONS_H_INCLUDED
#define COMPILER_UTILS_PASS_FUNCTIONS_H_INCLUDED

#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/IVDescriptors.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <array>
#include <functional>

namespace llvm {
class Argument;
class BasicBlock;
class Constant;
class ConstantExpr;
class Function;
class IntegerType;
class LLVMContext;
class Module;
class ModulePass;
class Type;
class Value;
class IRBuilderBase;
}  // namespace llvm

namespace compiler {
namespace utils {

/// @addtogroup utils
/// @{

/// @brief Calculate (approximately) the amount of private memory used by a
/// kernel.
///
/// @param fn The kernel function
///
/// @return uint64_t The private memory used by the kernel function in bytes.
uint64_t computeApproximatePrivateMemoryUsage(const llvm::Function &fn);

/// @brief Forces a constant expression or constant vector back to a normal
/// instruction
///
/// @param[in] constant to be replaced
void replaceConstantExpressionWithInstruction(llvm::Constant *const constant);

/// @brief remap operands of a constant expression
///
/// @note This will create a new constant expression and replace references to
/// the original constant with the new one
///
/// @param[in] expr Constant expression to be remapped
/// @param[in] from Constant which if found in expression will be
/// replaced
/// @param[in] to Constant which will replace any operands which are `from`
void remapConstantExpr(llvm::ConstantExpr *expr, llvm::Constant *from,
                       llvm::Constant *to);

/// @brief remap operands of a constant array
///
/// @note This will create a new constant array and replace references to
/// the original constant with the new one
///
/// @param[in] arr Constant array to be remapped
/// @param[in] from Constant which if found in array will be
/// replaced
/// @param[in] to Constant which will replace any operands which are `from`
void remapConstantArray(llvm::ConstantArray *arr, llvm::Constant *from,
                        llvm::Constant *to);

/// @brief Discover if input function references debug info metadata nodes
///
/// @param[in] func Function to check
/// @param[in,out] vmap Value map updated with identity mappings of any debug
/// info metadata found
///
/// @return bool True if function contains debug info, false otherwise
bool funcContainsDebugMetadata(const llvm::Function &func,
                               llvm::ValueToValueMapTy &vmap);

/// @brief Return a copy of a function's function, return, and parameter
/// attributes.
///
/// Only parameter attributes from indices 0 to numParams are copied. If
/// numParams is negative, all parameter attributes are copied.
llvm::AttributeList getCopiedFunctionAttrs(const llvm::Function &oldFn,
                                           int numParams = -1);

/// @brief Copy a function's attributes to a new function.
///
/// @param[in] oldFn Function to copy function attributes from.
/// @param[in] newFn Function to copy function attributes to.
/// @param[in] numParams number of parameters to copy attributes from, starting
/// from the first parameter. If set to a negative number, will copy all
/// parameter attributes.
void copyFunctionAttrs(const llvm::Function &oldFn, llvm::Function &newFn,
                       int numParams = -1);

using ParamTypeAttrsPair = std::pair<llvm::Type *, llvm::AttributeSet>;

using UpdateMDCallbackFn =
    std::function<void(llvm::Function &oldFn, llvm::Function &newFn, unsigned)>;

/// @brief Clone functions in a module and add an argument to them
///
/// @param module LLVM module containing the functions
/// @param paramTypeFunc Additional parameter to be added defined as a function
/// returning the type and set of attributes.
/// This function takes a module, primarily to access DataLayout
/// @param toBeClonedFunc function which dictates whether each function is
/// cloned
/// @param updateMetaDataCallback if set, is invoked with the old function, new
/// function and new argument index.
///
/// @return bool if the module has changed (currently always true)
///
/// This iterates through all the functions in a module but only clones and adds
/// the extra param for those that meet the following criteria after setting
/// `clonedNoBody` and `ClonedWithBody` from the toBeCloned expression:-
///
/// 1.  `!function` declaration or `ClonedNoBody` _or_ is a function
///     declaration and `ClonedWithBody`
/// 2.  Not already processed
bool cloneFunctionsAddArg(
    llvm::Module &module,
    std::function<ParamTypeAttrsPair(llvm::Module &)> paramTypeFunc,
    std::function<void(const llvm::Function &, bool &ClonedWithBody,
                       bool &ClonedNoBody)>
        toBeClonedFunc,
    const UpdateMDCallbackFn &updateMetaDataCallback = nullptr);

/// @brief Updates call instructions after to function clone to point to
/// `newFunc` instead of `oldFunc`, old call instructions are deleted.
///
/// @param[in] oldFunc Function which has been cloned
/// @param[in] newFunc Cloned function to point callsites to
/// @param[in] extraArg Whether the cloned callee has an extra argument added
void remapClonedCallsites(llvm::Function &oldFunc, llvm::Function &newFunc,
                          bool extraArg);

/// @brief Clone all functions in a module, appending an extra parameter to
/// them.
///
/// @param module llvm module containing the functions
/// @param newParamType Type of the parameter to be added
/// @param newParamAttrs Parameter attributes of the parameter to be added
/// @param updateMetaDataCallback if set, is invokved with the old function,
/// new function and new argument index.
///
/// @return bool if the module has changed (currently always true)
///
/// This iterates through all the functions in a module and clones all
/// functions with a body and adds the extra param at the end of their parameter
/// lists. Simpler version of `cloneFunctionsAddArg()` where the use case is
/// more limited.
bool addParamToAllFunctions(
    llvm::Module &module, llvm::Type *const newParamType,
    const llvm::AttributeSet &newParamAttrs,
    const UpdateMDCallbackFn &updateMetaDataCallback = nullptr);

using CreateLoopBodyFn = std::function<llvm::BasicBlock *(
    llvm::BasicBlock *, llvm::Value *, llvm::ArrayRef<llvm::Value *>,
    llvm::MutableArrayRef<llvm::Value *>)>;

struct CreateLoopOpts {
  /// @brief indexInc Value by which to increment the loop counter. If nullptr,
  /// then it is created as the constant 1, based on type of `indexStart`,
  /// which is a parameter to compiler::utils::createLoop proper.
  llvm::Value *indexInc = nullptr;
  /// @brief disableVectorize Sets loop metadata disabling further
  /// vectorization.
  bool disableVectorize = false;
  /// @brief headerName Optional name for the loop header block. Defaults to:
  /// "loopIR".
  llvm::StringRef headerName = "loopIR";
  /// @brief An optional list of incoming IV values.
  ///
  /// Each of these is used as the incoming value to a PHI created by
  /// createLoop. These PHIs are provided to the 'body' function of createLoop,
  /// which should in turn set the 'next' version of the IV.
  std::vector<llvm::Value *> IVs;
  /// @brief An optional list of IV names, to be set on the PHIs provided by
  /// 'IVs' field/parameter.
  ///
  /// If set, the names are assumed to correlate 1:1 with those IVs. The list
  /// may be shorter than the list of IVs, in which case the trailing IVs are
  /// not named.
  std::vector<std::string> loopIVNames;
};

/// @brief Create a loop around a body, creating an implicit induction variable
/// (IV) between specified start and end values, and incremented by a
/// user-specified amount. The loop thus has a trip count equal to the
/// following C-style loop: `for (auto i = start; i < end; i += incr)`.
///
/// Note that this helper always creates a CFG loop, even if the loop bounds
/// are known not to produce a loop at compile time. Users can use stock LLVM
/// optimizations to eliminate/simplify the loop in such a case.
///
/// @param entry Loop pre-header block. This block will be rewired to jump into
/// the new loop.
/// @param exit Loop exit block. The new loop will jump to this once it exits.
/// @param indexStart The start index
/// @param indexEnd The end index (we compare for <)
/// @param opts Set of options configuring the generation of this loop.
/// @param body Body of code to insert into loop.
///
/// The parameters of this function are as follows: the loop body BasicBlock;
/// the Value corresponding to the IV beginning at `indexStart` and incremented
/// each iteration by `indexInc` while less than `indexEnd`; the list of IVs
/// for this iteration of the loop (may or may not be PHIs, depending on the
/// loop bounds); the list of IVs for the next iteration of the loop (the
/// function is required to fill these in). Both these sets of IVs will be
/// arrays of equal length to the original list of IVs, in the same order. The
/// function returns the loop latch/exiting block: this block will be given the
/// branch that decides between continuing the loop and exiting from it.
///
/// @return llvm::BasicBlock* The exit block
llvm::BasicBlock *createLoop(llvm::BasicBlock *entry, llvm::BasicBlock *exit,
                             llvm::Value *indexStart, llvm::Value *indexEnd,
                             const CreateLoopOpts &opts, CreateLoopBodyFn body);

/// @brief Get the last argument of a function.
///
/// @param f An LLVM function to get an argument from.
///
/// @return An LLVM argument.
llvm::Argument *getLastArgument(llvm::Function *f);

/// @brief get the device-side size of size_t type in bytes.
unsigned getSizeTypeBytes(const llvm::Module &m);

/// @brief get a size_t type.
/// @return a LLVM IntegerType representing size_t.
llvm::IntegerType *getSizeType(const llvm::Module &m);

/// @brief Creates a wrapper function (without body), intended for calling @p F
/// @param M Containing module
/// @param F Kernel function which is being replaced
/// @param ArgTypes List of types to be used for the new function
/// @param Suffix String to which to append to the new function
/// @param OldSuffix String to which to append to the old function
/// @note This takes the metadata and debug from the original function.
///       This is intended to be used for creating a function which replaces
///       the original function but calls the original.
///
/// @note The name of the wrapper function is computed as the original name of
///       F followed by the Suffix. The original name of F is taken from F's
///       'mux-base-fn-name' attribute, if set, else it is F's name:
///
///         declare void @foo()
///         ; Function attrs "mux-base-fn-name"="baz"
///         declare void @bar()
///
///       With suffix '.wrapper', this function will produce:
///
///         declare void @foo.wrapper()
///         declare void @baz.wrapper()
///
///       With suffix '.new' and old suffix '.old', this function will produce:
///
///         declare void @foo.old()
///         ; Function attrs "mux-base-fn-name"="baz"
///         declare void @bar.old()
///
///         declare void @foo.new()
///         declare void @baz.new()
///
///       It is advised that the suffix begins with a character that may not
///       occur in the original source language, to avoid clashes with user
///       functions.
llvm::Function *createKernelWrapperFunction(
    llvm::Module &M, llvm::Function &F, llvm::ArrayRef<llvm::Type *> ArgTypes,
    llvm::StringRef Suffix, llvm::StringRef OldSuffix = "");

/// @brief As above, but creating a wrapper with the exact function signature
/// of @p F.
///
/// Copies over all parameter names and attributes.
llvm::Function *createKernelWrapperFunction(llvm::Function &F,
                                            llvm::StringRef Suffix,
                                            llvm::StringRef OldSuffix = "");

/// @brief Creates a call to a a wrapped function
///
/// Sets the calling convention and call-site attributes to match the wrapped
/// function.
///
/// @param WrappedF the function to call
/// @param Args the list of arguments to pass to the call
/// @param BB the basic block into which to insert the call. May be null, in
/// which case the call is not inserted anywhere.
/// @param InsertPt the point in BB at which to insert the call
/// @param Name the name of the call instruction. May be empty.
/// @return The call instruction
llvm::CallInst *createCallToWrappedFunction(
    llvm::Function &WrappedF, const llvm::SmallVectorImpl<llvm::Value *> &Args,
    llvm::BasicBlock *BB, llvm::BasicBlock::iterator InsertPt,
    llvm::StringRef Name = "");

/// @brief Create a binary operation corresponding to the given
/// `llvm::RecurKind` with the two provided arguments. It may not
/// necessarily return one of LLVM's in-built `BinaryOperator`s, or even one
/// operation: integer min/max operations may defer to multiple instructions or
/// intrinsics depending on the LLVM version.
///
/// @param[in] B the IRBuilder to build new instructions
/// @param[in] LHS the left-hand value for the operation
/// @param[in] RHS the right-hand value for the operation
/// @param[in] Kind the kind of operation to create
/// @return The binary operation.
llvm::Value *createBinOpForRecurKind(llvm::IRBuilderBase &B, llvm::Value *LHS,
                                     llvm::Value *RHS, llvm::RecurKind Kind);
/// @}
}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_PASS_FUNCTIONS_H_INCLUDED
