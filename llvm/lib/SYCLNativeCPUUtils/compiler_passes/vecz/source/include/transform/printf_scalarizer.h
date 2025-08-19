// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// @file

#ifndef VECZ_TRANSFORM_PRINTF_SCALARIZER_H_INCLUDED
#define VECZ_TRANSFORM_PRINTF_SCALARIZER_H_INCLUDED

#include <string>

namespace llvm {
class Module;
class User;
class Instruction;
template <typename T, unsigned N>
class SmallVector;
class GlobalVariable;
class Value;
class CallInst;
}  // namespace llvm

namespace vecz {

/// @brief An enumeration of errors that can occur when processing a format
/// string.
enum EnumPrintfError {
  kPrintfError_success,
  kPrintfError_fail,
  kPrintfError_invalidFormatString
};

/// @brief Retrieves a module-level global variable for a printf format string
/// from an Value.
/// @param[in] op The value that uses a global variable representing a printf
/// format string.
/// @return The module-level global variable for the printf format string.
llvm::GlobalVariable *GetFormatStringAsValue(llvm::Value *op);

/// @brief Extracts the raw string contents from a module-level global variable
/// containing a printf format string.
///
/// The @p op parameter must be an GlobalVariable with an initializer.
///
/// @param[in] op The module-level global variable for a printf format string.
/// @return The raw string contents of the format string global variable, or ""
/// if there was an error.
std::string GetFormatStringAsString(llvm::Value *op);

/// @brief Creates a global variable for a scalarized format string.
/// @param[in,out] module The parent module given to the pass.
/// @param[in] string_value The GlobalVariable for the old format string,
/// used to copy attributes over.
/// @param[in]  new_format_string The scalarized format string to create a
/// global variable from.
/// @return The newly created global variable for the format string.
llvm::GlobalVariable *GetNewFormatStringAsGlobalVar(
    llvm::Module &module, llvm::GlobalVariable *const string_value,
    const std::string &new_format_string);

/// @brief This function transforms an OpenCL printf format string into a
/// C99-conformant one.

/// Its main job is to scalarize vector format specifiers into scalarized form.
/// It does this by taking a vector specifier and determining the specifier
/// corresponding to each vector element. It then emits the element specifier
/// into the new format string for each element in the vector, separated by a
/// comma.
///
/// Special care needs to be taken for modifiers that aren't supported by C99
/// such as the 'hl' length modifier. The new format string will have 'hl'
/// stripped out.
///
/// Examples:
/// @code{.cpp}
/// // vector 2, 8-bit sized hexadecimal integers
/// "%v2hhx"  --> "%hhx,%hhx"
/// // vector 4, 32-bit sized floats
/// "%v4hlf"  --> "%f,%f,%f,%f"
/// @endcode
///
/// It also does some checking to ensure the printf string is conformant to the
/// OpenCL 1.2 specification, and returns an error if it is not.
/// @param[in] str The format string to scalarize and check.
/// @param[out] new_str The new, scalarized, format string.
/// @return The status of the scalarization (kPrintfError_success on success,
/// otherwise kPrintfError_invalidFormatString if we detected an illegal OpenCL
/// printf format string).
EnumPrintfError ScalarizeAndCheckFormatString(const std::string &str,
                                              std::string &new_str);

/// @brief Builds a new scalarized printf call given an existing call and a new
/// format string.
///
/// @param[in,out] module The parent module given to the pass.
/// @param[in] old_inst The old call to the printf function.
/// @param[in] new_format_string_gvar The module-level global variable for the
/// new format string.
/// @return A new call instruction to the new printf function.
llvm::Instruction *BuildNewPrintfCall(
    llvm::Module &module, llvm::CallInst *const old_inst,
    llvm::GlobalVariable *const new_format_string_gvar);
}  // namespace vecz

#endif  // VECZ_TRANSFORM_PRINTF_SCALARIZER_H_INCLUDED
