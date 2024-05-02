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
/// Various utlities to help with work-item and work-group scheduling.

#ifndef COMPILER_UTILS_SCHEDULING_H_INCLUDED
#define COMPILER_UTILS_SCHEDULING_H_INCLUDED

#include <cstddef>
#include <cstdint>

namespace llvm {
class Function;
class Module;
class StructType;
class Argument;
}  // namespace llvm

namespace compiler {
namespace utils {

namespace WorkItemInfoStructField {
enum Type : uint32_t {
  local_id,
  sub_group_id,
  num_sub_groups,
  max_sub_group_size,
  total
};
}

namespace WorkGroupInfoStructField {
enum Type : uint32_t {
  group_id = 0,
  num_groups,
  global_offset,
  local_size,
  work_dim,
  total
};
}

/// @brief Computes the work item info structure type for the given module.
llvm::StructType *getWorkItemInfoStructTy(llvm::Module &M);

/// @brief Computes the work item info structure type for the given module.
llvm::StructType *getWorkGroupInfoStructTy(llvm::Module &M);

/// @brief Populates an empty function with code to look up and return a value
/// from a pointer-to-struct argument.
///
/// The function may optionally have a 'rank', in which case the struct field
/// index is expected to be a 3D array of values. Ranked functions must have an
/// integer index as their first parameter. Any integer type is supported. The
/// generated code for ranked functions is given a bounds check to ensure the
/// index is less than 3. If the index is out of bounds, the default value is
/// returned.
///
/// The pointer-to-struct may be any parameter other than the index, which
/// comes first.
///
/// if !hasRankArg:
///   ; where structFieldIdx identifies the field.
///   %struct = type { ..., i64, ... }
///   declare i64 @foo(ptr %struct-ptr)
///
/// if hasRankArg:
///   ; where structFieldIdx identifies the field and the %idx parameter
///   ; identifies the sub-field.
///   %struct = type { ..., [i64, i64, i64], ... }
///   declare i64 @foo(i32 %idx, ptr %struct-ptr)
///
/// @param[in,out] F The function to define
/// @param[in] structPtrArg The pointer-to-struct argument
/// @param[in] structTy The underlying type of the pointer-to-struct argument,
/// used for offset calculations
/// @param[in] structFieldIdx The struct type's field index to load from
/// @param[in] hasRankArg True if the struct type's field index is a 3D array,
/// and thus the function's first parameter is an index parameter.
/// @param[in] defaultValue The default value returned if the index is out of
/// bounds. Only valid for ranked functions.
void populateStructGetterFunction(llvm::Function &F,
                                  llvm::Argument &structPtrArg,
                                  llvm::StructType *const structTy,
                                  uint32_t structFieldIdx, bool hasRankArg,
                                  size_t defaultValue = 0);

/// @brief Populates an empty function with code to store a value into a
/// pointer-to-struct argument.
///
/// The function may optionally have a 'rank', in which case the struct field
/// index is expected to be a 3D array of values. Ranked functions must have an
/// integer index as their first parameter. Any integer type is supported.
///
/// The value to store is the next parameter (either first or second) and the
/// pointer-to-struct may be any other unoccupied parameter.
///
/// if !hasRankArg:
///   ; where structFieldIdx identifies the field.
///   %struct = type { ..., i64, ... }
///   declare void @foo(i64 %val, ptr %struct-ptr)
///
/// if hasRankArg:
///   ; where structFieldIdx identifies the field and the %idx parameter
///   ; identifies the sub-field.
///   %struct = type { ..., [i64, i64, i64], ... }
///   declare void @foo(i32 %idx, i64 %val, ptr %struct-ptr)
///
/// Note that unlike populateStructGetterFunction, no bounds check is
/// performed. The setter functions are only available internally to the
/// compiler, and thus the indices are assumed to be within bounds.
///
/// @param[in,out] F The function to define
/// @param[in] structPtrArg The pointer-to-struct argument
/// @param[in] structTy The underlying type of the pointer-to-struct argument,
/// used for offset calculations
/// @param[in] structFieldIdx The struct type's field index to store to
/// @param[in] hasRankArg True if the struct type's field index is a 3D array,
/// and thus the function's first parameter is an index parameter.
void populateStructSetterFunction(llvm::Function &F,
                                  llvm::Argument &structPtrArg,
                                  llvm::StructType *const structTy,
                                  uint32_t structFieldIdx, bool hasRankArg);

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_SCHEDULING_H_INCLUDED
