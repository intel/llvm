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
/// Make opaque structure types unique.

#ifndef COMPILER_UTILS_UNIQUE_OPAQUE_STRUCTS_PASS_H_INCLUDED
#define COMPILER_UTILS_UNIQUE_OPAQUE_STRUCTS_PASS_H_INCLUDED

#include <llvm/IR/PassManager.h>

namespace compiler {
namespace utils {

/// @addtogroup utils
/// @{

/// @brief This pass replaces instances of suffixed opaque structure types
/// with unsuffixed versions if an unsuffixed version exists in the context.
///
/// When linking together two modules that declare the same opaque struct
/// type, or deserializing a module referencing an opaque struct type in a
/// context that already contains an opaque type with the same name, LLVM
/// will attempt to resolve the clash by appending a suffix to the name in
/// module. For example, deserializing a module referencing the
/// opencl.event_t in a context that already has this type will result in
/// the references all being renamed to opencl.event_t.0. This is
/// problematic if passes rely on the name of the struct to identify them.
/// This pass can be used to  resolve this issue by searching for
/// problematic types and replacing them with their unsuffixed version.
class UniqueOpaqueStructsPass
    : public llvm::PassInfoMixin<UniqueOpaqueStructsPass> {
 public:
  UniqueOpaqueStructsPass() = default;
  llvm::PreservedAnalyses run(llvm::Module &, llvm::ModuleAnalysisManager &);
};

}  // namespace utils
}  // namespace compiler

#endif  // COMPILER_UTILS_UNIQUE_OPAQUE_STRUCTS_PASS_H_INCLUDED
