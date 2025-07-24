//= SpecializationConstants.h - Processing of SYCL Specialization Constants ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Specialization constants processing consists of lowering and generation
// of new module with spec consts replaced by default values.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_POST_LINK_SPECIALIZATION_CONSTANTS_H
#define LLVM_SYCL_POST_LINK_SPECIALIZATION_CONSTANTS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"

#include <optional>

namespace llvm {
namespace sycl {

/// Handling consists of SpecConsts's lowering depending on the given
/// \p Mode. If \p Mode is std::nullopt, then no lowering happens.
/// If \p GenerateModuleDescWithDefaultSpecConsts is true, then a generation
/// of new modules with specialization constants replaced by default values
/// happens and the result is written in \p NewModuleDescs.
/// Otherwise, \p NewModuleDescs is expected to be nullptr.
///
/// \returns Boolean value indicating whether the lowering has changed the input
/// modules.
bool handleSpecializationConstants(
    llvm::SmallVectorImpl<module_split::ModuleDesc> &MDs,
    std::optional<SpecConstantsPass::HandlingMode> Mode,
    bool GenerateModuleDescWithDefaultSpecConsts = false,
    llvm::SmallVectorImpl<module_split::ModuleDesc> *NewModuleDescs = nullptr);

} // namespace sycl
} // namespace llvm

#endif // LLVM_SYCL_POST_LINK_SPECIALIZATION_CONSTANTS_H
