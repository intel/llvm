//===--------- ESIMDPostSplitProcessing.h - Post split ESIMD Processing ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Post split ESIMD processing contains of lowering ESIMD constructs and
// required optimimizations.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_POST_LINK_ESIMD_POST_SPLIT_PROCESSING_H
#define LLVM_SYCL_POST_LINK_ESIMD_POST_SPLIT_PROCESSING_H

#include "llvm/SYCLPostLink/ModuleSplitter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {
namespace sycl {

struct ESIMDProcessingOptions {
  llvm::module_split::IRSplitMode SplitMode =
      llvm::module_split::IRSplitMode::SPLIT_NONE;
  bool EmitOnlyKernelsAsEntryPoints = false;
  bool AllowDeviceImageDependencies = false;
  bool LowerESIMD = false;
  bool SplitESIMD = false;
  unsigned OptLevel = 0;
  bool ForceDisableESIMDOpt = false;
};

/// Lowers ESIMD constructs after separation from regular SYCL code.
/// \p Options.SplitESIMD identifies that ESIMD splitting is requested in the
/// compilation. Returns true if the given \p MD has been modified.
bool lowerESIMDConstructs(llvm::module_split::ModuleDesc &MD,
                          const ESIMDProcessingOptions &Options);

/// Performs ESIMD processing that happens in the following steps:
///  1) Separate ESIMD Module from SYCL code.
///     \p Options.EmitOnlyKernelsAsEntryPoints and
///     \p Options.AllowDeviceImageDependencies are being used in the splitting.
///  2) If \p Options.LowerESIMD is true then ESIMD lowering pipeline is applied
///  to the ESIMD Module.
///     If \p Options.OptLevel is not O0 then ESIMD Module is being optimized
///     after the lowering.
///  3.1) If \p Options.SplitESIMD is true then both ESIMD and non-ESIMD modules
///  are returned.
///  3.2) Otherwise, two Modules are being linked into one Module which is
///     returned. After the linking graphs become disjoint because functions
///     shared between graphs are cloned and renamed.
///
/// \p Modified value indicates whether the Module has been modified.
/// \p SplitOccurred value indicates whether split has occurred before or during
/// function's invocation.
Expected<SmallVector<std::unique_ptr<module_split::ModuleDesc>, 2>>
handleESIMD(std::unique_ptr<llvm::module_split::ModuleDesc> MDesc,
            const ESIMDProcessingOptions &Options, bool &Modified,
            bool &SplitOccurred);

} // namespace sycl
} // namespace llvm

#endif // LLVM_SYCL_POST_LINK_ESIMD_POST_SPLIT_PROCESSING_H
