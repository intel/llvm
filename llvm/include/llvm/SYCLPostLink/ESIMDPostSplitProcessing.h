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

#include "llvm/SYCLPostLink/ModuleSplitter.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace sycl {

/// Lowers ESIMD constructs after separation from regular SYCL code.
/// \p Optimize tells whether optimizations are allowed.
/// \p SplitESIMD identifies that ESIMD splitting is requested in the
/// compilation. Returns true if the given \p MD has been modified.
bool lowerESIMDConstructs(llvm::module_split::ModuleDesc &MD, bool Optimize,
                          bool SplitESIMD);

/// Performs ESIMD processing that happens in the following steps:
///  1) Separate ESIMD Module from SYCL code.
///     \p EmitOnlyKernelsAsEntryPoints and \p AllowDeviceImageDependencies are
///     being passed into splitting.
///  2) If \p LowerESIMD is true then ESIMD lowering pipeline is applied to the
///  ESIMD Module.
///     If \p OptimizeESIMD is true then ESIMD Module is being optimized after
///     the lowering.
///  3.1) If \p SplitESIMD is true then both ESIMD and non-ESIMD modules are
///  returned.
///  3.2) Otherwise, two Modules are being linked into one Module which is
///  returned. After the linking graphs become disjoint because functions
///  shared between graphs are cloned and renamed.
///
/// \p Modified value indicates whether the Module has been modified.
/// \p SplitOccured value indicates whether split has occured before or during
/// function's invocation.
Expected<SmallVector<module_split::ModuleDesc, 2>>
handleESIMD(llvm::module_split::ModuleDesc MDesc,
            llvm::module_split::IRSplitMode SplitMode,
            bool EmitOnlyKernelsAsEntryPoints,
            bool AllowDeviceImageDependencies, bool LowerESIMD, bool SplitESIMD,
            bool OptimizeESIMDModule, bool &Modified, bool &SplitOccurred);

} // namespace sycl
} // namespace llvm
