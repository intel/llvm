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

namespace llvm {
namespace sycl {

/// Lowers ESIMD constructs after separation from regular SYCL code.
/// \SplitESIMD identifies that ESIMD splitting is requested in the compilation.
/// Returns true if the given \MD has been modified.
bool lowerESIMDConstructs(llvm::module_split::ModuleDesc &MD, bool OptLevelO0,
                          bool SplitESIMD);

} // namespace sycl
} // namespace llvm
