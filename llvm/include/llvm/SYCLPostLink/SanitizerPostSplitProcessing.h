//===------------------ SanitizerPostSplitProcessing.h --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Post split sanitizer processing inserts a per-module sentinel kernel used by
// sanitizer runtimes (ASan, MSan, TSan) to uniquely identify a device image.
//===----------------------------------------------------------------------===//
#ifndef LLVM_SYCLPOSTLINK_SANITIZERPOSTSPLITPROCESSING_H
#define LLVM_SYCLPOSTLINK_SANITIZERPOSTSPLITPROCESSING_H

#include <llvm/ADT/SmallVector.h>
#include <llvm/SYCLPostLink/ModuleSplitter.h>

namespace llvm {
namespace sycl_post_link {

/// For each module descriptor that uses ASan, MSan, or TSan, inserts an empty
/// SPIR_KERNEL named "__sanitizerModule_<hash>" where the hash is an MD5
/// digest of the module's bitcode. This sentinel kernel gives the sanitizer
/// runtime a stable, unique identifier for the device image so that
/// per-module metadata (e.g. shadow memory descriptors) can be associated with
/// the correct image at runtime. Returns true if any module was modified.
bool handleSanitizers(
    llvm::SmallVectorImpl<std::unique_ptr<module_split::ModuleDesc>> &MDs);
} // namespace sycl_post_link
} // namespace llvm

#endif // LLVM_SYCLPOSTLINK_SANITIZERPOSTSPLITPROCESSING_H
