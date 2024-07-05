//===-------- RecordSYCLAspectNames.h - RecordSYCLAspectNames Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The !sycl_used_aspects metadata is populated from C++ attributes and
// further populated by the SYCLPropagateAspectsPass that describes which
// apsects a function uses. The format of this metadata initially just an
// integer value corresponding to the enum value in C++. The !sycl_aspects
// named metadata contains the associations from aspect values to aspect names.
// These associations are needed later in sycl-post-link, but we drop
// !sycl_aspects before that to avoid LLVM IR bloat, so this pass takes
// the associations from !sycl_aspects and then updates all the
// !sycl_used_aspects metadata to include the aspect names, which allows us
// to preserve these associations.
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_RECORD_SYCL_ASPECT_NAMES
#define LLVM_RECORD_SYCL_ASPECT_NAMES

#include "llvm/IR/PassManager.h"

namespace llvm {

class RecordSYCLAspectNamesPass
    : public PassInfoMixin<RecordSYCLAspectNamesPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // namespace llvm

#endif // LLVM_RECORD_SYCL_ASPECT_NAMES
