//===------------ SanitizerUtils.h - sanitizer utility functions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions for device sanitizers.
//===----------------------------------------------------------------------===//
#pragma once

#include "llvm/ADT/StringRef.h"

namespace llvm {

class Module;

namespace sycl {

constexpr StringRef ASAN_KERNEL_METADATA_PREFIX = "__AsanKernelMetadata";
constexpr StringRef MSAN_KERNEL_METADATA_PREFIX = "__MsanKernelMetadata";
constexpr StringRef TSAN_KERNEL_METADATA_PREFIX = "__TsanKernelMetadata";

bool isModuleUsingAsan(const Module &M);
bool isModuleUsingMsan(const Module &M);
bool isModuleUsingTsan(const Module &M);

} // namespace sycl
} // namespace llvm
