//===--- sanitizer_stackdepot.hpp - Stack traces keyed by origin id -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sanitizer_stacktrace.hpp"

namespace llvm::offload::sanitizer {

const uint32_t kHeapTypeCount = 5;
enum HeapType { DeviceUSM, HostUSM, SharedUSM, Local, ExportableMem };

inline const char *ToString(HeapType Type) {
  switch (Type) {
  case HeapType::DeviceUSM:
    return "Device USM";
  case HeapType::HostUSM:
    return "Host USM";
  case HeapType::SharedUSM:
    return "Shared USM";
  case HeapType::Local:
    return "Local Memory";
  case HeapType::ExportableMem:
    return "Exportable Memory";
  default:
    return "Unknown Heap Type";
  }
}

// Save/load stack with corresponding stack id
void StackDepotPut(uint32_t Id, StackTrace &Stack);
StackTrace StackDepotGet(uint32_t Id);

} // namespace llvm::offload::sanitizer
