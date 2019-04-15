//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_CONFIG_H
#define LLD_WASM_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Support/CachePruning.h"

namespace lld {
namespace wasm {

struct Configuration {
  bool AllowUndefined;
  bool CompressRelocations;
  bool Demangle;
  bool DisableVerify;
  bool ExportAll;
  bool ExportDynamic;
  bool ExportTable;
  bool GcSections;
  bool ImportMemory;
  bool SharedMemory;
  bool ImportTable;
  bool MergeDataSegments;
  bool Pie;
  bool PrintGcSections;
  bool Relocatable;
  bool SaveTemps;
  bool Shared;
  bool StripAll;
  bool StripDebug;
  bool StackFirst;
  bool Trace;
  uint32_t GlobalBase;
  uint32_t InitialMemory;
  uint32_t MaxMemory;
  uint32_t ZStackSize;
  unsigned LTOPartitions;
  unsigned LTOO;
  unsigned Optimize;
  unsigned ThinLTOJobs;
  llvm::StringRef Entry;
  llvm::StringRef OutputFile;
  llvm::StringRef ThinLTOCacheDir;

  llvm::StringSet<> AllowUndefinedSymbols;
  std::vector<llvm::StringRef> SearchPaths;
  llvm::CachePruningPolicy ThinLTOCachePolicy;

  // True if we are creating position-independent code.
  bool Pic;
};

// The only instance of Configuration struct.
extern Configuration *Config;

} // namespace wasm
} // namespace lld

#endif
