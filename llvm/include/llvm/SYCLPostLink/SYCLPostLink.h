// TODO: update header
//===--------- ModuleSplitter.h - split a module into callgraphs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module into call graphs. A callgraph here is a set
// of entry points with all functions reachable from them via a call. The result
// of the split is new modules containing corresponding callgraph.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_POST_LINK_H
#define LLVM_SYCL_POST_LINK_H

#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <memory>
#include <vector>
#include <string>

namespace llvm {

namespace sycl_post_link {

struct PostLinkSettings {
  StringRef OutputPrefix = "";
  bool OutputAssembly = false; // LLVM bc or LLVM IR;

  module_split::IRSplitMode SplitMode = module_split::IRSplitMode::SPLIT_NONE;

  std::optional<SpecConstantsPass::HandlingMode> SpecConstMode;
  bool GenerateModuleDescWithDefaultSpecConsts = false;
};

// TODO: add documentation.
Expected<module_split::SplitModule> saveModuleDesc(module_split::ModuleDesc &MD, std::string Prefix, bool OutputAssembly);

// TODO: add documentation.
Expected<module_split::SplitModule> saveModuleDesc(module_split::ModuleDesc &MD, std::string Prefix, bool OutputAssembly);

// TODO: move to SYCLPostLink.cpp
/// Parses the output table file from sycl-post-link tool.
Expected<std::vector<module_split::SplitModule>> parseSplitModulesFromFile(StringRef File);

Expected<std::vector<module_split::SplitModule>> PostLinkProcessing(
  std::unique_ptr<Module> M, PostLinkSettings Settings
);

} // namespace sycl_post_link

} // namespace llvm

#endif // LLVM_SYCL_POST_LINK_H
