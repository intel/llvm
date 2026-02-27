//===-------- SYCLPostLink.h - post-link processing for SYCL offloading ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The SYCL post-link processing performs various transformations on LLVM
// modules after linking, including:
// - Module splitting based on different criteria (per-kernel, per-source, etc.)
// - Specialization constant handling and optimization
// - Output file generation in LLVM IR or bitcode format
// - Symbol table generation for split modules
//
// The main entry point is PerformPostLinkProcessing() which orchestrates
// the entire post-link pipeline according to the provided settings.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_POST_LINK_H
#define LLVM_SYCL_POST_LINK_H

#include "llvm/ADT/StringRef.h"
#include "llvm/SYCLLowerIR/SpecConstants.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {

namespace sycl_post_link {

/// \brief Saves a module descriptor to a file and creates a split module
/// representation.
///
/// This function serializes a ModuleDesc to either LLVM IR assembly (.ll) or
/// bitcode (.bc) format, saves split information as metadata, and constructs
/// a SplitModule object containing the file path and symbol table.
///
/// \param MD The module descriptor to save.
///
/// \param Prefix The base filename prefix for the output file. The appropriate
///               file extension (.ll or .bc) will be automatically appended
///               based on the OutputAssembly parameter.
/// \param OutputAssembly If true, saves as LLVM IR assembly (.ll file).
///                       If false, saves as LLVM bitcode (.bc file).
///
/// \returns Expected<module_split::SplitModule> containing the created
///          SplitModule with file path and symbol table on success, or an Error
///          if the file save operation fails.
///
/// \note The function modifies the input ModuleDesc by calling
///       saveSplitInformationAsMetadata() before saving.
Expected<module_split::SplitModule> saveModuleDesc(module_split::ModuleDesc &MD,
                                                   std::string Prefix,
                                                   bool OutputAssembly);

/// Parses the output table file from sycl-post-link tool.
Expected<std::vector<module_split::SplitModule>>
parseSplitModulesFromFile(StringRef File);

struct PostLinkSettings {
  StringRef OutputPrefix = "";
  bool OutputAssembly = false; // LLVM bc or LLVM IR;

  module_split::IRSplitMode SplitMode = module_split::IRSplitMode::SPLIT_NONE;

  std::optional<SpecConstantsPass::HandlingMode> SpecConstMode;
  bool GenerateModuleDescWithDefaultSpecConsts = false;
};

std::string convertSettingsToString(const PostLinkSettings &Settings);

/// \brief Performs SYCL post-link processing on a LLVM module.
///
/// This function takes a LLVM module and applies various post-link
/// transformations including module splitting, specialization constant
/// handling. The processing pipeline includes:
/// * Splitting the input module according to the specified split mode
/// * Fixing linkage of direct invoke SIMD targets
/// * Handling specialization constants (if enabled)
/// * Saving processed modules to output files
///
/// \param M The input LLVM module to process.
/// \param Settings Configuration settings that control the post-link processing
///                 behavior, including split mode, specialization constant
///                 mode, output prefix, and assembly output options.
///
/// \returns The list containing the processed split modules on success, or an
///          Error on failure. Each SplitModule represents a processed module
///          fragment with associated metadata.
///
/// \note The function generates output files with names based on
//        Settings.OutputPrefix followed by an underscore and sequential ID
//        (e.g., "prefix_0", "prefix_1").
Expected<std::vector<module_split::SplitModule>>
performPostLinkProcessing(std::unique_ptr<Module> M, PostLinkSettings Settings);

} // namespace sycl_post_link

} // namespace llvm

#endif // LLVM_SYCL_POST_LINK_H
