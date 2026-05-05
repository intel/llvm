//===------------ Utils.h - Utility functions for SYCL Offloading ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Low-level utility functions for SYCL post-link processing.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_POST_LINK_UTILS_H
#define LLVM_SYCL_POST_LINK_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace sycl_post_link {

/// \brief Saves an LLVM module to a file in either bitcode or LLVM assembly
/// format.
///
/// \param M The LLVM module to be saved.
/// \param Filename The path where the module should be saved. If the file
///                 exists, it will be overwritten.
/// \param OutputAssembly If true, saves the module as human-readable LLVM IR
///                       assembly (.ll format). If false, saves as bitcode
///                       (.bc format).
///
/// \return Error::success() on successful write, or a StringError containing
///         details about the failure (typically file I/O errors).
llvm::Error saveModuleIR(Module &M, StringRef Filename, bool OutputAssembly);

/// Checks if the given target and module are compatible.
/// A target and module are compatible if all the optional kernel features
/// the module uses are supported by that target (i.e. that module can be
/// compiled for that target and then be executed on that target). This
/// information comes from the device config file (DeviceConfigFile.td).
/// For example, the intel_gpu_tgllp target does not support fp64 - therefore,
/// a module using fp64 would *not* be compatible with intel_gpu_tgllp.
bool isTargetCompatibleWithModule(const std::string &Target,
                                  module_split::ModuleDesc &IrMD);

/// \brief Saves module properties to a file with optional target specification.
///
/// \param MD Module descriptor containing the module and entry points
/// \param GlobProps Global binary image properties to include
/// \param Filename Output file path for the properties
/// \param Target Optional target name to add as compile_target requirement
/// \param AllowDeviceImageDependencies If true, preserves inter-module
/// dependencies
/// \param SplitMode The module splitting mode used
///
/// \return Error::success() on success, or error details on failure
llvm::Error saveModuleProperties(const module_split::ModuleDesc &MD,
                                 const sycl::GlobalBinImageProps &GlobProps,
                                 StringRef Filename, StringRef Target,
                                 bool AllowDeviceImageDependencies,
                                 module_split::IRSplitMode SplitMode);

/// \brief Saves the symbol table (entry point names) for a module to a file.
///
/// \param MD Module descriptor containing the module and entry points
/// \param Filename Output file path for the symbol table
///
/// \return Error::success() on success, or error details on failure
llvm::Error saveModuleSymbolTable(const module_split::ModuleDesc &MD,
                                  StringRef Filename);

} // namespace sycl_post_link
} // namespace llvm

#endif // LLVM_SYCL_POST_LINK_UTILS_H
