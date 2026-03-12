//===------------ Utils.h - Utility functions for SYCL Offloading ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functions for constructing the output from SYCL Offloading.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYCL_POST_LINK_UTILS_H
#define LLVM_SYCL_POST_LINK_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/SYCLPostLink/ComputeModuleRuntimeInfo.h"
#include "llvm/SYCLPostLink/ModuleSplitter.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/PropertySetIO.h"
#include "llvm/Support/SimpleTable.h"

#include <string>

namespace llvm {
namespace sycl_post_link {

/// \brief Saves an LLVM module to a file in either bitcode or LLVM assembly format.
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
///
/// \note This function is part of the SYCL post-link toolchain and is
///       typically used to save processed device code modules.
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

struct IrPropSymFilenameTriple {
  std::string Ir;
  std::string Prop;
  std::string Sym;
};

void addTableRow(util::SimpleTable &Table,
                 const IrPropSymFilenameTriple &RowData);

struct FinalModule {
  module_split::ModuleDesc &MD;
  std::string IR;                             // Filepath to saved IR
  std::string Symbols;                        // Symbol table
  llvm::util::PropertySetRegistry Properties;
};

/// \brief Generates a final module package with IR, symbols, and properties for SYCL post-link processing.
///
/// This function takes a module descriptor from the splitting phase and creates
/// a complete FinalModule containing all necessary components: IR code, symbol
/// table, and device properties. It handles module cleanup, filename generation,
/// and prepares the module for target-specific output generation.
///
/// \param MD Module descriptor containing the LLVM module and associated metadata
///           from the splitting phase. Will be modified during processing.
/// \param GenerateSymbols If true, computes and includes a symbol table containing
///                        exported/imported symbols for the module.
/// \param I Index used for generating unique filenames when multiple modules
///          are being processed (typically from module splitting).
/// \param OutputPrefix Base prefix for generated output filenames.
/// \param IRFilename If non-empty, uses this as the IR filename instead of
///                   generating a new one. The IR will not be saved to disk.
/// \param OutputAssembly If true, saves IR as human-readable assembly (.ll);
///                       if false, saves as bitcode (.bc).
/// \param AllowDeviceImageDependencies If true, preserves inter-module dependencies
///                                     during cleanup; if false, removes them.
/// \param GlobalProps Global binary image properties to be incorporated into
///                    the module's property set.
/// \param SplitMode The module splitting mode used, affects property generation.
///
/// \return Expected<FinalModule> containing the complete module package on success,
///         or an Error if IR saving or other operations fail.
///
/// \details The function performs the following operations:
/// 1. **Metadata Preservation**: Saves split information as LLVM metadata
/// 2. **Suffix Determination**: Adds "_esimd" suffix for ESIMD modules
/// 3. **IR Handling**: Either uses provided filename or generates new one and saves IR
/// 4. **Module Cleanup**: For non-deviceLib modules, performs cleanup to remove
///    unused code and handle dependencies based on AllowDeviceImageDependencies
/// 5. **Symbol Generation**: Optionally computes symbol table from module entries
/// 6. **Property Computation**: Generates comprehensive device properties including
///    global properties and split-mode specific settings
///
/// The generated IR filename follows the pattern:
/// `{OutputPrefix}[_esimd]_{I}.{ll|bc}`
///
/// \note SYCL device library modules skip the cleanup phase as they only contain
///       exported functions intended for use by other device modules.
Expected<FinalModule>
generateFinalModule(module_split::ModuleDesc &MD, bool GenerateSymbols, int I,
                    const Twine &OutputPrefix, StringRef IRFilename,
                    bool OutputAssembly, bool AllowDeviceImageDependencies,
                    sycl::GlobalBinImageProps GlobalProps,
                    module_split::IRSplitMode SplitMode);

/// A pair of target and filename used for filtering by target's aspects.
struct TargetFilenamePair {
  std::string Target;
  std::string Filename;
};

/// \brief Saves a final module and generates associated files for all compatible targets.
///
/// This function processes a finalized SYCL module and generates the necessary
/// output files (symbols, properties) for each compatible target. It populates
/// output tables with file information and handles target-specific property
/// generation including compile target requirements.
///
/// \param FM The final module containing IR, symbols, properties, and metadata
///           to be processed and saved.
/// \param OutTables Vector of output tables where file information will be
///                  recorded. Each table corresponds to a target in OutputFiles.
/// \param GenerateSymbols If true, generates and saves a symbols file (.sym)
///                        containing exported/imported symbols from the module.
/// \param I Index used for generating unique filenames (typically module or
///          split index).
/// \param OutputPrefix Base prefix for all generated output filenames.
/// \param OutputFiles Array of target-filename pairs specifying the targets
///                    and their associated output files.
/// \param DoPropGen If true, generates target-specific property files (.prop)
///                  with device requirements and compile target information.
/// \param Suffix Additional suffix to append to generated filenames for
///               disambiguation.
///
/// \return Error::success() on successful processing of all compatible targets,
///         or the first error encountered during file generation.
///
/// \details The function performs the following operations for each target:
/// 1. **Symbol Generation**: If enabled, writes module symbols to a .sym file
/// 2. **Target Compatibility**: Checks if each target is compatible with the
///    module's metadata before processing
/// 3. **Property Generation**: For compatible targets, creates .prop files with:
///    - Base module properties from FM.Properties
///    - Target-specific "compile_target" requirement (if target specified)
/// 4. **Table Population**: Adds file information (IR, properties, symbols) to
///    the corresponding output table
///
/// The generated filenames follow the pattern:
/// - Symbols: `{OutputPrefix}{Suffix}_{I}.sym`
/// - Properties: `{OutputPrefix}[_{Target}]_{I}.prop`
llvm::Error saveFinalModuleForEveryTarget(
    const FinalModule &FM,
    const std::vector<std::unique_ptr<util::SimpleTable>> &OutTables,
    bool GenerateSymbols, int I, const Twine &OutputPrefix,
    ArrayRef<TargetFilenamePair> OutputFiles, bool DoPropGen, StringRef Suffix);

} // namespace sycl_post_link
} // namespace llvm

#endif // LLVM_SYCL_POST_LINK_UTILS_H
