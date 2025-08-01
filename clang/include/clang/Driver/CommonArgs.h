//===--- CommonArgs.h - Args handling for multiple toolchains ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H

#include "clang/Basic/CodeGenOptions.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Multilib.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CodeGen.h"

namespace clang {
namespace driver {
namespace tools {

void addPathIfExists(const Driver &D, const Twine &Path,
                     ToolChain::path_list &Paths);

void AddLinkerInputs(const ToolChain &TC, const InputInfoList &Inputs,
                     const llvm::opt::ArgList &Args,
                     llvm::opt::ArgStringList &CmdArgs, const JobAction &JA);

void addLinkerCompressDebugSectionsOption(const ToolChain &TC,
                                          const llvm::opt::ArgList &Args,
                                          llvm::opt::ArgStringList &CmdArgs);

void claimNoWarnArgs(const llvm::opt::ArgList &Args);

bool addSanitizerRuntimes(const ToolChain &TC, const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs);

void linkSanitizerRuntimeDeps(const ToolChain &TC,
                              const llvm::opt::ArgList &Args,
                              llvm::opt::ArgStringList &CmdArgs);

bool addXRayRuntime(const ToolChain &TC, const llvm::opt::ArgList &Args,
                    llvm::opt::ArgStringList &CmdArgs);

void linkXRayRuntimeDeps(const ToolChain &TC, const llvm::opt::ArgList &Args,
                         llvm::opt::ArgStringList &CmdArgs);

void AddRunTimeLibs(const ToolChain &TC, const Driver &D,
                    llvm::opt::ArgStringList &CmdArgs,
                    const llvm::opt::ArgList &Args);

void AddStaticDeviceLibsLinking(Compilation &C, const Tool &T,
                                const JobAction &JA,
                                const InputInfoList &Inputs,
                                const llvm::opt::ArgList &DriverArgs,
                                llvm::opt::ArgStringList &CmdArgs,
                                StringRef Arch, StringRef Target,
                                bool isBitCodeSDL);
void AddStaticDeviceLibsPostLinking(const Driver &D,
                                    const llvm::opt::ArgList &DriverArgs,
                                    llvm::opt::ArgStringList &CmdArgs,
                                    StringRef Arch, StringRef Target,
                                    bool isBitCodeSDL, bool postClangLink);
void AddStaticDeviceLibs(Compilation *C, const Tool *T, const JobAction *JA,
                         const InputInfoList *Inputs, const Driver &D,
                         const llvm::opt::ArgList &DriverArgs,
                         llvm::opt::ArgStringList &CmdArgs, StringRef Arch,
                         StringRef Target, bool isBitCodeSDL);

const char *SplitDebugName(const JobAction &JA, const llvm::opt::ArgList &Args,
                           const InputInfo &Input, const InputInfo &Output);

void SplitDebugInfo(const ToolChain &TC, Compilation &C, const Tool &T,
                    const JobAction &JA, const llvm::opt::ArgList &Args,
                    const InputInfo &Output, const char *OutFile);

void addLTOOptions(const ToolChain &ToolChain, const llvm::opt::ArgList &Args,
                   llvm::opt::ArgStringList &CmdArgs, const InputInfo &Output,
                   const InputInfoList &Inputs, bool IsThinLTO);

const char *RelocationModelName(llvm::Reloc::Model Model);

std::tuple<llvm::Reloc::Model, unsigned, bool>
ParsePICArgs(const ToolChain &ToolChain, const llvm::opt::ArgList &Args);

unsigned ParseFunctionAlignment(const ToolChain &TC,
                                const llvm::opt::ArgList &Args);

void addDebugInfoKind(llvm::opt::ArgStringList &CmdArgs,
                      llvm::codegenoptions::DebugInfoKind DebugInfoKind);

llvm::codegenoptions::DebugInfoKind
debugLevelToInfoKind(const llvm::opt::Arg &A);

// Extract the integer N from a string spelled "-dwarf-N", returning 0
// on mismatch. The StringRef input (rather than an Arg) allows
// for use by the "-Xassembler" option parser.
unsigned DwarfVersionNum(StringRef ArgValue);
// Find a DWARF format version option.
// This function is a complementary for DwarfVersionNum().
const llvm::opt::Arg *getDwarfNArg(const llvm::opt::ArgList &Args);
unsigned getDwarfVersion(const ToolChain &TC, const llvm::opt::ArgList &Args);

void AddAssemblerKPIC(const ToolChain &ToolChain,
                      const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs);

void addArchSpecificRPath(const ToolChain &TC, const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs);
void addOpenMPRuntimeLibraryPath(const ToolChain &TC,
                                 const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs);
/// Returns true, if an OpenMP runtime has been added.
bool addOpenMPRuntime(const Compilation &C, llvm::opt::ArgStringList &CmdArgs,
                      const ToolChain &TC, const llvm::opt::ArgList &Args,
                      bool ForceStaticHostRuntime = false,
                      bool IsOffloadingHost = false, bool GompNeedsRT = false);

/// Adds offloading options for OpenMP host compilation to \p CmdArgs.
void addOpenMPHostOffloadingArgs(const Compilation &C, const JobAction &JA,
                                 const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs);

void addHIPRuntimeLibArgs(const ToolChain &TC, Compilation &C,
                          const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs);

void addAsNeededOption(const ToolChain &TC, const llvm::opt::ArgList &Args,
                       llvm::opt::ArgStringList &CmdArgs, bool as_needed);

llvm::opt::Arg *getLastCSProfileGenerateArg(const llvm::opt::ArgList &Args);
llvm::opt::Arg *getLastProfileUseArg(const llvm::opt::ArgList &Args);
llvm::opt::Arg *getLastProfileSampleUseArg(const llvm::opt::ArgList &Args);

bool isObjCAutoRefCount(const llvm::opt::ArgList &Args);

llvm::StringRef getLTOParallelism(const llvm::opt::ArgList &Args,
                                  const Driver &D);

bool areOptimizationsEnabled(const llvm::opt::ArgList &Args);

bool isDependentLibAdded(const llvm::opt::ArgList &Args, StringRef Lib);

bool isUseSeparateSections(const llvm::Triple &Triple);
// Parse -mtls-dialect=. Return true if the target supports both general-dynamic
// and TLSDESC, and TLSDESC is requested.
bool isTLSDESCEnabled(const ToolChain &TC, const llvm::opt::ArgList &Args);

/// \p EnvVar is split by system delimiter for environment variables.
/// If \p ArgName is "-I", "-L", or an empty string, each entry from \p EnvVar
/// is prefixed by \p ArgName then added to \p Args. Otherwise, for each
/// entry of \p EnvVar, \p ArgName is added to \p Args first, then the entry
/// itself is added.
void addDirectoryList(const llvm::opt::ArgList &Args,
                      llvm::opt::ArgStringList &CmdArgs, const char *ArgName,
                      const char *EnvVar);

void AddTargetFeature(const llvm::opt::ArgList &Args,
                      std::vector<StringRef> &Features,
                      llvm::opt::OptSpecifier OnOpt,
                      llvm::opt::OptSpecifier OffOpt, StringRef FeatureName);

std::string getCPUName(const Driver &D, const llvm::opt::ArgList &Args,
                       const llvm::Triple &T, bool FromAs = false);

void getTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                       const llvm::opt::ArgList &Args,
                       llvm::opt::ArgStringList &CmdArgs, bool ForAS,
                       bool IsAux = false);

/// Iterate \p Args and convert -mxxx to +xxx and -mno-xxx to -xxx and
/// append it to \p Features.
///
/// Note: Since \p Features may contain default values before calling
/// this function, or may be appended with entries to override arguments,
/// entries in \p Features are not unique.
void handleTargetFeaturesGroup(const Driver &D, const llvm::Triple &Triple,
                               const llvm::opt::ArgList &Args,
                               std::vector<StringRef> &Features,
                               llvm::opt::OptSpecifier Group);

/// If there are multiple +xxx or -xxx features, keep the last one.
SmallVector<StringRef> unifyTargetFeatures(ArrayRef<StringRef> Features);

/// Handles the -save-stats option and returns the filename to save statistics
/// to.
SmallString<128> getStatsFileName(const llvm::opt::ArgList &Args,
                                  const InputInfo &Output,
                                  const InputInfo &Input, const Driver &D);

/// \p Flag must be a flag accepted by the driver.
void addMultilibFlag(bool Enabled, const StringRef Flag,
                     Multilib::flags_list &Flags);

void addX86AlignBranchArgs(const Driver &D, const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs, bool IsLTO,
                           const StringRef PluginOptPrefix = "");

void checkAMDGPUCodeObjectVersion(const Driver &D,
                                  const llvm::opt::ArgList &Args);

unsigned getAMDGPUCodeObjectVersion(const Driver &D,
                                    const llvm::opt::ArgList &Args);

bool haveAMDGPUCodeObjectVersionArgument(const Driver &D,
                                         const llvm::opt::ArgList &Args);

void addMachineOutlinerArgs(const Driver &D, const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs,
                            const llvm::Triple &Triple, bool IsLTO,
                            const StringRef PluginOptPrefix = "");

void addOpenMPDeviceRTL(const Driver &D, const llvm::opt::ArgList &DriverArgs,
                        llvm::opt::ArgStringList &CC1Args,
                        StringRef BitcodeSuffix, const llvm::Triple &Triple,
                        const ToolChain &HostTC);

void addOutlineAtomicsArgs(const Driver &D, const ToolChain &TC,
                           const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs,
                           const llvm::Triple &Triple);
void addOffloadCompressArgs(const llvm::opt::ArgList &TCArgs,
                            llvm::opt::ArgStringList &CmdArgs);
void addMCModel(const Driver &D, const llvm::opt::ArgList &Args,
                const llvm::Triple &Triple,
                const llvm::Reloc::Model &RelocationModel,
                llvm::opt::ArgStringList &CmdArgs);

/// Handle the -f{no}-color-diagnostics and -f{no}-diagnostics-colors options.
void handleColorDiagnosticsArgs(const Driver &D, const llvm::opt::ArgList &Args,
                                llvm::opt::ArgStringList &CmdArgs);

/// Add backslashes to escape spaces and other backslashes.
/// This is used for the space-separated argument list specified with
/// the -dwarf-debug-flags option.
void escapeSpacesAndBackslashes(const char *Arg,
                                llvm::SmallVectorImpl<char> &Res);

/// Join the args in the given ArgList, escape spaces and backslashes and
/// return the joined string. This is used when saving the command line as a
/// result of using either the -frecord-command-line or -grecord-command-line
/// options. The lifetime of the returned c-string will match that of the Args
/// argument.
const char *renderEscapedCommandLine(const ToolChain &TC,
                                     const llvm::opt::ArgList &Args);

/// Check if the command line should be recorded in the object file. This is
/// done if either -frecord-command-line or -grecord-command-line options have
/// been passed. This also does some error checking since -frecord-command-line
/// is currently only supported on ELF platforms. The last two boolean
/// arguments are out parameters and will be set depending on the command
/// line options that were passed.
bool shouldRecordCommandLine(const ToolChain &TC,
                             const llvm::opt::ArgList &Args,
                             bool &FRecordCommandLine,
                             bool &GRecordCommandLine);

void renderCommonIntegerOverflowOptions(const llvm::opt::ArgList &Args,
                                        llvm::opt::ArgStringList &CmdArgs);

bool shouldEnableVectorizerAtOLevel(const llvm::opt::ArgList &Args,
                                    bool isSlpVec);

/// Enable -floop-interchange based on the optimization level selected.
void handleInterchangeLoopsArgs(const llvm::opt::ArgList &Args,
                                llvm::opt::ArgStringList &CmdArgs);

/// Enable -fvectorize based on the optimization level selected.
void handleVectorizeLoopsArgs(const llvm::opt::ArgList &Args,
                              llvm::opt::ArgStringList &CmdArgs);

/// Enable -fslp-vectorize based on the optimization level selected.
void handleVectorizeSLPArgs(const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs);

// Parse -mprefer-vector-width=. Return the Value string if well-formed.
// Otherwise, return an empty string and issue a diagnosic message if needed.
StringRef parseMPreferVectorWidthOption(clang::DiagnosticsEngine &Diags,
                                        const llvm::opt::ArgList &Args);

// Parse -mrecip. Return the Value string if well-formed.
// Otherwise, return an empty string and issue a diagnosic message if needed.
StringRef parseMRecipOption(clang::DiagnosticsEngine &Diags,
                            const llvm::opt::ArgList &Args);

} // end namespace tools
} // end namespace driver
} // end namespace clang

clang::CodeGenOptions::FramePointerKind
getFramePointerKind(const llvm::opt::ArgList &Args, const llvm::Triple &Triple);

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_COMMONARGS_H
