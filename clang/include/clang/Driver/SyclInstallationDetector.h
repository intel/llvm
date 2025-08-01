//===-- SyclInstallationDetector.h - SYCL Instalation Detector --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H
#define LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H

#include "clang/Basic/Cuda.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"

namespace clang {
namespace driver {

// This is a mapping between the user provided --offload-arch value for Intel
// GPU targets and the spir64_gen device name accepted by OCLOC (the Intel GPU
// AOT compiler).
StringRef mapIntelGPUArchName(StringRef ArchName);

class SYCLInstallationDetector {
public:
  SYCLInstallationDetector(const Driver &D);
  SYCLInstallationDetector(const Driver &D, const llvm::Triple &HostTriple,
                           const llvm::opt::ArgList &Args);

  /// \brief Find and return the path to the libspirv library for the target
  /// \return The path to the libspirv library if found, otherwise nullptr.
  /// The lifetime of the returned string is managed by \p Args.
  const char *findLibspirvPath(const llvm::Triple &DeviceTriple,
                               const llvm::opt::ArgList &Args,
                               const llvm::Triple &HostTriple) const;

  void addLibspirvLinkArgs(const llvm::Triple &DeviceTriple,
                           const llvm::opt::ArgList &DriverArgs,
                           const llvm::Triple &HostTriple,
                           llvm::opt::ArgStringList &CC1Args) const;

  void getSYCLDeviceLibPath(
      llvm::SmallVector<llvm::SmallString<128>, 4> &DeviceLibPaths) const;
  void addSYCLIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                          llvm::opt::ArgStringList &CC1Args) const;
  void print(llvm::raw_ostream &OS) const;

private:
  const Driver &D;
  llvm::SmallVector<llvm::SmallString<128>, 4> InstallationCandidates;
};


class Command;

namespace tools {
namespace SYCL {

void constructLLVMForeachCommand(Compilation &C, const JobAction &JA,
                                 std::unique_ptr<Command> InputCommand,
                                 const InputInfoList &InputFiles,
                                 const InputInfo &Output, const Tool *T,
                                 StringRef Increment, StringRef Ext = "out",
                                 StringRef ParallelJobs = "");

// Provides a vector of device library names that are associated with the
// given triple and AOT information.
SmallVector<std::string, 8> getDeviceLibraries(const Compilation &C,
                                               const llvm::Triple &TargetTriple,
                                               bool IsSpirvAOT);

// Populates the SYCL device traits macros.
void populateSYCLDeviceTraitsMacrosArgs(
    Compilation &C, const llvm::opt::ArgList &Args,
    const SmallVectorImpl<std::pair<const ToolChain *, StringRef>> &Targets);

bool shouldDoPerObjectFileLinking(const Compilation &C);
// Runs llvm-spirv to convert spirv to bc, llvm-link, which links multiple LLVM
// bitcode. Converts generated bc back to spirv using llvm-spirv, wraps with
// offloading information. Finally compiles to object using llc
class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("SYCL::Linker", "sycl-link", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  /// \return llvm-link output file name.
  const char *constructLLVMLinkCommand(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       const llvm::opt::ArgList &Args,
                                       llvm::StringRef SubArchName,
                                       llvm::StringRef OutputFilePrefix,
                                       const InputInfoList &InputFiles) const;
};

namespace gen {

class LLVM_LIBRARY_VISIBILITY BackendCompiler : public Tool {
public:
  BackendCompiler(const ToolChain &TC)
      : Tool("gen::BackendCompiler", "gen compiler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

StringRef resolveGenDevice(StringRef DeviceName);
SmallString<64> getGenDeviceMacro(StringRef DeviceName);
StringRef getGenGRFFlag(StringRef GRFMode);

// Prefix for GPU specific targets used for -fsycl-targets
constexpr char IntelGPU[] = "intel_gpu_";
constexpr char NvidiaGPU[] = "nvidia_gpu_";
constexpr char AmdGPU[] = "amd_gpu_";

template <auto GPUArh> std::optional<StringRef> isGPUTarget(StringRef Target) {
  // Handle target specifications that resemble '(intel, nvidia, amd)_gpu_*'
  // here.
  if (Target.starts_with(GPUArh)) {
    return resolveGenDevice(Target);
  }
  return std::nullopt;
}

} // end namespace gen

namespace x86_64 {

class LLVM_LIBRARY_VISIBILITY BackendCompiler : public Tool {
public:
  BackendCompiler(const ToolChain &TC)
      : Tool("x86_64::BackendCompiler", "x86_64 compiler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

} // end namespace x86_64

} // end namespace SYCL
} // end namespace tools

} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H
