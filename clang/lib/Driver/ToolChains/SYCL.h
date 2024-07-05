//===--- SYCL.h - SYCL ToolChain Implementations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H

#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

namespace clang {
namespace driver {

class SYCLInstallationDetector {
public:
  SYCLInstallationDetector(const Driver &D);
  void getSYCLDeviceLibPath(
      llvm::SmallVector<llvm::SmallString<128>, 4> &DeviceLibPaths) const;
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
  void constructLlcCommand(Compilation &C, const JobAction &JA,
                           const InputInfo &Output,
                           const char *InputFile) const;
};

/// Directly call FPGA Compiler and Linker
namespace fpga {

class LLVM_LIBRARY_VISIBILITY BackendCompiler : public Tool {
public:
  BackendCompiler(const ToolChain &TC)
      : Tool("fpga::BackendCompiler", "fpga compiler", TC) {}

  bool hasIntegratedCPP() const override { return false; }

  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;

private:
  void constructOpenCLAOTCommand(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &InputFiles,
                                 const llvm::opt::ArgList &Args) const;
};

} // end namespace fpga

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

// // Prefix for GPU specific targets used for -fsycl-targets
constexpr char IntelGPU[] = "intel_gpu_";
constexpr char NvidiaGPU[] = "nvidia_gpu_";
constexpr char AmdGPU[] = "amd_gpu_";

template <auto GPUArh> std::optional<StringRef> isGPUTarget(StringRef Target) {
  // Handle target specifications that resemble '(intel, nvidia, amd)_gpu_*'
  // here.
  if (Target.starts_with(GPUArh)) {
    return resolveGenDevice(Target);
  }
  return  std::nullopt;
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

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY SYCLToolChain : public ToolChain {
public:
  SYCLToolChain(const Driver &D, const llvm::Triple &Triple,
                const ToolChain &HostTC, const llvm::opt::ArgList &Args);

  const llvm::Triple *getAuxTriple() const override {
    return &HostTC.getTriple();
  }

  llvm::opt::DerivedArgList *
  TranslateArgs(const llvm::opt::DerivedArgList &Args, StringRef BoundArch,
                Action::OffloadKind DeviceOffloadKind) const override;
  void addClangTargetOptions(const llvm::opt::ArgList &DriverArgs,
                         llvm::opt::ArgStringList &CC1Args,
                         Action::OffloadKind DeviceOffloadKind) const override;
  void AddImpliedTargetArgs(const llvm::Triple &Triple,
                            const llvm::opt::ArgList &Args,
                            llvm::opt::ArgStringList &CmdArgs,
                            const JobAction &JA, const ToolChain &HostTC,
                            StringRef Device = "") const;
  void TranslateBackendTargetArgs(const llvm::Triple &Triple,
                                  const llvm::opt::ArgList &Args,
                                  llvm::opt::ArgStringList &CmdArgs,
                                  StringRef Device = "") const;
  void TranslateLinkerTargetArgs(const llvm::Triple &Triple,
                                 const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs,
                                 StringRef Device = "") const;
  void TranslateTargetOpt(const llvm::Triple &Triple,
                          const llvm::opt::ArgList &Args,
                          llvm::opt::ArgStringList &CmdArgs,
                          llvm::opt::OptSpecifier Opt,
                          llvm::opt::OptSpecifier Opt_EQ,
                          StringRef Device) const;

  bool useIntegratedAs() const override { return true; }
  bool isPICDefault() const override {
    if (this->IsSYCLNativeCPU)
      return this->HostTC.isPICDefault();
    return false;
  }
  llvm::codegenoptions::DebugInfoFormat getDefaultDebugFormat() const override {
    if (this->IsSYCLNativeCPU ||
        this->HostTC.getTriple().isWindowsMSVCEnvironment())
      return this->HostTC.getDefaultDebugFormat();
    return ToolChain::getDefaultDebugFormat();
  }
  bool isPIEDefault(const llvm::opt::ArgList &Args) const override {
    return false;
  }
  bool isPICDefaultForced() const override { return false; }

  void addClangWarningOptions(llvm::opt::ArgStringList &CC1Args) const override;
  CXXStdlibType GetCXXStdlibType(const llvm::opt::ArgList &Args) const override;
  static void AddSYCLIncludeArgs(const clang::driver::Driver &Driver,
                                 const llvm::opt::ArgList &DriverArgs,
                                 llvm::opt::ArgStringList &CC1Args);
  void AddClangSystemIncludeArgs(const llvm::opt::ArgList &DriverArgs,
                            llvm::opt::ArgStringList &CC1Args) const override;
  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CC1Args) const override;

  SanitizerMask getSupportedSanitizers() const override;

  const ToolChain &HostTC;
  const bool IsSYCLNativeCPU;

protected:
  Tool *buildBackendCompiler() const override;
  Tool *buildLinker() const override;

private:
  void TranslateGPUTargetOpt(const llvm::opt::ArgList &Args,
                             llvm::opt::ArgStringList &CmdArgs,
                             llvm::opt::OptSpecifier Opt_EQ) const;
};

} // end namespace toolchains

inline bool isSYCLNativeCPU(const llvm::opt::ArgList &Args) {
  if (auto SYCLTargets = Args.getLastArg(options::OPT_fsycl_targets_EQ)) {
    if (SYCLTargets->containsValue("native_cpu"))
      return true;
  }
  return false;
}

inline bool isSYCLNativeCPU(const llvm::Triple &HostT, const llvm::Triple &DevT) {
  return HostT == DevT;
}

inline bool isSYCLNativeCPU(const ToolChain &TC) {
  const llvm::Triple *const AuxTriple = TC.getAuxTriple();
  return AuxTriple && isSYCLNativeCPU(TC.getTriple(), *AuxTriple);
}
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
