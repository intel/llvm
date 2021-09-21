//===--- SYCL.h - SYCL ToolChain Implementations -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H

#include "clang/Driver/ToolChain.h"
#include "clang/Driver/Tool.h"

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
                                 StringRef Increment, StringRef Ext);

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
  /// \return llvm-spirv output file name.
  const char *constructLLVMSpirvCommand(Compilation &C, const JobAction &JA,
                                       const InputInfo &Output,
                                       llvm::StringRef OutputFilePrefix,
                                       bool isBc, const char *InputFile) const;
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
                            llvm::opt::ArgStringList &CmdArgs) const;
  void TranslateBackendTargetArgs(const llvm::Triple &Triple,
                                  const llvm::opt::ArgList &Args,
                                  llvm::opt::ArgStringList &CmdArgs) const;
  void TranslateLinkerTargetArgs(const llvm::Triple &Triple,
                                 const llvm::opt::ArgList &Args,
                                 llvm::opt::ArgStringList &CmdArgs) const;

  bool useIntegratedAs() const override { return true; }
  bool isPICDefault() const override { return false; }
  bool isPIEDefault() const override { return false; }
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


  const ToolChain &HostTC;
  const SYCLInstallationDetector SYCLInstallation;

protected:
  Tool *buildBackendCompiler() const override;
  Tool *buildLinker() const override;

private:
  void TranslateTargetOpt(const llvm::opt::ArgList &Args,
      llvm::opt::ArgStringList &CmdArgs, llvm::opt::OptSpecifier Opt,
      llvm::opt::OptSpecifier Opt_EQ) const;
};

} // end namespace toolchains
} // end namespace driver
} // end namespace clang

#endif // LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_SYCL_H
