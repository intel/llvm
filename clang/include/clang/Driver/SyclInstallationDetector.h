//===-- SyclInstallationDetector.h - SYCL Instalation Detector --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H
#define LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H

#include "clang/Driver/Driver.h"

namespace clang {
namespace driver {

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

} // namespace driver
} // namespace clang

#endif // LLVM_CLANG_DRIVER_SYCLINSTALLATIONDETECTOR_H
