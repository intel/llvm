//==------ program_manager.hpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/stl.hpp>

#include <map>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" void __tgt_register_lib(pi_device_binaries desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" void __tgt_unregister_lib(pi_device_binaries desc);

// +++ }

namespace cl {
namespace sycl {
class context;
namespace detail {

using DeviceImage = pi_device_binary_struct;

// Custom deleter for the DeviceImage. Must only be called for "orphan" images
// allocated by the runtime. Those Images which are part of binaries must not
// be attempted to de-allocate.
struct ImageDeleter;

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // Returns the single instance of the program manager for the entire process.
  // Can only be called after staticInit is done.
  static ProgramManager &getInstance();
  RT::PiProgram createOpenCLProgram(OSModuleHandle M, const context &Context,
                                 DeviceImage **I = nullptr) {
    return loadProgram(M, Context, I);
  }
  RT::PiProgram getBuiltOpenCLProgram(OSModuleHandle M, const context &Context);
  RT::PiKernel getOrCreateKernel(OSModuleHandle M, const context &Context,
                                  const string_class &KernelName);
  RT::PiProgram getClProgramFromClKernel(RT::PiKernel Kernel);

  void addImages(pi_device_binaries DeviceImages);
  void debugDumpBinaryImages() const;
  void debugDumpBinaryImage(const DeviceImage *Img) const;
  static string_class getProgramBuildLog(const RT::PiProgram &Program);

private:
  RT::PiProgram loadProgram(OSModuleHandle M, const context &Context,
                            DeviceImage **I = nullptr);
  void build(RT::PiProgram Program, const string_class &Options = "",
             std::vector<RT::PiDevice> Devices = std::vector<RT::PiDevice>());

  ProgramManager() = default;
  ~ProgramManager() = default;
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  /// Keeps all available device executable images added via \ref addImages.
  /// Organizes the images as a map from a module handle (.exe .dll) to the
  /// vector of images coming from the module.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::map<OSModuleHandle, std::unique_ptr<std::vector<DeviceImage *>>>
      m_DeviceImages;
  /// Keeps device images not bound to a particular module. Program manager
  /// allocated memory for these images, so they are auto-freed in destructor.
  /// No image can out-live the Program manager.
  std::vector<std::unique_ptr<DeviceImage, ImageDeleter>> m_OrphanDeviceImages;
};
} // namespace detail
} // namespace sycl
} // namespace cl
