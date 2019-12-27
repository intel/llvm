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

__SYCL_INLINE namespace cl {
namespace sycl {
class context;
namespace detail {

using DeviceImage = pi_device_binary_struct;

// Custom deleter for the DeviceImage. Must only be called for "orphan" images
// allocated by the runtime. Those Images which are part of binaries must not
// be attempted to de-allocate.
struct ImageDeleter;

enum DeviceLibExt {
  cl_intel_devicelib_assert = 0
};

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // Returns the single instance of the program manager for the entire process.
  // Can only be called after staticInit is done.
  static ProgramManager &getInstance();
  DeviceImage &getDeviceImage(OSModuleHandle M, const string_class &KernelName,
                              const context &Context);
  RT::PiProgram createPIProgram(const DeviceImage &Img, const context &Context);
  RT::PiProgram getBuiltPIProgram(OSModuleHandle M, const context &Context,
                                  const string_class &KernelName);
  RT::PiKernel getOrCreateKernel(OSModuleHandle M, const context &Context,
                                  const string_class &KernelName);
  RT::PiProgram getClProgramFromClKernel(RT::PiKernel Kernel);

  void addImages(pi_device_binaries DeviceImages);
  void debugDumpBinaryImages() const;
  void debugDumpBinaryImage(const DeviceImage *Img) const;
  static string_class getProgramBuildLog(const RT::PiProgram &Program);

private:
  ProgramManager();
  ~ProgramManager() = default;
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  DeviceImage &getDeviceImage(OSModuleHandle M, KernelSetId KSId,
                              const context &Context);
  using ProgramPtr = unique_ptr_class<remove_pointer_t<RT::PiProgram>,
                                      decltype(&::piProgramRelease)>;
  ProgramPtr build(ProgramPtr Program, RT::PiContext Context,
                   const string_class &Options,
                   const std::vector<RT::PiDevice> &Devices,
                   std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms,
                   bool LinkDeviceLibs = false);
  /// Provides a new kernel set id for grouping kernel names together
  KernelSetId getNextKernelSetId() const;
  /// Returns the kernel set associated with the kernel, handles some special
  /// cases (when reading images from file or using images with no entry info)
  KernelSetId getKernelSetId(OSModuleHandle M,
                             const string_class &KernelName) const;
  /// Returns the format of the binary image
  RT::PiDeviceBinaryType getFormat(const DeviceImage &Img) const;
  /// Dumps image to current directory
  void dumpImage(const DeviceImage &Img, KernelSetId KSId) const;

  /// The three maps below are used during kernel resolution. Any kernel is
  /// identified by its name and the OS module it's coming from, allowing
  /// kernels with identical names in different OS modules. The following
  /// assumption is made: for any two device images in a SYCL application their
  /// kernel sets are either identical or disjoint.
  /// Based on this assumption, m_KernelSets is used to group kernels together
  /// into sets by assigning a set ID to them during device image registration.
  /// This ID is then mapped to a vector of device images containing kernels
  /// from the set (m_DeviceImages).
  /// An exception is made for device images with no entry information: a
  /// special kernel set ID is used for them which is assigned to just the OS
  /// module. These kernel set ids are stored in m_OSModuleKernelSets and device
  /// images associated with them are assumed to contain all kernels coming from
  /// that OS module.

  /// Keeps all available device executable images added via \ref addImages.
  /// Organizes the images as a map from a kernel set id to the vector of images
  /// containing kernels from that set.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::map<KernelSetId, std::unique_ptr<std::vector<DeviceImage *>>> m_DeviceImages;

  using StrToKSIdMap = std::map<string_class, KernelSetId>;
  /// Maps names of kernels from a specific OS module (.exe .dll) to their set
  /// id (the sets are disjoint).
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::map<OSModuleHandle, StrToKSIdMap> m_KernelSets;

  /// Keeps kernel sets for OS modules containing images without entry info.
  /// Such images are assumed to contain all kernel associated with the module.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::map<OSModuleHandle, KernelSetId> m_OSModuleKernelSets;

  /// Keeps device images not bound to a particular module. Program manager
  /// allocated memory for these images, so they are auto-freed in destructor.
  /// No image can out-live the Program manager.
  std::vector<std::unique_ptr<DeviceImage, ImageDeleter>> m_OrphanDeviceImages;

  /// True iff a SPIRV file has been specified with an environment variable
  bool m_UseSpvFile = false;
};
} // namespace detail
} // namespace sycl
} // namespace cl
