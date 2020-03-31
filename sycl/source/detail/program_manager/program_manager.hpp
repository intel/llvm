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
#include <CL/sycl/detail/spec_constant_impl.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/stl.hpp>

#include <map>
#include <unordered_map>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" void __sycl_register_lib(pi_device_binaries desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" void __sycl_unregister_lib(pi_device_binaries desc);

// +++ }

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
class context;
namespace detail {

class context_impl;
using ContextImplPtr = std::shared_ptr<context_impl>;
class program_impl;

enum DeviceLibExt {
  cl_intel_devicelib_assert = 0,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64
};

// SYCL RT wrapper over PI binary image.
class RTDeviceBinaryImage : public pi::DeviceBinaryImage {
public:
  RTDeviceBinaryImage(OSModuleHandle ModuleHandle)
      : pi::DeviceBinaryImage(), ModuleHandle(ModuleHandle) {}
  RTDeviceBinaryImage(pi_device_binary Bin, OSModuleHandle ModuleHandle)
      : pi::DeviceBinaryImage(Bin), ModuleHandle(ModuleHandle) {}
  OSModuleHandle getOSModuleHandle() const { return ModuleHandle; }

  ~RTDeviceBinaryImage() override {}

  bool supportsSpecConstants() const {
    return getFormat() == PI_DEVICE_BINARY_TYPE_SPIRV;
  }

  const pi_device_binary_struct &getRawData() const { return *get(); }

  void print() const override {
    pi::DeviceBinaryImage::print();
    std::cerr << "    OSModuleHandle=" << ModuleHandle << "\n";
  }

protected:
  OSModuleHandle ModuleHandle;
};

// Dynamically allocated device binary image, which de-allocates its binary data
// in destructor.
class DynRTDeviceBinaryImage : public RTDeviceBinaryImage {
public:
  DynRTDeviceBinaryImage(std::unique_ptr<char[]> &&DataPtr, size_t DataSize,
                         OSModuleHandle M);
  ~DynRTDeviceBinaryImage() override;

  void print() const override {
    RTDeviceBinaryImage::print();
    std::cerr << "    DYNAMICALLY CREATED\n";
  }

protected:
  std::unique_ptr<char[]> Data;
};

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // Returns the single instance of the program manager for the entire
  // process. Can only be called after staticInit is done.
  static ProgramManager &getInstance();
  RTDeviceBinaryImage &getDeviceImage(OSModuleHandle M,
                                      const string_class &KernelName,
                                      const context &Context);
  RT::PiProgram createPIProgram(const RTDeviceBinaryImage &Img,
                                const context &Context);
  RT::PiProgram getBuiltPIProgram(OSModuleHandle M, const context &Context,
                                  const string_class &KernelName);
  RT::PiKernel getOrCreateKernel(OSModuleHandle M, const context &Context,
                                 const string_class &KernelName);
  RT::PiProgram getPiProgramFromPiKernel(RT::PiKernel Kernel,
                                         const ContextImplPtr Context);

  void addImages(pi_device_binaries DeviceImages);
  void debugPrintBinaryImages() const;
  static string_class getProgramBuildLog(const RT::PiProgram &Program,
                                         const ContextImplPtr Context);

  spec_constant_impl &resolveSpecConstant(const program_impl *P,
                                          const char *Name);

  // Takes current values of specialization constants and "injects" them into
  // the program via specialization constant managemment PI APIs
  void flushSpecConstants(pi::PiProgram Prg, context_impl &Ctx);

private:
  ProgramManager();
  ~ProgramManager() = default;
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  RTDeviceBinaryImage &getDeviceImage(OSModuleHandle M, KernelSetId KSId,
                                      const context &Context);
  using ProgramPtr = unique_ptr_class<remove_pointer_t<RT::PiProgram>,
                                      decltype(&::piProgramRelease)>;
  ProgramPtr build(ProgramPtr Program, const ContextImplPtr Context,
                   const string_class &CompileOptions,
                   const string_class &LinkOptions,
                   const std::vector<RT::PiDevice> &Devices,
                   std::map<DeviceLibExt, RT::PiProgram> &CachedLibPrograms,
                   bool LinkDeviceLibs = false);
  /// Provides a new kernel set id for grouping kernel names together
  KernelSetId getNextKernelSetId() const;
  /// Returns the kernel set associated with the kernel, handles some special
  /// cases (when reading images from file or using images with no entry info)
  KernelSetId getKernelSetId(OSModuleHandle M,
                             const string_class &KernelName) const;
  /// Dumps image to current directory
  void dumpImage(const RTDeviceBinaryImage &Img, KernelSetId KSId) const;

  void populateSpecConstRegistryImpl(const RTDeviceBinaryImage &Img);
  void populateSpecConstRegistry();

  /// The three maps below are used during kernel resolution. Any kernel is
  /// identified by its name and the OS module it's coming from, allowing
  /// kernels with identical names in different OS modules. The following
  /// assumption is made: for any two device images in a SYCL application
  /// their kernel sets are either identical or disjoint. Based on this
  /// assumption, m_KernelSets is used to group kernels together into sets by
  /// assigning a set ID to them during device image registration. This ID is
  /// then mapped to a vector of device images containing kernels from the set
  /// (m_DeviceImages). An exception is made for device images with no entry
  /// information: a special kernel set ID is used for them which is assigned
  /// to just the OS module. These kernel set ids are stored in
  /// m_OSModuleKernelSets and device images associated with them are assumed
  /// to contain all kernels coming from that OS module.

  using RTDeviceBinaryImageUPtr = std::unique_ptr<RTDeviceBinaryImage>;

  /// Keeps all available device executable images added via \ref addImages.
  /// Organizes the images as a map from a kernel set id to the vector of images
  /// containing kernels from that set.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::unordered_map<KernelSetId,
                     std::unique_ptr<std::vector<RTDeviceBinaryImageUPtr>>>
      m_DeviceImages;

  using StrToKSIdMap = std::unordered_map<string_class, KernelSetId>;
  /// Maps names of kernels from a specific OS module (.exe .dll) to their set
  /// id (the sets are disjoint).
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::unordered_map<OSModuleHandle, StrToKSIdMap> m_KernelSets;

  /// Keeps kernel sets for OS modules containing images without entry info.
  /// Such images are assumed to contain all kernel associated with the module.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::unordered_map<OSModuleHandle, KernelSetId> m_OSModuleKernelSets;

  // Maps specialization constant symbolic name to its integer ID.
  // NOTE: a key in this map is a pointer into some existing device binary image
  // and it is invalidated once the encompassing OS module gets unloaded.
  using SpecConstMapTy =
      std::unordered_map<const char *, spec_constant_impl, HashCStr, CmpCStr>;

  // Keeps specialization constant map for each operating system module.
  // The base approach is that there is a global name->id spec constant map for
  // each OS module. This map is populated from the name->id pairs found in
  // device binary image upon registration at startup for this OS module.
  // If multiple images beloning to the same OS module have a specialization
  // constant with the same name, its corresponding numeric id must also match.
  // It is modified (populated) once at startup, so no locks are necessary.
  std::unordered_map<OSModuleHandle, SpecConstMapTy> SpecConstRegistry;

  // Keeps track of pi_program to image correspondence. Needed for:
  // - knowing which specialization constants are used in the program and
  //   injecting their current values before compiling the SPIRV; the binary
  //   image object has info about all spec constants used in the module
  // NOTE: using RTDeviceBinaryImage raw pointers is OK, since they are not
  // referenced from outside SYCL runtime and RTDeviceBinaryImage object
  // lifetime matches program manager's one.
  // NOTE: keys in the map can be invalid (reference count went to zero and
  // the underlying program disposed of), so the map can't be used in any way
  // other than binary image lookup with known live PiProgram as the key.
  // NOTE: access is synchronized via the same lock as program cache
  std::unordered_map<pi::PiProgram, const RTDeviceBinaryImage *> NativePrograms;

  /// True iff a SPIRV file has been specified with an environment variable
  bool m_UseSpvFile = false;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
