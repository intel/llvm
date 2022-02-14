//==------ program_manager.hpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/device_binary_image.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/util.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <CL/sycl/stl.hpp>
#include <detail/device_global_map_entry.hpp>
#include <detail/spec_constant_impl.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" __SYCL_EXPORT void __sycl_register_lib(pi_device_binaries desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" __SYCL_EXPORT void __sycl_unregister_lib(pi_device_binaries desc);

// +++ }

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
class context;
namespace detail {

// This value must be the same as in libdevice/device_itt.h.
// See sycl/doc/extensions/ITTAnnotations/ITTAnnotations.rst for more info.
static constexpr uint32_t inline ITTSpecConstId = 0xFF747469;

class context_impl;
using ContextImplPtr = std::shared_ptr<context_impl>;
class device_impl;
using DeviceImplPtr = std::shared_ptr<device_impl>;
class program_impl;
// DeviceLibExt is shared between sycl runtime and sycl-post-link tool.
// If any update is made here, need to sync with DeviceLibExt definition
// in llvm/tools/sycl-post-link/sycl-post-link.cpp
enum class DeviceLibExt : std::uint32_t {
  cl_intel_devicelib_assert,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64,
  cl_intel_devicelib_cstring
};

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // TODO use a custom dynamic bitset instead to make initialization simpler.
  using KernelArgMask = std::vector<bool>;

  // Returns the single instance of the program manager for the entire
  // process. Can only be called after staticInit is done.
  static ProgramManager &getInstance();
  RTDeviceBinaryImage &getDeviceImage(OSModuleHandle M,
                                      const std::string &KernelName,
                                      const context &Context,
                                      const device &Device,
                                      bool JITCompilationIsRequired = false);
  RT::PiProgram createPIProgram(const RTDeviceBinaryImage &Img,
                                const context &Context, const device &Device);
  /// Creates a PI program using either a cached device code binary if present
  /// in the persistent cache or from the supplied device image otherwise.
  /// \param Img The device image to find a cached device code binary for or
  ///        create the PI program with.
  /// \param Context The context to find or create the PI program with.
  /// \param Device The device to find or create the PI program for.
  /// \param CompileAndLinkOptions The compile and linking options to be used
  ///        for building the PI program. These options must appear in the
  ///        mentioned order. This parameter is used as a partial key in the
  ///        cache and has no effect if no cached device code binary is found in
  ///        the persistent cache.
  /// \param SpecConsts Specialization constants associated with the device
  ///        image. This parameter is used  as a partial key in the cache and
  ///        has no effect if no cached device code binary is found in the
  ///        persistent cache.
  /// \return A pair consisting of the PI program created with the corresponding
  ///         device code binary and a boolean that is true if the device code
  ///         binary was found in the persistent cache and false otherwise.
  std::pair<RT::PiProgram, bool>
  getOrCreatePIProgram(const RTDeviceBinaryImage &Img, const context &Context,
                       const device &Device,
                       const std::string &CompileAndLinkOptions,
                       SerializedObj SpecConsts);
  /// Builds or retrieves from cache a program defining the kernel with given
  /// name.
  /// \param M identifies the OS module the kernel comes from (multiple OS
  ///        modules may have kernels with the same name)
  /// \param Context the context to build the program with
  /// \param Device the device for which the program is built
  /// \param KernelName the kernel's name
  /// \param Prg provides build context information, such as
  ///        current specialization constants settings; can be nullptr.
  ///        Passing as a raw pointer is OK, since it is not captured anywhere
  ///        once the function returns.
  /// \param JITCompilationIsRequired If JITCompilationIsRequired is true
  ///        add a check that kernel is compiled, otherwise don't add the check.
  RT::PiProgram getBuiltPIProgram(OSModuleHandle M,
                                  const ContextImplPtr &ContextImpl,
                                  const DeviceImplPtr &DeviceImpl,
                                  const std::string &KernelName,
                                  const program_impl *Prg = nullptr,
                                  bool JITCompilationIsRequired = false);

  RT::PiProgram getBuiltPIProgram(OSModuleHandle M, const context &Context,
                                  const device &Device,
                                  const std::string &KernelName,
                                  const property_list &PropList,
                                  bool JITCompilationIsRequired = false);

  std::tuple<RT::PiKernel, std::mutex *, RT::PiProgram>
  getOrCreateKernel(OSModuleHandle M, const ContextImplPtr &ContextImpl,
                    const DeviceImplPtr &DeviceImpl,
                    const std::string &KernelName, const program_impl *Prg);

  RT::PiProgram getPiProgramFromPiKernel(RT::PiKernel Kernel,
                                         const ContextImplPtr Context);

  void addImages(pi_device_binaries DeviceImages);
  void debugPrintBinaryImages() const;
  static std::string getProgramBuildLog(const RT::PiProgram &Program,
                                        const ContextImplPtr Context);

  /// Resolves given program to a device binary image and requests the program
  /// to flush constants the image depends on.
  /// \param Prg the program object to get spec constant settings from.
  ///        Passing program_impl by raw reference is OK, since it is not
  ///        captured anywhere once the function returns.
  /// \param NativePrg the native program, target for spec constant setting; if
  ///        not null then overrides the native program in Prg
  /// \param Img A source of the information about which constants need
  ///        setting and symboling->integer spec constant ID mapping. If not
  ///        null, overrides native program->binary image binding maintained by
  ///        the program manager.
  void flushSpecConstants(const program_impl &Prg,
                          pi::PiProgram NativePrg = nullptr,
                          const RTDeviceBinaryImage *Img = nullptr);
  uint32_t getDeviceLibReqMask(const RTDeviceBinaryImage &Img);

  /// Returns the mask for eliminated kernel arguments for the requested kernel
  /// within the native program.
  /// \param M identifies the OS module the kernel comes from (multiple OS
  ///        modules may have kernels with the same name).
  /// \param NativePrg the PI program associated with the kernel.
  /// \param KernelName the name of the kernel.
  KernelArgMask getEliminatedKernelArgMask(OSModuleHandle M,
                                           pi::PiProgram NativePrg,
                                           const std::string &KernelName);

  // The function returns the unique SYCL kernel identifier associated with a
  // kernel name.
  kernel_id getSYCLKernelID(const std::string &KernelName);

  // The function returns a vector containing all unique SYCL kernel identifiers
  // in SYCL device images.
  std::vector<kernel_id> getAllSYCLKernelIDs();

  // The function returns the unique SYCL kernel identifier associated with a
  // built-in kernel name.
  kernel_id getBuiltInKernelID(const std::string &KernelName);

  // The function inserts a device_global entry into the device_global map.
  void addDeviceGlobalEntry(void *DeviceGlobalPtr, const char *UniqueId);

  // The function returns a vector of SYCL device images that are compiled with
  // the required state and at least one device from the passed list of devices.
  std::vector<device_image_plain>
  getSYCLDeviceImagesWithCompatibleState(const context &Ctx,
                                         const std::vector<device> &Devs,
                                         bundle_state TargetState);

  // Brind images in the passed vector to the required state. Does it inplace
  void
  bringSYCLDeviceImagesToState(std::vector<device_image_plain> &DeviceImages,
                               bundle_state TargetState);

  // The function returns a vector of SYCL device images in required state,
  // which are compatible with at least one of the device from Devs.
  std::vector<device_image_plain>
  getSYCLDeviceImages(const context &Ctx, const std::vector<device> &Devs,
                      bundle_state State);

  // The function returns a vector of SYCL device images, for which Selector
  // callable returns true, in required state, which are compatible with at
  // least one of the device from Devs.
  std::vector<device_image_plain>
  getSYCLDeviceImages(const context &Ctx, const std::vector<device> &Devs,
                      const DevImgSelectorImpl &Selector,
                      bundle_state TargetState);

  // The function returns a vector of SYCL device images which represent at
  // least one kernel from kernel ids vector in required state, which are
  // compatible with at least one of the device from Devs.
  std::vector<device_image_plain>
  getSYCLDeviceImages(const context &Ctx, const std::vector<device> &Devs,
                      const std::vector<kernel_id> &KernelIDs,
                      bundle_state TargetState);

  // Produces new device image by convering input device image to the object
  // state
  device_image_plain compile(const device_image_plain &DeviceImage,
                             const std::vector<device> &Devs,
                             const property_list &PropList);

  // Produces set of device images by convering input device images to object
  // the executable state
  std::vector<device_image_plain>
  link(const std::vector<device_image_plain> &DeviceImages,
       const std::vector<device> &Devs, const property_list &PropList);

  // Produces new device image by converting input device image to the
  // executable state
  device_image_plain build(const device_image_plain &DeviceImage,
                           const std::vector<device> &Devs,
                           const property_list &PropList);

  std::pair<RT::PiKernel, std::mutex *>
  getOrCreateKernel(const context &Context, const std::string &KernelName,
                    const property_list &PropList, RT::PiProgram Program);

  ProgramManager();
  ~ProgramManager() = default;

  bool kernelUsesAssert(OSModuleHandle M, const std::string &KernelName) const;

private:
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  RTDeviceBinaryImage &getDeviceImage(OSModuleHandle M, KernelSetId KSId,
                                      const context &Context,
                                      const device &Device,
                                      bool JITCompilationIsRequired = false);
  using ProgramPtr = std::unique_ptr<remove_pointer_t<RT::PiProgram>,
                                     decltype(&::piProgramRelease)>;
  ProgramPtr build(ProgramPtr Program, const ContextImplPtr Context,
                   const std::string &CompileOptions,
                   const std::string &LinkOptions, const RT::PiDevice &Device,
                   std::map<std::pair<DeviceLibExt, RT::PiDevice>,
                            RT::PiProgram> &CachedLibPrograms,
                   uint32_t DeviceLibReqMask);
  /// Provides a new kernel set id for grouping kernel names together
  KernelSetId getNextKernelSetId() const;
  /// Returns the kernel set associated with the kernel, handles some special
  /// cases (when reading images from file or using images with no entry info)
  KernelSetId getKernelSetId(OSModuleHandle M,
                             const std::string &KernelName) const;
  /// Dumps image to current directory
  void dumpImage(const RTDeviceBinaryImage &Img, KernelSetId KSId) const;

  /// Add info on kernels using assert into cache
  void cacheKernelUsesAssertInfo(OSModuleHandle M, RTDeviceBinaryImage &Img);

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

  using StrToKSIdMap = std::unordered_map<std::string, KernelSetId>;
  /// Maps names of kernels from a specific OS module (.exe .dll) to their set
  /// id (the sets are disjoint).
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::unordered_map<OSModuleHandle, StrToKSIdMap> m_KernelSets;

  /// Keeps kernel sets for OS modules containing images without entry info.
  /// Such images are assumed to contain all kernel associated with the module.
  /// Access must be guarded by the \ref Sync::getGlobalLock()
  std::unordered_map<OSModuleHandle, KernelSetId> m_OSModuleKernelSets;

  /// Maps names of kernels to their unique kernel IDs.
  /// TODO: Use std::unordered_set with transparent hash and equality functions
  ///       when C++20 is enabled for the runtime library.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_map<std::string, kernel_id> m_KernelIDs;

  /// Protects kernel ID cache.
  /// NOTE: This may be acquired while \ref Sync::getGlobalLock() is held so to
  /// avoid deadlocks care must be taken not to acquire
  /// \ref Sync::getGlobalLock() while holding this mutex.
  std::mutex m_KernelIDsMutex;

  /// Caches all found service kernels to expedite future checks. A SYCL service
  /// kernel is a kernel that has not been defined by the user but is instead
  /// generated by the SYCL runtime. Service kernel name types must be declared
  /// in the sycl::detail::__sycl_service_kernel__ namespace which is
  /// exclusively used for this purpose.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_set<std::string> m_ServiceKernels;

  /// Caches all exported symbols to allow faster lookup when excluding these
  // from kernel bundles.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_set<std::string> m_ExportedSymbols;

  /// Maps names of built-in kernels to their unique kernel IDs.
  /// Access must be guarded by the m_BuiltInKernelIDsMutex mutex.
  std::unordered_map<std::string, kernel_id> m_BuiltInKernelIDs;

  /// Protects built-in kernel ID cache.
  std::mutex m_BuiltInKernelIDsMutex;

  // Keeps track of pi_program to image correspondence. Needed for:
  // - knowing which specialization constants are used in the program and
  //   injecting their current values before compiling the SPIR-V; the binary
  //   image object has info about all spec constants used in the module
  // - finding kernel argument masks for kernels associated with each
  //   pi_program
  // NOTE: using RTDeviceBinaryImage raw pointers is OK, since they are not
  // referenced from outside SYCL runtime and RTDeviceBinaryImage object
  // lifetime matches program manager's one.
  // NOTE: keys in the map can be invalid (reference count went to zero and
  // the underlying program disposed of), so the map can't be used in any way
  // other than binary image lookup with known live PiProgram as the key.
  // NOTE: access is synchronized via the MNativeProgramsMutex
  std::unordered_map<pi::PiProgram, const RTDeviceBinaryImage *> NativePrograms;

  /// Protects NativePrograms that can be changed by class' methods.
  std::mutex MNativeProgramsMutex;

  using KernelNameToArgMaskMap = std::unordered_map<std::string, KernelArgMask>;
  /// Maps binary image and kernel name pairs to kernel argument masks which
  /// specify which arguments were eliminated during device code optimization.
  std::unordered_map<const RTDeviceBinaryImage *, KernelNameToArgMaskMap>
      m_EliminatedKernelArgMasks;

  /// True iff a SPIR-V file has been specified with an environment variable
  bool m_UseSpvFile = false;

  using KernelNameWithOSModule = std::pair<std::string, OSModuleHandle>;
  std::set<KernelNameWithOSModule> m_KernelUsesAssert;

  // Map between device_global unique ids and associated information.
  std::unordered_map<std::string, DeviceGlobalMapEntry> m_DeviceGlobals;

  /// Protects m_DeviceGlobals.
  std::mutex m_DeviceGlobalsMutex;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
