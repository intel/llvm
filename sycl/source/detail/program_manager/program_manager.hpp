//==------ program_manager.hpp --- SYCL program manager---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/cg.hpp>
#include <detail/device_binary_image.hpp>
#include <detail/device_global_map_entry.hpp>
#include <detail/host_pipe_map_entry.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/spec_constant_impl.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/device_global_map.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/host_pipe_map.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/util.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

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

namespace sycl {
inline namespace _V1 {
class context;
namespace detail {

bool doesDevSupportDeviceRequirements(const device &Dev,
                                      const RTDeviceBinaryImage &BinImages);
std::optional<sycl::exception>
checkDevSupportDeviceRequirements(const device &Dev,
                                  const RTDeviceBinaryImage &BinImages,
                                  const NDRDescT &NDRDesc = {});

// This value must be the same as in libdevice/device_itt.h.
// See sycl/doc/design/ITTAnnotations.md for more info.
static constexpr uint32_t inline ITTSpecConstId = 0xFF747469;

class context_impl;
using ContextImplPtr = std::shared_ptr<context_impl>;
class device_impl;
using DeviceImplPtr = std::shared_ptr<device_impl>;
class queue_impl;
class event_impl;
// DeviceLibExt is shared between sycl runtime and sycl-post-link tool.
// If any update is made here, need to sync with DeviceLibExt definition
// in llvm/tools/sycl-post-link/sycl-post-link.cpp
enum class DeviceLibExt : std::uint32_t {
  cl_intel_devicelib_assert,
  cl_intel_devicelib_math,
  cl_intel_devicelib_math_fp64,
  cl_intel_devicelib_complex,
  cl_intel_devicelib_complex_fp64,
  cl_intel_devicelib_cstring,
  cl_intel_devicelib_imf,
  cl_intel_devicelib_imf_fp64,
  cl_intel_devicelib_imf_bf16,
  cl_intel_devicelib_bfloat16,
};

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // Returns the single instance of the program manager for the entire
  // process. Can only be called after staticInit is done.
  static ProgramManager &getInstance();

  RTDeviceBinaryImage &getDeviceImage(const std::string &KernelName,
                                      const context &Context,
                                      const device &Device,
                                      bool JITCompilationIsRequired = false);

  RTDeviceBinaryImage &getDeviceImage(
      const std::unordered_set<RTDeviceBinaryImage *> &ImagesToVerify,
      const context &Context, const device &Device,
      bool JITCompilationIsRequired = false);

  sycl::detail::pi::PiProgram createPIProgram(const RTDeviceBinaryImage &Img,
                                              const context &Context,
                                              const device &Device);
  /// Creates a PI program using either a cached device code binary if present
  /// in the persistent cache or from the supplied device image otherwise.
  /// \param Img The device image used to create the program.
  /// \param AllImages All images needed to build the program, used for cache
  ///        lookup.
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
  std::pair<sycl::detail::pi::PiProgram, bool> getOrCreatePIProgram(
      const RTDeviceBinaryImage &Img,
      const std::vector<const RTDeviceBinaryImage *> &AllImages,
      const context &Context, const device &Device,
      const std::string &CompileAndLinkOptions, SerializedObj SpecConsts);
  /// Builds or retrieves from cache a program defining the kernel with given
  /// name.
  /// \param M identifies the OS module the kernel comes from (multiple OS
  ///        modules may have kernels with the same name)
  /// \param Context the context to build the program with
  /// \param Device the device for which the program is built
  /// \param KernelName the kernel's name
  /// \param JITCompilationIsRequired If JITCompilationIsRequired is true
  ///        add a check that kernel is compiled, otherwise don't add the check.
  sycl::detail::pi::PiProgram
  getBuiltPIProgram(const ContextImplPtr &ContextImpl,
                    const DeviceImplPtr &DeviceImpl,
                    const std::string &KernelName, const NDRDescT &NDRDesc = {},
                    bool JITCompilationIsRequired = false);

  sycl::detail::pi::PiProgram
  getBuiltPIProgram(const context &Context, const device &Device,
                    const std::string &KernelName,
                    const property_list &PropList,
                    bool JITCompilationIsRequired = false);

  std::tuple<sycl::detail::pi::PiKernel, std::mutex *, const KernelArgMask *,
             sycl::detail::pi::PiProgram>
  getOrCreateKernel(const ContextImplPtr &ContextImpl,
                    const DeviceImplPtr &DeviceImpl,
                    const std::string &KernelName,
                    const NDRDescT &NDRDesc = {});

  sycl::detail::pi::PiProgram
  getPiProgramFromPiKernel(sycl::detail::pi::PiKernel Kernel,
                           const ContextImplPtr Context);

  void addImages(pi_device_binaries DeviceImages);
  void debugPrintBinaryImages() const;
  static std::string
  getProgramBuildLog(const sycl::detail::pi::PiProgram &Program,
                     const ContextImplPtr Context);

  uint32_t getDeviceLibReqMask(const RTDeviceBinaryImage &Img);

  /// Returns the mask for eliminated kernel arguments for the requested kernel
  /// within the native program.
  /// \param NativePrg the PI program associated with the kernel.
  /// \param KernelName the name of the kernel.
  const KernelArgMask *
  getEliminatedKernelArgMask(pi::PiProgram NativePrg,
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

  // The function inserts or initializes a device_global entry into the
  // device_global map.
  void addOrInitDeviceGlobalEntry(const void *DeviceGlobalPtr,
                                  const char *UniqueId);

  // Returns true if any available image is compatible with the device Dev.
  bool hasCompatibleImage(const device &Dev);

  // The function gets a device_global entry identified by the pointer to the
  // device_global object from the device_global map.
  DeviceGlobalMapEntry *getDeviceGlobalEntry(const void *DeviceGlobalPtr);

  // The function gets multiple device_global entries identified by their unique
  // IDs from the device_global map.
  std::vector<DeviceGlobalMapEntry *>
  getDeviceGlobalEntries(const std::vector<std::string> &UniqueIds,
                         bool ExcludeDeviceImageScopeDecorated = false);
  // The function inserts or initializes a host_pipe entry into the
  // host_pipe map.
  void addOrInitHostPipeEntry(const void *HostPipePtr, const char *UniqueId);

  // The function gets a host_pipe entry identified by the unique ID from
  // the host_pipe map.
  HostPipeMapEntry *getHostPipeEntry(const std::string &UniqueId);

  // The function gets a host_pipe entry identified by the pointer to the
  // host_pipe object from the host_pipe map.
  HostPipeMapEntry *getHostPipeEntry(const void *HostPipePtr);

  device_image_plain
  getDeviceImageFromBinaryImage(RTDeviceBinaryImage *BinImage,
                                const context &Ctx, const device &Dev);

  // The function returns a vector of SYCL device images that are compiled with
  // the required state and at least one device from the passed list of devices.
  std::vector<device_image_plain> getSYCLDeviceImagesWithCompatibleState(
      const context &Ctx, const std::vector<device> &Devs,
      bundle_state TargetState, const std::vector<kernel_id> &KernelIDs = {});

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
  std::vector<device_image_plain> link(const device_image_plain &DeviceImages,
                                       const std::vector<device> &Devs,
                                       const property_list &PropList);

  // Produces new device image by converting input device image to the
  // executable state
  device_image_plain build(const device_image_plain &DeviceImage,
                           const std::vector<device> &Devs,
                           const property_list &PropList);

  std::tuple<sycl::detail::pi::PiKernel, std::mutex *, const KernelArgMask *>
  getOrCreateKernel(const context &Context, const std::string &KernelName,
                    const property_list &PropList,
                    sycl::detail::pi::PiProgram Program);

  ProgramManager();
  ~ProgramManager() = default;

  bool kernelUsesAssert(const std::string &KernelName) const;

  bool kernelUsesAsan() const { return m_AsanFoundInImage; }

  std::set<RTDeviceBinaryImage *>
  getRawDeviceImages(const std::vector<kernel_id> &KernelIDs);

private:
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  using ProgramPtr =
      std::unique_ptr<remove_pointer_t<sycl::detail::pi::PiProgram>,
                      decltype(&::piProgramRelease)>;
  ProgramPtr
  build(ProgramPtr Program, const ContextImplPtr Context,
        const std::string &CompileOptions, const std::string &LinkOptions,
        const sycl::detail::pi::PiDevice &Device, uint32_t DeviceLibReqMask,
        const std::vector<sycl::detail::pi::PiProgram> &ProgramsToLink);
  /// Dumps image to current directory
  void dumpImage(const RTDeviceBinaryImage &Img, uint32_t SequenceID = 0) const;

  /// Add info on kernels using assert into cache
  void cacheKernelUsesAssertInfo(RTDeviceBinaryImage &Img);

  std::set<RTDeviceBinaryImage *>
  collectDeviceImageDepsForImportedSymbols(const RTDeviceBinaryImage &Img,
                                           device Dev);

  std::set<RTDeviceBinaryImage *>
  collectDependentDeviceImagesForVirtualFunctions(
      const RTDeviceBinaryImage &Img, device Dev);

  /// The three maps below are used during kernel resolution. Any kernel is
  /// identified by its name.
  using RTDeviceBinaryImageUPtr = std::unique_ptr<RTDeviceBinaryImage>;

  /// Maps names of kernels to their unique kernel IDs.
  /// TODO: Use std::unordered_set with transparent hash and equality functions
  ///       when C++20 is enabled for the runtime library.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  //
  std::unordered_map<std::string, kernel_id> m_KernelName2KernelIDs;

  // Maps KernelIDs to device binary images. There can be more than one image
  // in case of SPIRV + AOT.
  // Using shared_ptr to avoid expensive copy of the vector.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_multimap<kernel_id, RTDeviceBinaryImage *>
      m_KernelIDs2BinImage;

  // Maps device binary image to a vector of kernel ids in this image.
  // Using shared_ptr to avoid expensive copy of the vector.
  // The vector is initialized in addImages function and is supposed to be
  // immutable afterwards.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_map<RTDeviceBinaryImage *,
                     std::shared_ptr<std::vector<kernel_id>>>
      m_BinImg2KernelIDs;

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
  std::unordered_multimap<std::string, RTDeviceBinaryImage *> m_ServiceKernels;

  /// Caches all exported symbols to allow faster lookup when excluding these
  // from kernel bundles.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_multimap<std::string, RTDeviceBinaryImage *>
      m_ExportedSymbolImages;

  /// Keeps all device images we are refering to during program lifetime. Used
  /// for proper cleanup.
  std::unordered_set<RTDeviceBinaryImageUPtr> m_DeviceImages;

  /// Maps names of built-in kernels to their unique kernel IDs.
  /// Access must be guarded by the m_BuiltInKernelIDsMutex mutex.
  std::unordered_map<std::string, kernel_id> m_BuiltInKernelIDs;

  /// Caches list of device images that use or provide virtual functions from
  /// the same set. Used to simplify access.
  std::unordered_map<std::string, std::set<RTDeviceBinaryImage *>>
      m_VFSet2BinImage;

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
  std::unordered_multimap<pi::PiProgram, const RTDeviceBinaryImage *>
      NativePrograms;

  /// Protects NativePrograms that can be changed by class' methods.
  std::mutex MNativeProgramsMutex;

  using KernelNameToArgMaskMap = std::unordered_map<std::string, KernelArgMask>;
  /// Maps binary image and kernel name pairs to kernel argument masks which
  /// specify which arguments were eliminated during device code optimization.
  std::unordered_map<const RTDeviceBinaryImage *, KernelNameToArgMaskMap>
      m_EliminatedKernelArgMasks;

  /// True iff a SPIR-V file has been specified with an environment variable
  bool m_UseSpvFile = false;
  RTDeviceBinaryImageUPtr m_SpvFileImage;

  std::set<std::string> m_KernelUsesAssert;

  // True iff there is a device image compiled with AddressSanitizer
  bool m_AsanFoundInImage;

  // Maps between device_global identifiers and associated information.
  std::unordered_map<std::string, std::unique_ptr<DeviceGlobalMapEntry>>
      m_DeviceGlobals;
  std::unordered_map<const void *, DeviceGlobalMapEntry *> m_Ptr2DeviceGlobal;

  /// Protects m_DeviceGlobals and m_Ptr2DeviceGlobal.
  std::mutex m_DeviceGlobalsMutex;

  // Maps between host_pipe identifiers and associated information.
  std::unordered_map<std::string, std::unique_ptr<HostPipeMapEntry>>
      m_HostPipes;
  std::unordered_map<const void *, HostPipeMapEntry *> m_Ptr2HostPipe;

  /// Protects m_HostPipes and m_Ptr2HostPipe.
  std::mutex m_HostPipesMutex;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
