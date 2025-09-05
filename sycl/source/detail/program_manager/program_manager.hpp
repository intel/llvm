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
#include <detail/device_global_map.hpp>
#include <detail/device_global_map_entry.hpp>
#include <detail/device_kernel_info.hpp>
#include <detail/host_pipe_map_entry.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/spec_constant_impl.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/device_global_map.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/host_pipe_map.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/detail/util.hpp>
#include <sycl/device.hpp>
#include <sycl/kernel_bundle.hpp>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// +++ Entry points referenced by the offload wrapper object {

/// Executed as a part of current module's (.exe, .dll) static initialization.
/// Registers device executable images with the runtime.
extern "C" __SYCL_EXPORT void __sycl_register_lib(sycl_device_binaries desc);

/// Executed as a part of current module's (.exe, .dll) static
/// de-initialization.
/// Unregisters device executable images with the runtime.
extern "C" __SYCL_EXPORT void __sycl_unregister_lib(sycl_device_binaries desc);

// +++ }

// For testing purposes
class ProgramManagerTest;

namespace sycl {
inline namespace _V1 {
class context;
namespace detail {

bool doesDevSupportDeviceRequirements(const device_impl &Dev,
                                      const RTDeviceBinaryImage &BinImages);
std::optional<sycl::exception>
checkDevSupportDeviceRequirements(const device_impl &Dev,
                                  const RTDeviceBinaryImage &BinImages,
                                  const NDRDescT &NDRDesc = {});

bool doesImageTargetMatchDevice(const RTDeviceBinaryImage &Img,
                                const device_impl &DevImpl);

// This value must be the same as in libdevice/device_itt.h.
// See sycl/doc/design/ITTAnnotations.md for more info.
static constexpr uint32_t inline ITTSpecConstId = 0xFF747469;

class context_impl;
class device_impl;
class devices_range;
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

enum class SanitizerType {
  None,
  AddressSanitizer,
  MemorySanitizer,
  ThreadSanitizer
};

// A helper class for storing image/program objects and their dependencies
// and making their handling a bit more readable.
template <typename T> class ObjectWithDeps {
public:
  ObjectWithDeps(T Main) : Objs({std::move(Main)}) {}
  // Assumes 0th element is the main one.
  ObjectWithDeps(std::vector<T> AllObjs) : Objs{std::move(AllObjs)} {}

  T &getMain() { return *Objs.begin(); }
  const T &getMain() const { return *Objs.begin(); }
  const std::vector<T> &getAll() const { return Objs; }
  std::size_t size() const { return Objs.size(); }
  bool hasDeps() const { return Objs.size() > 1; }
  auto begin() { return Objs.begin(); }
  auto begin() const { return Objs.begin(); }
  auto end() { return Objs.end(); }
  auto end() const { return Objs.end(); }
  // TODO use a subrange once C++20 is available
  auto depsBegin() const { return Objs.begin() + 1; }
  auto depsEnd() const { return Objs.end(); }

private:
  std::vector<T> Objs;
};

using DevImgPlainWithDeps = ObjectWithDeps<device_image_plain>;
using BinImgWithDeps = ObjectWithDeps<const RTDeviceBinaryImage *>;

// Provides single loading and building OpenCL programs with unique contexts
// that is necessary for no interoperability cases with lambda.
class ProgramManager {
public:
  // Returns the single instance of the program manager for the entire
  // process. Can only be called after staticInit is done.
  static ProgramManager &getInstance();

  const RTDeviceBinaryImage &getDeviceImage(KernelNameStrRefT KernelName,
                                            context_impl &ContextImpl,
                                            const device_impl &DeviceImpl);

  const RTDeviceBinaryImage &getDeviceImage(
      const std::unordered_set<const RTDeviceBinaryImage *> &ImagesToVerify,
      context_impl &ContextImpl, const device_impl &DeviceImpl);

  Managed<ur_program_handle_t> createURProgram(const RTDeviceBinaryImage &Img,
                                               context_impl &ContextImpl,
                                               devices_range Devices);
  /// Creates a UR program using either a cached device code binary if present
  /// in the persistent cache or from the supplied device image otherwise.
  /// \param Img The device image used to create the program.
  /// \param AllImages All images needed to build the program, used for cache
  ///        lookup.
  /// \param Context The context to find or create the UR program with.
  /// \param Device The device to find or create the UR program for.
  /// \param CompileAndLinkOptions The compile and linking options to be used
  ///        for building the UR program. These options must appear in the
  ///        mentioned order. This parameter is used as a partial key in the
  ///        cache and has no effect if no cached device code binary is found in
  ///        the persistent cache.
  /// \param SpecConsts Specialization constants associated with the device
  ///        image. This parameter is used  as a partial key in the cache and
  ///        has no effect if no cached device code binary is found in the
  ///        persistent cache.
  /// \return A pair consisting of the UR program created with the corresponding
  ///         device code binary and a boolean that is true if the device code
  ///         binary was found in the persistent cache and false otherwise.
  std::pair<Managed<ur_program_handle_t>, bool> getOrCreateURProgram(
      const RTDeviceBinaryImage &Img,
      const std::vector<const RTDeviceBinaryImage *> &AllImages,
      context_impl &ContextImpl, devices_range Devices,
      const std::string &CompileAndLinkOptions, SerializedObj SpecConsts);
  /// Builds or retrieves from cache a program defining the kernel with given
  /// name.
  /// \param M identifies the OS module the kernel comes from (multiple OS
  ///        modules may have kernels with the same name)
  /// \param Context the context to build the program with
  /// \param Device the device for which the program is built
  /// \param KernelName the kernel's name
  Managed<ur_program_handle_t> getBuiltURProgram(context_impl &ContextImpl,
                                                 device_impl &DeviceImpl,
                                                 KernelNameStrRefT KernelName,
                                                 const NDRDescT &NDRDesc = {});

  /// Builds a program from a given set of images or retrieves that program from
  /// cache.
  /// \param ImgWithDeps is the main image the program is built with and its
  /// dependencies.
  /// \param Context is the context the program is built for.
  /// \param Devs is a vector of devices the program is built for.
  /// \param DevImgWithDeps is an optional DevImgPlainWithDeps pointer that
  /// represents the images.
  /// \param SpecConsts is an optional parameter containing spec constant values
  /// the program should be built with.
  Managed<ur_program_handle_t>
  getBuiltURProgram(const BinImgWithDeps &ImgWithDeps,
                    context_impl &ContextImpl, devices_range Devs,
                    const DevImgPlainWithDeps *DevImgWithDeps = nullptr,
                    const SerializedObj &SpecConsts = {});

  FastKernelCacheValPtr getOrCreateKernel(context_impl &ContextImpl,
                                          device_impl &DeviceImpl,
                                          KernelNameStrRefT KernelName,
                                          DeviceKernelInfo &DeviceKernelInfo,
                                          const NDRDescT &NDRDesc = {});

  ur_kernel_handle_t getCachedMaterializedKernel(
      KernelNameStrRefT KernelName,
      const std::vector<unsigned char> &SpecializationConsts);

  ur_kernel_handle_t getOrCreateMaterializedKernel(
      const RTDeviceBinaryImage &Img, const context &Context,
      const device &Device, KernelNameStrRefT KernelName,
      const std::vector<unsigned char> &SpecializationConsts);

  ur_program_handle_t getUrProgramFromUrKernel(ur_kernel_handle_t Kernel,
                                               context_impl &Context);

  void addImage(sycl_device_binary RawImg, bool RegisterImgExports = true,
                RTDeviceBinaryImage **OutImage = nullptr,
                std::vector<kernel_id> *OutKernelIDs = nullptr);
  void addImages(sycl_device_binaries DeviceImages);
  void removeImages(sycl_device_binaries DeviceImages);
  void debugPrintBinaryImages() const;
  static std::string getProgramBuildLog(const ur_program_handle_t &Program,
                                        context_impl &Context);

  uint32_t getDeviceLibReqMask(const RTDeviceBinaryImage &Img);

  /// Returns the mask for eliminated kernel arguments for the requested kernel
  /// within the native program.
  /// \param NativePrg the UR program associated with the kernel.
  /// \param KernelName the name of the kernel.
  const KernelArgMask *getEliminatedKernelArgMask(ur_program_handle_t NativePrg,
                                                  KernelNameStrRefT KernelName);

  // The function returns the unique SYCL kernel identifier associated with a
  // kernel name or nullopt if there is no such ID.
  std::optional<kernel_id> tryGetSYCLKernelID(KernelNameStrRefT KernelName);

  // The function returns the unique SYCL kernel identifier associated with a
  // kernel name or throws a sycl exception if there is no such ID.
  kernel_id getSYCLKernelID(KernelNameStrRefT KernelName);

  // The function returns a vector containing all unique SYCL kernel identifiers
  // in SYCL device images.
  std::vector<kernel_id> getAllSYCLKernelIDs();

  // The function returns the unique SYCL kernel identifier associated with a
  // built-in kernel name.
  kernel_id getBuiltInKernelID(KernelNameStrRefT KernelName);

  // The function inserts or initializes a device_global entry into the
  // device_global map.
  void addOrInitDeviceGlobalEntry(const void *DeviceGlobalPtr,
                                  const char *UniqueId);

  // The function inserts or initializes a kernel global desc into the
  // kernel global map.
  void registerKernelGlobalInfo(
      std::unordered_map<std::string_view, unsigned> &&GlobalInfoToCopy);

  // The function returns a pointer to the kernel global desc identified by
  // the unique ID from the kernel global map.
  std::optional<unsigned> getKernelGlobalInfoDesc(const char *UniqueId);

  // Returns true if any available image is compatible with the device Dev.
  bool hasCompatibleImage(const device_impl &DeviceImpl);

  // The function gets a device_global entry identified by the pointer to the
  // device_global object from the device_global map.
  DeviceGlobalMapEntry *getDeviceGlobalEntry(const void *DeviceGlobalPtr);

  // The function attempts to get a single device_global entry identified by its
  // unique ID from the device_global map. If no such entry is found, nullptr is
  // returned.
  DeviceGlobalMapEntry *
  tryGetDeviceGlobalEntry(const std::string &UniqueId,
                          bool ExcludeDeviceImageScopeDecorated = false);

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
  getDeviceImageFromBinaryImage(const RTDeviceBinaryImage *BinImage,
                                const context &Ctx, const device &Dev);

  // The function returns a vector of SYCL device images that are compiled with
  // the required state and at least one device from the passed list of devices.
  std::vector<DevImgPlainWithDeps> getSYCLDeviceImagesWithCompatibleState(
      const context &Ctx, devices_range Devs, bundle_state TargetState,
      const std::vector<kernel_id> &KernelIDs = {});

  // Creates a new dependency image for a given dependency binary image.
  device_image_plain createDependencyImage(const context &Ctx,
                                           devices_range Devs,
                                           const RTDeviceBinaryImage *DepImage,
                                           bundle_state DepState);

  // Bring image to the required state. Does it inplace
  void bringSYCLDeviceImageToState(DevImgPlainWithDeps &DeviceImage,
                                   bundle_state TargetState);

  // Bring images in the passed vector to the required state. Does it inplace
  void
  bringSYCLDeviceImagesToState(std::vector<DevImgPlainWithDeps> &DeviceImages,
                               bundle_state TargetState);

  // The function returns a vector of SYCL device images in required state,
  // which are compatible with at least one of the device from Devs.
  std::vector<DevImgPlainWithDeps> getSYCLDeviceImages(const context &Ctx,
                                                       devices_range Devs,
                                                       bundle_state State);

  // The function returns a vector of SYCL device images, for which Selector
  // callable returns true, in required state, which are compatible with at
  // least one of the device from Devs.
  std::vector<DevImgPlainWithDeps>
  getSYCLDeviceImages(const context &Ctx, devices_range Devs,
                      const DevImgSelectorImpl &Selector,
                      bundle_state TargetState);

  // The function returns a vector of SYCL device images which represent at
  // least one kernel from kernel ids vector in required state, which are
  // compatible with at least one of the device from Devs.
  std::vector<DevImgPlainWithDeps>
  getSYCLDeviceImages(const context &Ctx, devices_range Devs,
                      const std::vector<kernel_id> &KernelIDs,
                      bundle_state TargetState);

  // Produces new device image by convering input device image to the object
  // state
  DevImgPlainWithDeps compile(const DevImgPlainWithDeps &ImgWithDeps,
                              devices_range Devs,
                              const property_list &PropList);

  // Produces set of device images by convering input device images to object
  // the executable state
  std::vector<device_image_plain>
  link(const std::vector<device_image_plain> &Imgs, devices_range Devs,
       const property_list &PropList);

  // Produces new device image by converting input device image to the
  // executable state
  device_image_plain build(const DevImgPlainWithDeps &ImgWithDeps,
                           devices_range Devs, const property_list &PropList);

  std::tuple<Managed<ur_kernel_handle_t>, std::mutex *, const KernelArgMask *>
  getOrCreateKernel(const context &Context, KernelNameStrRefT KernelName,
                    const property_list &PropList, ur_program_handle_t Program);

  ProgramManager();
  ~ProgramManager() = default;

  template <typename NameT>
  bool kernelUsesAssert(const NameT &KernelName) const {
    return m_KernelUsesAssert.find(KernelName) != m_KernelUsesAssert.end();
  }

  SanitizerType kernelUsesSanitizer() const { return m_SanitizerFoundInImage; }

  std::optional<int>
  kernelImplicitLocalArgPos(KernelNameStrRefT KernelName) const;

  DeviceKernelInfo &
  getOrCreateDeviceKernelInfo(const CompileTimeKernelInfoTy &Info);
  DeviceKernelInfo &getOrCreateDeviceKernelInfo(KernelNameStrRefT KernelName);

  std::set<const RTDeviceBinaryImage *>
  getRawDeviceImages(const std::vector<kernel_id> &KernelIDs);

  std::set<const RTDeviceBinaryImage *>
  collectDeviceImageDeps(const RTDeviceBinaryImage &Img, const device_impl &Dev,
                         bool ErrorOnUnresolvableImport = true);
  std::set<const RTDeviceBinaryImage *>
  collectDeviceImageDepsForImportedSymbols(const RTDeviceBinaryImage &Img,
                                           const device_impl &Dev,
                                           bool ErrorOnUnresolvableImport);

  static bundle_state getBinImageState(const RTDeviceBinaryImage *BinImage);

private:
  ProgramManager(ProgramManager const &) = delete;
  ProgramManager &operator=(ProgramManager const &) = delete;

  Managed<ur_program_handle_t>
  build(Managed<ur_program_handle_t> Program, context_impl &Context,
        const std::string &CompileOptions, const std::string &LinkOptions,
        std::vector<ur_device_handle_t> &Devices, uint32_t DeviceLibReqMask,
        const std::vector<Managed<ur_program_handle_t>> &ProgramsToLink,
        bool CreatedFromBinary = false);

  /// Dumps image to current directory
  void dumpImage(const RTDeviceBinaryImage &Img, uint32_t SequenceID = 0) const;

  /// Add info on kernels using assert into cache
  void cacheKernelUsesAssertInfo(const RTDeviceBinaryImage &Img);

  /// Add info on kernels using local arg into cache
  void cacheKernelImplicitLocalArg(const RTDeviceBinaryImage &Img);

  std::set<const RTDeviceBinaryImage *>
  collectDependentDeviceImagesForVirtualFunctions(
      const RTDeviceBinaryImage &Img, const device_impl &Dev);

  bool isBfloat16DeviceImage(const RTDeviceBinaryImage *BinImage);
  bool shouldBF16DeviceImageBeUsed(const RTDeviceBinaryImage *BinImage,
                                   const device_impl &DeviceImpl);

protected:
  /// The three maps below are used during kernel resolution. Any kernel is
  /// identified by its name.
  using RTDeviceBinaryImageUPtr = std::unique_ptr<RTDeviceBinaryImage>;
  using DynRTDeviceBinaryImageUPtr = std::unique_ptr<DynRTDeviceBinaryImage>;
  /// Maps names of kernels to their unique kernel IDs.
  /// TODO: Use std::unordered_set with transparent hash and equality functions
  ///       when C++20 is enabled for the runtime library.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  //
  std::unordered_map<KernelNameStrT, kernel_id> m_KernelName2KernelIDs;

  // Maps KernelIDs to device binary images. There can be more than one image
  // in case of SPIRV + AOT.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_multimap<kernel_id, const RTDeviceBinaryImage *>
      m_KernelIDs2BinImage;

  // Maps device binary image to a vector of kernel ids in this image.
  // Using shared_ptr to avoid expensive copy of the vector.
  // The vector is initialized in addImages function and is supposed to be
  // immutable afterwards.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_map<const RTDeviceBinaryImage *,
                     std::shared_ptr<std::vector<kernel_id>>>
      m_BinImg2KernelIDs;

  /// Protects kernel ID cache.
  /// NOTE: This may be acquired while \ref Sync::getGlobalLock() is held so to
  /// avoid deadlocks care must be taken not to acquire
  /// \ref Sync::getGlobalLock() while holding this mutex.
  std::mutex m_KernelIDsMutex;

  /// Keeps track of binary image to kernel name reference count.
  /// Used for checking if the last image referencing the kernel name
  /// is removed in order to trigger cleanup of kernel specific information.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_map<KernelNameStrT, int> m_KernelNameRefCount;

  /// Caches all found service kernels to expedite future checks. A SYCL service
  /// kernel is a kernel that has not been defined by the user but is instead
  /// generated by the SYCL runtime. Service kernel name types must be declared
  /// in the sycl::detail::__sycl_service_kernel__ namespace which is
  /// exclusively used for this purpose.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_multimap<KernelNameStrT, const RTDeviceBinaryImage *>
      m_ServiceKernels;

  /// Caches all exported symbols to allow faster lookup when excluding these
  /// from kernel bundles.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  /// Owns its keys to support the bfloat16 use case with dynamic images,
  /// where the symbol is taken from another image (that might be unloaded).
  std::unordered_multimap<std::string, const RTDeviceBinaryImage *>
      m_ExportedSymbolImages;

  /// Keeps all device images we are refering to during program lifetime. Used
  /// for proper cleanup.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_map<sycl_device_binary, RTDeviceBinaryImageUPtr>
      m_DeviceImages;

  /// Maps names of built-in kernels to their unique kernel IDs.
  /// Access must be guarded by the m_BuiltInKernelIDsMutex mutex.
  std::unordered_map<KernelNameStrT, kernel_id> m_BuiltInKernelIDs;

  /// Caches list of device images that use or provide virtual functions from
  /// the same set. Used to simplify access.
  /// Access must be guarded by the m_KernelIDsMutex mutex.
  std::unordered_map<std::string, std::set<const RTDeviceBinaryImage *>>
      m_VFSet2BinImage;

  /// Protects built-in kernel ID cache.
  std::mutex m_BuiltInKernelIDsMutex;

  // Keeps track of ur_program to image correspondence. Needed for:
  // - knowing which specialization constants are used in the program and
  //   injecting their current values before compiling the SPIR-V; the binary
  //   image object has info about all spec constants used in the module
  // - finding kernel argument masks for kernels associated with each
  //   ur_program
  // NOTE: using RTDeviceBinaryImage raw pointers is OK, since they are not
  // referenced from outside SYCL runtime and RTDeviceBinaryImage object
  // lifetime matches program manager's one.
  // NOTE: keys in the map can be invalid (reference count went to zero and
  // the underlying program disposed of), so the map can't be used in any way
  // other than binary image lookup with known live UrProgram as the key.
  // NOTE: access is synchronized via the MNativeProgramsMutex
  std::unordered_multimap<
      ur_program_handle_t,
      std::pair<std::weak_ptr<context_impl>, const RTDeviceBinaryImage *>>
      NativePrograms;

  /// Protects NativePrograms that can be changed by class' methods.
  std::mutex MNativeProgramsMutex;

  using KernelNameToArgMaskMap =
      std::unordered_map<KernelNameStrT, KernelArgMask>;
  /// Maps binary image and kernel name pairs to kernel argument masks which
  /// specify which arguments were eliminated during device code optimization.
  std::unordered_map<const RTDeviceBinaryImage *, KernelNameToArgMaskMap>
      m_EliminatedKernelArgMasks;

  /// True iff a SPIR-V file has been specified with an environment variable
  bool m_UseSpvFile = false;
  RTDeviceBinaryImageUPtr m_SpvFileImage;

  // std::less<> is a transparent comparator that enabled comparison between
  // different types without temporary key_type object creation. This includes
  // standard overloads, such as comparison between std::string and
  // std::string_view or just char*.
  using KernelUsesAssertSet = std::set<KernelNameStrT, std::less<>>;
  KernelUsesAssertSet m_KernelUsesAssert;
  std::unordered_map<KernelNameStrT, int> m_KernelImplicitLocalArgPos;

  // Map for storing device kernel information. Runtime lookup should be avoided
  // by caching the pointers when possible.
  std::unordered_map<KernelNameStrT, DeviceKernelInfo> m_DeviceKernelInfoMap;

  // Protects m_DeviceKernelInfoMap.
  std::mutex m_DeviceKernelInfoMapMutex;

  // Sanitizer type used in device image
  SanitizerType m_SanitizerFoundInImage;

  // Maps between device_global identifiers and associated information.
  // The ownership of entry resources is taken to allow contexts to cleanup
  // their associated entry resources when they die.
  DeviceGlobalMap m_DeviceGlobals{/*OwnerControlledCleanup=*/true};

  // Maps between free function kernel name and associated kernel global
  // information.
  std::unordered_map<std::string_view, unsigned> m_FreeFunctionKernelGlobalInfo;

  // Maps between host_pipe identifiers and associated information.
  std::unordered_map<std::string, std::unique_ptr<HostPipeMapEntry>>
      m_HostPipes;
  std::unordered_map<const void *, HostPipeMapEntry *> m_Ptr2HostPipe;

  /// Protects m_HostPipes and m_Ptr2HostPipe.
  std::mutex m_HostPipesMutex;

  using MaterializedEntries =
      std::map<std::vector<unsigned char>, Managed<ur_kernel_handle_t>>;
  std::unordered_map<KernelNameStrT, MaterializedEntries> m_MaterializedKernels;

  // Holds bfloat16 device library images, the 1st element is for fallback
  // version and 2nd is for native version. These bfloat16 device library
  // images are provided by compiler long time ago, we expect no further
  // update, so keeping 1 copy should be OK.
  std::array<RTDeviceBinaryImageUPtr, 2> m_Bfloat16DeviceLibImages;

  friend class ::ProgramManagerTest;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
