//==---------------- context_impl.hpp - SYCL context ------------*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <detail/device_impl.hpp>
#include <detail/kernel_program_cache.hpp>
#include <detail/memory_pool_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/property_list.hpp>

#include <map>
#include <memory>
#include <optional>
#include <set>

namespace sycl {
inline namespace _V1 {
// Forward declaration
class device;
namespace detail {
class context_impl : public std::enable_shared_from_this<context_impl> {
  struct private_tag {
    explicit private_tag() = default;
  };

public:
  /// Constructs a context_impl using a list of SYCL devices.
  ///
  /// Newly created instance will save each SYCL device in the list. This
  /// requres that all devices in the list are associated with the same
  /// SYCL platform.
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  /// PropList carries the properties of the constructed context_impl.
  ///
  /// \param DeviceList is a list of SYCL device instances.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  context_impl(const std::vector<sycl::device> DeviceList,
               async_handler AsyncHandler, const property_list &PropList,
               private_tag);

  /// Construct a context_impl using plug-in interoperability handle.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  ///
  /// \param UrContext is an instance of a valid plug-in context handle.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param Adapter is the reference to the underlying Adapter that this
  /// \param OwnedByRuntime is the flag if ownership is kept by user or
  /// transferred to runtime
  context_impl(ur_context_handle_t UrContext, async_handler AsyncHandler,
               const AdapterPtr &Adapter,
               const std::vector<sycl::device> &DeviceList, bool OwnedByRuntime,
               private_tag);

  context_impl(ur_context_handle_t UrContext, async_handler AsyncHandler,
               const AdapterPtr &Adapter, private_tag tag)
      : context_impl(UrContext, AsyncHandler, Adapter,
                     std::vector<sycl::device>{},
                     /*OwnedByRuntime*/ true, tag) {}

  // Single variadic method works because all the ctors are expected to be
  // "public" except the `private_tag` part restricting the creation to
  // `std::shared_ptr` allocations.
  template <typename... Ts>
  static std::shared_ptr<context_impl> create(Ts &&...args) {
    return std::make_shared<context_impl>(std::forward<Ts>(args)...,
                                          private_tag{});
  }

  ~context_impl();

  /// Gets OpenCL interoperability context handle.
  ///
  /// \return an instance of OpenCL cl_context.
  cl_context get() const;

  /// Gets asynchronous exception handler.
  ///
  /// \return an instance of SYCL async_handler.
  const async_handler &get_async_handler() const;

  /// \return the Adapter associated with the platform of this context.
  const AdapterPtr &getAdapter() const { return MPlatform->getAdapter(); }

  /// \return the PlatformImpl associated with this context.
  platform_impl &getPlatformImpl() const { return *MPlatform; }

  /// Queries this context for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Queries SYCL queue for SYCL backend-specific information.
  ///
  /// The return type depends on information being queried.
  template <typename Param>
  typename Param::return_type get_backend_info() const;

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// \return an instance of raw UR context handle.
  ur_context_handle_t &getHandleRef();

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// \return an instance of raw UR context handle.
  const ur_context_handle_t &getHandleRef() const;

  /// Unlike `get_info<info::context::devices>', this function returns a
  /// reference.
  const std::vector<device> &getDevices() const { return MDevices; }

  using CachedLibProgramsT =
      std::map<std::pair<DeviceLibExt, ur_device_handle_t>,
               ur_program_handle_t>;

  /// In contrast to user programs, which are compiled from user code, library
  /// programs come from the SYCL runtime. They are identified by the
  /// corresponding extension:
  ///
  ///  cl_intel_devicelib_assert -> #<ur program with assert functions>
  ///  cl_intel_devicelib_complex -> #<ur program with complex functions>
  ///  etc.
  ///
  /// See `doc/design/DeviceLibExtensions.rst' for
  /// more details.
  ///
  /// \returns an instance of sycl::detail::Locked which wraps a map with device
  /// library programs and the corresponding lock for synchronized access.
  Locked<CachedLibProgramsT> acquireCachedLibPrograms() {
    return {MCachedLibPrograms, MCachedLibProgramsMutex};
  }

  KernelProgramCache &getKernelProgramCache() const;

  /// Returns true if and only if context contains the given device.
  bool hasDevice(const detail::device_impl &Device) const;

  /// Returns true if and only if the device can be used within this context.
  /// For OpenCL this is currently equivalent to hasDevice, for other backends
  /// it returns true if the device is either a member of the context or a
  /// descendant of a member.
  bool isDeviceValid(detail::device_impl &Device) {
    detail::device_impl *CurrDevice = &Device;
    while (!hasDevice(*CurrDevice)) {
      if (CurrDevice->isRootDevice()) {
        if (CurrDevice->has(aspect::ext_oneapi_is_component)) {
          // Component devices should be implicitly usable in context created
          // for a composite device they belong to.
          auto CompositeDevice = CurrDevice->get_info<
              ext::oneapi::experimental::info::device::composite_device>();
          return hasDevice(*detail::getSyclObjImpl(CompositeDevice));
        }

        return false;
      } else if (CurrDevice->getBackend() == backend::opencl) {
        // OpenCL does not support using descendants of context members within
        // that context yet. We make the exception in case it supports
        // component/composite devices.
        // TODO remove once this limitation is lifted
        return false;
      }
      CurrDevice = detail::getSyclObjImpl(
                       CurrDevice->get_info<info::device::parent_device>())
                       .get();
    }

    return true;
  }

  // Returns the backend of this context
  backend getBackend() const {
    assert(MPlatform && "MPlatform must be not null");
    return MPlatform->getBackend();
  }

  /// Given a UR device, returns the matching shared_ptr<device_impl>
  /// within this context. May return nullptr if no match discovered.
  device_impl *findMatchingDeviceImpl(ur_device_handle_t &DeviceUR) const;

  /// Gets the native handle of the SYCL context.
  ///
  /// \return a native handle.
  ur_native_handle_t getNative() const;

  // Returns true if buffer_location property is supported by devices
  bool isBufferLocationSupported() const;

  /// Adds an associated device global to the tracked associates.
  void addAssociatedDeviceGlobal(const void *DeviceGlobalPtr);

  /// Removes an associated device global from the tracked associates.
  void removeAssociatedDeviceGlobal(const void *DeviceGlobalPtr);

  /// Adds a device global initializer.
  void addDeviceGlobalInitializer(ur_program_handle_t Program,
                                  const std::vector<device> &Devs,
                                  const RTDeviceBinaryImage *BinImage);

  /// Initializes device globals for a program on the associated queue.
  std::vector<ur_event_handle_t>
  initializeDeviceGlobals(ur_program_handle_t NativePrg,
                          const std::shared_ptr<queue_impl> &QueueImpl);

  void memcpyToHostOnlyDeviceGlobal(device_impl &DeviceImpl,
                                    const void *DeviceGlobalPtr,
                                    const void *Src, size_t DeviceGlobalTSize,
                                    bool IsDeviceImageScoped, size_t NumBytes,
                                    size_t Offset);

  void memcpyFromHostOnlyDeviceGlobal(device_impl &DeviceImpl, void *Dest,
                                      const void *DeviceGlobalPtr,
                                      bool IsDeviceImageScoped, size_t NumBytes,
                                      size_t Offset);

  /// Gets a program associated with a device global from the cache.
  std::optional<ur_program_handle_t>
  getProgramForDeviceGlobal(const device &Device,
                            DeviceGlobalMapEntry *DeviceGlobalEntry);
  /// Gets a program associated with a HostPipe Entry from the cache.
  std::optional<ur_program_handle_t>
  getProgramForHostPipe(const device &Device, HostPipeMapEntry *HostPipeEntry);

  /// Gets a program associated with Dev / Images pairs.
  std::optional<ur_program_handle_t>
  getProgramForDevImgs(const device &Device,
                       const std::set<std::uintptr_t> &ImgIdentifiers,
                       const std::string &ObjectTypeName);

  bool isOwnedByRuntime() { return MOwnedByRuntime; };

  enum PropertySupport { NotSupported = 0, Supported = 1, NotChecked = 2 };

  const property_list &getPropList() const { return MPropList; }

  std::shared_ptr<sycl::ext::oneapi::experimental::detail::memory_pool_impl>
  get_default_memory_pool(const context &Context, const device &Device,
                          const usm::alloc &Kind);

private:
  bool MOwnedByRuntime;
  async_handler MAsyncHandler;
  std::vector<device> MDevices;
  ur_context_handle_t MContext;
  // TODO: Make it a reference instead, but that needs a bit more refactoring:
  std::shared_ptr<platform_impl> MPlatform;
  property_list MPropList;
  CachedLibProgramsT MCachedLibPrograms;
  std::mutex MCachedLibProgramsMutex;
  mutable KernelProgramCache MKernelProgramCache;
  mutable PropertySupport MSupportBufferLocationByDevices;

  // Device pools.
  // Weak_ptr preventing circular dependency between memory_pool_impl and
  // context_impl.
  std::vector<std::pair<
      device,
      std::weak_ptr<sycl::ext::oneapi::experimental::detail::memory_pool_impl>>>
      MMemPoolImplPtrs;

  std::set<const void *> MAssociatedDeviceGlobals;
  std::mutex MAssociatedDeviceGlobalsMutex;

  struct DeviceGlobalInitializer {
    DeviceGlobalInitializer() = default;
    DeviceGlobalInitializer(const RTDeviceBinaryImage *BinImage)
        : MBinImage(BinImage) {
      // If there are no device globals, they are trivially fully initialized.
      // Note: Lock is not needed during construction.
      MDeviceGlobalsFullyInitialized = BinImage->getDeviceGlobals().size() == 0;
    }

    /// Clears all events of the initializer. This will not acquire the lock.
    void ClearEvents(const AdapterPtr &Adapter);

    /// The binary image of the program.
    const RTDeviceBinaryImage *MBinImage = nullptr;

    /// Mutex for protecting initialization of device globals for the image.
    std::mutex MDeviceGlobalInitMutex;

    /// Flag signalling whether or not the device globals have been initialized.
    /// This is effectively the same as checking that MDeviceGlobalInitEvents
    /// is empty, however it does not require MDeviceGlobalInitMutex to be held
    /// when checked as it will only be true when initialization is guaranteed
    /// to be done.
    /// MDeviceGlobalInitMutex must be held when determining if initialization
    /// has been begun or when setting the value.
    bool MDeviceGlobalsFullyInitialized = false;

    /// A vector of events associated with the initialization of device globals.
    /// MDeviceGlobalInitMutex must be held when accessing this.
    std::vector<ur_event_handle_t> MDeviceGlobalInitEvents;
  };

  using HandleDevicePair = std::pair<ur_program_handle_t, ur_device_handle_t>;

  struct HandleDevicePairHash {
    std::size_t operator()(const HandleDevicePair &Key) const {
      return std::hash<ur_program_handle_t>{}(Key.first) ^
             std::hash<ur_device_handle_t>{}(Key.second);
    }
  };

  std::unordered_map<HandleDevicePair, DeviceGlobalInitializer,
                     HandleDevicePairHash>
      MDeviceGlobalInitializers;
  std::mutex MDeviceGlobalInitializersMutex;
  // The number of device globals that have not been initialized yet.
  std::atomic<size_t> MDeviceGlobalNotInitializedCnt = 0;

  // For device_global variables that are not used in any kernel code we still
  // allow copy operations on them. MDeviceGlobalUnregisteredData stores the
  // associated writes.
  // The key to this map is a combination of a the pointer to the device_global
  // and optionally a device if the device_global has device image scope.
  std::map<std::pair<const void *, std::optional<ur_device_handle_t>>,
           std::unique_ptr<std::byte[]>>
      MDeviceGlobalUnregisteredData;
  std::mutex MDeviceGlobalUnregisteredDataMutex;

  void verifyProps(const property_list &Props) const;
};

template <typename T, typename Capabilities>
void GetCapabilitiesIntersectionSet(const std::vector<sycl::device> &Devices,
                                    std::vector<T> &CapabilityList) {
  for (const sycl::device &Device : Devices) {
    std::vector<T> NewCapabilityList;
    std::vector<T> DeviceCapabilities = Device.get_info<Capabilities>();
    std::set_intersection(
        CapabilityList.begin(), CapabilityList.end(),
        DeviceCapabilities.begin(), DeviceCapabilities.end(),
        std::inserter(NewCapabilityList, NewCapabilityList.begin()));
    CapabilityList = NewCapabilityList;
  }
  CapabilityList.shrink_to_fit();
}

} // namespace detail

// We're under sycl/source and these won't be exported but it's way more
// convenient to be able to reference them without extra `detail::`.
inline auto get_ur_handles(const sycl::context &syclContext) {
  sycl::detail::context_impl &Ctx = *sycl::detail::getSyclObjImpl(syclContext);
  ur_context_handle_t urCtx = Ctx.getHandleRef();
  const sycl::detail::Adapter *Adapter = Ctx.getAdapter().get();
  return std::tuple{urCtx, Adapter};
}
inline auto get_ur_handles(const sycl::device &syclDevice,
                           const sycl::context &syclContext) {
  auto [urCtx, Adapter] = get_ur_handles(syclContext);
  ur_device_handle_t urDevice =
      sycl::detail::getSyclObjImpl(syclDevice)->getHandleRef();
  return std::tuple{urDevice, urCtx, Adapter};
}
} // namespace _V1
} // namespace sycl
