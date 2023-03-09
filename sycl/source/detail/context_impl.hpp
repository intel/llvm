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
#include <detail/platform_impl.hpp>
#include <detail/program_manager/program_manager.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/property_list.hpp>
#include <sycl/stl.hpp>

#include <map>
#include <memory>
#include <optional>
#include <set>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
// Forward declaration
class device;
namespace detail {
using PlatformImplPtr = std::shared_ptr<detail::platform_impl>;
class context_impl {
public:
  /// Constructs a context_impl using a single SYCL devices.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  /// PropList carries the properties of the constructed context_impl.
  ///
  /// \param Device is an instance of SYCL device.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param PropList is an instance of property_list.
  context_impl(const device &Device, async_handler AsyncHandler,
               const property_list &PropList);

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
               async_handler AsyncHandler, const property_list &PropList);

  /// Construct a context_impl using plug-in interoperability handle.
  ///
  /// The constructed context_impl will use the AsyncHandler parameter to
  /// handle exceptions.
  ///
  /// \param PiContext is an instance of a valid plug-in context handle.
  /// \param AsyncHandler is an instance of async_handler.
  /// \param Plugin is the reference to the underlying Plugin that this
  /// context is associated with.
  context_impl(RT::PiContext PiContext, async_handler AsyncHandler,
               const plugin &Plugin);

  ~context_impl();

  /// Checks if this context_impl has a property of type propertyT.
  ///
  /// \return true if this context_impl has a property of type propertyT.
  template <typename propertyT> bool has_property() const noexcept {
    return MPropList.has_property<propertyT>();
  }

  /// Gets the specified property of this context_impl.
  ///
  /// Throws invalid_object_error if this context_impl does not have a property
  /// of type propertyT.
  ///
  /// \return a copy of the property of type propertyT.
  template <typename propertyT> propertyT get_property() const {
    return MPropList.get_property<propertyT>();
  }

  /// Gets OpenCL interoperability context handle.
  ///
  /// \return an instance of OpenCL cl_context.
  cl_context get() const;

  /// Checks if this context is a host context.
  ///
  /// \return true if this context is a host context.
  bool is_host() const;

  /// Gets asynchronous exception handler.
  ///
  /// \return an instance of SYCL async_handler.
  const async_handler &get_async_handler() const;

  /// \return the Plugin associated with the platform of this context.
  const plugin &getPlugin() const { return MPlatform->getPlugin(); }

  /// \return the PlatformImpl associated with this context.
  PlatformImplPtr getPlatformImpl() const { return MPlatform; }

  /// Queries this context for information.
  ///
  /// The return type depends on information being queried.
  template <typename Param> typename Param::return_type get_info() const;

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// \return an instance of raw plug-in context handle.
  RT::PiContext &getHandleRef();

  /// Gets the underlying context object (if any) without reference count
  /// modification.
  ///
  /// Caller must ensure the returned object lives on stack only. It can also
  /// be safely passed to the underlying native runtime API. Warning. Returned
  /// reference will be invalid if context_impl was destroyed.
  ///
  /// \return an instance of raw plug-in context handle.
  const RT::PiContext &getHandleRef() const;

  /// Unlike `get_info<info::context::devices>', this function returns a
  /// reference.
  const std::vector<device> &getDevices() const { return MDevices; }

  using CachedLibProgramsT =
      std::map<std::pair<DeviceLibExt, RT::PiDevice>, RT::PiProgram>;

  /// In contrast to user programs, which are compiled from user code, library
  /// programs come from the SYCL runtime. They are identified by the
  /// corresponding extension:
  ///
  ///  cl_intel_devicelib_assert -> #<pi_program with assert functions>
  ///  cl_intel_devicelib_complex -> #<pi_program with complex functions>
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
  bool hasDevice(std::shared_ptr<detail::device_impl> Device) const;

  /// Returns true if and only if the device can be used within this context.
  /// For OpenCL this is currently equivalent to hasDevice, for other backends
  /// it returns true if the device is either a member of the context or a
  /// descendant of a member.
  bool isDeviceValid(DeviceImplPtr Device) {
    // OpenCL does not support using descendants of context members within that
    // context yet.
    // TODO remove once this limitation is lifted
    if (!is_host() && getPlugin().getBackend() == backend::opencl)
      return hasDevice(Device);

    while (!hasDevice(Device)) {
      if (Device->isRootDevice())
        return false;
      Device = detail::getSyclObjImpl(
          Device->get_info<info::device::parent_device>());
    }

    return true;
  }

  /// Given a PiDevice, returns the matching shared_ptr<device_impl>
  /// within this context. May return nullptr if no match discovered.
  DeviceImplPtr findMatchingDeviceImpl(RT::PiDevice &DevicePI) const;

  /// Gets the native handle of the SYCL context.
  ///
  /// \return a native handle.
  pi_native_handle getNative() const;

  // Returns true if buffer_location property is supported by devices
  bool isBufferLocationSupported() const;

  /// Adds an associated device global to the tracked associates.
  void addAssociatedDeviceGlobal(const void *DeviceGlobalPtr);

  /// Adds a device global initializer.
  void addDeviceGlobalInitializer(RT::PiProgram Program,
                                  const std::vector<device> &Devs,
                                  const RTDeviceBinaryImage *BinImage);

  /// Initializes device globals for a program on the associated queue.
  std::vector<RT::PiEvent>
  initializeDeviceGlobals(pi::PiProgram NativePrg,
                          const std::shared_ptr<queue_impl> &QueueImpl);

  /// Gets a program associated with a device global from the cache.
  std::optional<RT::PiProgram>
  getProgramForDeviceGlobal(const device &Device,
                            DeviceGlobalMapEntry *DeviceGlobalEntry);

  enum PropertySupport { NotSupported = 0, Supported = 1, NotChecked = 2 };

private:
  async_handler MAsyncHandler;
  std::vector<device> MDevices;
  RT::PiContext MContext;
  PlatformImplPtr MPlatform;
  property_list MPropList;
  bool MHostContext;
  CachedLibProgramsT MCachedLibPrograms;
  std::mutex MCachedLibProgramsMutex;
  mutable KernelProgramCache MKernelProgramCache;
  mutable PropertySupport MSupportBufferLocationByDevices;

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
    void ClearEvents(const plugin &Plugin);

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
    std::vector<RT::PiEvent> MDeviceGlobalInitEvents;
  };

  std::map<std::pair<RT::PiProgram, RT::PiDevice>, DeviceGlobalInitializer>
      MDeviceGlobalInitializers;
  std::mutex MDeviceGlobalInitializersMutex;
};

} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
