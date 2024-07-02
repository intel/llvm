//==---------------- context_impl.cpp - SYCL context -----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <detail/context_impl.hpp>
#include <detail/context_info.hpp>
#include <detail/event_info.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/cuda_definitions.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/platform.hpp>
#include <sycl/properties/context_properties.hpp>
#include <sycl/property_list.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {
namespace detail {

context_impl::context_impl(const device &Device, async_handler AsyncHandler,
                           const property_list &PropList)
    : MOwnedByRuntime(true), MAsyncHandler(AsyncHandler), MDevices(1, Device),
      MContext(nullptr),
      MPlatform(detail::getSyclObjImpl(Device.get_platform())),
      MPropList(PropList), MSupportBufferLocationByDevices(NotChecked) {
  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(const std::vector<sycl::device> Devices,
                           async_handler AsyncHandler,
                           const property_list &PropList)
    : MOwnedByRuntime(true), MAsyncHandler(AsyncHandler), MDevices(Devices),
      MContext(nullptr), MPlatform(), MPropList(PropList),
      MSupportBufferLocationByDevices(NotChecked) {
  MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  std::vector<sycl::detail::pi::PiDevice> DeviceIds;
  for (const auto &D : MDevices) {
    if (D.has(aspect::ext_oneapi_is_composite)) {
      // Component devices are considered to be descendent devices from a
      // composite device and therefore context created for a composite
      // device should also work for a component device.
      // In order to achieve that, we implicitly add all component devices to
      // the list if a composite device was passed by user to us.
      std::vector<device> ComponentDevices = D.get_info<
          ext::oneapi::experimental::info::device::component_devices>();
      for (const auto &CD : ComponentDevices)
        DeviceIds.push_back(getSyclObjImpl(CD)->getHandleRef());
    }

    DeviceIds.push_back(getSyclObjImpl(D)->getHandleRef());
  }

  if (getBackend() == backend::ext_oneapi_cuda) {
    const bool UseCUDAPrimaryContext = MPropList.has_property<
        ext::oneapi::cuda::property::context::use_primary_context>();
    const pi_context_properties Props[] = {
        static_cast<pi_context_properties>(
            __SYCL_PI_CONTEXT_PROPERTIES_CUDA_PRIMARY),
        static_cast<pi_context_properties>(UseCUDAPrimaryContext), 0};

    getPlugin()->call<PiApiKind::piContextCreate>(
        Props, DeviceIds.size(), DeviceIds.data(), nullptr, nullptr, &MContext);
  } else {
    getPlugin()->call<PiApiKind::piContextCreate>(nullptr, DeviceIds.size(),
                                                  DeviceIds.data(), nullptr,
                                                  nullptr, &MContext);
  }

  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(sycl::detail::pi::PiContext PiContext,
                           async_handler AsyncHandler, const PluginPtr &Plugin,
                           const std::vector<sycl::device> &DeviceList,
                           bool OwnedByRuntime)
    : MOwnedByRuntime(OwnedByRuntime), MAsyncHandler(AsyncHandler),
      MDevices(DeviceList), MContext(PiContext), MPlatform(),
      MSupportBufferLocationByDevices(NotChecked) {
  if (!MDevices.empty()) {
    MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  } else {
    std::vector<sycl::detail::pi::PiDevice> DeviceIds;
    uint32_t DevicesNum = 0;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin->call<PiApiKind::piContextGetInfo>(
        MContext, PI_CONTEXT_INFO_NUM_DEVICES, sizeof(DevicesNum), &DevicesNum,
        nullptr);
    DeviceIds.resize(DevicesNum);
    // TODO catch an exception and put it to list of asynchronous exceptions
    Plugin->call<PiApiKind::piContextGetInfo>(
        MContext, PI_CONTEXT_INFO_DEVICES,
        sizeof(sycl::detail::pi::PiDevice) * DevicesNum, &DeviceIds[0],
        nullptr);

    if (!DeviceIds.empty()) {
      std::shared_ptr<detail::platform_impl> Platform =
          platform_impl::getPlatformFromPiDevice(DeviceIds[0], Plugin);
      for (sycl::detail::pi::PiDevice Dev : DeviceIds) {
        MDevices.emplace_back(createSyclObjFromImpl<device>(
            Platform->getOrMakeDeviceImpl(Dev, Platform)));
      }
      MPlatform = Platform;
    } else {
      throw invalid_parameter_error(
          "No devices in the provided device list and native context.",
          PI_ERROR_INVALID_VALUE);
    }
  }
  // TODO catch an exception and put it to list of asynchronous exceptions
  // getPlugin() will be the same as the Plugin passed. This should be taken
  // care of when creating device object.
  //
  // TODO: Move this backend-specific retain of the context to SYCL-2020 style
  //       make_context<backend::opencl> interop, when that is created.
  if (getBackend() == sycl::backend::opencl) {
    getPlugin()->call<PiApiKind::piContextRetain>(MContext);
  }
  MKernelProgramCache.setContextPtr(this);
}

cl_context context_impl::get() const {
  // TODO catch an exception and put it to list of asynchronous exceptions
  getPlugin()->call<PiApiKind::piContextRetain>(MContext);
  return pi::cast<cl_context>(MContext);
}

context_impl::~context_impl() {
  try {
    // Free all events associated with the initialization of device globals.
    for (auto &DeviceGlobalInitializer : MDeviceGlobalInitializers)
      DeviceGlobalInitializer.second.ClearEvents(getPlugin());
    // Free all device_global USM allocations associated with this context.
    for (const void *DeviceGlobal : MAssociatedDeviceGlobals) {
      DeviceGlobalMapEntry *DGEntry =
          detail::ProgramManager::getInstance().getDeviceGlobalEntry(
              DeviceGlobal);
      DGEntry->removeAssociatedResources(this);
    }
    for (auto LibProg : MCachedLibPrograms) {
      assert(LibProg.second && "Null program must not be kept in the cache");
      getPlugin()->call<PiApiKind::piProgramRelease>(LibProg.second);
    }
    // TODO catch an exception and put it to list of asynchronous exceptions
    getPlugin()->call<PiApiKind::piContextRelease>(MContext);
  } catch (std::exception &e) {
    __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~context_impl", e);
  }
}

const async_handler &context_impl::get_async_handler() const {
  return MAsyncHandler;
}

template <>
uint32_t context_impl::get_info<info::context::reference_count>() const {
  return get_context_info<info::context::reference_count>(this->getHandleRef(),
                                                          this->getPlugin());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  return createSyclObjFromImpl<platform>(MPlatform);
}
template <>
std::vector<sycl::device>
context_impl::get_info<info::context::devices>() const {
  return MDevices;
}
template <>
std::vector<sycl::memory_order>
context_impl::get_info<info::context::atomic_memory_order_capabilities>()
    const {
  std::vector<sycl::memory_order> CapabilityList{
      sycl::memory_order::relaxed, sycl::memory_order::acquire,
      sycl::memory_order::release, sycl::memory_order::acq_rel,
      sycl::memory_order::seq_cst};

  GetCapabilitiesIntersectionSet<
      sycl::memory_order, info::device::atomic_memory_order_capabilities>(
      MDevices, CapabilityList);

  return CapabilityList;
}
template <>
std::vector<sycl::memory_scope>
context_impl::get_info<info::context::atomic_memory_scope_capabilities>()
    const {
  std::vector<sycl::memory_scope> CapabilityList{
      sycl::memory_scope::work_item, sycl::memory_scope::sub_group,
      sycl::memory_scope::work_group, sycl::memory_scope::device,
      sycl::memory_scope::system};

  GetCapabilitiesIntersectionSet<
      sycl::memory_scope, info::device::atomic_memory_scope_capabilities>(
      MDevices, CapabilityList);

  return CapabilityList;
}
template <>
std::vector<sycl::memory_order>
context_impl::get_info<info::context::atomic_fence_order_capabilities>() const {
  std::vector<sycl::memory_order> CapabilityList{
      sycl::memory_order::relaxed, sycl::memory_order::acquire,
      sycl::memory_order::release, sycl::memory_order::acq_rel,
      sycl::memory_order::seq_cst};

  GetCapabilitiesIntersectionSet<sycl::memory_order,
                                 info::device::atomic_fence_order_capabilities>(
      MDevices, CapabilityList);

  return CapabilityList;
}
template <>
std::vector<sycl::memory_scope>
context_impl::get_info<info::context::atomic_fence_scope_capabilities>() const {
  std::vector<sycl::memory_scope> CapabilityList{
      sycl::memory_scope::work_item, sycl::memory_scope::sub_group,
      sycl::memory_scope::work_group, sycl::memory_scope::device,
      sycl::memory_scope::system};

  GetCapabilitiesIntersectionSet<sycl::memory_scope,
                                 info::device::atomic_fence_scope_capabilities>(
      MDevices, CapabilityList);

  return CapabilityList;
}

template <>
typename info::platform::version::return_type
context_impl::get_backend_info<info::platform::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::platform::version info descriptor can "
                          "only be queried with an OpenCL backend");
  }
  return MDevices[0].get_platform().get_info<info::platform::version>();
}

device select_device(DSelectorInvocableType DeviceSelectorInvocable,
                     std::vector<device> &Devices);

template <>
typename info::device::version::return_type
context_impl::get_backend_info<info::device::version>() const {
  if (getBackend() != backend::opencl) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::version info descriptor can only "
                          "be queried with an OpenCL backend");
  }
  auto Devices = get_info<info::context::devices>();
  if (Devices.empty()) {
    return "No available device";
  }
  // Use default selector to pick a device.
  return select_device(default_selector_v, Devices)
      .get_info<info::device::version>();
}

template <>
typename info::device::backend_version::return_type
context_impl::get_backend_info<info::device::backend_version>() const {
  if (getBackend() != backend::ext_oneapi_level_zero) {
    throw sycl::exception(errc::backend_mismatch,
                          "the info::device::backend_version info descriptor "
                          "can only be queried with a Level Zero backend");
  }
  return "";
  // Currently The Level Zero backend does not define the value of this
  // information descriptor and implementations are encouraged to return the
  // empty string as per specification.
}

sycl::detail::pi::PiContext &context_impl::getHandleRef() { return MContext; }
const sycl::detail::pi::PiContext &context_impl::getHandleRef() const {
  return MContext;
}

KernelProgramCache &context_impl::getKernelProgramCache() const {
  return MKernelProgramCache;
}

bool context_impl::hasDevice(
    std::shared_ptr<detail::device_impl> Device) const {
  for (auto D : MDevices)
    if (getSyclObjImpl(D) == Device)
      return true;
  return false;
}

DeviceImplPtr context_impl::findMatchingDeviceImpl(
    sycl::detail::pi::PiDevice &DevicePI) const {
  for (device D : MDevices)
    if (getSyclObjImpl(D)->getHandleRef() == DevicePI)
      return getSyclObjImpl(D);

  return nullptr;
}

pi_native_handle context_impl::getNative() const {
  const auto &Plugin = getPlugin();
  if (getBackend() == backend::opencl)
    Plugin->call<PiApiKind::piContextRetain>(getHandleRef());
  pi_native_handle Handle;
  Plugin->call<PiApiKind::piextContextGetNativeHandle>(getHandleRef(), &Handle);
  return Handle;
}

bool context_impl::isBufferLocationSupported() const {
  if (MSupportBufferLocationByDevices != NotChecked)
    return MSupportBufferLocationByDevices == Supported ? true : false;
  // Check that devices within context have support of buffer location
  MSupportBufferLocationByDevices = Supported;
  for (auto &Device : MDevices) {
    if (!Device.has_extension("cl_intel_mem_alloc_buffer_location")) {
      MSupportBufferLocationByDevices = NotSupported;
      break;
    }
  }
  return MSupportBufferLocationByDevices == Supported ? true : false;
}

void context_impl::addAssociatedDeviceGlobal(const void *DeviceGlobalPtr) {
  std::lock_guard<std::mutex> Lock{MAssociatedDeviceGlobalsMutex};
  MAssociatedDeviceGlobals.insert(DeviceGlobalPtr);
}

void context_impl::addDeviceGlobalInitializer(
    sycl::detail::pi::PiProgram Program, const std::vector<device> &Devs,
    const RTDeviceBinaryImage *BinImage) {
  std::lock_guard<std::mutex> Lock(MDeviceGlobalInitializersMutex);
  for (const device &Dev : Devs) {
    auto Key = std::make_pair(Program, getSyclObjImpl(Dev)->getHandleRef());
    MDeviceGlobalInitializers.emplace(Key, BinImage);
  }
}

std::vector<sycl::detail::pi::PiEvent> context_impl::initializeDeviceGlobals(
    pi::PiProgram NativePrg, const std::shared_ptr<queue_impl> &QueueImpl) {
  const PluginPtr &Plugin = getPlugin();
  const DeviceImplPtr &DeviceImpl = QueueImpl->getDeviceImplPtr();
  std::lock_guard<std::mutex> NativeProgramLock(MDeviceGlobalInitializersMutex);
  auto ImgIt = MDeviceGlobalInitializers.find(
      std::make_pair(NativePrg, DeviceImpl->getHandleRef()));
  if (ImgIt == MDeviceGlobalInitializers.end() ||
      ImgIt->second.MDeviceGlobalsFullyInitialized)
    return {};

  DeviceGlobalInitializer &InitRef = ImgIt->second;
  {
    std::lock_guard<std::mutex> InitLock(InitRef.MDeviceGlobalInitMutex);
    std::vector<sycl::detail::pi::PiEvent> &InitEventsRef =
        InitRef.MDeviceGlobalInitEvents;
    if (!InitEventsRef.empty()) {
      // Initialization has begun but we do not know if the events are done.
      auto NewEnd = std::remove_if(
          InitEventsRef.begin(), InitEventsRef.end(),
          [&Plugin](const sycl::detail::pi::PiEvent &Event) {
            return get_event_info<info::event::command_execution_status>(
                       Event, Plugin) == info::event_command_status::complete;
          });
      // Release the removed events.
      for (auto EventIt = NewEnd; EventIt != InitEventsRef.end(); ++EventIt)
        Plugin->call<PiApiKind::piEventRelease>(*EventIt);
      // Remove them from the collection.
      InitEventsRef.erase(NewEnd, InitEventsRef.end());
      // If there are no more events, we can mark it as fully initialized.
      if (InitEventsRef.empty())
        InitRef.MDeviceGlobalsFullyInitialized = true;
      return InitEventsRef;
    } else if (InitRef.MDeviceGlobalsFullyInitialized) {
      // MDeviceGlobalsFullyInitialized could have been set while we were
      // waiting on the lock and since there were no init events we are done.
      return {};
    }

    // There were no events and it was not set as fully initialized, so this is
    // responsible for intializing the device globals.
    auto DeviceGlobals = InitRef.MBinImage->getDeviceGlobals();
    std::vector<std::string> DeviceGlobalIds;
    DeviceGlobalIds.reserve(DeviceGlobals.size());
    for (const pi_device_binary_property &DeviceGlobal : DeviceGlobals)
      DeviceGlobalIds.push_back(DeviceGlobal->Name);
    std::vector<DeviceGlobalMapEntry *> DeviceGlobalEntries =
        detail::ProgramManager::getInstance().getDeviceGlobalEntries(
            DeviceGlobalIds,
            /*ExcludeDeviceImageScopeDecorated=*/true);

    // If there were no device globals without device_image_scope the device
    // globals are trivially fully initialized and we can end early.
    if (DeviceGlobalEntries.empty()) {
      InitRef.MDeviceGlobalsFullyInitialized = true;
      return {};
    }

    // We may have reserved too much for DeviceGlobalEntries, but now that we
    // know number of device globals to initialize, we can use that for the
    // list.
    InitEventsRef.reserve(DeviceGlobalEntries.size());

    // Device global map entry pointers will not die before the end of the
    // program and the pointers will stay the same, so we do not need
    // m_DeviceGlobalsMutex here.
    for (DeviceGlobalMapEntry *DeviceGlobalEntry : DeviceGlobalEntries) {
      // Get or allocate the USM memory associated with the device global.
      DeviceGlobalUSMMem &DeviceGlobalUSM =
          DeviceGlobalEntry->getOrAllocateDeviceGlobalUSM(QueueImpl);

      // If the device global still has a initialization event it should be
      // added to the initialization events list. Since initialization events
      // are cleaned up separately from cleaning up the device global USM memory
      // this must retain the event.
      {
        if (OwnedPiEvent ZIEvent = DeviceGlobalUSM.getInitEvent(Plugin))
          InitEventsRef.push_back(ZIEvent.TransferOwnership());
      }
      // Write the pointer to the device global and store the event in the
      // initialize events list.
      sycl::detail::pi::PiEvent InitEvent;
      void *const &USMPtr = DeviceGlobalUSM.getPtr();
      Plugin->call<PiApiKind::piextEnqueueDeviceGlobalVariableWrite>(
          QueueImpl->getHandleRef(), NativePrg,
          DeviceGlobalEntry->MUniqueId.c_str(), false, sizeof(void *), 0,
          &USMPtr, 0, nullptr, &InitEvent);

      InitEventsRef.push_back(InitEvent);
    }
    return InitEventsRef;
  }
}

void context_impl::DeviceGlobalInitializer::ClearEvents(
    const PluginPtr &Plugin) {
  for (const sycl::detail::pi::PiEvent &Event : MDeviceGlobalInitEvents)
    Plugin->call<PiApiKind::piEventRelease>(Event);
  MDeviceGlobalInitEvents.clear();
}

void context_impl::memcpyToHostOnlyDeviceGlobal(
    const std::shared_ptr<device_impl> &DeviceImpl, const void *DeviceGlobalPtr,
    const void *Src, size_t DeviceGlobalTSize, bool IsDeviceImageScoped,
    size_t NumBytes, size_t Offset) {
  std::optional<sycl::detail::pi::PiDevice> KeyDevice = std::nullopt;
  if (IsDeviceImageScoped)
    KeyDevice = DeviceImpl->getHandleRef();
  auto Key = std::make_pair(DeviceGlobalPtr, KeyDevice);

  std::lock_guard<std::mutex> InitLock(MDeviceGlobalUnregisteredDataMutex);

  auto UnregisteredDataIt = MDeviceGlobalUnregisteredData.find(Key);
  if (UnregisteredDataIt == MDeviceGlobalUnregisteredData.end()) {
    std::unique_ptr<std::byte[]> NewData =
        std::make_unique<std::byte[]>(DeviceGlobalTSize);
    UnregisteredDataIt =
        MDeviceGlobalUnregisteredData.insert({Key, std::move(NewData)}).first;
  }
  std::byte *ValuePtr = UnregisteredDataIt->second.get();
  std::memcpy(ValuePtr + Offset, Src, NumBytes);
}

void context_impl::memcpyFromHostOnlyDeviceGlobal(
    const std::shared_ptr<device_impl> &DeviceImpl, void *Dest,
    const void *DeviceGlobalPtr, bool IsDeviceImageScoped, size_t NumBytes,
    size_t Offset) {

  std::optional<sycl::detail::pi::PiDevice> KeyDevice = std::nullopt;
  if (IsDeviceImageScoped)
    KeyDevice = DeviceImpl->getHandleRef();
  auto Key = std::make_pair(DeviceGlobalPtr, KeyDevice);

  std::lock_guard<std::mutex> InitLock(MDeviceGlobalUnregisteredDataMutex);

  auto UnregisteredDataIt = MDeviceGlobalUnregisteredData.find(Key);
  if (UnregisteredDataIt == MDeviceGlobalUnregisteredData.end()) {
    // If there is no entry we do not need to add it as it would just be
    // zero-initialized.
    char *FillableDest = reinterpret_cast<char *>(Dest);
    std::fill(FillableDest, FillableDest + NumBytes, 0);
    return;
  }
  std::byte *ValuePtr = UnregisteredDataIt->second.get();
  std::memcpy(Dest, ValuePtr + Offset, NumBytes);
}

std::optional<sycl::detail::pi::PiProgram> context_impl::getProgramForDevImgs(
    const device &Device, const std::set<std::uintptr_t> &ImgIdentifiers,
    const std::string &ObjectTypeName) {

  KernelProgramCache::ProgramBuildResultPtr BuildRes = nullptr;
  {
    auto LockedCache = MKernelProgramCache.acquireCachedPrograms();
    auto &KeyMap = LockedCache.get().KeyMap;
    auto &Cache = LockedCache.get().Cache;
    sycl::detail::pi::PiDevice &DevHandle =
        getSyclObjImpl(Device)->getHandleRef();
    for (std::uintptr_t ImageIDs : ImgIdentifiers) {
      auto OuterKey = std::make_pair(ImageIDs, DevHandle);
      size_t NProgs = KeyMap.count(OuterKey);
      if (NProgs == 0)
        continue;
      // If the cache has multiple programs for the identifiers or if we have
      // already found a program in the cache with the device_global or host
      // pipe we cannot proceed.
      if (NProgs > 1 || (BuildRes && NProgs == 1))
        throw sycl::exception(make_error_code(errc::invalid),
                              "More than one image exists with the " +
                                  ObjectTypeName + ".");

      auto KeyMappingsIt = KeyMap.find(OuterKey);
      assert(KeyMappingsIt != KeyMap.end());
      auto CachedProgIt = Cache.find(KeyMappingsIt->second);
      assert(CachedProgIt != Cache.end());
      BuildRes = CachedProgIt->second;
    }
  }
  if (!BuildRes)
    return std::nullopt;
  using BuildState = KernelProgramCache::BuildState;
  BuildState NewState = BuildRes->waitUntilTransition();
  if (NewState == BuildState::BS_Failed)
    throw compile_program_error(BuildRes->Error.Msg, BuildRes->Error.Code);

  assert(NewState == BuildState::BS_Done);
  return BuildRes->Val;
}

std::optional<sycl::detail::pi::PiProgram>
context_impl::getProgramForDeviceGlobal(
    const device &Device, DeviceGlobalMapEntry *DeviceGlobalEntry) {
  return getProgramForDevImgs(Device, DeviceGlobalEntry->MImageIdentifiers,
                              "device_global");
}
/// Gets a program associated with a HostPipe Entry from the cache.
std::optional<sycl::detail::pi::PiProgram>
context_impl::getProgramForHostPipe(const device &Device,
                                    HostPipeMapEntry *HostPipeEntry) {
  // One HostPipe entry belongs to one Img
  std::set<std::uintptr_t> ImgIdentifiers;
  ImgIdentifiers.insert(HostPipeEntry->getDevBinImage()->getImageID());
  return getProgramForDevImgs(Device, ImgIdentifiers, "host_pipe");
}

} // namespace detail
} // namespace _V1
} // namespace sycl
