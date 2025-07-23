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
#include <detail/memory_pool_impl.hpp>
#include <detail/platform_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/exception_list.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>
#include <sycl/info/info_desc.hpp>
#include <sycl/platform.hpp>
#include <sycl/property_list.hpp>

#include <algorithm>
#include <set>

namespace sycl {
inline namespace _V1 {
namespace detail {

context_impl::context_impl(const std::vector<sycl::device> Devices,
                           async_handler AsyncHandler,
                           const property_list &PropList, private_tag)
    : MOwnedByRuntime(true), MAsyncHandler(std::move(AsyncHandler)),
      MDevices(std::move(Devices)), MContext(nullptr),
      MPlatform(detail::getSyclObjImpl(MDevices[0].get_platform())),
      MPropList(PropList), MSupportBufferLocationByDevices(NotChecked) {
  verifyProps(PropList);
  std::vector<ur_device_handle_t> DeviceIds;
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

  getAdapter().call<UrApiKind::urContextCreate>(
      DeviceIds.size(), DeviceIds.data(), nullptr, &MContext);

  MKernelProgramCache.setContextPtr(this);
}

context_impl::context_impl(ur_context_handle_t UrContext,
                           async_handler AsyncHandler, adapter_impl &Adapter,
                           const std::vector<sycl::device> &DeviceList,
                           bool OwnedByRuntime, private_tag)
    : MOwnedByRuntime(OwnedByRuntime), MAsyncHandler(std::move(AsyncHandler)),
      MDevices(DeviceList), MContext(UrContext), MPlatform(),
      MSupportBufferLocationByDevices(NotChecked) {
  if (!MDevices.empty()) {
    MPlatform = detail::getSyclObjImpl(MDevices[0].get_platform());
  } else {
    std::vector<ur_device_handle_t> DeviceIds;
    uint32_t DevicesNum = 0;
    // TODO catch an exception and put it to list of asynchronous exceptions
    Adapter.call<UrApiKind::urContextGetInfo>(
        MContext, UR_CONTEXT_INFO_NUM_DEVICES, sizeof(DevicesNum), &DevicesNum,
        nullptr);
    DeviceIds.resize(DevicesNum);
    // TODO catch an exception and put it to list of asynchronous exceptions
    Adapter.call<UrApiKind::urContextGetInfo>(
        MContext, UR_CONTEXT_INFO_DEVICES,
        sizeof(ur_device_handle_t) * DevicesNum, &DeviceIds[0], nullptr);

    if (DeviceIds.empty())
      throw exception(
          make_error_code(errc::invalid),
          "No devices in the provided device list and native context.");

    platform_impl &Platform =
        platform_impl::getPlatformFromUrDevice(DeviceIds[0], Adapter);
    for (ur_device_handle_t Dev : DeviceIds) {
      MDevices.emplace_back(
          createSyclObjFromImpl<device>(Platform.getOrMakeDeviceImpl(Dev)));
    }
    MPlatform = Platform.shared_from_this();
  }
  // TODO catch an exception and put it to list of asynchronous exceptions
  // getAdapter() will be the same as the Adapter passed. This should be taken
  // care of when creating device object.
  //
  // TODO: Move this backend-specific retain of the context to SYCL-2020 style
  //       make_context<backend::opencl> interop, when that is created.
  if (getBackend() == sycl::backend::opencl) {
    getAdapter().call<UrApiKind::urContextRetain>(MContext);
  }
  MKernelProgramCache.setContextPtr(this);
}

cl_context context_impl::get() const {
  // TODO catch an exception and put it to list of asynchronous exceptions
  getAdapter().call<UrApiKind::urContextRetain>(MContext);
  ur_native_handle_t nativeHandle = 0;
  getAdapter().call<UrApiKind::urContextGetNativeHandle>(MContext,
                                                         &nativeHandle);
  return ur::cast<cl_context>(nativeHandle);
}

context_impl::~context_impl() {
  try {
    // Free all events associated with the initialization of device globals.
    for (auto &DeviceGlobalInitializer : MDeviceGlobalInitializers)
      DeviceGlobalInitializer.second.ClearEvents(getAdapter());
    // Free all device_global USM allocations associated with this context.
    for (const void *DeviceGlobal : MAssociatedDeviceGlobals) {
      DeviceGlobalMapEntry *DGEntry =
          detail::ProgramManager::getInstance().getDeviceGlobalEntry(
              DeviceGlobal);
      DGEntry->removeAssociatedResources(this);
    }
    for (auto LibProg : MCachedLibPrograms) {
      assert(LibProg.second && "Null program must not be kept in the cache");
      getAdapter().call<UrApiKind::urProgramRelease>(LibProg.second);
    }
    // TODO catch an exception and put it to list of asynchronous exceptions
    getAdapter().call_nocheck<UrApiKind::urContextRelease>(MContext);
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
                                                          this->getAdapter());
}
template <> platform context_impl::get_info<info::context::platform>() const {
  return createSyclObjFromImpl<platform>(*MPlatform);
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

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
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
#endif

device select_device(DSelectorInvocableType DeviceSelectorInvocable,
                     std::vector<device> &Devices);

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
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
#endif

#ifndef __INTEL_PREVIEW_BREAKING_CHANGES
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
#endif

ur_context_handle_t &context_impl::getHandleRef() { return MContext; }
const ur_context_handle_t &context_impl::getHandleRef() const {
  return MContext;
}

KernelProgramCache &context_impl::getKernelProgramCache() const {
  return MKernelProgramCache;
}

bool context_impl::hasDevice(const detail::device_impl &Device) const {
  for (auto D : MDevices)
    if (getSyclObjImpl(D).get() == &Device)
      return true;
  return false;
}

device_impl *
context_impl::findMatchingDeviceImpl(ur_device_handle_t &DeviceUR) const {
  for (device D : MDevices)
    if (getSyclObjImpl(D)->getHandleRef() == DeviceUR)
      return getSyclObjImpl(D).get();

  return nullptr;
}

ur_native_handle_t context_impl::getNative() const {
  detail::adapter_impl &Adapter = getAdapter();
  ur_native_handle_t Handle;
  Adapter.call<UrApiKind::urContextGetNativeHandle>(getHandleRef(), &Handle);
  if (getBackend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainContext, ur::cast<cl_context>(Handle));
  }
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

void context_impl::removeAssociatedDeviceGlobal(const void *DeviceGlobalPtr) {
  std::lock_guard<std::mutex> Lock{MAssociatedDeviceGlobalsMutex};
  MAssociatedDeviceGlobals.erase(DeviceGlobalPtr);
}

void context_impl::addDeviceGlobalInitializer(
    ur_program_handle_t Program, devices_range Devs,
    const RTDeviceBinaryImage *BinImage) {
  if (BinImage->getDeviceGlobals().empty())
    return;
  std::lock_guard<std::mutex> Lock(MDeviceGlobalInitializersMutex);
  for (device_impl &Dev : Devs) {
    auto Key = std::make_pair(Program, Dev.getHandleRef());
    auto [Iter, Inserted] = MDeviceGlobalInitializers.emplace(Key, BinImage);
    if (Inserted && !Iter->second.MDeviceGlobalsFullyInitialized)
      ++MDeviceGlobalNotInitializedCnt;
  }
}

std::vector<ur_event_handle_t> context_impl::initializeDeviceGlobals(
    ur_program_handle_t NativePrg, queue_impl &QueueImpl,
    detail::kernel_bundle_impl *KernelBundleImplPtr) {
  if (!MDeviceGlobalNotInitializedCnt.load(std::memory_order_acquire))
    return {};

  detail::adapter_impl &Adapter = getAdapter();
  device_impl &DeviceImpl = QueueImpl.getDeviceImpl();
  std::lock_guard<std::mutex> NativeProgramLock(MDeviceGlobalInitializersMutex);
  auto ImgIt = MDeviceGlobalInitializers.find(
      std::make_pair(NativePrg, DeviceImpl.getHandleRef()));
  if (ImgIt == MDeviceGlobalInitializers.end() ||
      ImgIt->second.MDeviceGlobalsFullyInitialized)
    return {};

  DeviceGlobalInitializer &InitRef = ImgIt->second;
  {
    std::lock_guard<std::mutex> InitLock(InitRef.MDeviceGlobalInitMutex);
    std::vector<ur_event_handle_t> &InitEventsRef =
        InitRef.MDeviceGlobalInitEvents;
    if (!InitEventsRef.empty()) {
      // Initialization has begun but we do not know if the events are done.
      auto NewEnd = std::remove_if(
          InitEventsRef.begin(), InitEventsRef.end(),
          [&Adapter](const ur_event_handle_t &Event) {
            return get_event_info<info::event::command_execution_status>(
                       Event, Adapter) == info::event_command_status::complete;
          });
      // Release the removed events.
      for (auto EventIt = NewEnd; EventIt != InitEventsRef.end(); ++EventIt)
        Adapter.call<UrApiKind::urEventRelease>(*EventIt);
      // Remove them from the collection.
      InitEventsRef.erase(NewEnd, InitEventsRef.end());
      // If there are no more events, we can mark it as fully initialized.
      if (InitEventsRef.empty()) {
        InitRef.MDeviceGlobalsFullyInitialized = true;
        --MDeviceGlobalNotInitializedCnt;
      }
      return InitEventsRef;
    } else if (InitRef.MDeviceGlobalsFullyInitialized) {
      // MDeviceGlobalsFullyInitialized could have been set while we were
      // waiting on the lock and since there were no init events we are done.
      return {};
    }

    // There were no events and it was not set as fully initialized, so this is
    // responsible for initializing the device globals.
    auto DeviceGlobals = InitRef.MBinImage->getDeviceGlobals();
    std::vector<std::string> DeviceGlobalIds;
    DeviceGlobalIds.reserve(DeviceGlobals.size());
    for (const sycl_device_binary_property &DeviceGlobal : DeviceGlobals)
      DeviceGlobalIds.push_back(DeviceGlobal->Name);
    std::vector<DeviceGlobalMapEntry *> DeviceGlobalEntries =
        detail::ProgramManager::getInstance().getDeviceGlobalEntries(
            DeviceGlobalIds,
            /*ExcludeDeviceImageScopeDecorated=*/true);
    // Kernel bundles may have isolated device globals. They need to be
    // initialized too.
    if (KernelBundleImplPtr && KernelBundleImplPtr->getDeviceGlobalMap().size())
      KernelBundleImplPtr->getDeviceGlobalMap().getEntries(
          DeviceGlobalIds, /*ExcludeDeviceImageScopeDecorated=*/true,
          DeviceGlobalEntries);

    // If there were no device globals without device_image_scope the device
    // globals are trivially fully initialized and we can end early.
    if (DeviceGlobalEntries.empty()) {
      InitRef.MDeviceGlobalsFullyInitialized = true;
      --MDeviceGlobalNotInitializedCnt;
      return {};
    }

    // We may have reserved too much for DeviceGlobalEntries, but now that we
    // know number of device globals to initialize, we can use that for the
    // list.
    InitEventsRef.reserve(DeviceGlobalEntries.size());

    // Device global map entry pointers will not die before the end of the
    // program and the pointers will stay the same, so we do not need
    // to lock the device global map here.
    // The lifetimes of device global map entries representing globals in
    // runtime-compiled code will be tied to the kernel bundle, so the
    // assumption holds in that setting as well.
    for (DeviceGlobalMapEntry *DeviceGlobalEntry : DeviceGlobalEntries) {
      // Get or allocate the USM memory associated with the device global.
      DeviceGlobalUSMMem &DeviceGlobalUSM =
          DeviceGlobalEntry->getOrAllocateDeviceGlobalUSM(QueueImpl);

      // If the device global still has a initialization event it should be
      // added to the initialization events list. Since initialization events
      // are cleaned up separately from cleaning up the device global USM memory
      // this must retain the event.
      {
        if (OwnedUrEvent ZIEvent = DeviceGlobalUSM.getInitEvent(Adapter))
          InitEventsRef.push_back(ZIEvent.TransferOwnership());
      }
      // Write the pointer to the device global and store the event in the
      // initialize events list.
      ur_event_handle_t InitEvent;
      void *const &USMPtr = DeviceGlobalUSM.getPtr();
      Adapter.call<UrApiKind::urEnqueueDeviceGlobalVariableWrite>(
          QueueImpl.getHandleRef(), NativePrg,
          DeviceGlobalEntry->MUniqueId.c_str(), false, sizeof(void *), 0u,
          &USMPtr, 0u, nullptr, &InitEvent);

      InitEventsRef.push_back(InitEvent);
    }
    return InitEventsRef;
  }
}

void context_impl::DeviceGlobalInitializer::ClearEvents(adapter_impl &Adapter) {
  for (const ur_event_handle_t &Event : MDeviceGlobalInitEvents)
    Adapter.call<UrApiKind::urEventRelease>(Event);
  MDeviceGlobalInitEvents.clear();
}

void context_impl::memcpyToHostOnlyDeviceGlobal(
    device_impl &DeviceImpl, const void *DeviceGlobalPtr, const void *Src,
    size_t DeviceGlobalTSize, bool IsDeviceImageScoped, size_t NumBytes,
    size_t Offset) {
  std::optional<ur_device_handle_t> KeyDevice = std::nullopt;
  if (IsDeviceImageScoped)
    KeyDevice = DeviceImpl.getHandleRef();
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
    device_impl &DeviceImpl, void *Dest, const void *DeviceGlobalPtr,
    bool IsDeviceImageScoped, size_t NumBytes, size_t Offset) {

  std::optional<ur_device_handle_t> KeyDevice = std::nullopt;
  if (IsDeviceImageScoped)
    KeyDevice = DeviceImpl.getHandleRef();
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

std::optional<ur_program_handle_t> context_impl::getProgramForDevImgs(
    const device &Device, const std::set<std::uintptr_t> &ImgIdentifiers,
    const std::string &ObjectTypeName) {

  KernelProgramCache::ProgramBuildResultPtr BuildRes = nullptr;
  {
    auto LockedCache = MKernelProgramCache.acquireCachedPrograms();
    auto &KeyMap = LockedCache.get().KeyMap;
    auto &Cache = LockedCache.get().Cache;
    ur_device_handle_t &DevHandle = getSyclObjImpl(Device)->getHandleRef();
    for (std::uintptr_t ImageIDs : ImgIdentifiers) {
      auto OuterKey =
          std::make_pair(ImageIDs, std::set<ur_device_handle_t>{DevHandle});
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
      auto CachedProgIt = Cache.find((*KeyMappingsIt).second);
      assert(CachedProgIt != Cache.end());
      BuildRes = CachedProgIt->second;
    }
  }
  if (!BuildRes)
    return std::nullopt;
  using BuildState = KernelProgramCache::BuildState;
  BuildState NewState = BuildRes->waitUntilTransition();
  if (NewState == BuildState::BS_Failed)
    throw detail::set_ur_error(
        exception(make_error_code(errc::build), BuildRes->Error.Msg),
        BuildRes->Error.Code);

  assert(NewState == BuildState::BS_Done);
  return BuildRes->Val;
}

std::optional<ur_program_handle_t> context_impl::getProgramForDeviceGlobal(
    const device &Device, DeviceGlobalMapEntry *DeviceGlobalEntry) {
  return getProgramForDevImgs(Device, DeviceGlobalEntry->MImageIdentifiers,
                              "device_global");
}
/// Gets a program associated with a HostPipe Entry from the cache.
std::optional<ur_program_handle_t>
context_impl::getProgramForHostPipe(const device &Device,
                                    HostPipeMapEntry *HostPipeEntry) {
  // One HostPipe entry belongs to one Img
  std::set<std::uintptr_t> ImgIdentifiers;
  ImgIdentifiers.insert(HostPipeEntry->getDevBinImage()->getImageID());
  return getProgramForDevImgs(Device, ImgIdentifiers, "host_pipe");
}

void context_impl::verifyProps(const property_list &Props) const {
  auto NoAllowedPropertiesCheck = [](int) { return false; };
  detail::PropertyValidator::checkPropsAndThrow(Props, NoAllowedPropertiesCheck,
                                                NoAllowedPropertiesCheck);
}

// The handle for a device default pool is retrieved once on first request.
// Subsequent requests are returned immediately without calling the backend.
std::shared_ptr<sycl::ext::oneapi::experimental::detail::memory_pool_impl>
context_impl::get_default_memory_pool(const context &Context,
                                      const device &Device,
                                      [[maybe_unused]] const usm::alloc &Kind) {

  assert(Kind == usm::alloc::device);

  detail::device_impl &DevImpl = *detail::getSyclObjImpl(Device);
  ur_device_handle_t DeviceHandle = DevImpl.getHandleRef();
  detail::adapter_impl &Adapter = this->getAdapter();

  // Check dev is already in our list of device pool pairs.
  if (auto it = std::find_if(MMemPoolImplPtrs.begin(), MMemPoolImplPtrs.end(),
                             [&](auto &pair) { return Device == pair.first; });
      it != MMemPoolImplPtrs.end()) {
    // Check if the shared_ptr of memory_pool_impl has not been destroyed.
    if (!it->second.expired())
      return it->second.lock();
  }

  // The memory_pool_impl does not exist for this device yet.
  ur_usm_pool_handle_t PoolHandle;
  Adapter.call<sycl::errc::runtime,
               sycl::detail::UrApiKind::urUSMPoolGetDefaultDevicePoolExp>(
      this->getHandleRef(), DeviceHandle, &PoolHandle);

  auto MemPoolImplPtr = std::make_shared<
      sycl::ext::oneapi::experimental::detail::memory_pool_impl>(
      Context, Device, sycl::usm::alloc::device, PoolHandle,
      true /*Default pool*/);

  // Hold onto a weak_ptr of the memory_pool_impl. Prevents circular
  // dependencies between the context_impl and memory_pool_impl.
  MMemPoolImplPtrs.push_back(std::pair(Device, MemPoolImplPtr));

  return MemPoolImplPtr;
}

} // namespace detail
} // namespace _V1
} // namespace sycl
