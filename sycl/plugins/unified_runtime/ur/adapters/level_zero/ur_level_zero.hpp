//===--------- ur_level_zero.hpp - Level Zero Adapter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "ur_level_zero_common.hpp"

struct _ur_platform_handle_t;
// using ur_platform_handle_t = _ur_platform_handle_t *;
struct _ur_device_handle_t;
// using ur_device_handle_t = _ur_device_handle_t *;

struct _ur_platform_handle_t : public _ur_platform {
  _ur_platform_handle_t(ze_driver_handle_t Driver) : ZeDriver{Driver} {}
  // Performs initialization of a newly constructed PI platform.
  ur_result_t initialize();

  // Level Zero lacks the notion of a platform, but there is a driver, which is
  // a pretty good fit to keep here.
  ze_driver_handle_t ZeDriver;

  // Cache versions info from zeDriverGetProperties.
  std::string ZeDriverVersion;
  std::string ZeDriverApiVersion;
  ze_api_version_t ZeApiVersion;

  // Cache driver extensions
  std::unordered_map<std::string, uint32_t> zeDriverExtensionMap;

  // Flags to tell whether various Level Zero platform extensions are available.
  bool ZeDriverGlobalOffsetExtensionFound{false};
  bool ZeDriverModuleProgramExtensionFound{false};

  // Cache UR devices for reuse
  std::vector<std::unique_ptr<ur_device_handle_t_>> PiDevicesCache;
  ur_shared_mutex PiDevicesCacheMutex;
  bool DeviceCachePopulated = false;

  // Check the device cache and load it if necessary.
  ur_result_t populateDeviceCacheIfNeeded();

  // Return the PI device from cache that represents given native device.
  // If not found, then nullptr is returned.
  ur_device_handle_t getDeviceFromNativeHandle(ze_device_handle_t);
};

enum EventsScope {
  // All events are created host-visible.
  AllHostVisible,
  // All events are created with device-scope and only when
  // host waits them or queries their status that a proxy
  // host-visible event is created and set to signal after
  // original event signals.
  OnDemandHostVisibleProxy,
  // All events are created with device-scope and only
  // when a batch of commands is submitted for execution a
  // last command in that batch is added to signal host-visible
  // completion of each command in this batch (the default mode).
  LastCommandInBatchHostVisible
};

struct _ur_device_handle_t : _ur_object {
  _ur_device_handle_t(ze_device_handle_t Device, ur_platform_handle_t Plt,
                      ur_device_handle_t ParentDevice = nullptr)
      : ZeDevice{Device}, Platform{Plt}, RootDevice{ParentDevice},
        ZeDeviceProperties{}, ZeDeviceComputeProperties{} {
    // NOTE: one must additionally call initialize() to complete
    // UR device creation.
  }

  // The helper structure that keeps info about a command queue groups of the
  // device. It is not changed after it is initialized.
  struct queue_group_info_t {
    enum type {
      MainCopy,
      LinkCopy,
      Compute,
      Size // must be last
    };

    // Keep the ordinal of the commands group as returned by
    // zeDeviceGetCommandQueueGroupProperties. A value of "-1" means that
    // there is no such queue group available in the Level Zero runtime.
    int32_t ZeOrdinal{-1};

    // Keep the index of the specific queue in this queue group where
    // all the command enqueues of the corresponding type should go to.
    // The value of "-1" means that no hard binding is defined and
    // implementation can choose specific queue index on its own.
    int32_t ZeIndex{-1};

    // Keeps the queue group properties.
    ZeStruct<ze_command_queue_group_properties_t> ZeProperties;
  };

  std::vector<queue_group_info_t> QueueGroup =
      std::vector<queue_group_info_t>(queue_group_info_t::Size);

  // This returns "true" if a main copy engine is available for use.
  bool hasMainCopyEngine() const {
    return QueueGroup[queue_group_info_t::MainCopy].ZeOrdinal >= 0;
  }

  // This returns "true" if a link copy engine is available for use.
  bool hasLinkCopyEngine() const {
    return QueueGroup[queue_group_info_t::LinkCopy].ZeOrdinal >= 0;
  }

  // This returns "true" if a main or link copy engine is available for use.
  bool hasCopyEngine() const {
    return hasMainCopyEngine() || hasLinkCopyEngine();
  }

  // Initialize the entire UR device.
  // Optional param `SubSubDeviceOrdinal` `SubSubDeviceIndex` are the compute
  // command queue ordinal and index respectively, used to initialize
  // sub-sub-devices.
  ur_result_t initialize(int SubSubDeviceOrdinal = -1,
                         int SubSubDeviceIndex = -1);

  // Level Zero device handle.
  // This field is only set at _ur_device_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // _ur_device_handle_t.
  const ze_device_handle_t ZeDevice;

  // Keep the subdevices that are partitioned from this ur_device_handle_t for
  // reuse The order of sub-devices in this vector is repeated from the
  // ze_device_handle_t array that are returned from zeDeviceGetSubDevices()
  // call, which will always return sub-devices in the fixed same order.
  std::vector<ur_device_handle_t> SubDevices;

  // PI platform to which this device belongs.
  // This field is only set at _ur_device_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // _ur_device_handle_t.
  ur_platform_handle_t Platform;

  // Root-device of a sub-device, null if this is not a sub-device.
  // This field is only set at _ur_device_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // _ur_device_handle_t.
  const ur_device_handle_t RootDevice;

  enum ImmCmdlistMode {
    // Immediate commandlists are not used.
    NotUsed = 0,
    // One set of compute and copy immediate commandlists per queue.
    PerQueue,
    // One set of compute and copy immediate commandlists per host thread that
    // accesses the queue.
    PerThreadPerQueue
  };
  // Read env settings to select immediate commandlist mode.
  ImmCmdlistMode useImmediateCommandLists();

  // Returns whether immediate command lists are used on this device.
  ImmCmdlistMode ImmCommandListUsed{};

  // Scope of events used for events on the device
  // Can be adjusted with UR_DEVICE_SCOPE_EVENTS
  // for non-immediate command lists
  EventsScope ZeEventsScope = AllHostVisible;

  bool isSubDevice() { return RootDevice != nullptr; }

  // Is this a Data Center GPU Max series (aka PVC)?
  // TODO: change to use
  // https://spec.oneapi.io/level-zero/latest/core/api.html#ze-device-ip-version-ext-t
  // when that is stable.
  bool isPVC() {
    return (ZeDeviceProperties->deviceId & 0xff0) == 0xbd0 ||
           (ZeDeviceProperties->deviceId & 0xff0) == 0xb60;
  }

  // Does this device represent a single compute slice?
  bool isCCS() const {
    return QueueGroup[_ur_device_handle_t::queue_group_info_t::Compute]
               .ZeIndex >= 0;
  }

  // Cache of the immutable device properties.
  ZeCache<ZeStruct<ze_device_properties_t>> ZeDeviceProperties;
  ZeCache<ZeStruct<ze_device_compute_properties_t>> ZeDeviceComputeProperties;
  ZeCache<ZeStruct<ze_device_image_properties_t>> ZeDeviceImageProperties;
  ZeCache<ZeStruct<ze_device_module_properties_t>> ZeDeviceModuleProperties;
  ZeCache<std::pair<std::vector<ZeStruct<ze_device_memory_properties_t>>,
                    std::vector<ZeStruct<ze_device_memory_ext_properties_t>>>>
      ZeDeviceMemoryProperties;
  ZeCache<ZeStruct<ze_device_memory_access_properties_t>>
      ZeDeviceMemoryAccessProperties;
  ZeCache<ZeStruct<ze_device_cache_properties_t>> ZeDeviceCacheProperties;
};
