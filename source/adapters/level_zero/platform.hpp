//===--------- platform.hpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "common.hpp"
#include "ur_api.h"
#include "ze_api.h"
#include "ze_ddi.h"
#include "zes_api.h"

struct ur_device_handle_t_;

typedef size_t DeviceId;

struct ur_zes_device_handle_data_t {
  zes_device_handle_t ZesDevice;
  uint32_t SubDeviceId;
  ze_bool_t SubDevice = false;
};

struct ur_platform_handle_t_ : public _ur_platform {
  ur_platform_handle_t_(ze_driver_handle_t Driver)
      : ZeDriver{Driver}, ZeApiVersion{ZE_API_VERSION_CURRENT} {}
  // Performs initialization of a newly constructed PI platform.
  ur_result_t initialize();

  // Level Zero lacks the notion of a platform, but there is a driver, which is
  // a pretty good fit to keep here.
  ze_driver_handle_t ZeDriver;

  // Cache of the ZesDevices mapped to the ZeDevices for use in zes apis calls
  // based on a ze device handle.
  std::unordered_map<ze_device_handle_t, ur_zes_device_handle_data_t>
      ZedeviceToZesDeviceMap;

  // Given a multi driver scenario, the driver handle must be translated to the
  // internal driver handle to allow calls to driver experimental apis.
  ze_driver_handle_t ZeDriverHandleExpTranslated;

  // Helper wrapper for working with Driver Version String extension in Level
  // Zero.
  ZeDriverVersionStringExtension ZeDriverVersionString;

  // Cache versions info from zeDriverGetProperties.
  std::string ZeDriverVersion;
  std::string ZeDriverApiVersion;
  ze_api_version_t ZeApiVersion;

  // Cache driver extensions
  std::unordered_map<std::string, uint32_t> zeDriverExtensionMap;

  // Flags to tell whether various Level Zero platform extensions are available.
  bool ZeDriverGlobalOffsetExtensionFound{false};
  bool ZeDriverModuleProgramExtensionFound{false};
  bool ZeDriverEventPoolCountingEventsExtensionFound{false};
  bool zeDriverImmediateCommandListAppendFound{false};

  // Cache UR devices for reuse
  std::vector<std::unique_ptr<ur_device_handle_t_>> URDevicesCache;
  ur_shared_mutex URDevicesCacheMutex;
  bool DeviceCachePopulated = false;

  // Check the device cache and load it if necessary.
  ur_result_t populateDeviceCacheIfNeeded();

  size_t getNumDevices();

  ur_device_handle_t getDeviceById(DeviceId);

  // Return the PI device from cache that represents given native device.
  // If not found, then nullptr is returned.
  ur_device_handle_t getDeviceFromNativeHandle(ze_device_handle_t);

  /// Checks the version of the level-zero driver.
  bool isDriverVersionNewerOrSimilar(uint32_t VersionMajor,
                                     uint32_t VersionMinor,
                                     uint32_t VersionBuild);

  // Keep track of all contexts in the platform. This is needed to manage
  // a lifetime of memory allocations in each context when there are kernels
  // with indirect access.
  // TODO: should be deleted when memory isolation in the context is implemented
  // in the driver.
  std::list<ur_context_handle_t> Contexts;
  ur_shared_mutex ContextsMutex;

  // Structure with function pointers for mutable command list extension.
  // Not all drivers may support it, so considering that the platform object is
  // associated with particular Level Zero driver, store this extension here.
  struct ZeMutableCmdListExtension {
    bool Supported = false;
    ze_result_t (*zexCommandListGetNextCommandIdExp)(
        ze_command_list_handle_t, const ze_mutable_command_id_exp_desc_t *,
        uint64_t *) = nullptr;
    ze_result_t (*zexCommandListUpdateMutableCommandsExp)(
        ze_command_list_handle_t,
        const ze_mutable_commands_exp_desc_t *) = nullptr;
    ze_result_t (*zexCommandListUpdateMutableCommandSignalEventExp)(
        ze_command_list_handle_t, uint64_t, ze_event_handle_t) = nullptr;
    ze_result_t (*zexCommandListUpdateMutableCommandWaitEventsExp)(
        ze_command_list_handle_t, uint64_t, uint32_t,
        ze_event_handle_t *) = nullptr;
    ze_result_t (*zexCommandListUpdateMutableCommandKernelsExp)(
        ze_command_list_handle_t, uint32_t, uint64_t *,
        ze_kernel_handle_t *) = nullptr;
    ze_result_t (*zexCommandListGetNextCommandIdWithKernelsExp)(
        ze_command_list_handle_t, const ze_mutable_command_id_exp_desc_t *,
        uint32_t, ze_kernel_handle_t *, uint64_t *) = nullptr;
  } ZeMutableCmdListExt;

  // Structure with function pointers for External Semaphore Extension.
  struct ZeExternalSemaphoreExtension {
    bool Supported = false;
    ze_result_t (*zexImportExternalSemaphoreExp)(
        ze_device_handle_t, const ze_intel_external_semaphore_exp_desc_t *,
        ze_intel_external_semaphore_exp_handle_t *);
    ze_result_t (*zexCommandListAppendWaitExternalSemaphoresExp)(
        ze_command_list_handle_t, unsigned int,
        const ze_intel_external_semaphore_exp_handle_t *,
        const ze_intel_external_semaphore_wait_exp_params_t *,
        ze_event_handle_t, uint32_t, ze_event_handle_t *);
    ze_result_t (*zexCommandListAppendSignalExternalSemaphoresExp)(
        ze_command_list_handle_t, size_t,
        const ze_intel_external_semaphore_exp_handle_t *,
        const ze_intel_external_semaphore_signal_exp_params_t *,
        ze_event_handle_t, uint32_t, ze_event_handle_t *);
    ze_result_t (*zexDeviceReleaseExternalSemaphoreExp)(
        ze_intel_external_semaphore_exp_handle_t);
  } ZeExternalSemaphoreExt;
};