//===--------- device.cpp - Level Zero Adapter v2 ------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <unified-runtime/ur_api.h>
#include <ur/ur.hpp>

#include "../common/device.hpp"
#include "../common/platform.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero::v2 {

ur_result_t urDeviceGetInfo(::ur_device_handle_t hDeviceOpque,
                            ::ur_device_info_t propName, size_t propSize,
                            void *propValue, size_t *propSizeRet) {
  auto hDevice = ur::level_zero::common_cast(hDeviceOpque);
  UrReturnHelper ReturnValue(propSize, propValue, propSizeRet);

  switch (propName) {
  case UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP: {
    if (!hDevice->Platform->ZeGraphExt.Supported) {
      return ReturnValue(false);
    }

    // The experimental variant of the extension reports its capabilities
    // through a structure with a different type value; an older driver would
    // not recognize the stable one and would leave graphFlags unset. The
    // structure layout (stype, pNext, graphFlags) is identical between the two
    // variants, so the stable type can be reused with the experimental value.
    constexpr ze_structure_type_t ZeStructTypeRecordReplayGraphExpProperties =
        static_cast<ze_structure_type_t>(0x00030029);
    ze_record_replay_graph_ext_properties_t GraphProperties{};
    GraphProperties.stype =
        hDevice->Platform->ZeGraphExt.UsesLegacyExperimentalApi
            ? ZeStructTypeRecordReplayGraphExpProperties
            : ZE_STRUCTURE_TYPE_RECORD_REPLAY_GRAPH_EXT_PROPERTIES;
    GraphProperties.pNext = nullptr;
    ZeStruct<ze_device_properties_t> DeviceProperties;
    DeviceProperties.pNext = &GraphProperties;
    ZE2UR_CALL(zeDeviceGetProperties, (hDevice->ZeDevice, &DeviceProperties));

    constexpr ze_record_replay_graph_ext_flags_t GraphModeMask =
        ZE_RECORD_REPLAY_GRAPH_EXT_FLAG_IMMUTABLE_GRAPH |
        ZE_RECORD_REPLAY_GRAPH_EXT_FLAG_MUTABLE_GRAPH;
    return ReturnValue(static_cast<ur_bool_t>(
        (GraphProperties.graphFlags & GraphModeMask) != 0));
  }
  case UR_DEVICE_INFO_ENQUEUE_HOST_TASK_SUPPORT_EXP:
    return ReturnValue(
        static_cast<ur_bool_t>(hDevice->Platform->ZeHostTaskExt.Supported));
  case UR_DEVICE_INFO_REUSABLE_EVENTS_SUPPORT_EXP:
    return ReturnValue(static_cast<ur_bool_t>(true));
  case UR_DEVICE_INFO_PER_EVENT_PROFILING_SUPPORT_EXP:
    return ReturnValue(true);
  case UR_DEVICE_INFO_USM_HOST_ALLOC_REGISTER_SUPPORT_EXP:
    // Registering existing host memory as a USM host allocation relies on the
    // external system memory mapping extension being supported by the driver.
    return ReturnValue(
        hDevice->Platform->ZeExternalMemoryMappingExtensionSupported);
  case UR_DEVICE_INFO_IPC_PHYSICAL_MEMORY_SUPPORT_EXP:
#if defined(__linux__)
    return ReturnValue(true);
#else
    return ReturnValue(false);
#endif
  case UR_DEVICE_INFO_IPC_EVENT_SUPPORT_EXP: {
#if defined(__linux__)
    constexpr uint32_t MinDriverBuild = 38646;
    ZeStruct<ze_driver_properties_t> ZeDriverProperties;
    ZE2UR_CALL(zeDriverGetProperties,
               (hDevice->Platform->ZeDriver, &ZeDriverProperties));
    const uint32_t DriverBuild = ZeDriverProperties.driverVersion & 0xFFFF;
    return ReturnValue(static_cast<ur_bool_t>(hDevice->isBMGOrNewer() &&
                                              DriverBuild >= MinDriverBuild));
#else
    return ReturnValue(false);
#endif
  }
  default:
    return ur::level_zero::urDeviceGetInfo(hDeviceOpque, propName, propSize,
                                           propValue, propSizeRet);
  }
}

} // namespace ur::level_zero::v2
