//===--------- device.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <ur_api.h>

namespace ur::level_zero::common {

ur_result_t urDeviceGet(ur_platform_handle_t hPlatform,
                        ur_device_type_t DeviceType, uint32_t NumEntries,
                        ur_device_handle_t *phDevices, uint32_t *pNumDevices);
ur_result_t urDeviceGetInfo(ur_device_handle_t hDevice,
                            ur_device_info_t propName, size_t propSize,
                            void *pPropValue, size_t *pPropSizeRet);
ur_result_t urDeviceRetain(ur_device_handle_t hDevice);
ur_result_t urDeviceRelease(ur_device_handle_t hDevice);
ur_result_t
urDevicePartition(ur_device_handle_t hDevice,
                  const ur_device_partition_properties_t *pProperties,
                  uint32_t NumDevices, ur_device_handle_t *phSubDevices,
                  uint32_t *pNumDevicesRet);
ur_result_t urDeviceSelectBinary(ur_device_handle_t hDevice,
                                 const ur_device_binary_t *pBinaries,
                                 uint32_t NumBinaries,
                                 uint32_t *pSelectedBinary);
ur_result_t urDeviceGetNativeHandle(ur_device_handle_t hDevice,
                                    ur_native_handle_t *phNativeDevice);
ur_result_t
urDeviceCreateWithNativeHandle(ur_native_handle_t hNativeDevice,
                               ur_adapter_handle_t hAdapter,
                               const ur_device_native_properties_t *pProperties,
                               ur_device_handle_t *phDevice);
ur_result_t urDeviceGetGlobalTimestamps(ur_device_handle_t hDevice,
                                        uint64_t *pDeviceTimestamp,
                                        uint64_t *pHostTimestamp);
ur_result_t urDeviceWaitExp(ur_device_handle_t hDevice);

} // namespace ur::level_zero::common
