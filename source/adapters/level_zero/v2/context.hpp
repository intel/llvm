//===--------- context.hpp - Level Zero Adapter --------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <ur_api.h>

#include "command_list_cache.hpp"
#include "common.hpp"
#include "event_pool_cache.hpp"
#include "usm.hpp"

struct ur_context_handle_t_ : _ur_object {
  ur_context_handle_t_(ze_context_handle_t hContext, uint32_t numDevices,
                       const ur_device_handle_t *phDevices, bool ownZeContext);

  ur_result_t retain();
  ur_result_t release();

  inline ze_context_handle_t getZeHandle() const { return hContext.get(); }
  ur_platform_handle_t getPlatform() const;
  const std::vector<ur_device_handle_t> &getDevices() const;
  ur_usm_pool_handle_t getDefaultUSMPool();
  const std::vector<ur_device_handle_t> &
  getP2PDevices(ur_device_handle_t hDevice) const;

  // Checks if Device is covered by this context.
  // For that the Device or its root devices need to be in the context.
  bool isValidDevice(ur_device_handle_t Device) const;

  v2::command_list_cache_t commandListCache;
  v2::event_pool_cache eventPoolCache;

  // pool used for urEventCreateWithNativeHandle when native handle is NULL
  // (uses non-counter based events to allow for signaling from host)
  v2::event_pool nativeEventsPool;

private:
  const v2::raii::ze_context_handle_t hContext;
  const std::vector<ur_device_handle_t> hDevices;

  // P2P devices for each device in the context, indexed by device id.
  const std::vector<std::vector<ur_device_handle_t>> p2pAccessDevices;

  ur_usm_pool_handle_t_ defaultUSMPool;
};
