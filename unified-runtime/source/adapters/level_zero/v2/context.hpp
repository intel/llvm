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

enum class PoolCacheType { Immediate, Regular };

struct ur_context_handle_t_ : ur_object {
  ur_context_handle_t_(ze_context_handle_t hContext, uint32_t numDevices,
                       const ur_device_handle_t *phDevices, bool ownZeContext);

  ur_result_t retain();
  ur_result_t release();

  inline ze_context_handle_t getZeHandle() const { return hContext.get(); }
  ur_platform_handle_t getPlatform() const;

  const std::vector<ur_device_handle_t> &getDevices() const;
  ur_usm_pool_handle_t getDefaultUSMPool();
  ur_usm_pool_handle_t getAsyncPool();

  void addUsmPool(ur_usm_pool_handle_t hPool);
  void removeUsmPool(ur_usm_pool_handle_t hPool);

  template <typename Func> void forEachUsmPool(Func func) {
    std::shared_lock<ur_shared_mutex> lock(Mutex);
    for (const auto &hPool : usmPoolHandles) {
      if (!func(hPool))
        break;
    }
  }

  const std::vector<ur_device_handle_t> &
  getP2PDevices(ur_device_handle_t hDevice) const;

  v2::event_pool &getNativeEventsPool() { return nativeEventsPool; }
  v2::command_list_cache_t &getCommandListCache() { return commandListCache; }
  v2::event_pool_cache &getEventPoolCache(PoolCacheType type) {
    switch (type) {
    case PoolCacheType::Immediate:
      return eventPoolCacheImmediate;
    case PoolCacheType::Regular:
      return eventPoolCacheRegular;
    default:
      assert(false && "Requested invalid event pool cache type");
      throw UR_RESULT_ERROR_INVALID_VALUE;
    }
  }
  // Checks if Device is covered by this context.
  // For that the Device or its root devices need to be in the context.
  bool isValidDevice(ur_device_handle_t Device) const;

private:
  const v2::raii::ze_context_handle_t hContext;
  const std::vector<ur_device_handle_t> hDevices;
  v2::command_list_cache_t commandListCache;
  v2::event_pool_cache eventPoolCacheImmediate;
  v2::event_pool_cache eventPoolCacheRegular;

  // pool used for urEventCreateWithNativeHandle when native handle is NULL
  // (uses non-counter based events to allow for signaling from host)
  v2::event_pool nativeEventsPool;

  // P2P devices for each device in the context, indexed by device id.
  const std::vector<std::vector<ur_device_handle_t>> p2pAccessDevices;

  ur_usm_pool_handle_t_ defaultUSMPool;
  ur_usm_pool_handle_t_ asyncPool;
  std::list<ur_usm_pool_handle_t> usmPoolHandles;
};
