//===--------- context.hpp - Level Zero Adapter --------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <unified-runtime/ur_api.h>

#include <ze_api.h>

#include "../common/interfaces.hpp"
#include "command_list_cache.hpp"
#include "common.hpp"
#include "common/ur_ref_count.hpp"
#include "event_pool_cache.hpp"
#include "logger/ur_logger.hpp"
#include "usm.hpp"

namespace ur::level_zero::v2 {

enum class PoolCacheType { Immediate, Regular };

struct ur_context_handle_t_ : ur_context_common_t {
  ur_context_handle_t_(ze_context_handle_t hContext, uint32_t numDevices,
                       const ur_device_handle_t *phDevices, bool ownZeContext);

  ur_result_t retain();

  ur_usm_pool_handle_t getDefaultUSMPool();
  ur_usm_pool_handle_t getAsyncPool();

  void addUsmPool(ur_usm_pool_handle_t hPool);
  void removeUsmPool(ur_usm_pool_handle_t hPool);
  void changeResidentDevice(ur_device_handle_t hDevice,
                            ur_device_handle_t peerDevice, bool isAdding);

  template <typename Func> void forEachUsmPool(Func func) {
    std::shared_lock<ur_shared_mutex> lock(Mutex);
    for (const auto &hPool : usmPoolHandles) {
      if (!func(hPool))
        break;
    }
  }

  std::vector<ur_device_handle_t>
  getDevicesWhoseAllocationsCanBeAccessedFrom(ur_device_handle_t hDevice);

  std::vector<ur_device_handle_t>
  getDevicesWhichCanAccessAllocationsPresentOn(ur_device_handle_t hDevice);

  event_pool &getNativeEventsPool() { return nativeEventsPool; }
  command_list_cache_t &getCommandListCache() { return commandListCache; }
  event_pool_cache &getEventPoolCache(PoolCacheType type) {
    switch (type) {
    case PoolCacheType::Immediate:
      return eventPoolCacheImmediate;
    case PoolCacheType::Regular:
      return eventPoolCacheRegular;
    }
    UR_FFAILURE("Requested invalid event pool cache type");
  }

  event_pool_cache &getReusableEventPoolCache() {
    return reusableEventPoolCache;
  }

  ur_exp_graph_handle_t getGraphFromZeHandle(ze_graph_handle_t zeGraph) {
    auto it = zeToUrGraphMap.find(zeGraph);
    return (it != zeToUrGraphMap.end()) ? it->second : nullptr;
  }

  void registerGraph(ze_graph_handle_t zeGraph, ur_exp_graph_handle_t hGraph) {
    zeToUrGraphMap[zeGraph] = hGraph;
  }

  void unregisterGraph(ze_graph_handle_t zeGraph) {
    zeToUrGraphMap.erase(zeGraph);
  }

  ur::RefCount RefCount;

  ur_shared_mutex GraphMapMutex;

private:
  const raii::ze_context_handle_t hContext;
  command_list_cache_t commandListCache;
  event_pool_cache eventPoolCacheImmediate;
  event_pool_cache eventPoolCacheRegular;
  event_pool_cache reusableEventPoolCache;

  // pool used for urEventCreateWithNativeHandle when native handle is NULL
  // (uses non-counter based events to allow for signaling from host)
  event_pool nativeEventsPool;

  ur_usm_pool_handle_t_ defaultUSMPool;
  ur_usm_pool_handle_t_ asyncPool;
  std::list<ur_usm_pool_handle_t> usmPoolHandles;

  // Graph fork-join may occur from direct L0 submissions, so we must map
  // L0 graph handles to UR handles to query across different command list
  // managers. Caller must protect accesses with GraphMapMutex.
  std::unordered_map<ze_graph_handle_t, ur_exp_graph_handle_t> zeToUrGraphMap;
};

} // namespace ur::level_zero::v2
