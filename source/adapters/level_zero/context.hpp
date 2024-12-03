//===--------- context.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <list>
#include <map>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ur/ur.hpp>
#include <ur_ddi.h>
#include <ze_api.h>
#include <zes_api.h>

#include "common.hpp"
#include "queue.hpp"

#include <umf_helpers.hpp>

struct l0_command_list_cache_info {
  ZeStruct<ze_command_queue_desc_t> ZeQueueDesc;
  bool InOrderList = false;
  bool IsImmediate = false;
};

typedef uint32_t ze_intel_event_sync_mode_exp_flags_t;
typedef enum _ze_intel_event_sync_mode_exp_flag_t {
  ZE_INTEL_EVENT_SYNC_MODE_EXP_FLAG_LOW_POWER_WAIT = ZE_BIT(0),
  ZE_INTEL_EVENT_SYNC_MODE_EXP_FLAG_SIGNAL_INTERRUPT = ZE_BIT(1),
  ZE_INTEL_EVENT_SYNC_MODE_EXP_EXP_FLAG_FORCE_UINT32 = 0x7fffffff

} ze_intel_event_sync_mode_exp_flag_t;

#define ZE_INTEL_STRUCTURE_TYPE_EVENT_SYNC_MODE_EXP_DESC                       \
  (ze_structure_type_t)0x00030016

typedef struct _ze_intel_event_sync_mode_exp_desc_t {
  ze_structure_type_t stype;
  const void *pNext;

  ze_intel_event_sync_mode_exp_flags_t syncModeFlags;
} ze_intel_event_sync_mode_exp_desc_t;

struct ur_context_handle_t_ : _ur_object {
  ur_context_handle_t_(ze_context_handle_t ZeContext, uint32_t NumDevices,
                       const ur_device_handle_t *Devs, bool OwnZeContext)
      : ZeContext{ZeContext}, Devices{Devs, Devs + NumDevices},
        NumDevices{NumDevices} {
    OwnNativeHandle = OwnZeContext;
  }

  ur_context_handle_t_(ze_context_handle_t ZeContext) : ZeContext{ZeContext} {}

  // A L0 context handle is primarily used during creation and management of
  // resources that may be used by multiple devices.
  // This field is only set at ur_context_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // ur_context_handle_t.
  const ze_context_handle_t ZeContext{};

  // Keep the PI devices this PI context was created for.
  // This field is only set at ur_context_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // ur_context_handle_t. const std::vector<ur_device_handle_t> Devices;
  std::vector<ur_device_handle_t> Devices;
  uint32_t NumDevices{};

  // Immediate Level Zero command list for the device in this context, to be
  // used for initializations. To be created as:
  // - Immediate command list: So any command appended to it is immediately
  //   offloaded to the device.
  // - Synchronous: So implicit synchronization is made inside the level-zero
  //   driver.
  // There will be a list of immediate command lists (for each device) when
  // support of the multiple devices per context will be added.
  ze_command_list_handle_t ZeCommandListInit{};

  // Mutex for the immediate command list. Per the Level Zero spec memory copy
  // operations submitted to an immediate command list are not allowed to be
  // called from simultaneous threads.
  ur_mutex ImmediateCommandListMutex;

  // Mutex Lock for the Command List Cache. This lock is used to control both
  // compute and copy command list caches.
  ur_mutex ZeCommandListCacheMutex;

  // If context contains one device or sub-devices of the same device, we want
  // to save this device.
  // This field is only set at ur_context_handle_t creation time, and cannot
  // change. Therefore it can be accessed without holding a lock on this
  // ur_context_handle_t.
  ur_device_handle_t SingleRootDevice = nullptr;

  // Cache of all currently available/completed command/copy lists.
  // Note that command-list can only be re-used on the same device.
  //
  // TODO: explore if we should use root-device for creating command-lists
  // as spec says that in that case any sub-device can re-use it: "The
  // application must only use the command list for the device, or its
  // sub-devices, which was provided during creation."
  //
  std::unordered_map<ze_device_handle_t,
                     std::list<std::pair<ze_command_list_handle_t,
                                         l0_command_list_cache_info>>>
      ZeComputeCommandListCache;
  std::unordered_map<ze_device_handle_t,
                     std::list<std::pair<ze_command_list_handle_t,
                                         l0_command_list_cache_info>>>
      ZeCopyCommandListCache;

  std::unordered_map<ur_device_handle_t, std::list<ur_device_handle_t>>
      P2PDeviceCache;

  // Store USM pool for USM shared and device allocations. There is 1 memory
  // pool per each pair of (context, device) per each memory type.
  std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
      DeviceMemPools;
  std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
      SharedMemPools;
  std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
      SharedReadOnlyMemPools;

  // Store the host memory pool. It does not depend on any device.
  umf::pool_unique_handle_t HostMemPool;

  // Allocation-tracking proxy pools for direct allocations. No pooling used.
  std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
      DeviceMemProxyPools;
  std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
      SharedMemProxyPools;
  std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
      SharedReadOnlyMemProxyPools;
  umf::pool_unique_handle_t HostMemProxyPool;

  // Map associating pools created with urUsmPoolCreate and internal pools
  std::list<ur_usm_pool_handle_t> UsmPoolHandles{};

  // We need to store all memory allocations in the context because there could
  // be kernels with indirect access. Kernels with indirect access start to
  // reference all existing memory allocations at the time when they are
  // submitted to the device. Referenced memory allocations can be released only
  // when kernel has finished execution.
  std::unordered_map<void *, MemAllocRecord> MemAllocs;

  // Following member variables are used to manage assignment of events
  // to event pools.
  //
  // TODO: Create ur_event_pool class to encapsulate working with pools.
  // This will avoid needing the use of maps below, and cleanup the
  // ur_context_handle_t overall.
  //

  // The cache of event pools from where new events are allocated from.
  // The head event pool is where the next event would be added to if there
  // is still some room there. If there is no room in the head then
  // the following event pool is taken (guranteed to be empty) and made the
  // head. In case there is no next pool, a new pool is created and made the
  // head.
  //
  // Cache of event pools to which host-visible events are added to.
  std::vector<std::list<ze_event_pool_handle_t>> ZeEventPoolCache{30};
  std::vector<std::unordered_map<ze_device_handle_t, size_t>>
      ZeEventPoolCacheDeviceMap{30};

  // This map will be used to determine if a pool is full or not
  // by storing number of empty slots available in the pool.
  std::unordered_map<ze_event_pool_handle_t, uint32_t>
      NumEventsAvailableInEventPool;
  // This map will be used to determine number of unreleased events in the pool.
  // We use separate maps for number of event slots available in the pool from
  // the number of events unreleased in the pool.
  // This will help when we try to make the code thread-safe.
  std::unordered_map<ze_event_pool_handle_t, uint32_t>
      NumEventsUnreleasedInEventPool;

  // Mutex to control operations on event pool caches and the helper maps
  // holding the current pool usage counts.
  ur_mutex ZeEventPoolCacheMutex;

  // Initialize the PI context.
  ur_result_t initialize();

  // If context contains one device then return this device.
  // If context contains sub-devices of the same device, then return this parent
  // device. Return nullptr if context consists of several devices which are not
  // sub-devices of the same device. We call returned device the root device of
  // a context.
  // TODO: get rid of this when contexts with multiple devices are supported for
  // images.
  ur_device_handle_t getRootDevice() const;

  // Finalize the PI context
  ur_result_t finalize();

  // Return the Platform, which is the same for all devices in the context
  ur_platform_handle_t getPlatform() const;

  // Get vector of devices from this context
  const std::vector<ur_device_handle_t> &getDevices() const;

  // Get index of the free slot in the available pool. If there is no available
  // pool then create new one. The HostVisible parameter tells if we need a
  // slot for a host-visible event. The ProfilingEnabled tells is we need a
  // slot for an event with profiling capabilities.
  ur_result_t getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &, size_t &,
                                             bool HostVisible,
                                             bool ProfilingEnabled,
                                             ur_device_handle_t Device,
                                             bool CounterBasedEventEnabled,
                                             bool UsingImmCmdList,
                                             bool InterruptBasedEventEnabled);

  // Get ur_event_handle_t from cache.
  ur_event_handle_t getEventFromContextCache(bool HostVisible,
                                             bool WithProfiling,
                                             ur_device_handle_t Device,
                                             bool CounterBasedEventEnabled,
                                             bool InterruptBasedEventEnabled);

  // Add ur_event_handle_t to cache.
  void addEventToContextCache(ur_event_handle_t);

  enum EventPoolCacheType {
    HostVisibleCacheType,
    HostInvisibleCacheType,
    HostVisibleCounterBasedRegularCacheType,
    HostInvisibleCounterBasedRegularCacheType,
    HostVisibleCounterBasedImmediateCacheType,
    HostInvisibleCounterBasedImmediateCacheType,

    HostVisibleInterruptBasedRegularCacheType,
    HostInvisibleInterruptBasedRegularCacheType,
    HostVisibleInterruptBasedImmediateCacheType,
    HostInvisibleInterruptBasedImmediateCacheType,

    HostVisibleInterruptAndCounterBasedRegularCacheType,
    HostInvisibleInterruptAndCounterBasedRegularCacheType,
    HostVisibleInterruptAndCounterBasedImmediateCacheType,
    HostInvisibleInterruptAndCounterBasedImmediateCacheType
  };

  std::list<ze_event_pool_handle_t> *
  getZeEventPoolCache(bool HostVisible, bool WithProfiling,
                      bool CounterBasedEventEnabled, bool UsingImmediateCmdList,
                      bool InterruptBasedEventEnabled,
                      ze_device_handle_t ZeDevice) {
    EventPoolCacheType CacheType;

    calculateCacheIndex(HostVisible, CounterBasedEventEnabled,
                        UsingImmediateCmdList, InterruptBasedEventEnabled,
                        CacheType);
    if (ZeDevice) {
      auto ZeEventPoolCacheMap =
          WithProfiling ? &ZeEventPoolCacheDeviceMap[CacheType * 2]
                        : &ZeEventPoolCacheDeviceMap[CacheType * 2 + 1];
      if (ZeEventPoolCacheMap->find(ZeDevice) == ZeEventPoolCacheMap->end()) {
        ZeEventPoolCache.emplace_back();
        ZeEventPoolCacheMap->insert(
            std::make_pair(ZeDevice, ZeEventPoolCache.size() - 1));
      }
      return &ZeEventPoolCache[(*ZeEventPoolCacheMap)[ZeDevice]];
    } else {
      return WithProfiling ? &ZeEventPoolCache[CacheType * 2]
                           : &ZeEventPoolCache[CacheType * 2 + 1];
    }
  }

  ur_result_t calculateCacheIndex(bool HostVisible,
                                  bool CounterBasedEventEnabled,
                                  bool UsingImmediateCmdList,
                                  bool InterruptBasedEventEnabled,
                                  EventPoolCacheType &CacheType) {
    if (InterruptBasedEventEnabled) {
      if (CounterBasedEventEnabled) {
        if (HostVisible) {
          if (UsingImmediateCmdList) {
            CacheType = HostVisibleInterruptAndCounterBasedImmediateCacheType;
          } else {
            CacheType = HostVisibleInterruptAndCounterBasedRegularCacheType;
          }
        } else {
          if (UsingImmediateCmdList) {
            CacheType = HostInvisibleInterruptAndCounterBasedImmediateCacheType;
          } else {
            CacheType = HostInvisibleInterruptAndCounterBasedRegularCacheType;
          }
        }
      } else {
        if (HostVisible) {
          if (UsingImmediateCmdList) {
            CacheType = HostVisibleInterruptBasedImmediateCacheType;
          } else {
            CacheType = HostVisibleInterruptBasedRegularCacheType;
          }
        } else {
          if (UsingImmediateCmdList) {
            CacheType = HostInvisibleInterruptBasedImmediateCacheType;
          } else {
            CacheType = HostInvisibleInterruptBasedRegularCacheType;
          }
        }
      }
    } else {
      if (CounterBasedEventEnabled && HostVisible && !UsingImmediateCmdList) {
        CacheType = HostVisibleCounterBasedRegularCacheType;
      } else if (CounterBasedEventEnabled && !HostVisible &&
                 !UsingImmediateCmdList) {
        CacheType = HostInvisibleCounterBasedRegularCacheType;
      } else if (CounterBasedEventEnabled && HostVisible &&
                 UsingImmediateCmdList) {
        CacheType = HostVisibleCounterBasedImmediateCacheType;
      } else if (CounterBasedEventEnabled && !HostVisible &&
                 UsingImmediateCmdList) {
        CacheType = HostInvisibleCounterBasedImmediateCacheType;
      } else if (!CounterBasedEventEnabled && HostVisible) {
        CacheType = HostVisibleCacheType;
      } else {
        CacheType = HostInvisibleCacheType;
      }
    }

    return UR_RESULT_SUCCESS;
  }

  // Decrement number of events living in the pool upon event destroy
  // and return the pool to the cache if there are no unreleased events.
  ur_result_t decrementUnreleasedEventsInPool(ur_event_handle_t Event);

  // Retrieves a command list for executing on this device along with
  // a fence to be used in tracking the execution of this command list.
  // If a command list has been created on this device which has
  // completed its commands, then that command list and its associated fence
  // will be reused. Otherwise, a new command list and fence will be created for
  // running on this device. L0 fences are created on a L0 command queue so the
  // caller must pass a command queue to create a new fence for the new command
  // list if a command list/fence pair is not available. All Command Lists &
  // associated fences are destroyed at Device Release.
  // If UseCopyEngine is true, the command will eventually be executed in a
  // copy engine. Otherwise, the command will be executed in a compute engine.
  // If AllowBatching is true, then the command list returned may already have
  // command in it, if AllowBatching is false, any open command lists that
  // already exist in Queue will be closed and executed.
  // If ForcedCmdQueue is not nullptr, the resulting command list must be tied
  // to the contained command queue. This option is ignored if immediate
  // command lists are used.
  // When using immediate commandlists, retrieves an immediate command list
  // for executing on this device. Immediate commandlists are created only
  // once for each SYCL Queue and after that they are reused.
  ur_result_t getAvailableCommandList(
      ur_queue_handle_t Queue, ur_command_list_ptr_t &CommandList,
      bool UseCopyEngine, uint32_t NumEventsInWaitList,
      const ur_event_handle_t *EventWaitList, bool AllowBatching,
      ze_command_queue_handle_t *ForcedCmdQueue);

  // Checks if Device is covered by this context.
  // For that the Device or its root devices need to be in the context.
  bool isValidDevice(ur_device_handle_t Device) const;

  // Get handle to the L0 context
  ze_context_handle_t getZeHandle() const;

private:
  enum EventFlags {
    EVENT_FLAG_HOST_VISIBLE = UR_BIT(0),
    EVENT_FLAG_WITH_PROFILING = UR_BIT(1),
    EVENT_FLAG_COUNTER = UR_BIT(2),
    EVENT_FLAG_INTERRUPT = UR_BIT(3),
    EVENT_FLAG_DEVICE = UR_BIT(5), // if set, subsequent bits are device id
    MAX_EVENT_FLAG_BITS =
        6, // this is used as an offset for embedding device id
  };

  // Mutex to control operations on event caches.
  ur_mutex EventCacheMutex;

  // Caches for events.
  using EventCache = std::list<ur_event_handle_t>;
  std::vector<EventCache> EventCaches;

  // Get the cache of events for a provided scope and profiling mode.
  EventCache *getEventCache(bool HostVisible, bool WithProfiling,
                            ur_device_handle_t Device, bool Counter,
                            bool Interrupt) {

    size_t index = 0;
    if (HostVisible) {
      index |= EVENT_FLAG_HOST_VISIBLE;
    }
    if (WithProfiling) {
      index |= EVENT_FLAG_WITH_PROFILING;
    }
    if (Counter) {
      index |= EVENT_FLAG_COUNTER;
    }
    if (Interrupt) {
      index |= EVENT_FLAG_INTERRUPT;
    }
    if (Device) {
      index |= EVENT_FLAG_DEVICE | (*Device->Id << MAX_EVENT_FLAG_BITS);
    }

    if (index >= EventCaches.size()) {
      EventCaches.resize(index + 1);
    }

    return &EventCaches[index];
  }
};

// Helper function to release the context, a caller must lock the platform-level
// mutex guarding the container with contexts because the context can be removed
// from the list of tracked contexts.
ur_result_t ContextReleaseHelper(ur_context_handle_t Context);
