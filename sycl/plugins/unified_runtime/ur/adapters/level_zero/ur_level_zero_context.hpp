//===--------- ur_level_zero_context.hpp - Level Zero Adapter ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <list>
#include <map>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "ur_level_zero_common.hpp"
#include "ur_level_zero_queue.hpp"
#include <ur/usm_allocator.hpp>

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
  // This field is only set at _pi_context creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_context.
  const ze_context_handle_t ZeContext{};

  // Keep the PI devices this PI context was created for.
  // This field is only set at _pi_context creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_context.
  // const std::vector<ur_device_handle_t> Devices;
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
  // This field is only set at _pi_context creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_context.
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
                                         ZeStruct<ze_command_queue_desc_t>>>>
      ZeComputeCommandListCache;
  std::unordered_map<ze_device_handle_t,
                     std::list<std::pair<ze_command_list_handle_t,
                                         ZeStruct<ze_command_queue_desc_t>>>>
      ZeCopyCommandListCache;

  // Store USM allocator context(internal allocator structures)
  // for USM shared and device allocations. There is 1 allocator context
  // per each pair of (context, device) per each memory type.
  std::unordered_map<ze_device_handle_t, USMAllocContext>
      DeviceMemAllocContexts;
  std::unordered_map<ze_device_handle_t, USMAllocContext>
      SharedMemAllocContexts;
  std::unordered_map<ze_device_handle_t, USMAllocContext>
      SharedReadOnlyMemAllocContexts;

  // Since L0 native runtime does not distinguisg "shared device_read_only"
  // vs regular "shared" allocations, we have keep track of it to use
  // proper USMAllocContext when freeing allocations.
  std::unordered_set<void *> SharedReadOnlyAllocs;

  // Store the host allocator context. It does not depend on any device.
  std::unique_ptr<USMAllocContext> HostMemAllocContext;

  // We need to store all memory allocations in the context because there could
  // be kernels with indirect access. Kernels with indirect access start to
  // reference all existing memory allocations at the time when they are
  // submitted to the device. Referenced memory allocations can be released only
  // when kernel has finished execution.
  std::unordered_map<void *, MemAllocRecord> MemAllocs;

  // Following member variables are used to manage assignment of events
  // to event pools.
  //
  // TODO: Create pi_event_pool class to encapsulate working with pools.
  // This will avoid needing the use of maps below, and cleanup the
  // pi_context overall.
  //

  // The cache of event pools from where new events are allocated from.
  // The head event pool is where the next event would be added to if there
  // is still some room there. If there is no room in the head then
  // the following event pool is taken (guranteed to be empty) and made the
  // head. In case there is no next pool, a new pool is created and made the
  // head.
  //
  // Cache of event pools to which host-visible events are added to.
  std::vector<std::list<ze_event_pool_handle_t>> ZeEventPoolCache{4};

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

  // Mutex to control operations on event caches.
  ur_mutex EventCacheMutex;

  // Caches for events.
  std::vector<std::list<ur_event_handle_t>> EventCaches{4};

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

  // Get index of the free slot in the available pool. If there is no available
  // pool then create new one. The HostVisible parameter tells if we need a
  // slot for a host-visible event. The ProfilingEnabled tells is we need a
  // slot for an event with profiling capabilities.
  ur_result_t getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &, size_t &,
                                             bool HostVisible,
                                             bool ProfilingEnabled);

  // Get pi_event from cache.
  ur_event_handle_t getEventFromContextCache(bool HostVisible,
                                             bool WithProfiling);

  // Add pi_event to cache.
  void addEventToContextCache(ur_event_handle_t);

  auto getZeEventPoolCache(bool HostVisible, bool WithProfiling) {
    if (HostVisible)
      return WithProfiling ? &ZeEventPoolCache[0] : &ZeEventPoolCache[1];
    else
      return WithProfiling ? &ZeEventPoolCache[2] : &ZeEventPoolCache[3];
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
  ur_result_t
  getAvailableCommandList(ur_queue_handle_t Queue,
                          ur_command_list_ptr_t &CommandList,
                          bool UseCopyEngine, bool AllowBatching = false,
                          ze_command_queue_handle_t *ForcedCmdQueue = nullptr);

  // Checks if Device is covered by this context.
  // For that the Device or its root devices need to be in the context.
  bool isValidDevice(ur_device_handle_t Device) const;

private:
  // Get the cache of events for a provided scope and profiling mode.
  auto getEventCache(bool HostVisible, bool WithProfiling) {
    if (HostVisible)
      return WithProfiling ? &EventCaches[0] : &EventCaches[1];
    else
      return WithProfiling ? &EventCaches[2] : &EventCaches[3];
  }
};

// Helper function to release the context, a caller must lock the platform-level
// mutex guarding the container with contexts because the context can be removed
// from the list of tracked contexts.
ur_result_t ContextReleaseHelper(ur_context_handle_t Context);
