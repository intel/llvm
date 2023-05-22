//===--------- pi_level_zero.hpp - Level Zero Plugin ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

/// \defgroup sycl_pi_level_zero Level Zero Plugin
/// \ingroup sycl_pi

/// \file pi_level_zero.hpp
/// Declarations for Level Zero Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying Level Zero runtime.
///
/// \ingroup sycl_pi_level_zero

#ifndef PI_LEVEL_ZERO_HPP
#define PI_LEVEL_ZERO_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_LEVEL_ZERO_PLUGIN_VERSION 1

#define _PI_LEVEL_ZERO_PLUGIN_VERSION_STRING                                   \
  _PI_PLUGIN_VERSION_STRING(_PI_LEVEL_ZERO_PLUGIN_VERSION)

#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <sycl/detail/pi.h>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <sycl/detail/iostream_proxy.hpp>
#include <ze_api.h>
#include <zes_api.h>

// Share code between this PI L0 Plugin and UR L0 Adapter
#include <pi2ur.hpp>
#include <ur/adapters/level_zero/ur_level_zero.hpp>
#include <ur/usm_allocator.hpp>

// Define the types that are opaque in pi.h in a manner suitabale for Level Zero
// plugin

struct _pi_platform : public _ur_platform_handle_t {
  using _ur_platform_handle_t::_ur_platform_handle_t;

  // Keep track of all contexts in the platform. This is needed to manage
  // a lifetime of memory allocations in each context when there are kernels
  // with indirect access.
  // TODO: should be deleted when memory isolation in the context is implemented
  // in the driver.
  std::list<pi_context> Contexts;
  ur_shared_mutex ContextsMutex;
};

// Implements memory allocation via L0 RT for USM allocator interface.
class USMMemoryAllocBase : public SystemMemory {
protected:
  pi_context Context;
  pi_device Device;
  // Internal allocation routine which must be implemented for each allocation
  // type
  virtual pi_result allocateImpl(void **ResultPtr, size_t Size,
                                 pi_uint32 Alignment) = 0;

public:
  USMMemoryAllocBase(pi_context Ctx, pi_device Dev)
      : Context{Ctx}, Device{Dev} {}
  void *allocate(size_t Size) override final;
  void *allocate(size_t Size, size_t Alignment) override final;
  void deallocate(void *Ptr) override final;
};

// Allocation routines for shared memory type
class USMSharedMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;

public:
  USMSharedMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for shared memory type that is only modified from host.
class USMSharedReadOnlyMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;

public:
  USMSharedReadOnlyMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for device memory type
class USMDeviceMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;

public:
  USMDeviceMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for host memory type
class USMHostMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;

public:
  USMHostMemoryAlloc(pi_context Ctx) : USMMemoryAllocBase(Ctx, nullptr) {}
};

struct _pi_device : _ur_device_handle_t {
  using _ur_device_handle_t::_ur_device_handle_t;
};

// Structure describing the specific use of a command-list in a queue.
// This is because command-lists are re-used across multiple queues
// in the same context.
struct pi_command_list_info_t {
  // The Level-Zero fence that will be signalled at completion.
  // Immediate commandlists do not have an associated fence.
  // A nullptr for the fence indicates that this is an immediate commandlist.
  ze_fence_handle_t ZeFence{nullptr};
  // Record if the fence is in use.
  // This is needed to avoid leak of the tracked command-list if the fence
  // was not yet signaled at the time all events in that list were already
  // completed (we are polling the fence at events completion). The fence
  // may be still "in-use" due to sporadic delay in HW.
  bool ZeFenceInUse{false};

  // Indicates if command list is in closed state. This is needed to avoid
  // appending commands to the closed command list.
  bool IsClosed{false};

  // Record the queue to which the command list will be submitted.
  ze_command_queue_handle_t ZeQueue{nullptr};
  // Keeps the ordinal of the ZeQueue queue group. Invalid if ZeQueue==nullptr
  uint32_t ZeQueueGroupOrdinal{0};
  // Helper functions to tell if this is a copy command-list.
  bool isCopy(pi_queue Queue) const;

  // Keeps events created by commands submitted into this command-list.
  // TODO: use this for explicit wait/cleanup of events at command-list
  // completion.
  // TODO: use this for optimizing events in the same command-list, e.g.
  // only have last one visible to the host.
  std::vector<pi_event> EventList{};
  size_t size() const { return EventList.size(); }
  void append(pi_event Event) { EventList.push_back(Event); }
};

// The map type that would track all command-lists in a queue.
using pi_command_list_map_t =
    std::unordered_map<ze_command_list_handle_t, pi_command_list_info_t>;
// The iterator pointing to a specific command-list in use.
using pi_command_list_ptr_t = pi_command_list_map_t::iterator;

struct _pi_context : _ur_object {
  _pi_context(ze_context_handle_t ZeContext, pi_uint32 NumDevices,
              const pi_device *Devs, bool OwnZeContext)
      : ZeContext{ZeContext}, OwnZeContext{OwnZeContext},
        Devices{Devs, Devs + NumDevices}, SingleRootDevice(getRootDevice()),
        ZeCommandListInit{nullptr} {
    // NOTE: one must additionally call initialize() to complete
    // PI context creation.
  }

  // Initialize the PI context.
  pi_result initialize();

  // Finalize the PI context
  pi_result finalize();

  // Return the Platform, which is the same for all devices in the context
  pi_platform getPlatform() const;

  // A L0 context handle is primarily used during creation and management of
  // resources that may be used by multiple devices.
  // This field is only set at _pi_context creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_context.
  const ze_context_handle_t ZeContext;

  // Indicates if we own the ZeContext or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeContext;

  // Keep the PI devices this PI context was created for.
  // This field is only set at _pi_context creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_context.
  const std::vector<pi_device> Devices;

  // Checks if Device is covered by this context.
  // For that the Device or its root devices need to be in the context.
  bool isValidDevice(pi_device Device) const;

  // If context contains one device or sub-devices of the same device, we want
  // to save this device.
  // This field is only set at _pi_context creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_context.
  const pi_device SingleRootDevice = nullptr;

  // Immediate Level Zero command list for the device in this context, to be
  // used for initializations. To be created as:
  // - Immediate command list: So any command appended to it is immediately
  //   offloaded to the device.
  // - Synchronous: So implicit synchronization is made inside the level-zero
  //   driver.
  // There will be a list of immediate command lists (for each device) when
  // support of the multiple devices per context will be added.
  ze_command_list_handle_t ZeCommandListInit;

  // Mutex for the immediate command list. Per the Level Zero spec memory copy
  // operations submitted to an immediate command list are not allowed to be
  // called from simultaneous threads.
  ur_mutex ImmediateCommandListMutex;

  // Mutex Lock for the Command List Cache. This lock is used to control both
  // compute and copy command list caches.
  ur_mutex ZeCommandListCacheMutex;
  // Cache of all currently available/completed command/copy lists.
  // Note that command-list can only be re-used on the same device.
  //
  // TODO: explore if we should use root-device for creating command-lists
  // as spec says that in that case any sub-device can re-use it: "The
  // application must only use the command list for the device, or its
  // sub-devices, which was provided during creation."
  //
  std::unordered_map<ze_device_handle_t, std::list<ze_command_list_handle_t>>
      ZeComputeCommandListCache;
  std::unordered_map<ze_device_handle_t, std::list<ze_command_list_handle_t>>
      ZeCopyCommandListCache;

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
  pi_result
  getAvailableCommandList(pi_queue Queue, pi_command_list_ptr_t &CommandList,
                          bool UseCopyEngine, bool AllowBatching = false,
                          ze_command_queue_handle_t *ForcedCmdQueue = nullptr);

  // Get index of the free slot in the available pool. If there is no available
  // pool then create new one. The HostVisible parameter tells if we need a
  // slot for a host-visible event. The ProfilingEnabled tells is we need a
  // slot for an event with profiling capabilities.
  pi_result getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &, size_t &,
                                           bool HostVisible,
                                           bool ProfilingEnabled);

  // Decrement number of events living in the pool upon event destroy
  // and return the pool to the cache if there are no unreleased events.
  pi_result decrementUnreleasedEventsInPool(pi_event Event);

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

  // Get pi_event from cache.
  pi_event getEventFromContextCache(bool HostVisible, bool WithProfiling);

  // Add pi_event to cache.
  void addEventToContextCache(pi_event);

private:
  // If context contains one device then return this device.
  // If context contains sub-devices of the same device, then return this parent
  // device. Return nullptr if context consists of several devices which are not
  // sub-devices of the same device. We call returned device the root device of
  // a context.
  // TODO: get rid of this when contexts with multiple devices are supported for
  // images.
  pi_device getRootDevice() const;

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
  auto getZeEventPoolCache(bool HostVisible, bool WithProfiling) {
    if (HostVisible)
      return WithProfiling ? &ZeEventPoolCache[0] : &ZeEventPoolCache[1];
    else
      return WithProfiling ? &ZeEventPoolCache[2] : &ZeEventPoolCache[3];
  }

  // This map will be used to determine if a pool is full or not
  // by storing number of empty slots available in the pool.
  std::unordered_map<ze_event_pool_handle_t, pi_uint32>
      NumEventsAvailableInEventPool;
  // This map will be used to determine number of unreleased events in the pool.
  // We use separate maps for number of event slots available in the pool from
  // the number of events unreleased in the pool.
  // This will help when we try to make the code thread-safe.
  std::unordered_map<ze_event_pool_handle_t, pi_uint32>
      NumEventsUnreleasedInEventPool;

  // Mutex to control operations on event pool caches and the helper maps
  // holding the current pool usage counts.
  ur_mutex ZeEventPoolCacheMutex;

  // Mutex to control operations on event caches.
  ur_mutex EventCacheMutex;

  // Caches for events.
  std::vector<std::list<pi_event>> EventCaches{4};

  // Get the cache of events for a provided scope and profiling mode.
  auto getEventCache(bool HostVisible, bool WithProfiling) {
    if (HostVisible)
      return WithProfiling ? &EventCaches[0] : &EventCaches[1];
    else
      return WithProfiling ? &EventCaches[2] : &EventCaches[3];
  }
};

struct _pi_queue : _ur_object {
  // ForceComputeIndex, if non-negative, indicates that the queue must be fixed
  // to that particular compute CCS.
  _pi_queue(std::vector<ze_command_queue_handle_t> &ComputeQueues,
            std::vector<ze_command_queue_handle_t> &CopyQueues,
            pi_context Context, pi_device Device, bool OwnZeCommandQueue,
            pi_queue_properties Properties = 0, int ForceComputeIndex = -1);

  using queue_type = _pi_device::queue_group_info_t::type;

  // PI queue is in general a one to many mapping to L0 native queues.
  struct pi_queue_group_t {
    pi_queue Queue;
    pi_queue_group_t() = delete;

    // The Queue argument captures the enclosing PI queue.
    // The Type argument specifies the type of this queue group.
    // The actual ZeQueues are populated at PI queue construction.
    pi_queue_group_t(pi_queue Queue, queue_type Type)
        : Queue(Queue), Type(Type) {}

    // The type of the queue group.
    queue_type Type;
    bool isCopy() const { return Type != queue_type::Compute; }

    // Level Zero command queue handles.
    std::vector<ze_command_queue_handle_t> ZeQueues;

    // Immediate commandlist handles, one per Level Zero command queue handle.
    // These are created only once, along with the L0 queues (see above)
    // and reused thereafter.
    std::vector<pi_command_list_ptr_t> ImmCmdLists;

    // Return the index of the next queue to use based on a
    // round robin strategy and the queue group ordinal.
    // If QueryOnly is true then return index values but don't update internal
    // index data members of the queue.
    uint32_t getQueueIndex(uint32_t *QueueGroupOrdinal, uint32_t *QueueIndex,
                           bool QueryOnly = false);

    // Get the ordinal for a command queue handle.
    int32_t getCmdQueueOrdinal(ze_command_queue_handle_t CmdQueue);

    // This function will return one of possibly multiple available native
    // queues and the value of the queue group ordinal.
    ze_command_queue_handle_t &getZeQueue(uint32_t *QueueGroupOrdinal);

    // This function sets an immediate commandlist from the interop interface.
    void setImmCmdList(ze_command_list_handle_t);

    // This function returns the next immediate commandlist to use.
    pi_command_list_ptr_t &getImmCmdList();

    // These indices are to filter specific range of the queues to use,
    // and to organize round-robin across them.
    uint32_t UpperIndex{0};
    uint32_t LowerIndex{0};
    uint32_t NextIndex{0};
  };

  // Helper class to facilitate per-thread queue groups
  // We maintain a hashtable of queue groups if requested to do them per-thread.
  // Otherwise it is just single entry used for all threads.
  struct pi_queue_group_by_tid_t
      : public std::unordered_map<std::thread::id, pi_queue_group_t> {
    bool PerThread = false;

    // Returns thread id if doing per-thread, or a generic id that represents
    // all the threads.
    std::thread::id tid() const {
      return PerThread ? std::this_thread::get_id() : std::thread::id();
    }

    // Make the specified queue group be the master
    void set(const pi_queue_group_t &QueueGroup) {
      const auto &Device = QueueGroup.Queue->Device;
      PerThread = Device->ImmCommandListUsed == _pi_device::PerThreadPerQueue;
      assert(empty());
      insert({tid(), QueueGroup});
    }

    // Get a queue group to use for this thread
    pi_queue_group_t &get() {
      assert(!empty());
      auto It = find(tid());
      if (It != end()) {
        return It->second;
      }
      // Add new queue group for this thread initialized from a master entry.
      auto QueueGroup = begin()->second;
      // Create space for queues and immediate commandlists, which are created
      // on demand.
      QueueGroup.ZeQueues = std::vector<ze_command_queue_handle_t>(
          QueueGroup.ZeQueues.size(), nullptr);
      QueueGroup.ImmCmdLists = std::vector<pi_command_list_ptr_t>(
          QueueGroup.ZeQueues.size(), QueueGroup.Queue->CommandListMap.end());

      std::tie(It, std::ignore) = insert({tid(), QueueGroup});
      return It->second;
    }
  };

  // A map of compute groups containing compute queue handles, one per thread.
  // When a queue is accessed from multiple host threads, a separate queue group
  // is created for each thread. The key used for mapping is the thread ID.
  pi_queue_group_by_tid_t ComputeQueueGroupsByTID;

  // A group containing copy queue handles. The main copy engine, if available,
  // comes first followed by link copy engines, if available.
  // When a queue is accessed from multiple host threads, a separate queue group
  // is created for each thread. The key used for mapping is the thread ID.
  pi_queue_group_by_tid_t CopyQueueGroupsByTID;

  // Wait for all commandlists associated with this Queue to finish operations.
  pi_result synchronize();

  // Return the queue group to use based on standard/immediate commandlist mode,
  // and if immediate mode, the thread-specific group.
  pi_queue_group_t &getQueueGroup(bool UseCopyEngine);

  // This function considers multiple factors including copy engine
  // availability and user preference and returns a boolean that is used to
  // specify if copy engine will eventually be used for a particular command.
  bool useCopyEngine(bool PreferCopyEngine = true) const;

  // Keeps the PI context to which this queue belongs.
  // This field is only set at _pi_queue creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_queue.
  const pi_context Context;

  // Keeps the PI device to which this queue belongs.
  // This field is only set at _pi_queue creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_queue.
  const pi_device Device;

  // A queue may use either standard or immediate commandlists. At queue
  // construction time this is set based on the device and any env var settings
  // that change the default for the device type. When an interop queue is
  // constructed, the caller chooses the type of commandlists to use.
  bool UsingImmCmdLists;

  // Keeps track of the event associated with the last enqueued command into
  // this queue. this is used to add dependency with the last command to add
  // in-order semantics and updated with the latest event each time a new
  // command is enqueued.
  pi_event LastCommandEvent = nullptr;

  // Kernel is not necessarily submitted for execution during
  // piEnqueueKernelLaunch, it may be batched. That's why we need to save the
  // list of kernels which is going to be submitted but have not been submitted
  // yet. This is needed to capture memory allocations for each kernel with
  // indirect access in the list at the moment when kernel is really submitted
  // for execution.
  std::vector<pi_kernel> KernelsToBeSubmitted;

  // Update map of memory references made by the kernels about to be submitted
  void CaptureIndirectAccesses();

  // Indicates if we own the ZeCommandQueue or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeCommandQueue;

  // Map of all command lists used in this queue.
  pi_command_list_map_t CommandListMap;

  // Helper data structure to hold all variables related to batching
  struct command_batch {
    // These two members are used to keep track of how often the
    // batching closes and executes a command list before reaching the
    // QueueComputeBatchSize limit, versus how often we reach the limit.
    // This info might be used to vary the QueueComputeBatchSize value.
    pi_uint32 NumTimesClosedEarly = {0};
    pi_uint32 NumTimesClosedFull = {0};

    // Open command list fields for batching commands into this queue.
    pi_command_list_ptr_t OpenCommandList{};

    // Approximate number of commands that are allowed to be batched for
    // this queue.
    // Added this member to the queue rather than using a global variable
    // so that future implementation could use heuristics to change this on
    // a queue specific basis. And by putting it in the queue itself, this
    // is thread safe because of the locking of the queue that occurs.
    pi_uint32 QueueBatchSize = {0};
  };

  // ComputeCommandBatch holds data related to batching of non-copy commands.
  // CopyCommandBatch holds data related to batching of copy commands.
  command_batch ComputeCommandBatch, CopyCommandBatch;

  // Returns true if any commands for this queue are allowed to
  // be batched together.
  // For copy commands, IsCopy is set to 'true'.
  // For non-copy commands, IsCopy is set to 'false'.
  bool isBatchingAllowed(bool IsCopy) const;

  // Keeps the properties of this queue.
  pi_queue_properties Properties;

  // Returns true if the queue is a in-order queue.
  bool isInOrderQueue() const;

  // Returns true if the queue has discard events property.
  bool isDiscardEvents() const;

  // Returns true if the queue has explicit priority set by user.
  bool isPriorityLow() const;
  bool isPriorityHigh() const;

  // adjust the queue's batch size, knowing that the current command list
  // is being closed with a full batch.
  // For copy commands, IsCopy is set to 'true'.
  // For non-copy commands, IsCopy is set to 'false'.
  void adjustBatchSizeForFullBatch(bool IsCopy);

  // adjust the queue's batch size, knowing that the current command list
  // is being closed with only a partial batch of commands.
  // For copy commands, IsCopy is set to 'true'.
  // For non-copy commands, IsCopy is set to 'false'.
  void adjustBatchSizeForPartialBatch(bool IsCopy);

  // Helper function to create a new command-list to this queue and associated
  // fence tracking its completion. This command list & fence are added to the
  // map of command lists in this queue with ZeFenceInUse = false.
  // The caller must hold a lock of the queue already.
  pi_result
  createCommandList(bool UseCopyEngine, pi_command_list_ptr_t &CommandList,
                    ze_command_queue_handle_t *ForcedCmdQueue = nullptr);

  /// @brief Resets the command list and associated fence in the map and removes
  /// events from the command list.
  /// @param CommandList The caller must verify that this command list and fence
  /// have been signalled.
  /// @param MakeAvailable If the reset command list should be made available,
  /// then MakeAvailable needs to be set to true.
  /// @param EventListToCleanup  The EventListToCleanup contains a list of
  /// events from the command list which need to be cleaned up.
  /// @param CheckStatus Hint informing whether we need to check status of the
  /// events before removing them from the immediate command list. This is
  /// needed because immediate command lists are not associated with fences and
  /// in general status of the event needs to be checked.
  /// @return PI_SUCCESS if successful, PI error code otherwise.
  pi_result resetCommandList(pi_command_list_ptr_t CommandList,
                             bool MakeAvailable,
                             std::vector<pi_event> &EventListToCleanup,
                             bool CheckStatus = true);

  // Returns true if an OpenCommandList has commands that need to be submitted.
  // If IsCopy is 'true', then the OpenCommandList containing copy commands is
  // checked. Otherwise, the OpenCommandList containing compute commands is
  // checked.
  bool hasOpenCommandList(bool IsCopy) const {
    auto CommandBatch = (IsCopy) ? CopyCommandBatch : ComputeCommandBatch;
    return CommandBatch.OpenCommandList != CommandListMap.end();
  }
  // Attach a command list to this queue.
  // For non-immediate commandlist also close and execute it.
  // Note that this command list cannot be appended to after this.
  // The "IsBlocking" tells if the wait for completion is required.
  // If OKToBatchCommand is true, then this command list may be executed
  // immediately, or it may be left open for other future command to be
  // batched into.
  // If IsBlocking is true, then batching will not be allowed regardless
  // of the value of OKToBatchCommand
  //
  // For immediate commandlists, no close and execute is necessary.
  pi_result executeCommandList(pi_command_list_ptr_t CommandList,
                               bool IsBlocking = false,
                               bool OKToBatchCommand = false);

  // If there is an open command list associated with this queue,
  // close it, execute it, and reset the corresponding OpenCommandList.
  // If IsCopy is 'true', then the OpenCommandList containing copy commands is
  // executed. Otherwise OpenCommandList containing compute commands is
  // executed.
  pi_result executeOpenCommandList(bool IsCopy);

  // Gets the open command containing the event, or CommandListMap.end()
  pi_command_list_ptr_t eventOpenCommandList(pi_event Event);

  // Wrapper function to execute both OpenCommandLists (Copy and Compute).
  // This wrapper is helpful when all 'open' commands need to be executed.
  // Call-sites instances: piQuueueFinish, piQueueRelease, etc.
  pi_result executeAllOpenCommandLists() {
    using IsCopy = bool;
    if (auto Res = executeOpenCommandList(IsCopy{false}))
      return Res;
    if (auto Res = executeOpenCommandList(IsCopy{true}))
      return Res;
    return PI_SUCCESS;
  }

  // Inserts a barrier waiting for all unfinished events in ActiveBarriers into
  // CmdList. Any finished events will be removed from ActiveBarriers.
  pi_result insertActiveBarriers(pi_command_list_ptr_t &CmdList,
                                 bool UseCopyEngine);

  // A helper structure to keep active barriers of the queue.
  // It additionally manages ref-count of events in this list.
  struct active_barriers {
    std::vector<pi_event> Events;
    void add(pi_event &Event);
    pi_result clear();
    bool empty() { return Events.empty(); }
    std::vector<pi_event> &vector() { return Events; }
  };
  // A collection of currently active barriers.
  // These should be inserted into a command list whenever an available command
  // list is needed for a command.
  active_barriers ActiveBarriers;

  // Besides each PI object keeping a total reference count in
  // _ur_object::RefCount we keep special track of the queue *external*
  // references. This way we are able to tell when the queue is being finished
  // externally, and can wait for internal references to complete, and do proper
  // cleanup of the queue.
  // This counter doesn't track the lifetime of a queue object, it only tracks
  // the number of external references. I.e. even if it reaches zero a queue
  // object may not be destroyed and can be used internally in the plugin.
  // That's why we intentionally don't use atomic type for this counter to
  // enforce guarding with a mutex all the work involving this counter.
  pi_uint32 RefCountExternal{1};

  // Indicates that the queue is healthy and all operations on it are OK.
  bool Healthy{true};

  // The following data structures and methods are used only for handling
  // in-order queue with discard_events property. Some commands in such queue
  // may have discarded event. Which means that event is not visible outside of
  // the plugin. It is possible to reset and reuse discarded events in the same
  // in-order queue because of the dependency between commands. We don't have to
  // wait event completion to do this. We use the following 2-event model to
  // reuse events inside each command list:
  //
  // Operation1 = zeCommantListAppendMemoryCopy (signal ze_event1)
  // zeCommandListAppendBarrier(wait for ze_event1)
  // zeCommandListAppendEventReset(ze_event1)
  // # Create new pi_event using ze_event1 and append to the cache.
  //
  // Operation2 = zeCommandListAppendMemoryCopy (signal ze_event2)
  // zeCommandListAppendBarrier(wait for ze_event2)
  // zeCommandListAppendEventReset(ze_event2)
  // # Create new pi_event using ze_event2 and append to the cache.
  //
  // # Get pi_event from the beginning of the cache because there are two events
  // # there. So it is guaranteed that we do round-robin between two events -
  // # event from the last command is appended to the cache.
  // Operation3 = zeCommandListAppendMemoryCopy (signal ze_event1)
  // # The same ze_event1 is used for Operation1 and Operation3.
  //
  // When we switch to a different command list we need to signal new event and
  // wait for it in the new command list using barrier.
  // [CmdList1]
  // Operation1 = zeCommantListAppendMemoryCopy (signal event1)
  // zeCommandListAppendBarrier(wait for event1)
  // zeCommandListAppendEventReset(event1)
  // zeCommandListAppendSignalEvent(NewEvent)
  //
  // [CmdList2]
  // zeCommandListAppendBarrier(wait for NewEvent)
  //
  // This barrier guarantees that command list execution starts only after
  // completion of previous command list which signals aforementioned event. It
  // allows to reset and reuse same event handles inside all command lists in
  // scope of the queue. It means that we need 2 reusable events of each type
  // (host-visible and device-scope) per queue at maximum.

  // This data member keeps track of the last used command list and allows to
  // handle switch of immediate command lists because immediate command lists
  // are never closed unlike regular command lists.
  pi_command_list_ptr_t LastUsedCommandList = CommandListMap.end();

  // Vector of 2 lists of reusable events: host-visible and device-scope.
  // They are separated to allow faster access to stored events depending on
  // requested type of event. Each list contains events which can be reused
  // inside all command lists in the queue as described in the 2-event model.
  // Leftover events in the cache are relased at the queue destruction.
  std::vector<std::list<pi_event>> EventCaches{2};

  // Get event from the queue's cache.
  // Returns nullptr if the cache doesn't contain any reusable events or if the
  // cache contains only one event which corresponds to the previous command and
  // can't be used for the current command because we can't use the same event
  // two times in a row and have to do round-robin between two events. Otherwise
  // it picks an event from the beginning of the cache and returns it. Event
  // from the last command is always appended to the end of the list.
  pi_event getEventFromQueueCache(bool HostVisible);

  // Put pi_event to the cache. Provided pi_event object is not used by
  // any command but its ZeEvent is used by many pi_event objects.
  // Commands to wait and reset ZeEvent must be submitted to the queue before
  // calling this method.
  pi_result addEventToQueueCache(pi_event Event);

  // Append command to provided command list to wait and reset the last event if
  // it is discarded and create new pi_event wrapper using the same native event
  // and put it to the cache. We call this method after each command submission
  // to make native event available to use by next commands.
  pi_result resetDiscardedEvent(pi_command_list_ptr_t);

  // Append command to the command list to signal new event if the last event in
  // the command list is discarded. While we submit commands in scope of the
  // same command list we can reset and reuse events but when we switch to a
  // different command list we currently need to signal new event and wait for
  // it in the new command list using barrier.
  pi_result signalEventFromCmdListIfLastEventDiscarded(pi_command_list_ptr_t);

  // Insert a barrier waiting for the last command event into the beginning of
  // command list. This barrier guarantees that command list execution starts
  // only after completion of previous command list which signals aforementioned
  // event. It allows to reset and reuse same event handles inside all command
  // lists in the queue.
  pi_result
  insertStartBarrierIfDiscardEventsMode(pi_command_list_ptr_t &CmdList);

  // Helper method telling whether we need to reuse discarded event in this
  // queue.
  bool doReuseDiscardedEvents();
};

struct _pi_mem : _ur_object {
  // Keeps the PI context of this memory handle.
  pi_context Context;

  // Enumerates all possible types of accesses.
  enum access_mode_t { unknown, read_write, read_only, write_only };

  // Interface of the _pi_mem object

  // Get the Level Zero handle of the current memory object
  virtual pi_result getZeHandle(char *&ZeHandle, access_mode_t,
                                pi_device Device = nullptr) = 0;

  // Get a pointer to the Level Zero handle of the current memory object
  virtual pi_result getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                   pi_device Device = nullptr) = 0;

  // Method to get type of the derived object (image or buffer)
  virtual bool isImage() const = 0;

  virtual ~_pi_mem() = default;

protected:
  _pi_mem(pi_context Ctx) : Context{Ctx} {}
};

struct _pi_buffer;
using pi_buffer = _pi_buffer *;

struct _pi_buffer final : _pi_mem {
  // Buffer constructor
  _pi_buffer(pi_context Context, size_t Size, char *HostPtr,
             bool ImportedHostPtr = false)
      : _pi_mem(Context), Size(Size), SubBuffer{nullptr, 0} {

    // We treat integrated devices (physical memory shared with the CPU)
    // differently from discrete devices (those with distinct memories).
    // For integrated devices, allocating the buffer in the host memory
    // enables automatic access from the device, and makes copying
    // unnecessary in the map/unmap operations. This improves performance.
    OnHost = Context->Devices.size() == 1 &&
             Context->Devices[0]->ZeDeviceProperties->flags &
                 ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;

    // Fill the host allocation data.
    if (HostPtr) {
      MapHostPtr = HostPtr;
      // If this host ptr is imported to USM then use this as a host
      // allocation for this buffer.
      if (ImportedHostPtr) {
        Allocations[nullptr].ZeHandle = HostPtr;
        Allocations[nullptr].Valid = true;
        Allocations[nullptr].ReleaseAction = _pi_buffer::allocation_t::unimport;
      }
    }

    // This initialization does not end up with any valid allocation yet.
    LastDeviceWithValidAllocation = nullptr;
  }

  // Sub-buffer constructor
  _pi_buffer(pi_buffer Parent, size_t Origin, size_t Size)
      : _pi_mem(Parent->Context), Size(Size), SubBuffer{Parent, Origin} {}

  // Interop-buffer constructor
  _pi_buffer(pi_context Context, size_t Size, pi_device Device,
             char *ZeMemHandle, bool OwnZeMemHandle)
      : _pi_mem(Context), Size(Size), SubBuffer{nullptr, 0} {

    // Device == nullptr means host allocation
    Allocations[Device].ZeHandle = ZeMemHandle;
    Allocations[Device].Valid = true;
    Allocations[Device].ReleaseAction =
        OwnZeMemHandle ? allocation_t::free_native : allocation_t::keep;

    // Check if this buffer can always stay on host
    OnHost = false;
    if (!Device) { // Host allocation
      if (Context->Devices.size() == 1 &&
          Context->Devices[0]->ZeDeviceProperties->flags &
              ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) {
        OnHost = true;
        MapHostPtr = ZeMemHandle; // map to this allocation
      }
    }
    LastDeviceWithValidAllocation = Device;
  }

  // Returns a pointer to the USM allocation representing this PI buffer
  // on the specified Device. If Device is nullptr then the returned
  // USM allocation is on the device where this buffer was used the latest.
  // The returned allocation is always valid, i.e. its contents is
  // up-to-date and any data copies needed for that are performed under
  // the hood.
  //
  virtual pi_result getZeHandle(char *&ZeHandle, access_mode_t,
                                pi_device Device = nullptr) override;
  virtual pi_result getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                   pi_device Device = nullptr) override;

  bool isImage() const override { return false; }

  bool isSubBuffer() const { return SubBuffer.Parent != nullptr; }

  // Frees all allocations made for the buffer.
  pi_result free();

  // Information about a single allocation representing this buffer.
  struct allocation_t {
    // Level Zero memory handle is really just a naked pointer.
    // It is just convenient to have it char * to simplify offset arithmetics.
    char *ZeHandle{nullptr};
    // Indicates if this allocation's data is valid.
    bool Valid{false};
    // Specifies the action that needs to be taken for this
    // allocation at buffer destruction.
    enum {
      keep,       // do nothing, the allocation is not owned by us
      unimport,   // release of the imported allocation
      free,       // free from the pooling context (default)
      free_native // free with a native call
    } ReleaseAction{free};
  };

  // We maintain multiple allocations on possibly all devices in the context.
  // The "nullptr" device identifies a host allocation representing buffer.
  // Sub-buffers don't maintain own allocations but rely on parent buffer.
  std::unordered_map<pi_device, allocation_t> Allocations;
  pi_device LastDeviceWithValidAllocation{nullptr};

  // Flag to indicate that this memory is allocated in host memory.
  // Integrated device accesses this memory.
  bool OnHost{false};

  // Tells the host allocation to use for buffer map operations.
  char *MapHostPtr{nullptr};

  // Supplementary data to keep track of the mappings of this buffer
  // created with piEnqueueMemBufferMap.
  struct Mapping {
    // The offset in the buffer giving the start of the mapped region.
    size_t Offset;
    // The size of the mapped region.
    size_t Size;
  };

  // The key is the host pointer representing an active mapping.
  // The value is the information needed to maintain/undo the mapping.
  std::unordered_map<void *, Mapping> Mappings;

  // The size and alignment of the buffer
  size_t Size;
  size_t getAlignment() const;

  struct {
    _pi_mem *Parent;
    size_t Origin; // only valid if Parent != nullptr
  } SubBuffer;
};

struct _pi_image;
using pi_image = _pi_image *;

// TODO: add proper support for images on context with multiple devices.
struct _pi_image final : _pi_mem {
  // Image constructor
  _pi_image(pi_context Ctx, ze_image_handle_t Image, bool OwnNativeHandle)
      : _pi_mem(Ctx), ZeImage{Image}, OwnZeMemHandle{OwnNativeHandle} {}

  virtual pi_result getZeHandle(char *&ZeHandle, access_mode_t,
                                pi_device = nullptr) override {
    ZeHandle = ur_cast<char *>(ZeImage);
    return PI_SUCCESS;
  }
  virtual pi_result getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                   pi_device = nullptr) override {
    ZeHandlePtr = ur_cast<char **>(&ZeImage);
    return PI_SUCCESS;
  }

  bool isImage() const override { return true; }

#ifndef NDEBUG
  // Keep the descriptor of the image (for debugging purposes)
  ZeStruct<ze_image_desc_t> ZeImageDesc;
#endif // !NDEBUG

  // Level Zero image handle.
  ze_image_handle_t ZeImage;

  bool OwnZeMemHandle;
};

struct _pi_ze_event_list_t {
  // List of level zero events for this event list.
  ze_event_handle_t *ZeEventList = {nullptr};

  // List of pi_events for this event list.
  pi_event *PiEventList = {nullptr};

  // length of both the lists.  The actual allocation of these lists
  // may be longer than this length.  This length is the actual number
  // of elements in the above arrays that are valid.
  pi_uint32 Length = {0};

  // A mutex is needed for destroying the event list.
  // Creation is already thread-safe because we only create the list
  // when an event is initially created.  However, it might be
  // possible to have multiple threads racing to destroy the list,
  // so this will be used to make list destruction thread-safe.
  ur_mutex PiZeEventListMutex;

  // Initialize this using the array of events in EventList, and retain
  // all the pi_events in the created data structure.
  // CurQueue is the pi_queue that the command with this event wait
  // list is going to be added to.  That is needed to flush command
  // batches for wait events that are in other queues.
  // UseCopyEngine indicates if the next command (the one that this
  // event wait-list is for) is going to go to copy or compute
  // queue. This is used to properly submit the dependent open
  // command-lists.
  pi_result createAndRetainPiZeEventList(pi_uint32 EventListLength,
                                         const pi_event *EventList,
                                         pi_queue CurQueue, bool UseCopyEngine);

  // Add all the events in this object's PiEventList to the end
  // of the list EventsToBeReleased. Destroy pi_ze_event_list_t data
  // structure fields making it look empty.
  pi_result collectEventsForReleaseAndDestroyPiZeEventList(
      std::list<pi_event> &EventsToBeReleased);

  // Had to create custom assignment operator because the mutex is
  // not assignment copyable. Just field by field copy of the other
  // fields.
  _pi_ze_event_list_t &operator=(const _pi_ze_event_list_t &other) {
    if (this != &other) {
      this->ZeEventList = other.ZeEventList;
      this->PiEventList = other.PiEventList;
      this->Length = other.Length;
    }
    return *this;
  }
};

struct _pi_event : _ur_object {
  _pi_event(ze_event_handle_t ZeEvent, ze_event_pool_handle_t ZeEventPool,
            pi_context Context, pi_command_type CommandType, bool OwnZeEvent)
      : ZeEvent{ZeEvent}, OwnZeEvent{OwnZeEvent}, ZeEventPool{ZeEventPool},
        CommandType{CommandType}, Context{Context}, CommandData{nullptr} {}

  // Level Zero event handle.
  ze_event_handle_t ZeEvent;

  // Indicates if we own the ZeEvent or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeEvent;

  // Level Zero event pool handle.
  ze_event_pool_handle_t ZeEventPool;

  // In case we use device-only events this holds their host-visible
  // counterpart. If this event is itself host-visble then HostVisibleEvent
  // points to this event. If this event is not host-visible then this field can
  // be: 1) null, meaning that a host-visible event wasn't yet created 2) a PI
  // event created internally that host will actually be redirected
  //    to wait/query instead of this PI event.
  //
  // The HostVisibleEvent is a reference counted PI event and can be used more
  // than by just this one event, depending on the mode (see EventsScope).
  //
  pi_event HostVisibleEvent = {nullptr};
  bool isHostVisible() const { return this == HostVisibleEvent; }

  // Get the host-visible event or create one and enqueue its signal.
  pi_result getOrCreateHostVisibleEvent(ze_event_handle_t &HostVisibleEvent);

  // Tells if this event is with profiling capabilities.
  bool isProfilingEnabled() const {
    return !Queue || // tentatively assume user events are profiling enabled
           (Queue->Properties & PI_QUEUE_FLAG_PROFILING_ENABLE) != 0;
  }

  // Keeps the command-queue and command associated with the event.
  // These are NULL for the user events.
  pi_queue Queue = {nullptr};
  pi_command_type CommandType;
  // Provide direct access to Context, instead of going via queue.
  // Not every PI event has a queue, and we need a handle to Context
  // to get to event pool related information.
  pi_context Context;

  // Opaque data to hold any data needed for CommandType.
  void *CommandData;

  // List of events that were in the wait list of the command that will
  // signal this event.  These events must be retained when the command is
  // enqueued, and must then be released when this event has signalled.
  // This list must be destroyed once the event has signalled.
  _pi_ze_event_list_t WaitList;

  // Command list associated with the pi_event.
  std::optional<pi_command_list_ptr_t> CommandList;

  // Tracks if the needed cleanup was already performed for
  // a completed event. This allows to control that some cleanup
  // actions are performed only once.
  //
  bool CleanedUp = {false};

  // Indicates that this PI event had already completed in the sense
  // that no other synchromization is needed. Note that the underlying
  // L0 event (if any) is not guranteed to have been signalled, or
  // being visible to the host at all.
  bool Completed = {false};

  // Indicates that this event is discarded, i.e. it is not visible outside of
  // plugin.
  bool IsDiscarded = {false};

  // Besides each PI object keeping a total reference count in
  // _ur_object::RefCount we keep special track of the event *external*
  // references. This way we are able to tell when the event is not referenced
  // externally anymore, i.e. it can't be passed as a dependency event to
  // piEnqueue* functions and explicitly waited meaning that we can do some
  // optimizations:
  // 1. For in-order queues we can reset and reuse event even if it was not yet
  // completed by submitting a reset command to the queue (since there are no
  // external references, we know that nobody can wait this event somewhere in
  // parallel thread or pass it as a dependency which may lead to hang)
  // 2. We can avoid creating host proxy event.
  // This counter doesn't track the lifetime of an event object. Even if it
  // reaches zero an event object may not be destroyed and can be used
  // internally in the plugin.
  std::atomic<pi_uint32> RefCountExternal{0};

  bool hasExternalRefs() { return RefCountExternal != 0; }

  // Reset _pi_event object.
  pi_result reset();
};

struct _pi_program : _ur_object {
  // Possible states of a program.
  typedef enum {
    // The program has been created from intermediate language (SPIR-V), but it
    // is not yet compiled.
    IL,

    // The program has been created by loading native code, but it has not yet
    // been built.  This is equivalent to an OpenCL "program executable" that
    // is loaded via clCreateProgramWithBinary().
    Native,

    // The program was notionally compiled from SPIR-V form.  However, since we
    // postpone compilation until the module is linked, the internal state
    // still represents the module as SPIR-V.
    Object,

    // The program has been built or linked, and it is represented as a Level
    // Zero module.
    Exe,

    // An error occurred during piProgramLink, but we created a _pi_program
    // object anyways in order to hold the ZeBuildLog.  Note that the ZeModule
    // may or may not be nullptr in this state, depending on the error.
    Invalid
  } state;

  // A utility class that converts specialization constants into the form
  // required by the Level Zero driver.
  class SpecConstantShim {
  public:
    SpecConstantShim(pi_program Program) {
      ZeSpecConstants.numConstants = Program->SpecConstants.size();
      ZeSpecContantsIds.reserve(ZeSpecConstants.numConstants);
      ZeSpecContantsValues.reserve(ZeSpecConstants.numConstants);

      for (auto &SpecConstant : Program->SpecConstants) {
        ZeSpecContantsIds.push_back(SpecConstant.first);
        ZeSpecContantsValues.push_back(SpecConstant.second);
      }
      ZeSpecConstants.pConstantIds = ZeSpecContantsIds.data();
      ZeSpecConstants.pConstantValues = ZeSpecContantsValues.data();
    }

    const ze_module_constants_t *ze() { return &ZeSpecConstants; }

  private:
    std::vector<uint32_t> ZeSpecContantsIds;
    std::vector<const void *> ZeSpecContantsValues;
    ze_module_constants_t ZeSpecConstants;
  };

  // Construct a program in IL or Native state.
  _pi_program(state St, pi_context Context, const void *Input, size_t Length)
      : Context{Context}, OwnZeModule{true}, State{St},
        Code{new uint8_t[Length]}, CodeLength{Length}, ZeModule{nullptr},
        ZeBuildLog{nullptr} {
    std::memcpy(Code.get(), Input, Length);
  }

  // Construct a program in Exe or Invalid state.
  _pi_program(state St, pi_context Context, ze_module_handle_t ZeModule,
              ze_module_build_log_handle_t ZeBuildLog)
      : Context{Context}, OwnZeModule{true}, State{St}, ZeModule{ZeModule},
        ZeBuildLog{ZeBuildLog} {}

  // Construct a program in Exe state (interop).
  _pi_program(state St, pi_context Context, ze_module_handle_t ZeModule,
              bool OwnZeModule)
      : Context{Context}, OwnZeModule{OwnZeModule}, State{St},
        ZeModule{ZeModule}, ZeBuildLog{nullptr} {}

  // Construct a program in Invalid state with a custom error message.
  _pi_program(state St, pi_context Context, const std::string &ErrorMessage)
      : Context{Context}, OwnZeModule{true}, ErrorMessage{ErrorMessage},
        State{St}, ZeModule{nullptr}, ZeBuildLog{nullptr} {}

  ~_pi_program();

  const pi_context Context; // Context of the program.

  // Indicates if we own the ZeModule or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  const bool OwnZeModule;

  // This error message is used only in Invalid state to hold a custom error
  // message from a call to piProgramLink.
  const std::string ErrorMessage;

  state State;

  // In IL and Object states, this contains the SPIR-V representation of the
  // module.  In Native state, it contains the native code.
  std::unique_ptr<uint8_t[]> Code; // Array containing raw IL / native code.
  size_t CodeLength{0};            // Size (bytes) of the array.

  // Used only in IL and Object states.  Contains the SPIR-V specialization
  // constants as a map from the SPIR-V "SpecID" to a buffer that contains the
  // associated value.  The caller of the PI layer is responsible for
  // maintaining the storage of this buffer.
  std::unordered_map<uint32_t, const void *> SpecConstants;

  // Used only in Object state.  Contains the build flags from the last call to
  // piProgramCompile().
  std::string BuildFlags;

  // The Level Zero module handle.  Used primarily in Exe state.
  ze_module_handle_t ZeModule;

  // The Level Zero build log from the last call to zeModuleCreate().
  ze_module_build_log_handle_t ZeBuildLog;
};

struct _pi_kernel : _ur_object {
  _pi_kernel(ze_kernel_handle_t Kernel, bool OwnZeKernel, pi_program Program)
      : ZeKernel{Kernel}, OwnZeKernel{OwnZeKernel}, Program{Program},
        MemAllocs{}, SubmissionsCount{0} {}

  // Completed initialization of PI kernel. Must be called after construction.
  pi_result initialize();

  // Returns true if kernel has indirect access, false otherwise.
  bool hasIndirectAccess() {
    // Currently indirect access flag is set for all kernels and there is no API
    // to check if kernel actually indirectly access smth.
    return true;
  }

  // Level Zero function handle.
  ze_kernel_handle_t ZeKernel;

  // Indicates if we own the ZeKernel or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeKernel;

  // Keep the program of the kernel.
  pi_program Program;

  // Hash function object for the unordered_set below.
  struct Hash {
    size_t operator()(const std::pair<void *const, MemAllocRecord> *P) const {
      return std::hash<void *>()(P->first);
    }
  };

  // If kernel has indirect access we need to make a snapshot of all existing
  // memory allocations to defer deletion of these memory allocations to the
  // moment when kernel execution has finished.
  // We store pointers to the elements because pointers are not invalidated by
  // insert/delete for std::unordered_map (iterators are invalidated). We need
  // to take a snapshot instead of just reference-counting the allocations,
  // because picture of active allocations can change during kernel execution
  // (new allocations can be added) and we need to know which memory allocations
  // were retained by this kernel to release them (and don't touch new
  // allocations) at kernel completion. Same kernel may be submitted several
  // times and retained allocations may be different at each submission. That's
  // why we have a set of memory allocations here and increase ref count only
  // once even if kernel is submitted many times. We don't want to know how many
  // times and which allocations were retained by each submission. We release
  // all allocations in the set only when SubmissionsCount == 0.
  std::unordered_set<std::pair<void *const, MemAllocRecord> *, Hash> MemAllocs;

  // Counter to track the number of submissions of the kernel.
  // When this value is zero, it means that kernel is not submitted for an
  // execution - at this time we can release memory allocations referenced by
  // this kernel. We can do this when RefCount turns to 0 but it is too late
  // because kernels are cached in the context by SYCL RT and they are released
  // only during context object destruction. Regular RefCount is not usable to
  // track submissions because user/SYCL RT can retain kernel object any number
  // of times. And that's why there is no value of RefCount which can mean zero
  // submissions.
  std::atomic<pi_uint32> SubmissionsCount;

  // Keeps info about an argument to the kernel enough to set it with
  // zeKernelSetArgumentValue.
  struct ArgumentInfo {
    uint32_t Index;
    size_t Size;
    const pi_mem Value;
    _pi_mem::access_mode_t AccessMode{_pi_mem::unknown};
  };
  // Arguments that still need to be set (with zeKernelSetArgumentValue)
  // before kernel is enqueued.
  std::vector<ArgumentInfo> PendingArguments;

  // Cache of the kernel properties.
  ZeCache<ZeStruct<ze_kernel_properties_t>> ZeKernelProperties;
  ZeCache<std::string> ZeKernelName;
};

struct _pi_sampler : _ur_object {
  _pi_sampler(ze_sampler_handle_t Sampler) : ZeSampler{Sampler} {}

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;
};

#endif // PI_LEVEL_ZERO_HPP
