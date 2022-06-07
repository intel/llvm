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

#include <CL/sycl/detail/pi.h>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#include "usm_allocator.hpp"

template <class To, class From> To pi_cast(From Value) {
  // TODO: see if more sanity checks are possible.
  assert(sizeof(From) == sizeof(To));
  return (To)(Value);
}

template <> uint32_t pi_cast(uint64_t Value) {
  // Cast value and check that we don't lose any information.
  uint32_t CastedValue = (uint32_t)(Value);
  assert((uint64_t)CastedValue == Value);
  return CastedValue;
}

// TODO: Currently die is defined in each plugin. Probably some
// common header file with utilities should be created.
[[noreturn]] void die(const char *Message) {
  std::cerr << "die: " << Message << std::endl;
  std::terminate();
}

// Returns the ze_structure_type_t to use in .stype of a structured descriptor.
// Intentionally not defined; will give an error if no proper specialization
template <class T> ze_structure_type_t getZeStructureType();
template <class T> zes_structure_type_t getZesStructureType();

template <> ze_structure_type_t getZeStructureType<ze_event_pool_desc_t>() {
  return ZE_STRUCTURE_TYPE_EVENT_POOL_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_fence_desc_t>() {
  return ZE_STRUCTURE_TYPE_FENCE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_command_list_desc_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_context_desc_t>() {
  return ZE_STRUCTURE_TYPE_CONTEXT_DESC;
}
template <>
ze_structure_type_t
getZeStructureType<ze_relaxed_allocation_limits_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_host_mem_alloc_desc_t>() {
  return ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_mem_alloc_desc_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_command_queue_desc_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_image_desc_t>() {
  return ZE_STRUCTURE_TYPE_IMAGE_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_module_desc_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_DESC;
}
template <>
ze_structure_type_t getZeStructureType<ze_module_program_exp_desc_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_PROGRAM_EXP_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_kernel_desc_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_event_desc_t>() {
  return ZE_STRUCTURE_TYPE_EVENT_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_sampler_desc_t>() {
  return ZE_STRUCTURE_TYPE_SAMPLER_DESC;
}
template <> ze_structure_type_t getZeStructureType<ze_driver_properties_t>() {
  return ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_device_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_compute_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_command_queue_group_properties_t>() {
  return ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_image_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_module_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_cache_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_device_memory_access_properties_t>() {
  return ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_module_properties_t>() {
  return ZE_STRUCTURE_TYPE_MODULE_PROPERTIES;
}
template <> ze_structure_type_t getZeStructureType<ze_kernel_properties_t>() {
  return ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
}
template <>
ze_structure_type_t getZeStructureType<ze_memory_allocation_properties_t>() {
  return ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
}

template <> zes_structure_type_t getZesStructureType<zes_pci_properties_t>() {
  return ZES_STRUCTURE_TYPE_PCI_PROPERTIES;
}

// The helpers to properly default initialize Level-Zero descriptor and
// properties structures.
template <class T> struct ZeStruct : public T {
  ZeStruct() : T{} { // zero initializes base struct
    this->stype = getZeStructureType<T>();
    this->pNext = nullptr;
  }
};
template <class T> struct ZesStruct : public T {
  ZesStruct() : T{} { // zero initializes base struct
    this->stype = getZesStructureType<T>();
    this->pNext = nullptr;
  }
};

// The wrapper for immutable Level-Zero data.
// The data is initialized only once at first access (via ->) with the
// initialization function provided in Init. All subsequent access to
// the data just returns the already stored data.
//
template <class T> struct ZeCache : private T {
  // The initialization function takes a reference to the data
  // it is going to initialize, since it is private here in
  // order to disallow access other than through "->".
  //
  typedef std::function<void(T &)> InitFunctionType;
  InitFunctionType Compute{nullptr};
  bool Computed{false};

  ZeCache() : T{} {}

  // Access to the fields of the original T data structure.
  T *operator->() {
    if (!Computed) {
      Compute(*this);
      Computed = true;
    }
    return this;
  }
};

// A single-threaded app has an opportunity to enable this mode to avoid
// overhead from mutex locking. Default value is 0 which means that single
// thread mode is disabled.
static const bool SingleThreadMode = [] {
  const char *Ret = std::getenv("SYCL_PI_LEVEL_ZERO_SINGLE_THREAD_MODE");
  const bool RetVal = Ret ? std::stoi(Ret) : 0;
  return RetVal;
}();

// Class which acts like shared_mutex if SingleThreadMode variable is not set.
// If SingleThreadMode variable is set then mutex operations are turned into
// nop.
class pi_shared_mutex : public std::shared_mutex {
public:
  void lock() {
    if (!SingleThreadMode)
      std::shared_mutex::lock();
  }
  bool try_lock() {
    return SingleThreadMode ? true : std::shared_mutex::try_lock();
  }
  void unlock() {
    if (!SingleThreadMode)
      std::shared_mutex::unlock();
  }

  void lock_shared() {
    if (!SingleThreadMode)
      std::shared_mutex::lock_shared();
  }
  bool try_lock_shared() {
    return SingleThreadMode ? true : std::shared_mutex::try_lock_shared();
  }
  void unlock_shared() {
    if (!SingleThreadMode)
      std::shared_mutex::unlock_shared();
  }
};

// This wrapper around std::atomic is created to limit operations with reference
// counter and to make allowed operations more transparent in terms of
// thread-safety in the plugin. increment() and load() operations do not need a
// mutex guard around them since the underlying data is already atomic.
// decrementAndTest() method is used to guard a code which needs to be
// executed when object's ref count becomes zero after release. This method also
// doesn't need a mutex guard because decrement operation is atomic and only one
// thread can reach ref count equal to zero, i.e. only a single thread can pass
// through this check.
struct ReferenceCounter {
  ReferenceCounter(pi_uint32 InitVal) : RefCount{InitVal} {}

  // Used when retaining an object.
  void increment() { RefCount++; }

  // Supposed to be used in pi*GetInfo* methods where ref count value is
  // requested.
  pi_uint32 load() { return RefCount.load(); }

  // This method allows to guard a code which needs to be executed when object's
  // ref count becomes zero after release. It is important to notice that only a
  // single thread can pass through this check. This is true because of several
  // reasons:
  //   1. Decrement operation is executed atomically.
  //   2. It is not allowed to retain an object after its refcount reaches zero.
  //   3. It is not allowed to release an object more times than the value of
  //   the ref count.
  // 2. and 3. basically means that we can't use an object at all as soon as its
  // refcount reaches zero. Using this check guarantees that code for deleting
  // an object and releasing its resources is executed once by a single thread
  // and we don't need to use any mutexes to guard access to this object in the
  // scope after this check. Of course if we access another objects in this code
  // (not the one which is being deleted) then access to these objects must be
  // guarded, for example with a mutex.
  bool decrementAndTest() { return --RefCount == 0; }

private:
  std::atomic<pi_uint32> RefCount;
};

// Base class to store common data
struct _pi_object {
  _pi_object() : RefCount{1} {}

  // Level Zero doesn't do the reference counting, so we have to do.
  // Must be atomic to prevent data race when incrementing/decrementing.
  ReferenceCounter RefCount;

  // This mutex protects accesses to all the non-const member variables.
  // Exclusive access is required to modify any of these members.
  //
  // To get shared access to the object in a scope use std::shared_lock:
  //    std::shared_lock Lock(Obj->Mutex);
  // To get exclusive access to the object in a scope use std::scoped_lock:
  //    std::scoped_lock Lock(Obj->Mutex);
  //
  // If several pi objects are accessed in a scope then each object's mutex must
  // be locked. For example, to get write access to Obj1 and Obj2 and read
  // access to Obj3 in a scope use the following approach:
  //   std::shared_lock Obj3Lock(Obj3->Mutex, std::defer_lock);
  //   std::scoped_lock LockAll(Obj1->Mutex, Obj2->Mutex, Obj3Lock);
  pi_shared_mutex Mutex;
};

// Record for a memory allocation. This structure is used to keep information
// for each memory allocation.
struct MemAllocRecord : _pi_object {
  MemAllocRecord(pi_context Context, bool OwnZeMemHandle = true)
      : Context(Context), OwnZeMemHandle(OwnZeMemHandle) {}
  // Currently kernel can reference memory allocations from different contexts
  // and we need to know the context of a memory allocation when we release it
  // in piKernelRelease.
  // TODO: this should go away when memory isolation issue is fixed in the Level
  // Zero runtime.
  pi_context Context;

  // Indicates if we own the native memory handle or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeMemHandle;
};

// Define the types that are opaque in pi.h in a manner suitabale for Level Zero
// plugin

struct _pi_platform {
  _pi_platform(ze_driver_handle_t Driver) : ZeDriver{Driver} {}
  // Performs initialization of a newly constructed PI platform.
  pi_result initialize();

  // Level Zero lacks the notion of a platform, but there is a driver, which is
  // a pretty good fit to keep here.
  ze_driver_handle_t ZeDriver;

  // Cache versions info from zeDriverGetProperties.
  std::string ZeDriverVersion;
  std::string ZeDriverApiVersion;
  ze_api_version_t ZeApiVersion;

  // Cache driver extensions
  std::unordered_map<std::string, uint32_t> zeDriverExtensionMap;

  // Cache pi_devices for reuse
  std::vector<std::unique_ptr<_pi_device>> PiDevicesCache;
  std::mutex PiDevicesCacheMutex;
  bool DeviceCachePopulated = false;

  // Check the device cache and load it if necessary.
  pi_result populateDeviceCacheIfNeeded();

  // Return the PI device from cache that represents given native device.
  // If not found, then nullptr is returned.
  pi_device getDeviceFromNativeHandle(ze_device_handle_t);

  // Current number of L0 Command Lists created on this platform.
  // this number must not exceed ZeMaxCommandListCache.
  std::atomic<int> ZeGlobalCommandListCount{0};

  // Keep track of all contexts in the platform. This is needed to manage
  // a lifetime of memory allocations in each context when there are kernels
  // with indirect access.
  // TODO: should be deleted when memory isolation in the context is implemented
  // in the driver.
  std::list<pi_context> Contexts;
  std::mutex ContextsMutex;
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
  virtual MemType getMemTypeImpl() = 0;

public:
  USMMemoryAllocBase(pi_context Ctx, pi_device Dev)
      : Context{Ctx}, Device{Dev} {}
  void *allocate(size_t Size) override final;
  void *allocate(size_t Size, size_t Alignment) override final;
  void deallocate(void *Ptr, bool OwnZeMemHandle) override final;
  MemType getMemType() override final;
};

// Allocation routines for shared memory type
class USMSharedMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;
  MemType getMemTypeImpl() override;

public:
  USMSharedMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for shared memory type that is only modified from host.
class USMSharedReadOnlyMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;
  MemType getMemTypeImpl() override { return SystemMemory::SharedReadOnly; }

public:
  USMSharedReadOnlyMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for device memory type
class USMDeviceMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;
  MemType getMemTypeImpl() override;

public:
  USMDeviceMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

// Allocation routines for host memory type
class USMHostMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;
  MemType getMemTypeImpl() override;

public:
  USMHostMemoryAlloc(pi_context Ctx) : USMMemoryAllocBase(Ctx, nullptr) {}
};

struct _pi_device : _pi_object {
  _pi_device(ze_device_handle_t Device, pi_platform Plt,
             pi_device ParentDevice = nullptr)
      : ZeDevice{Device}, Platform{Plt}, RootDevice{ParentDevice},
        ZeDeviceProperties{}, ZeDeviceComputeProperties{} {
    // NOTE: one must additionally call initialize() to complete
    // PI device creation.
  }

  // The helper structure that keeps info about a command queue groups of the
  // device. It is not changed after it is initialized.
  struct queue_group_info_t {
    typedef enum {
      MainCopy,
      LinkCopy,
      Compute,
      Size // must be last
    } type;

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

  // Initialize the entire PI device.
  // Optional param `SubSubDeviceOrdinal` `SubSubDeviceIndex` are the compute
  // command queue ordinal and index respectively, used to initialize
  // sub-sub-devices.
  pi_result initialize(int SubSubDeviceOrdinal = -1,
                       int SubSubDeviceIndex = -1);

  // Level Zero device handle.
  ze_device_handle_t ZeDevice;

  // Keep the subdevices that are partitioned from this pi_device for reuse
  // The order of sub-devices in this vector is repeated from the
  // ze_device_handle_t array that are returned from zeDeviceGetSubDevices()
  // call, which will always return sub-devices in the fixed same order.
  std::vector<pi_device> SubDevices;

  // PI platform to which this device belongs.
  pi_platform Platform;

  // Root-device of a sub-device, null if this is not a sub-device.
  pi_device RootDevice;
  bool isSubDevice() { return RootDevice != nullptr; }

  // Cache of the immutable device properties.
  ZeCache<ZeStruct<ze_device_properties_t>> ZeDeviceProperties;
  ZeCache<ZeStruct<ze_device_compute_properties_t>> ZeDeviceComputeProperties;
  ZeCache<ZeStruct<ze_device_image_properties_t>> ZeDeviceImageProperties;
  ZeCache<ZeStruct<ze_device_module_properties_t>> ZeDeviceModuleProperties;
  ZeCache<std::vector<ZeStruct<ze_device_memory_properties_t>>>
      ZeDeviceMemoryProperties;
  ZeCache<ZeStruct<ze_device_memory_access_properties_t>>
      ZeDeviceMemoryAccessProperties;
  ZeCache<ZeStruct<ze_device_cache_properties_t>> ZeDeviceCacheProperties;
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
  bool InUse{false};

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
typedef std::unordered_map<ze_command_list_handle_t, pi_command_list_info_t>
    pi_command_list_map_t;
// The iterator pointing to a specific command-list in use.
typedef pi_command_list_map_t::iterator pi_command_list_ptr_t;

struct _pi_context : _pi_object {
  _pi_context(ze_context_handle_t ZeContext, pi_uint32 NumDevices,
              const pi_device *Devs, bool OwnZeContext)
      : ZeContext{ZeContext}, OwnZeContext{OwnZeContext},
        Devices{Devs, Devs + NumDevices}, ZeCommandListInit{nullptr} {
    // NOTE: one must additionally call initialize() to complete
    // PI context creation.

    // Create USM allocator context for each pair (device, context).
    for (uint32_t I = 0; I < NumDevices; I++) {
      pi_device Device = Devs[I];
      SharedMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(Device),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMSharedMemoryAlloc(this, Device))));
      SharedReadOnlyMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(Device),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMSharedReadOnlyMemoryAlloc(this, Device))));
      DeviceMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(Device),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMDeviceMemoryAlloc(this, Device))));
    }
    // Create USM allocator context for host. Device and Shared USM allocations
    // are device-specific. Host allocations are not device-dependent therefore
    // we don't need a map with device as key.
    HostMemAllocContext = std::make_unique<USMAllocContext>(
        std::unique_ptr<SystemMemory>(new USMHostMemoryAlloc(this)));

    if (NumDevices == 1) {
      SingleRootDevice = Devices[0];
    } else {

      // Check if we have context with subdevices of the same device (context
      // may include root device itself as well)
      SingleRootDevice =
          Devices[0]->RootDevice ? Devices[0]->RootDevice : Devices[0];

      // For context with sub subdevices, the SingleRootDevice might still
      // not be the root device.
      // Check whether the SingleRootDevice is the subdevice or root device.
      if (SingleRootDevice->isSubDevice()) {
        SingleRootDevice = SingleRootDevice->RootDevice;
      }

      for (auto &Device : Devices) {
        if ((!Device->RootDevice && Device != SingleRootDevice) ||
            (Device->RootDevice && Device->RootDevice != SingleRootDevice)) {
          SingleRootDevice = nullptr;
          break;
        }
      }
    }

    // We may allocate memory to this root device so create allocators.
    if (SingleRootDevice && DeviceMemAllocContexts.find(SingleRootDevice) ==
                                DeviceMemAllocContexts.end()) {
      SharedMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(SingleRootDevice),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMSharedMemoryAlloc(this, SingleRootDevice))));
      DeviceMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(SingleRootDevice),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMDeviceMemoryAlloc(this, SingleRootDevice))));
    }
  }

  // Initialize the PI context.
  pi_result initialize();

  // Finalize the PI context
  pi_result finalize();

  // Return the Platform, which is the same for all devices in the context
  pi_platform getPlatform() const { return Devices[0]->Platform; }

  // A L0 context handle is primarily used during creation and management of
  // resources that may be used by multiple devices.
  ze_context_handle_t ZeContext;

  // Indicates if we own the ZeContext or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnZeContext;

  // Keep the PI devices this PI context was created for.
  std::vector<pi_device> Devices;

  // If context contains one device or sub-devices of the same device, we want
  // to save this device.
  pi_device SingleRootDevice = nullptr;

  // Immediate Level Zero command list for the device in this context, to be
  // used for initializations. To be created as:
  // - Immediate command list: So any command appended to it is immediately
  //   offloaded to the device.
  // - Synchronous: So implicit synchronization is made inside the level-zero
  //   driver.
  // There will be a list of immediate command lists (for each device) when
  // support of the multiple devices per context will be added.
  ze_command_list_handle_t ZeCommandListInit;

  // Mutex Lock for the Command List Cache. This lock is used to control both
  // compute and copy command list caches.
  std::mutex ZeCommandListCacheMutex;
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
  // When using immediate commandlists, retrieves an immediate command list
  // for executing on this device. Immediate commandlists are created only
  // once for each SYCL Queue and after that they are reused.
  pi_result getAvailableCommandList(pi_queue Queue,
                                    pi_command_list_ptr_t &CommandList,
                                    bool UseCopyEngine,
                                    bool AllowBatching = false);

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
  std::unordered_map<pi_device, USMAllocContext> DeviceMemAllocContexts;
  std::unordered_map<pi_device, USMAllocContext> SharedMemAllocContexts;
  std::unordered_map<pi_device, USMAllocContext> SharedReadOnlyMemAllocContexts;

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

private:
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
  std::mutex ZeEventPoolCacheMutex;
};

struct _pi_queue : _pi_object {
  _pi_queue(std::vector<ze_command_queue_handle_t> &ComputeQueues,
            std::vector<ze_command_queue_handle_t> &CopyQueues,
            pi_context Context, pi_device Device, bool OwnZeCommandQueue,
            pi_queue_properties Properties = 0);

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
    uint32_t getQueueIndex(uint32_t *QueueGroupOrdinal, uint32_t *QueueIndex);

    // This function will return one of possibly multiple available native
    // queues and the value of the queue group ordinal.
    ze_command_queue_handle_t &getZeQueue(uint32_t *QueueGroupOrdinal);

    // This function returns the next immediate commandlist to use.
    pi_command_list_ptr_t &getImmCmdList();

    // These indices are to filter specific range of the queues to use,
    // and to organize round-robin across them.
    uint32_t UpperIndex{0};
    uint32_t LowerIndex{0};
    uint32_t NextIndex{0};
  };

  pi_queue_group_t ComputeQueueGroup{this, queue_type::Compute};

  // Vector of Level Zero copy command command queue handles.
  // In this vector, main copy engine, if available, come first followed by
  // link copy engines, if available.
  pi_queue_group_t CopyQueueGroup{this, queue_type::MainCopy};

  // Wait for all commandlists associated with this Queue to finish operations.
  pi_result synchronize();

  pi_queue_group_t &getQueueGroup(bool UseCopyEngine) {
    return UseCopyEngine ? CopyQueueGroup : ComputeQueueGroup;
  }

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
  typedef struct CommandBatch {
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
  } command_batch;

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

  // Resets the Command List and Associated fence in the ZeCommandListFenceMap.
  // If the reset command list should be made available, then MakeAvailable
  // needs to be set to true. The caller must verify that this command list and
  // fence have been signalled. The EventListToCleanup contains a list of events
  // from the command list which need to be cleaned up.
  pi_result resetCommandList(pi_command_list_ptr_t CommandList,
                             bool MakeAvailable,
                             std::vector<_pi_event *> &EventListToCleanup);

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

  // Besides each PI object keeping a total reference count in
  // _pi_object::RefCount we keep special track of the queue *external*
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
};

struct _pi_mem : _pi_object {
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

    // Make first device in the context be the master. Mark that
    // allocation (yet to be made) having "valid" data. And real
    // allocation and initialization should follow the buffer
    // construction with a "write_only" access copy.
    LastDeviceWithValidAllocation = Context->Devices[0];
    Allocations[LastDeviceWithValidAllocation].Valid = true;
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

// TODO: add proper support for images on context with multiple devices.
struct _pi_image final : _pi_mem {
  // Image constructor
  _pi_image(pi_context Ctx, ze_image_handle_t Image)
      : _pi_mem(Ctx), ZeImage{Image} {}

  virtual pi_result getZeHandle(char *&ZeHandle, access_mode_t,
                                pi_device = nullptr) override {
    ZeHandle = pi_cast<char *>(ZeImage);
    return PI_SUCCESS;
  }
  virtual pi_result getZeHandlePtr(char **&ZeHandlePtr, access_mode_t,
                                   pi_device = nullptr) override {
    ZeHandlePtr = pi_cast<char **>(&ZeImage);
    return PI_SUCCESS;
  }

  bool isImage() const override { return true; }

#ifndef NDEBUG
  // Keep the descriptor of the image (for debugging purposes)
  ZeStruct<ze_image_desc_t> ZeImageDesc;
#endif // !NDEBUG

  // Level Zero image handle.
  ze_image_handle_t ZeImage;
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
  std::mutex PiZeEventListMutex;

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
    this->ZeEventList = other.ZeEventList;
    this->PiEventList = other.PiEventList;
    this->Length = other.Length;
    return *this;
  }
};

struct _pi_event : _pi_object {
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
           (Queue->Properties & PI_QUEUE_PROFILING_ENABLE) != 0;
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
};

struct _pi_program : _pi_object {
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
      : Context{Context},
        OwnZeModule{true}, State{St}, Code{new uint8_t[Length]},
        CodeLength{Length}, ZeModule{nullptr}, ZeBuildLog{nullptr} {
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
  size_t CodeLength;               // Size (bytes) of the array.

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

struct _pi_kernel : _pi_object {
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

struct _pi_sampler : _pi_object {
  _pi_sampler(ze_sampler_handle_t Sampler) : ZeSampler{Sampler} {}

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;
};

#endif // PI_LEVEL_ZERO_HPP
