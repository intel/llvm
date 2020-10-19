//===------- pi_level_zero.hpp - Level Zero Plugin -------------------===//
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
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <level_zero/ze_api.h>

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

// Base class to store common data
struct _pi_object {
  _pi_object() : RefCount{1} {}

  // Level Zero doesn't do the reference counting, so we have to do.
  // Must be atomic to prevent data race when incrementing/decrementing.
  std::atomic<pi_uint32> RefCount;
};

// Define the types that are opaque in pi.h in a manner suitabale for Level Zero
// plugin

struct _pi_platform {
  _pi_platform(ze_driver_handle_t Driver) : ZeDriver{Driver} {}

  // Level Zero lacks the notion of a platform, but there is a driver, which is
  // a pretty good fit to keep here.
  ze_driver_handle_t ZeDriver;

  // Cache versions info from zeDriverGetProperties.
  std::string ZeDriverVersion;
  std::string ZeDriverApiVersion;

  // Cache pi_devices for reuse
  std::vector<std::unique_ptr<_pi_device>> PiDevicesCache;
  std::mutex PiDevicesCacheMutex;
  pi_device getDeviceFromNativeHandle(ze_device_handle_t);
  bool DeviceCachePopulated = false;

  // Maximum Number of Command Lists that can be created.
  // This Value is initialized to 20000, but can be changed by the user
  // thru the environment variable SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE
  // ie SYCL_PI_LEVEL_ZERO_MAX_COMMAND_LIST_CACHE =10000.
  int ZeMaxCommandListCache = 0;

  // Current number of L0 Command Lists created on this platform.
  // this number must not exceed ZeMaxCommandListCache.
  std::atomic<int> ZeGlobalCommandListCount{0};
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

// Allocation routines for device memory type
class USMDeviceMemoryAlloc : public USMMemoryAllocBase {
protected:
  pi_result allocateImpl(void **ResultPtr, size_t Size,
                         pi_uint32 Alignment) override;

public:
  USMDeviceMemoryAlloc(pi_context Ctx, pi_device Dev)
      : USMMemoryAllocBase(Ctx, Dev) {}
};

struct _pi_device : _pi_object {
  _pi_device(ze_device_handle_t Device, pi_platform Plt,
             bool isSubDevice = false)
      : ZeDevice{Device}, Platform{Plt}, IsSubDevice{isSubDevice},
        ZeDeviceProperties{}, ZeDeviceComputeProperties{} {
    // NOTE: one must additionally call initialize() to complete
    // PI device creation.
  }

  // Keep the ordinal of a "compute" commands group, where we send all
  // commands currently.
  // TODO[1.0]: discover "copy" command group as well to use for memory
  // copying operations exclusively.
  //
  uint32_t ZeComputeQueueGroupIndex;

  // Initialize the entire PI device.
  pi_result initialize();

  // Level Zero device handle.
  ze_device_handle_t ZeDevice;

  // PI platform to which this device belongs.
  pi_platform Platform;

  // Mutex Lock for the Command List Cache
  std::mutex ZeCommandListCacheMutex;
  // Cache of all currently Available Command Lists for use by PI APIs
  std::list<ze_command_list_handle_t> ZeCommandListCache;

  // Indicates if this is a root-device or a sub-device.
  // Technically this information can be queried from a device handle, but it
  // seems better to just keep it here.
  bool IsSubDevice;

  // Retrieves a command list for executing on this device along with
  // a fence to be used in tracking the execution of this command list.
  // If a command list has been created on this device which has
  // completed its commands, then that command list and its associated fence
  // will be reused. Otherwise, a new command list and fence will be created for
  // running on this device. L0 fences are created on a L0 command queue so the
  // caller must pass a command queue to create a new fence for the new command
  // list if a command list/fence pair is not available. All Command Lists &
  // associated fences are destroyed at Device Release.
  // If AllowBatching is true, then the command list returned may already have
  // command in it, if AllowBatching is false, any open command lists that
  // already exist in Queue will be closed and executed.
  pi_result getAvailableCommandList(pi_queue Queue,
                                    ze_command_list_handle_t *ZeCommandList,
                                    ze_fence_handle_t *ZeFence,
                                    bool AllowBatching = false);

  // Cache of the immutable device properties.
  ze_device_properties_t ZeDeviceProperties;
  ze_device_compute_properties_t ZeDeviceComputeProperties;
};

struct _pi_context : _pi_object {
  _pi_context(pi_uint32 NumDevices, const pi_device *Devs)
      : Devices{Devs, Devs + NumDevices}, ZeCommandListInit{nullptr},
        ZeEventPool{nullptr}, NumEventsAvailableInEventPool{},
        NumEventsLiveInEventPool{} {
    // Create USM allocator context for each pair (device, context).
    for (uint32_t I = 0; I < NumDevices; I++) {
      pi_device Device = Devs[I];
      SharedMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(Device),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMSharedMemoryAlloc(this, Device))));
      DeviceMemAllocContexts.emplace(
          std::piecewise_construct, std::make_tuple(Device),
          std::make_tuple(std::unique_ptr<SystemMemory>(
              new USMDeviceMemoryAlloc(this, Device))));
    }
  }

  // A L0 context handle is primarily used during creation and management of
  // resources that may be used by multiple devices.
  ze_context_handle_t ZeContext;

  // Keep the PI devices this PI context was created for.
  std::vector<pi_device> Devices;

  // Immediate Level Zero command list for the device in this context, to be
  // used for initializations. To be created as:
  // - Immediate command list: So any command appended to it is immediately
  //   offloaded to the device.
  // - Synchronous: So implicit synchronization is made inside the level-zero
  //   driver.
  // There will be a list of immediate command lists (for each device) when
  // support of the multiple devices per context will be added.
  ze_command_list_handle_t ZeCommandListInit;

  // Get index of the free slot in the available pool. If there is no avialble
  // pool then create new one.
  ze_result_t getFreeSlotInExistingOrNewPool(ze_event_pool_handle_t &,
                                             size_t &);

  // If event is destroyed then decrement number of events living in the pool
  // and destroy the pool if there are no alive events.
  ze_result_t decrementAliveEventsInPool(ze_event_pool_handle_t pool);

  // Store USM allocator context(internal allocator structures)
  // for USM shared/host and device allocations. There is 1 allocator context
  // per each pair of (context, device) per each memory type.
  std::unordered_map<pi_device, USMAllocContext> SharedMemAllocContexts;
  std::unordered_map<pi_device, USMAllocContext> DeviceMemAllocContexts;

private:
  // Following member variables are used to manage assignment of events
  // to event pools.
  // TODO: These variables may be moved to pi_device and pi_platform
  // if appropriate.

  // Event pool to which events are being added to.
  ze_event_pool_handle_t ZeEventPool;
  // This map will be used to determine if a pool is full or not
  // by storing number of empty slots available in the pool.
  std::unordered_map<ze_event_pool_handle_t, pi_uint32>
      NumEventsAvailableInEventPool;
  // This map will be used to determine number of live events in the pool.
  // We use separate maps for number of event slots available in the pool.
  // number of events live in the pool live.
  // This will help when we try to make the code thread-safe.
  std::unordered_map<ze_event_pool_handle_t, pi_uint32>
      NumEventsLiveInEventPool;

  // TODO: we'd like to create a thread safe map class instead of mutex + map,
  // that must be carefully used together.

  // Mutex to control operations on NumEventsAvailableInEventPool map.
  std::mutex NumEventsAvailableInEventPoolMutex;

  // Mutex to control operations on NumEventsLiveInEventPool.
  std::mutex NumEventsLiveInEventPoolMutex;
};

struct _pi_queue : _pi_object {
  _pi_queue(ze_command_queue_handle_t Queue, pi_context Context,
            pi_device Device, pi_uint32 QueueBatchSize)
      : ZeCommandQueue{Queue}, Context{Context}, Device{Device},
        QueueBatchSize{QueueBatchSize} {}

  // Level Zero command queue handle.
  ze_command_queue_handle_t ZeCommandQueue;

  // Keeps the PI context to which this queue belongs.
  // This field is only set at _pi_queue creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_queue.
  const pi_context Context;

  // Keeps the PI device to which this queue belongs.
  // This field is only set at _pi_queue creation time, and cannot change.
  // Therefore it can be accessed without holding a lock on this _pi_queue.
  const pi_device Device;

  // Mutex to be locked on entry to a _pi_queue API call, and unlocked
  // prior to exit.  Access to all state of a queue is done only after
  // this lock has been acquired, and this must be released upon exit
  // from a pi_queue API call.  No other mutexes/locking should be
  // needed/used for the queue data structures.
  std::mutex PiQueueMutex;

  // Open command list field for batching commands into this queue.
  ze_command_list_handle_t ZeOpenCommandList = {nullptr};
  ze_fence_handle_t ZeOpenCommandListFence = {nullptr};
  pi_uint32 ZeOpenCommandListSize = {0};

  // Approximate number of commands that are allowed to be batched for
  // this queue.
  // Added this member to the queue rather than using a global variable
  // so that future implementation could use heuristics to change this on
  // a queue specific basis. And by putting it in the queue itself, this
  // is thread safe because of the locking of the queue that occurs.
  pi_uint32 QueueBatchSize = {0};

  // Map of all Command lists created with their associated Fence used for
  // tracking when the command list is available for use again.
  std::map<ze_command_list_handle_t, ze_fence_handle_t> ZeCommandListFenceMap;

  // Returns true if any commands for this queue are allowed to
  // be batched together.
  bool isBatchingAllowed();

  // Resets the Command List and Associated fence in the ZeCommandListFenceMap.
  // If the reset command list should be made available, then MakeAvailable
  // needs to be set to true. The caller must verify that this command list and
  // fence have been signalled.
  pi_result resetCommandListFenceEntry(ze_command_list_handle_t ZeCommandList,
                                       bool MakeAvailable);

  // Attach a command list to this queue and allow it to remain open
  // and used for further batching.  It may be executed immediately,
  // or it may be left open for other future command to be batched into.
  pi_result batchCommandList(ze_command_list_handle_t ZeCommandList,
                             ze_fence_handle_t ZeFence);

  // Attach a command list to this queue, close, and execute it.
  // Note that this command list cannot be appended to after this.
  // The "IsBlocking" tells if the wait for completion is requested.
  // The "ZeFence" passed is used to track when the command list passed
  // has completed execution on the device and can be reused.
  pi_result executeCommandList(ze_command_list_handle_t ZeCommandList,
                               ze_fence_handle_t ZeFence,
                               bool IsBlocking = false);

  // If there is an open command list associated with this queue,
  // close it, exceute it, and reset ZeOpenCommandList, ZeCommandListFence,
  // and ZeOpenCommandListSize.
  pi_result executeOpenCommandList();
};

struct _pi_mem : _pi_object {
  // Keeps the PI context of this memory handle.
  pi_context Context;

  // Keeps the host pointer where the buffer will be mapped to,
  // if created with PI_MEM_FLAGS_HOST_PTR_USE (see
  // piEnqueueMemBufferMap for details).
  char *MapHostPtr;

  // Supplementary data to keep track of the mappings of this memory
  // created with piEnqueueMemBufferMap and piEnqueueMemImageMap.
  struct Mapping {
    // The offset in the buffer giving the start of the mapped region.
    size_t Offset;
    // The size of the mapped region.
    size_t Size;
  };

  // Interface of the _pi_mem object

  // Get the Level Zero handle of the current memory object
  virtual void *getZeHandle() = 0;

  // Get a pointer to the Level Zero handle of the current memory object
  virtual void *getZeHandlePtr() = 0;

  // Method to get type of the derived object (image or buffer)
  virtual bool isImage() const = 0;

  virtual ~_pi_mem() = default;

  // Thread-safe methods to work with memory mappings
  pi_result addMapping(void *MappedTo, size_t Size, size_t Offset);
  pi_result removeMapping(void *MappedTo, Mapping &MapInfo);

protected:
  _pi_mem(pi_context Ctx, char *HostPtr)
      : Context{Ctx}, MapHostPtr{HostPtr}, Mappings{} {}

private:
  // The key is the host pointer representing an active mapping.
  // The value is the information needed to maintain/undo the mapping.
  std::unordered_map<void *, Mapping> Mappings;

  // TODO: we'd like to create a thread safe map class instead of mutex + map,
  // that must be carefully used together.
  // The mutex that is used for thread-safe work with Mappings.
  std::mutex MappingsMutex;
};

struct _pi_buffer final : _pi_mem {
  // Buffer/Sub-buffer constructor
  _pi_buffer(pi_context Ctx, char *Mem, char *HostPtr,
             _pi_mem *Parent = nullptr, size_t Origin = 0, size_t Size = 0)
      : _pi_mem(Ctx, HostPtr), ZeMem{Mem}, SubBuffer{Parent, Origin, Size} {}

  void *getZeHandle() override { return ZeMem; }

  void *getZeHandlePtr() override { return &ZeMem; }

  bool isImage() const override { return false; }

  bool isSubBuffer() const { return SubBuffer.Parent != nullptr; }

  // Level Zero memory handle is really just a naked pointer.
  // It is just convenient to have it char * to simplify offset arithmetics.
  char *ZeMem;

  struct {
    _pi_mem *Parent;
    size_t Origin; // only valid if Parent != nullptr
    size_t Size;   // only valid if Parent != nullptr
  } SubBuffer;
};

struct _pi_image final : _pi_mem {
  // Image constructor
  _pi_image(pi_context Ctx, ze_image_handle_t Image, char *HostPtr)
      : _pi_mem(Ctx, HostPtr), ZeImage{Image} {}

  void *getZeHandle() override { return ZeImage; }

  void *getZeHandlePtr() override { return &ZeImage; }

  bool isImage() const override { return true; }

#ifndef NDEBUG
  // Keep the descriptor of the image (for debugging purposes)
  ze_image_desc_t ZeImageDesc;
#endif // !NDEBUG

  // Level Zero image handle.
  ze_image_handle_t ZeImage;
};

struct _pi_event : _pi_object {
  _pi_event(ze_event_handle_t ZeEvent, ze_event_pool_handle_t ZeEventPool,
            pi_context Context, pi_command_type CommandType)
      : ZeEvent{ZeEvent}, ZeEventPool{ZeEventPool}, ZeCommandList{nullptr},
        CommandType{CommandType}, Context{Context}, CommandData{nullptr} {}

  // Level Zero event handle.
  ze_event_handle_t ZeEvent;
  // Level Zero event pool handle.
  ze_event_pool_handle_t ZeEventPool;

  // Level Zero command list where the command signaling this event was appended
  // to. This is currently used to remember/destroy the command list after all
  // commands in it are completed, i.e. this event signaled.
  ze_command_list_handle_t ZeCommandList;

  // Keeps the command-queue and command associated with the event.
  // These are NULL for the user events.
  pi_queue Queue;
  pi_command_type CommandType;
  // Provide direct access to Context, instead of going via queue.
  // Not every PI event has a queue, and we need a handle to Context
  // to get to event pool related information.
  pi_context Context;

  // Opaque data to hold any data needed for CommandType.
  void *CommandData;

  // Methods for translating PI events list into Level Zero events list
  static ze_event_handle_t *createZeEventList(pi_uint32, const pi_event *);
  static void deleteZeEventList(ze_event_handle_t *);
};

struct _pi_program : _pi_object {
  // Possible states of a program.
  typedef enum {
    // The program has been created from intermediate language (SPIR-v), but it
    // is not yet compiled.
    IL,

    // The program has been created by loading native code, but it has not yet
    // been built.  This is equivalent to an OpenCL "program executable" that
    // is loaded via clCreateProgramWithBinary().
    Native,

    // The program consists of native code (typically compiled from SPIR-v),
    // but it has unresolved external dependencies which need to be resolved
    // by linking with other Object state program(s).  Programs in this state
    // have a single Level Zero module.
    Object,

    // The program consists of native code with no external dependencies.
    // Programs in this state have a single Level Zero module, but no linking
    // is needed in order to run kernels.
    Exe,

    // The program consists of several Level Zero modules, each of which
    // contains native code.  Some modules may import external symbols and
    // other modules may export definitions of those external symbols.  All of
    // the modules have been linked together, so the imported references are
    // resolved by the exported definitions.
    //
    // Module linking in Level Zero is quite different from program linking in
    // OpenCL.  OpenCL statically links several program objects together to
    // form a new program that contains the linked result.  Level Zero is more
    // similar to shared libraries.  When several Level Zero modules are linked
    // together, each module is modified "in place" such that external
    // references from one are linked to external definitions in another.
    // Linking in Level Zero does not produce a new Level Zero module that
    // represents the linked result, therefore a program in LinkedExe state
    // holds a list of all the pi_programs that were linked together.  Queries
    // about the linked program need to query all the pi_programs in this list.
    LinkedExe
  } state;

  // This is a wrapper class used for programs in LinkedExe state.  Such a
  // program contains a list of pi_programs in Object state that have been
  // linked together.  The program in LinkedExe state increments the reference
  // counter for each of the Object state programs, thus "retaining" a
  // reference to them, and it may also set the "HasImportsAndIsLinked" flag
  // in these Object state programs.  The purpose of this wrapper is to
  // decrement the reference count and clear the flag when the LinkedExe
  // program is destroyed, so all the interesting code is in the wrapper's
  // destructor.
  //
  // In order to ensure that the reference count is never decremented more
  // than once, the wrapper has no copy constructor or copy assignment
  // operator.  Instead, we only allow move semantics for the wrapper.
  class LinkedReleaser {
  public:
    LinkedReleaser(pi_program Prog) : Prog(Prog) {}
    LinkedReleaser(LinkedReleaser &&Other) {
      Prog = Other.Prog;
      Other.Prog = nullptr;
    }
    LinkedReleaser(const LinkedReleaser &Other) = delete;
    LinkedReleaser &operator=(LinkedReleaser &&Other) {
      std::swap(Prog, Other.Prog);
      return *this;
    }
    LinkedReleaser &operator=(const LinkedReleaser &Other) = delete;
    ~LinkedReleaser();

    pi_program operator->() const { return Prog; }

  private:
    pi_program Prog;
  };

  // A utility class that iterates over the Level Zero modules contained by
  // the program.  This helps hide the difference between programs in Object
  // or Exe state (which have one module) and programs in LinkedExe state
  // (which have several modules).
  class ModuleIterator {
  public:
    ModuleIterator(pi_program Prog)
        : Prog(Prog), It(Prog->LinkedPrograms.begin()) {
      if (Prog->State == LinkedExe) {
        NumMods = Prog->LinkedPrograms.size();
        IsDone = (It == Prog->LinkedPrograms.end());
        Mod = IsDone ? nullptr : (*It)->ZeModule;
      } else if (Prog->State == IL || Prog->State == Native) {
        NumMods = 0;
        IsDone = true;
        Mod = nullptr;
      } else {
        NumMods = 1;
        IsDone = false;
        Mod = Prog->ZeModule;
      }
    }

    bool Done() const { return IsDone; }
    size_t Count() const { return NumMods; }
    ze_module_handle_t operator*() const { return Mod; }

    void operator++(int) {
      if (!IsDone && (Prog->State == LinkedExe) &&
          (++It != Prog->LinkedPrograms.end())) {
        Mod = (*It)->ZeModule;
      } else {
        Mod = nullptr;
        IsDone = true;
      }
    }

  private:
    pi_program Prog;
    ze_module_handle_t Mod;
    size_t NumMods;
    bool IsDone;
    std::vector<LinkedReleaser>::iterator It;
  };

  // Construct a program in IL or Native state.
  _pi_program(pi_context Context, const void *Input, size_t Length, state St)
      : State(St), Context(Context), Code(new uint8_t[Length]),
        CodeLength(Length), ZeModule(nullptr), HasImports(false),
        HasImportsAndIsLinked(false), ZeBuildLog(nullptr) {

    std::memcpy(Code.get(), Input, Length);
  }

  // Construct a program in either Object or Exe state.
  _pi_program(pi_context Context, ze_module_handle_t ZeModule, state St,
              bool HasImports = false)
      : State(St), Context(Context), ZeModule(ZeModule), HasImports(HasImports),
        HasImportsAndIsLinked(false), ZeBuildLog(nullptr) {}

  // Construct a program in LinkedExe state.
  _pi_program(pi_context Context, std::vector<LinkedReleaser> &&Inputs,
              ze_module_build_log_handle_t ZeLog)
      : State(LinkedExe), Context(Context), ZeModule(nullptr),
        HasImports(false), HasImportsAndIsLinked(false),
        LinkedPrograms(std::move(Inputs)), ZeBuildLog(ZeLog) {}

  ~_pi_program();

  // Used for programs in all states.
  state State;
  pi_context Context; // Context of the program.

  // Used for programs in IL or Native states.
  std::unique_ptr<uint8_t[]> Code; // Array containing raw IL / native code.
  size_t CodeLength;               // Size (bytes) of the array.

  // Level Zero specialization constants, used for programs in IL state.
  std::unordered_map<uint32_t, uint64_t> ZeSpecConstants;
  std::mutex MutexZeSpecConstants; // Protects access to this field.

  // Used for programs in Object or Exe state.
  ze_module_handle_t ZeModule; // Level Zero module handle.
  bool HasImports;             // Tells if module imports any symbols.

  // Used for programs in Object state.  Tells if this module imports any
  // symbols AND it is linked into some other program that has state LinkedExe.
  // Such an Object is linked into exactly one other LinkedExe program.  Access
  // to this field needs to be locked in case there are two threads that try to
  // simultaneously link with this module.
  bool HasImportsAndIsLinked;
  std::mutex MutexHasImportsAndIsLinked; // Protects access to this field.

  // Used for programs in LinkedExe state.  This is the set of Object programs
  // that are linked together.
  //
  // Note that the Object programs in this vector might also be linked into
  // other LinkedExe programs!
  std::vector<LinkedReleaser> LinkedPrograms;

  // Level Zero build or link log, used for programs in Obj, Exe, or LinkedExe
  // state.
  ze_module_build_log_handle_t ZeBuildLog;
};

struct _pi_kernel : _pi_object {
  _pi_kernel(ze_kernel_handle_t Kernel, pi_program Program,
             const char *KernelName)
      : ZeKernel{Kernel}, Program{Program}, KernelName(KernelName) {}

  // Level Zero function handle.
  ze_kernel_handle_t ZeKernel;

  // Keep the program of the kernel.
  pi_program Program;

  // TODO: remove when bug in the Level Zero runtime will be fixed.
  std::string KernelName;
};

struct _pi_sampler : _pi_object {
  _pi_sampler(ze_sampler_handle_t Sampler) : ZeSampler{Sampler} {}

  // Level Zero sampler handle.
  ze_sampler_handle_t ZeSampler;
};

#endif // PI_LEVEL_ZERO_HPP
