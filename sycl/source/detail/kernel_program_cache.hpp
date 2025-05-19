//==--- kernel_program_cache.hpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/exception.hpp"
#include <detail/config.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <detail/kernel_name_based_cache_t.hpp>
#include <detail/platform_impl.hpp>
#include <detail/unordered_multimap.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/kernel_name_str_t.hpp>
#include <sycl/detail/locked.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/ur.hpp>
#include <sycl/detail/util.hpp>

#include <atomic>
#include <condition_variable>
#include <iomanip>
#include <list>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include <type_traits>

// For testing purposes
class MockKernelProgramCache;

namespace sycl {
inline namespace _V1 {
namespace detail {
class context_impl;

// During SYCL program execution SYCL runtime will create internal objects
// representing kernels and programs, it may also invoke JIT compiler to bring
// kernels in a program to executable state. Those runtime operations are quite
// expensive. To avoid redundant operations and to speed up the execution, SYCL
// runtime employs in-memory cache for kernels and programs. When a kernel is
// invoked multiple times, the runtime will fetch the kernel from the cache
// instead of creating it from scratch.
// By default, there  is no upper bound on the cache size.
// When the system runs out of memory, the cache will be cleared. Alternatively,
// the cache size can be limited by setting SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD
// to a positive value. When the cache size exceeds the threshold, the least
// recently used programs, and associated kernels, will be evicted from the
// cache.
class KernelProgramCache {
public:
  /// Denotes build error data. The data is filled in from sycl::exception
  /// class instance.
  struct BuildError {
    std::string Msg;
    int32_t Code;

    bool isFilledIn() const { return !Msg.empty(); }
  };

  /// Denotes the state of a build.
  enum class BuildState { BS_Initial, BS_InProgress, BS_Done, BS_Failed };

  /// Denotes pointer to some entity with its general state and build error.
  /// The pointer is not null if and only if the entity is usable.
  /// State of the entity is provided by the user of cache instance.
  /// Currently there is only a single user - ProgramManager class.
  template <typename T> struct BuildResult {
    T Val;
    std::atomic<BuildState> State{BuildState::BS_Initial};
    BuildError Error{"", 0};

    /// Condition variable to signal that build result is ready.
    /// A per-object (i.e. kernel or program) condition variable is employed
    /// instead of global one in order to eliminate the following deadlock.
    /// A thread T1 awaiting for build result BR1 to be ready may be awakened by
    /// another thread (due to use of global condition variable), which made
    /// build result BR2 ready. Meanwhile, a thread which made build result BR1
    /// ready notifies everyone via a global condition variable and T1 will skip
    /// this notification as it's not in condition_variable::wait()'s wait cycle
    /// now. Now T1 goes to sleep again and will wait until either a spurious
    /// wake-up or another thread will wake it up.
    std::condition_variable MBuildCV;
    /// A mutex to be employed along with MBuildCV.
    std::mutex MBuildResultMutex;

    BuildState
    waitUntilTransition(BuildState From = BuildState::BS_InProgress) {
      BuildState To;
      std::unique_lock<std::mutex> Lock(MBuildResultMutex);
      MBuildCV.wait(Lock, [&] {
        To = State;
        return State != From;
      });
      return To;
    }

    void updateAndNotify(BuildState DesiredState) {
      {
        std::lock_guard<std::mutex> Lock(MBuildResultMutex);
        State.store(DesiredState);
      }
      MBuildCV.notify_all();
    }
  };

  struct ProgramBuildResult : public BuildResult<ur_program_handle_t> {
    std::weak_ptr<Adapter> AdapterWeakPtr;
    ProgramBuildResult(const AdapterPtr &Adapter) : AdapterWeakPtr(Adapter) {
      Val = nullptr;
    }
    ProgramBuildResult(const AdapterPtr &Adapter, BuildState InitialState)
        : AdapterWeakPtr(Adapter) {
      Val = nullptr;
      this->State.store(InitialState);
    }
    ~ProgramBuildResult() {
      try {
        if (Val) {
          AdapterPtr AdapterSharedPtr = AdapterWeakPtr.lock();
          if (AdapterSharedPtr) {
            ur_result_t Err =
                AdapterSharedPtr->call_nocheck<UrApiKind::urProgramRelease>(
                    Val);
            __SYCL_CHECK_UR_CODE_NO_EXC(Err);
          }
        }
      } catch (std::exception &e) {
        __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~ProgramBuildResult",
                                          e);
      }
    }
  };
  using ProgramBuildResultPtr = std::shared_ptr<ProgramBuildResult>;

  /* Drop LinkOptions and CompileOptions from CacheKey since they are only used
   * when debugging environment variables are set and we can just ignore them
   * since all kernels will have their build options overridden with the same
   * string*/
  using ProgramCacheKeyT = std::pair<std::pair<SerializedObj, std::uintptr_t>,
                                     std::set<ur_device_handle_t>>;
  using CommonProgramKeyT =
      std::pair<std::uintptr_t, std::set<ur_device_handle_t>>;

  // A custom hashing and equality function for ProgramCacheKeyT.
  // These are used to compare and hash the keys in the cache.
  struct ProgramCacheKeyHash {
    std::size_t operator()(const ProgramCacheKeyT &Key) const {
      std::size_t Hash = 0;
      // Hash the serialized object, representing spec consts.
      for (const auto &Elem : Key.first.first)
        Hash ^= std::hash<unsigned char>{}(Elem);

      // Hash the imageId.
      Hash ^= std::hash<std::uintptr_t>{}(Key.first.second);

      // Hash the devices.
      for (const auto &Elem : Key.second)
        Hash ^= std::hash<void *>{}(static_cast<void *>(Elem));
      return Hash;
    }
  };

  struct ProgramCacheKeyEqual {
    bool operator()(const ProgramCacheKeyT &LHS,
                    const ProgramCacheKeyT &RHS) const {
      // Check equality of SerializedObj (Spec const)
      return std::equal(LHS.first.first.begin(), LHS.first.first.end(),
                        RHS.first.first.begin()) &&
             // Check equality of imageId
             LHS.first.second == RHS.first.second &&
             // Check equality of devices
             std::equal(LHS.second.begin(), LHS.second.end(),
                        RHS.second.begin(), RHS.second.end());
    }
  };

  struct ProgramCache {
    emhash8::HashMap<ProgramCacheKeyT, ProgramBuildResultPtr> Cache;
    UnorderedMultimap<CommonProgramKeyT, ProgramCacheKeyT> KeyMap;
    // Mapping between a UR program and its size.
    std::unordered_map<ur_program_handle_t, size_t> ProgramSizeMap;

    size_t ProgramCacheSizeInBytes = 0;
    inline size_t GetProgramCacheSizeInBytes() const noexcept {
      return ProgramCacheSizeInBytes;
    }

    // Returns number of entries in the cache.
    size_t size() const noexcept { return Cache.size(); }
  };

  using ContextPtr = context_impl *;

  using KernelArgMaskPairT =
      std::pair<ur_kernel_handle_t, const KernelArgMask *>;
  struct KernelBuildResult : public BuildResult<KernelArgMaskPairT> {
    std::weak_ptr<Adapter> AdapterWeakPtr;
    KernelBuildResult(const AdapterPtr &Adapter) : AdapterWeakPtr(Adapter) {
      Val.first = nullptr;
    }
    ~KernelBuildResult() {
      try {
        if (Val.first) {
          AdapterPtr AdapterSharedPtr = AdapterWeakPtr.lock();
          if (AdapterSharedPtr) {
            ur_result_t Err =
                AdapterSharedPtr->call_nocheck<UrApiKind::urKernelRelease>(
                    Val.first);
            __SYCL_CHECK_UR_CODE_NO_EXC(Err);
          }
        }
      } catch (std::exception &e) {
        __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~KernelBuildResult", e);
      }
    }
  };
  using KernelBuildResultPtr = std::shared_ptr<KernelBuildResult>;

  using KernelByNameT = emhash8::HashMap<KernelNameStrT, KernelBuildResultPtr>;
  using KernelCacheT = emhash8::HashMap<ur_program_handle_t, KernelByNameT>;

  class FastKernelSubcacheWrapper {
  public:
    FastKernelSubcacheWrapper(FastKernelSubcacheT *CachePtr,
                              ur_context_handle_t UrContext)
        : MSubcachePtr{CachePtr}, MUrContext{UrContext} {
      if (!MSubcachePtr) {
        MOwnsSubcache = true;
        MSubcachePtr = new FastKernelSubcacheT();
      }
    }
    FastKernelSubcacheWrapper(const FastKernelSubcacheWrapper &) = delete;
    FastKernelSubcacheWrapper(FastKernelSubcacheWrapper &&Other)
        : MSubcachePtr{Other.MSubcachePtr}, MOwnsSubcache{Other.MOwnsSubcache},
          MUrContext{Other.MUrContext} {
      Other.MSubcachePtr = nullptr;
    }
    FastKernelSubcacheWrapper &
    operator=(const FastKernelSubcacheWrapper &) = delete;
    FastKernelSubcacheWrapper &operator=(FastKernelSubcacheWrapper &&Other) {
      MSubcachePtr = Other.MSubcachePtr;
      MOwnsSubcache = Other.MOwnsSubcache;
      MUrContext = Other.MUrContext;
      Other.MSubcachePtr = nullptr;
      return *this;
    };

    ~FastKernelSubcacheWrapper() {
      if (!MSubcachePtr)
        return;

      if (MOwnsSubcache) {
        delete MSubcachePtr;
        return;
      }

      // Single subcache might be used by different contexts.
      FastKernelSubcacheMapT &CacheMap = MSubcachePtr->Map;
      FastKernelSubcacheWriteLockT Lock{MSubcachePtr->Mutex};
      for (auto it = CacheMap.begin(); it != CacheMap.end();) {
        if (it->first.second == MUrContext) {
          it = CacheMap.erase(it);
        } else {
          ++it;
        }
      }
    }

    FastKernelSubcacheT &get() { return *MSubcachePtr; }

  private:
    FastKernelSubcacheT *MSubcachePtr = nullptr;
    bool MOwnsSubcache = false;
    ur_context_handle_t MUrContext = nullptr;
  };

  using FastKernelCacheT =
      emhash8::HashMap<KernelNameStrT, FastKernelSubcacheWrapper>;

  // DS to hold data and functions related to Program cache eviction.
  struct EvictionList {
  private:
    // Linked list of cache entries to be evicted in case of cache overflow.
    std::list<ProgramCacheKeyT> MProgramEvictionList;

    // Mapping between program handle and the iterator to the eviction list.
    std::unordered_map<ProgramCacheKeyT, std::list<ProgramCacheKeyT>::iterator,
                       ProgramCacheKeyHash, ProgramCacheKeyEqual>
        MProgramToEvictionListMap;

  public:
    std::list<ProgramCacheKeyT> &getProgramEvictionList() {
      return MProgramEvictionList;
    }

    void clear() {
      MProgramEvictionList.clear();
      MProgramToEvictionListMap.clear();
    }

    void emplaceBack(const ProgramCacheKeyT &CacheKey) {
      MProgramEvictionList.emplace_back(CacheKey);

      // In std::list, the iterators are not invalidated when elements are
      // added/removed/moved to the list. So, we can safely store the iterators.
      MProgramToEvictionListMap[CacheKey] =
          std::prev(MProgramEvictionList.end());
      traceProgram("Program added to the end of eviction list.", CacheKey);
    }

    // This function is called on the hot path, whenever a kernel/program
    // is accessed. So, it should be very fast.
    void moveToEnd(const ProgramCacheKeyT &CacheKey) {
      auto It = MProgramToEvictionListMap.find(CacheKey);
      if (It != MProgramToEvictionListMap.end()) {
        MProgramEvictionList.splice(MProgramEvictionList.end(),
                                    MProgramEvictionList, It->second);
        traceProgram("Program moved to the end of eviction list.", CacheKey);
      }
      // else: This can happen if concurrently the program is removed from
      // eviction list by another thread.
    }

    bool empty() { return MProgramEvictionList.empty(); }

    size_t size() { return MProgramEvictionList.size(); }

    void popFront() {
      if (!MProgramEvictionList.empty()) {
        MProgramToEvictionListMap.erase(MProgramEvictionList.front());
        MProgramEvictionList.pop_front();
      }
    }

    void erase(const ProgramCacheKeyT &CacheKey) {
      MProgramEvictionList.remove(CacheKey);
      MProgramToEvictionListMap.erase(CacheKey);
    }
  };

  ~KernelProgramCache() = default;

  void setContextPtr(const ContextPtr &AContext) { MParentContext = AContext; }

  // Sends message to std:cerr stream when SYCL_CACHE_TRACE environemnt is
  // set.
  static inline void traceProgram(const std::string &Msg,
                                  const ProgramCacheKeyT &CacheKey) {
    if (!SYCLConfig<SYCL_CACHE_TRACE>::isTraceInMemCache())
      return;

    int ImageId = CacheKey.first.second;
    std::stringstream DeviceList;
    std::vector<unsigned char> SerializedObjVec = CacheKey.first.first;

    // Convert spec constants to string. Spec constants are stored as
    // ASCII values, so we need need to convert them to int and then to
    // string.
    std::string SerializedObjString;
    SerializedObjString.reserve(SerializedObjVec.size() * sizeof(size_t));
    for (unsigned char c : SerializedObjVec)
      SerializedObjString += std::to_string((int)c) + ",";

    for (const auto &Device : CacheKey.second)
      DeviceList << "0x" << std::setbase(16)
                 << reinterpret_cast<uintptr_t>(Device) << ",";

    std::string Identifier = "[Key:{imageId = " + std::to_string(ImageId) +
                             ",urDevice = " + DeviceList.str() +
                             ", serializedObj = " + SerializedObjString +
                             "}]: ";

    std::cerr << "[In-Memory Cache][Thread Id:" << std::this_thread::get_id()
              << "][Program Cache]" << Identifier << Msg << std::endl;
  }

  // Sends message to std:cerr stream when SYCL_CACHE_TRACE environemnt is
  // set.
  static inline void traceKernel(std::string_view Msg,
                                 std::string_view KernelName,
                                 bool IsFastKernelCache = false) {
    if (!SYCLConfig<SYCL_CACHE_TRACE>::isTraceInMemCache())
      return;

    std::string Identifier =
        "[IsFastCache: " + std::to_string(IsFastKernelCache) +
        "][Key:{Name = " + KernelName.data() + "}]: ";

    std::cerr << "[In-Memory Cache][Thread Id:" << std::this_thread::get_id()
              << "][Kernel Cache]" << Identifier << Msg << std::endl;
  }

  Locked<ProgramCache> acquireCachedPrograms() {
    return {MCachedPrograms, MProgramCacheMutex};
  }

  Locked<KernelCacheT> acquireKernelsPerProgramCache() {
    return {MKernelsPerProgramCache, MKernelsPerProgramCacheMutex};
  }

  Locked<EvictionList> acquireEvictionList() {
    return {MEvictionList, MProgramEvictionListMutex};
  }

  std::pair<ProgramBuildResultPtr, bool>
  getOrInsertProgram(const ProgramCacheKeyT &CacheKey) {
    auto LockedCache = acquireCachedPrograms();
    auto &ProgCache = LockedCache.get();
    auto [It, DidInsert] = ProgCache.Cache.try_emplace(CacheKey, nullptr);
    if (DidInsert) {
      It->second = std::make_shared<ProgramBuildResult>(getAdapter());
      // Save reference between the common key and the full key.
      CommonProgramKeyT CommonKey =
          std::make_pair(CacheKey.first.second, CacheKey.second);
      ProgCache.KeyMap.emplace(CommonKey, CacheKey);
      traceProgram("Program inserted.", CacheKey);
    } else
      traceProgram("Program fetched.", CacheKey);
    return std::make_pair(It->second, DidInsert);
  }

  // Used in situation where you have several cache keys corresponding to the
  // same program. An example would be a multi-device build, or use of virtual
  // functions in kernels.
  //
  // Returns whether or not an insertion took place.
  bool insertBuiltProgram(const ProgramCacheKeyT &CacheKey,
                          ur_program_handle_t Program) {
    auto LockedCache = acquireCachedPrograms();
    auto &ProgCache = LockedCache.get();
    auto [It, DidInsert] = ProgCache.Cache.try_emplace(CacheKey, nullptr);
    if (DidInsert) {
      It->second = std::make_shared<ProgramBuildResult>(getAdapter(),
                                                        BuildState::BS_Done);
      It->second->Val = Program;
      // Save reference between the common key and the full key.
      CommonProgramKeyT CommonKey =
          std::make_pair(CacheKey.first.second, CacheKey.second);
      ProgCache.KeyMap.emplace(CommonKey, CacheKey);
      traceProgram("Program inserted.", CacheKey);
    }
    return DidInsert;
  }

  std::pair<KernelBuildResultPtr, bool>
  getOrInsertKernel(ur_program_handle_t Program, KernelNameStrRefT KernelName) {
    auto LockedCache = acquireKernelsPerProgramCache();
    auto &Cache = LockedCache.get()[Program];
    auto [It, DidInsert] = Cache.try_emplace(KernelName, nullptr);
    if (DidInsert) {
      It->second = std::make_shared<KernelBuildResult>(getAdapter());
      traceKernel("Kernel inserted.", KernelName);
    } else
      traceKernel("Kernel fetched.", KernelName);
    return std::make_pair(It->second, DidInsert);
  }

  FastKernelCacheValT
  tryToGetKernelFast(KernelNameStrRefT KernelName, ur_device_handle_t Device,
                     FastKernelSubcacheT *KernelSubcacheHint) {
    FastKernelCacheWriteLockT Lock(MFastKernelCacheMutex);
    if (!KernelSubcacheHint) {
      auto It = MFastKernelCache.try_emplace(
          KernelName,
          FastKernelSubcacheWrapper(KernelSubcacheHint, getURContext()));
      KernelSubcacheHint = &It.first->second.get();
    }

    const FastKernelSubcacheMapT &SubcacheMap = KernelSubcacheHint->Map;
    FastKernelSubcacheReadLockT SubcacheLock{KernelSubcacheHint->Mutex};
    ur_context_handle_t Context = getURContext();
    auto It = SubcacheMap.find(FastKernelCacheKeyT(Device, Context));
    if (It != SubcacheMap.end()) {
      traceKernel("Kernel fetched.", KernelName, true);
      return It->second;
    }
    return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
  }

  void saveKernel(KernelNameStrRefT KernelName, ur_device_handle_t Device,
                  FastKernelCacheValT CacheVal,
                  FastKernelSubcacheT *KernelSubcacheHint) {
    ur_program_handle_t Program = std::get<3>(CacheVal);
    if (SYCLConfig<SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD>::
            isProgramCacheEvictionEnabled()) {
      // Save kernel in fast cache only if the corresponding program is also
      // in the cache.
      auto LockedCache = acquireCachedPrograms();
      auto &ProgCache = LockedCache.get();
      if (ProgCache.ProgramSizeMap.find(Program) ==
          ProgCache.ProgramSizeMap.end())
        return;
    }

    // Save reference between the program and the fast cache key.
    FastKernelCacheWriteLockT Lock(MFastKernelCacheMutex);
    MProgramToFastKernelCacheKeyMap[Program].emplace_back(KernelName, Device);

    // if no insertion took place, then some other thread has already inserted
    // smth in the cache
    traceKernel("Kernel inserted.", KernelName, true);
    auto It = MFastKernelCache.try_emplace(
        KernelName,
        FastKernelSubcacheWrapper(KernelSubcacheHint, getURContext()));
    KernelSubcacheHint = &It.first->second.get();

    FastKernelSubcacheWriteLockT SubcacheLock{KernelSubcacheHint->Mutex};
    ur_context_handle_t Context = getURContext();
    KernelSubcacheHint->Map.emplace(FastKernelCacheKeyT(Device, Context),
                                    std::move(CacheVal));
  }

  // Expects locked program cache
  size_t removeProgramByKey(const ProgramCacheKeyT &CacheKey,
                            ProgramCache &ProgCache) {
    auto It = ProgCache.Cache.find(CacheKey);

    if (It != ProgCache.Cache.end()) {
      // We are about to remove this program now.
      // (1) Remove it from KernelPerProgram cache.
      // (2) Remove corresponding entries from FastKernelCache.
      // (3) Remove it from ProgramCache KeyMap.
      // (4) Remove it from the ProgramCache.
      // (5) Remove it from ProgramSizeMap.
      // (6) Update the cache size.

      // Remove entry from the KernelsPerProgram cache.
      ur_program_handle_t NativePrg = It->second->Val;
      {
        auto LockedCacheKP = acquireKernelsPerProgramCache();
        // List kernels that are to be removed from the cache, if tracing is
        // enabled.
        if (SYCLConfig<SYCL_CACHE_TRACE>::isTraceInMemCache()) {
          for (const auto &Kernel : LockedCacheKP.get()[NativePrg])
            traceKernel("Kernel evicted.", Kernel.first);
        }
        LockedCacheKP.get().erase(NativePrg);
      }

      {
        // Remove corresponding entries from FastKernelCache.
        FastKernelCacheWriteLockT Lock(MFastKernelCacheMutex);
        if (auto FastCacheKeysItr =
                MProgramToFastKernelCacheKeyMap.find(NativePrg);
            FastCacheKeysItr != MProgramToFastKernelCacheKeyMap.end()) {
          ur_context_handle_t Context = getURContext();
          for (const auto &FastCacheKey : FastCacheKeysItr->second) {
            if (auto FastKernelCacheItr =
                    MFastKernelCache.find(FastCacheKey.first);
                FastKernelCacheItr != MFastKernelCache.end()) {
              FastKernelSubcacheT &Subcache = FastKernelCacheItr->second.get();
              bool RemoveSubcache = false;
              {
                FastKernelSubcacheWriteLockT SubcacheLock{Subcache.Mutex};
                Subcache.Map.erase(
                    FastKernelCacheKeyT(FastCacheKey.second, Context));
                traceKernel("Kernel evicted.", FastCacheKey.first, true);

                // Remove the subcache wrapper from this kernel program cache if
                // the subcache no longer contains entries for this context.
                RemoveSubcache = std::none_of(
                    Subcache.Map.begin(), Subcache.Map.end(),
                    [&](const auto &It) { return It.first.second == Context; });
              }
              if (RemoveSubcache)
                MFastKernelCache.erase(FastKernelCacheItr);
            }
          }
          MProgramToFastKernelCacheKeyMap.erase(FastCacheKeysItr);
        }
      }

      // Remove entry from ProgramCache KeyMap.
      CommonProgramKeyT CommonKey =
          std::make_pair(CacheKey.first.second, CacheKey.second);
      // Since KeyMap is a multi-map, we need to iterate over all entries
      // with this CommonKey and remove those that match the CacheKey.
      auto KeyMapItrRange = ProgCache.KeyMap.equal_range(CommonKey);
      for (auto KeyMapItr = KeyMapItrRange.first;
           KeyMapItr != KeyMapItrRange.second; ++KeyMapItr) {
        if ((*KeyMapItr).second == CacheKey) {
          ProgCache.KeyMap.erase(KeyMapItr);
          break;
        }
      }

      // Get size of the program.
      size_t ProgramSize = MCachedPrograms.ProgramSizeMap[It->second->Val];
      // Evict program from the cache.
      ProgCache.Cache.erase(It);
      // Remove program size from the cache size.
      MCachedPrograms.ProgramCacheSizeInBytes -= ProgramSize;
      MCachedPrograms.ProgramSizeMap.erase(NativePrg);

      traceProgram("Program evicted.", CacheKey);
    } else
      // This should never happen.
      throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                            "Program not found in the cache.");

    return MCachedPrograms.ProgramCacheSizeInBytes;
  }

  // Evict programs from cache to free up space.
  void evictPrograms(size_t DesiredCacheSize, size_t CurrentCacheSize) {

    // Figure out how many programs from the beginning we need to evict.
    if (CurrentCacheSize < DesiredCacheSize || MCachedPrograms.Cache.empty())
      return;

    // Evict programs from the beginning of the cache.
    {
      std::lock_guard<std::mutex> Lock(MProgramEvictionListMutex);
      auto &ProgramEvictionList = MEvictionList.getProgramEvictionList();
      size_t CurrCacheSize = MCachedPrograms.ProgramCacheSizeInBytes;

      // Traverse the eviction list and remove the LRU programs.
      // The LRU programs will be at the front of the list.
      while (CurrCacheSize > DesiredCacheSize && !MEvictionList.empty()) {
        ProgramCacheKeyT CacheKey = ProgramEvictionList.front();
        auto LockedCache = acquireCachedPrograms();
        auto &ProgCache = LockedCache.get();
        CurrCacheSize = removeProgramByKey(CacheKey, ProgCache);
        // Remove the program from the eviction list.
        MEvictionList.popFront();
      }
    }
  }

  // Register that a program has been fetched from the cache.
  // If it is the first time the program is fetched, add it to the eviction
  // list.
  void registerProgramFetch(const ProgramCacheKeyT &CacheKey,
                            const ur_program_handle_t &Program,
                            const bool IsBuilt) {

    size_t ProgramCacheEvictionThreshold =
        SYCLConfig<SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD>::getProgramCacheSize();

    // No need to populate the eviction list if eviction is disabled.
    if (ProgramCacheEvictionThreshold == 0)
      return;

    // If the program is not in the cache, add it to the cache.
    if (IsBuilt) {
      // This is the first time we are adding this entry. Add it to the end of
      // eviction list.
      {
        std::lock_guard<std::mutex> Lock(MProgramEvictionListMutex);
        MEvictionList.emplaceBack(CacheKey);
      }

      // Store size of the program and check if we need to evict some entries.
      // Get Size of the program.
      size_t ProgramSize = 0;
      auto Adapter = getAdapter();

      try {
        // Get number of devices this program was built for.
        unsigned int DeviceNum = 0;
        Adapter->call<UrApiKind::urProgramGetInfo>(
            Program, UR_PROGRAM_INFO_NUM_DEVICES, sizeof(DeviceNum), &DeviceNum,
            nullptr);

        // Get binary sizes for each device.
        std::vector<size_t> BinarySizes(DeviceNum);
        Adapter->call<UrApiKind::urProgramGetInfo>(
            Program, UR_PROGRAM_INFO_BINARY_SIZES,
            sizeof(size_t) * BinarySizes.size(), BinarySizes.data(), nullptr);

        // Sum up binary sizes.
        ProgramSize =
            std::accumulate(BinarySizes.begin(), BinarySizes.end(), 0);
      } catch (const exception &Ex) {
        std::cerr << "Failed to get program size: " << Ex.what() << std::endl;
        std::rethrow_exception(std::current_exception());
      }
      // Store program size in the cache.
      size_t CurrCacheSize = 0;
      {
        std::lock_guard<std::mutex> Lock(MProgramCacheMutex);
        MCachedPrograms.ProgramSizeMap[Program] = ProgramSize;
        MCachedPrograms.ProgramCacheSizeInBytes += ProgramSize;
        CurrCacheSize = MCachedPrograms.ProgramCacheSizeInBytes;
      }

      // Evict programs if the cache size exceeds the threshold.
      if (CurrCacheSize > ProgramCacheEvictionThreshold)
        evictPrograms(ProgramCacheEvictionThreshold, CurrCacheSize);
    }
    // If the program is already in the cache, move it to the end of the list.
    // Since we are following LRU eviction policy, we need to move the program
    // to the end of the list. Items in the front of the list are the least
    // recently This code path is "hot" and should be very fast.
    else {
      std::lock_guard<std::mutex> Lock(MProgramEvictionListMutex);
      MEvictionList.moveToEnd(CacheKey);
    }
  }

  /// Clears cache state.
  ///
  /// This member function should only be used in unit tests.
  void reset() {
    std::lock_guard<std::mutex> EvictionListLock(MProgramEvictionListMutex);
    std::lock_guard<std::mutex> L1(MProgramCacheMutex);
    std::lock_guard<std::mutex> L2(MKernelsPerProgramCacheMutex);
    FastKernelCacheWriteLockT L3(MFastKernelCacheMutex);
    MCachedPrograms = ProgramCache{};
    MKernelsPerProgramCache = KernelCacheT{};
    MFastKernelCache = FastKernelCacheT{};
    MProgramToFastKernelCacheKeyMap.clear();
    // Clear the eviction lists and its mutexes.
    MEvictionList.clear();
  }

  /// Try to fetch entity (kernel or program) from cache. If there is no such
  /// entity try to build it. Throw any exception build process may throw.
  /// This method eliminates unwanted builds by employing atomic variable with
  /// build state and waiting until the entity is built in another thread.
  /// If the building thread has failed the awaiting thread will fail either.
  /// Exception thrown by build procedure are rethrown.
  ///
  /// \tparam RetT type of entity to get
  /// \tparam Errc error code of exception to throw on awaiting thread if the
  ///         building thread fails build step.
  /// \tparam KeyT key (in cache) to fetch built entity with
  /// \tparam AcquireFT type of function which will acquire the locked version
  /// of
  ///         the cache. Accept reference to KernelProgramCache.
  /// \tparam GetCacheFT type of function which will fetch proper cache from
  ///         locked version. Accepts reference to locked version of cache.
  /// \tparam BuildFT type of function which will build the entity if it is not
  /// in
  ///         cache. Accepts nothing. Return pointer to built entity.
  ///
  /// \return a pointer to cached build result, return value must not be
  /// nullptr.
  template <errc Errc, typename GetCachedBuildFT, typename BuildFT,
            typename EvictFT = void *>
  auto getOrBuild(GetCachedBuildFT &&GetCachedBuild, BuildFT &&Build,
                  EvictFT &&EvictFunc = nullptr) {
    using BuildState = KernelProgramCache::BuildState;
    constexpr size_t MaxAttempts = 2;
    for (size_t AttemptCounter = 0;; ++AttemptCounter) {
      auto Res = GetCachedBuild();
      auto &BuildResult = Res.first;
      BuildState Expected = BuildState::BS_Initial;
      BuildState Desired = BuildState::BS_InProgress;
      if (!BuildResult->State.compare_exchange_strong(Expected, Desired)) {
        // no insertion took place, thus some other thread has already inserted
        // smth in the cache
        BuildState NewState = BuildResult->waitUntilTransition();

        // Build succeeded.
        if (NewState == BuildState::BS_Done) {
          if constexpr (!std::is_same_v<EvictFT, void *>)
            EvictFunc(BuildResult->Val, /*IsBuilt=*/false);
          return BuildResult;
        }

        // Build failed, or this is the last attempt.
        if (NewState == BuildState::BS_Failed ||
            AttemptCounter + 1 == MaxAttempts) {
          if (BuildResult->Error.isFilledIn())
            throw detail::set_ur_error(
                exception(make_error_code(Errc), BuildResult->Error.Msg),
                BuildResult->Error.Code);
          else
            throw exception();
        }

        // NewState == BuildState::BS_Initial
        // Build state was set back to the initial state,
        // which means to go back to the beginning of the
        // loop and try again.
        continue;
      }

      // only the building thread will run this
      try {
        BuildResult->Val = Build();

        if constexpr (!std::is_same_v<EvictFT, void *>)
          EvictFunc(BuildResult->Val, /*IsBuilt=*/true);

        BuildResult->updateAndNotify(BuildState::BS_Done);
        return BuildResult;
      } catch (const exception &Ex) {
        BuildResult->Error.Msg = Ex.what();
        BuildResult->Error.Code = detail::get_ur_error(Ex);
        if (Ex.code() == errc::memory_allocation ||
            BuildResult->Error.Code == UR_RESULT_ERROR_OUT_OF_RESOURCES ||
            BuildResult->Error.Code == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY ||
            BuildResult->Error.Code == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY) {
          reset();
          BuildResult->updateAndNotify(BuildState::BS_Initial);
          continue;
        }

        BuildResult->updateAndNotify(BuildState::BS_Failed);
        std::rethrow_exception(std::current_exception());
      } catch (...) {
        BuildResult->updateAndNotify(BuildState::BS_Initial);
        std::rethrow_exception(std::current_exception());
      }
    }
  }

  void removeAllRelatedEntries(uint32_t ImageId) {
    auto LockedCache = acquireCachedPrograms();
    auto &ProgCache = LockedCache.get();

    auto It = std::find_if(
        ProgCache.KeyMap.begin(), ProgCache.KeyMap.end(),
        [&ImageId](const auto &Entry) { return ImageId == Entry.first.first; });
    if (It == ProgCache.KeyMap.end())
      return;

    auto Key = (*It).second;
    removeProgramByKey(Key, ProgCache);
    {
      auto LockedEvictionList = acquireEvictionList();
      LockedEvictionList.get().erase(Key);
    }
  }

private:
  std::mutex MProgramCacheMutex;
  std::mutex MKernelsPerProgramCacheMutex;

  ProgramCache MCachedPrograms;
  KernelCacheT MKernelsPerProgramCache;
  ContextPtr MParentContext;

  using FastKernelCacheMutexT = SpinLock;
  using FastKernelCacheReadLockT = std::lock_guard<FastKernelCacheMutexT>;
  using FastKernelCacheWriteLockT = std::lock_guard<FastKernelCacheMutexT>;
  FastKernelCacheMutexT MFastKernelCacheMutex;
  FastKernelCacheT MFastKernelCache;

  // Map between fast kernel cache keys and program handle.
  // MFastKernelCacheMutex will be used for synchronization.
  std::unordered_map<ur_program_handle_t,
                     std::vector<std::pair<KernelNameStrT, ur_device_handle_t>>>
      MProgramToFastKernelCacheKeyMap;

  EvictionList MEvictionList;
  // Mutexes that will be used when accessing the eviction lists.
  std::mutex MProgramEvictionListMutex;

  friend class ::MockKernelProgramCache;

  const AdapterPtr &getAdapter();
  ur_context_handle_t getURContext() const;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
