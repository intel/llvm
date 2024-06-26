//==--- kernel_program_cache.hpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/kernel_arg_mask.hpp>
#include <detail/platform_impl.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/locked.hpp>
#include <sycl/detail/os_util.hpp>
#include <sycl/detail/pi.hpp>
#include <sycl/detail/util.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <type_traits>

#include <boost/unordered/unordered_flat_map.hpp>
#include <boost/unordered_map.hpp>

// For testing purposes
class MockKernelProgramCache;

namespace sycl {
inline namespace _V1 {
namespace detail {
class context_impl;
class KernelProgramCache {
public:
  /// Denotes build error data. The data is filled in from sycl::exception
  /// class instance.
  struct BuildError {
    std::string Msg;
    pi_int32 Code;

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

  struct ProgramBuildResult : public BuildResult<sycl::detail::pi::PiProgram> {
    PluginPtr Plugin;
    ProgramBuildResult(const PluginPtr &Plugin) : Plugin(Plugin) {
      Val = nullptr;
    }
    ~ProgramBuildResult() {
      if (Val) {
        sycl::detail::pi::PiResult Err =
            Plugin->call_nocheck<PiApiKind::piProgramRelease>(Val);
        __SYCL_CHECK_OCL_CODE_NO_EXC(Err);
      }
    }
  };
  using ProgramBuildResultPtr = std::shared_ptr<ProgramBuildResult>;

  /* Drop LinkOptions and CompileOptions from CacheKey since they are only used
   * when debugging environment variables are set and we can just ignore them
   * since all kernels will have their build options overridden with the same
   * string*/
  using ProgramCacheKeyT = std::pair<std::pair<SerializedObj, std::uintptr_t>,
                                     sycl::detail::pi::PiDevice>;
  using CommonProgramKeyT =
      std::pair<std::uintptr_t, sycl::detail::pi::PiDevice>;

  struct ProgramCache {
    ::boost::unordered_map<ProgramCacheKeyT, ProgramBuildResultPtr> Cache;
    ::boost::unordered_multimap<CommonProgramKeyT, ProgramCacheKeyT> KeyMap;

    size_t size() const noexcept { return Cache.size(); }
  };

  using ContextPtr = context_impl *;

  using KernelArgMaskPairT =
      std::pair<sycl::detail::pi::PiKernel, const KernelArgMask *>;
  struct KernelBuildResult : public BuildResult<KernelArgMaskPairT> {
    PluginPtr Plugin;
    KernelBuildResult(const PluginPtr &Plugin) : Plugin(Plugin) {
      Val.first = nullptr;
    }
    ~KernelBuildResult() {
      if (Val.first) {
        sycl::detail::pi::PiResult Err =
            Plugin->call_nocheck<PiApiKind::piKernelRelease>(Val.first);
        __SYCL_CHECK_OCL_CODE_NO_EXC(Err);
      }
    }
  };
  using KernelBuildResultPtr = std::shared_ptr<KernelBuildResult>;

  using KernelByNameT =
      ::boost::unordered_map<std::string, KernelBuildResultPtr>;
  using KernelCacheT =
      ::boost::unordered_map<sycl::detail::pi::PiProgram, KernelByNameT>;

  using KernelFastCacheKeyT =
      std::tuple<SerializedObj, sycl::detail::pi::PiDevice, std::string,
                 std::string>;
  using KernelFastCacheValT =
      std::tuple<sycl::detail::pi::PiKernel, std::mutex *,
                 const KernelArgMask *, sycl::detail::pi::PiProgram>;
  // This container is used as a fast path for retrieving cached kernels.
  // unordered_flat_map is used here to reduce lookup overhead.
  // The slow path is used only once for each newly created kernel, so the
  // higher overhead of insertion that comes with unordered_flat_map is more
  // of an issue there. For that reason, those use regular unordered maps.
  using KernelFastCacheT =
      ::boost::unordered_flat_map<KernelFastCacheKeyT, KernelFastCacheValT>;

  ~KernelProgramCache() = default;

  void setContextPtr(const ContextPtr &AContext) { MParentContext = AContext; }

  Locked<ProgramCache> acquireCachedPrograms() {
    return {MCachedPrograms, MProgramCacheMutex};
  }

  Locked<KernelCacheT> acquireKernelsPerProgramCache() {
    return {MKernelsPerProgramCache, MKernelsPerProgramCacheMutex};
  }

  std::pair<ProgramBuildResultPtr, bool>
  getOrInsertProgram(const ProgramCacheKeyT &CacheKey) {
    auto LockedCache = acquireCachedPrograms();
    auto &ProgCache = LockedCache.get();
    auto [It, DidInsert] = ProgCache.Cache.try_emplace(CacheKey, nullptr);
    if (DidInsert) {
      It->second = std::make_shared<ProgramBuildResult>(getPlugin());
      // Save reference between the common key and the full key.
      CommonProgramKeyT CommonKey =
          std::make_pair(CacheKey.first.second, CacheKey.second);
      ProgCache.KeyMap.emplace(CommonKey, CacheKey);
    }
    return std::make_pair(It->second, DidInsert);
  }

  std::pair<KernelBuildResultPtr, bool>
  getOrInsertKernel(sycl::detail::pi::PiProgram Program,
                    const std::string &KernelName) {
    auto LockedCache = acquireKernelsPerProgramCache();
    auto &Cache = LockedCache.get()[Program];
    auto [It, DidInsert] = Cache.try_emplace(KernelName, nullptr);
    if (DidInsert)
      It->second = std::make_shared<KernelBuildResult>(getPlugin());
    return std::make_pair(It->second, DidInsert);
  }

  template <typename KeyT>
  KernelFastCacheValT tryToGetKernelFast(KeyT &&CacheKey) {
    std::unique_lock<std::mutex> Lock(MKernelFastCacheMutex);
    auto It = MKernelFastCache.find(CacheKey);
    if (It != MKernelFastCache.end()) {
      return It->second;
    }
    return std::make_tuple(nullptr, nullptr, nullptr, nullptr);
  }

  template <typename KeyT, typename ValT>
  void saveKernel(KeyT &&CacheKey, ValT &&CacheVal) {
    std::unique_lock<std::mutex> Lock(MKernelFastCacheMutex);
    // if no insertion took place, thus some other thread has already inserted
    // smth in the cache
    MKernelFastCache.emplace(CacheKey, CacheVal);
  }

  /// Clears cache state.
  ///
  /// This member function should only be used in unit tests.
  void reset() {
    std::lock_guard<std::mutex> L1(MProgramCacheMutex);
    std::lock_guard<std::mutex> L2(MKernelsPerProgramCacheMutex);
    std::lock_guard<std::mutex> L3(MKernelFastCacheMutex);
    MCachedPrograms = ProgramCache{};
    MKernelsPerProgramCache = KernelCacheT{};
    MKernelFastCache = KernelFastCacheT{};
  }

  /// Try to fetch entity (kernel or program) from cache. If there is no such
  /// entity try to build it. Throw any exception build process may throw.
  /// This method eliminates unwanted builds by employing atomic variable with
  /// build state and waiting until the entity is built in another thread.
  /// If the building thread has failed the awaiting thread will fail either.
  /// Exception thrown by build procedure are rethrown.
  ///
  /// \tparam RetT type of entity to get
  /// \tparam ExceptionT type of exception to throw on awaiting thread if the
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
  template <typename ExceptionT, typename GetCachedBuildFT, typename BuildFT>
  auto getOrBuild(GetCachedBuildFT &&GetCachedBuild, BuildFT &&Build) {
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
        if (NewState == BuildState::BS_Done)
          return BuildResult;

        // Build failed, or this is the last attempt.
        if (NewState == BuildState::BS_Failed ||
            AttemptCounter + 1 == MaxAttempts) {
          if (BuildResult->Error.isFilledIn())
            throw ExceptionT(BuildResult->Error.Msg, BuildResult->Error.Code);
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

        BuildResult->updateAndNotify(BuildState::BS_Done);
        return BuildResult;
      } catch (const exception &Ex) {
        BuildResult->Error.Msg = Ex.what();
        BuildResult->Error.Code = Ex.get_cl_code();
        if (BuildResult->Error.Code == PI_ERROR_OUT_OF_RESOURCES ||
            BuildResult->Error.Code == PI_ERROR_OUT_OF_HOST_MEMORY) {
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

private:
  std::mutex MProgramCacheMutex;
  std::mutex MKernelsPerProgramCacheMutex;

  ProgramCache MCachedPrograms;
  KernelCacheT MKernelsPerProgramCache;
  ContextPtr MParentContext;

  std::mutex MKernelFastCacheMutex;
  KernelFastCacheT MKernelFastCache;
  friend class ::MockKernelProgramCache;

  const PluginPtr &getPlugin();
};
} // namespace detail
} // namespace _V1
} // namespace sycl
