//==--- kernel_program_cache.hpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/locked.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/util.hpp>
#include <detail/platform_impl.hpp>

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
class context_impl;
class KernelProgramCache {
public:
  /// Denotes build error data. The data is filled in from cl::sycl::exception
  /// class instance.
  struct BuildError {
    std::string Msg;
    pi_int32 Code;

    bool isFilledIn() const {
      return !Msg.empty();
    }
  };

  /// Denotes pointer to some entity with its general state and build error.
  /// The pointer is not null if and only if the entity is usable.
  /// State of the entity is provided by the user of cache instance.
  /// Currently there is only a single user - ProgramManager class.
  template <typename T> struct BuildResult {
    std::atomic<T *> Ptr;
    std::atomic<int> State;
    BuildError Error;

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

    BuildResult(T* P, int S) : Ptr{P}, State{S}, Error{"", 0} {}
  };

  using PiProgramT = std::remove_pointer<RT::PiProgram>::type;
  using PiProgramPtrT = std::atomic<PiProgramT *>;
  using ProgramWithBuildStateT = BuildResult<PiProgramT>;
  using ProgramCacheKeyT = std::pair<std::pair<SerializedObj, KernelSetId>,
                                     std::pair<RT::PiDevice, std::string>>;
  using ProgramCacheT = std::map<ProgramCacheKeyT, ProgramWithBuildStateT>;
  using ContextPtr = context_impl *;

  using PiKernelT = std::remove_pointer<RT::PiKernel>::type;

  using PiKernelPtrT = std::atomic<PiKernelT *>;
  using KernelWithBuildStateT = BuildResult<PiKernelT>;
  using KernelByNameT = std::map<string_class, KernelWithBuildStateT>;
  using KernelCacheT = std::map<RT::PiProgram, KernelByNameT>;

  ~KernelProgramCache();

  void setContextPtr(const ContextPtr &AContext) { MParentContext = AContext; }

  Locked<ProgramCacheT> acquireCachedPrograms() {
    return {MCachedPrograms, MProgramCacheMutex};
  }

  Locked<KernelCacheT> acquireKernelsPerProgramCache() {
    return {MKernelsPerProgramCache, MKernelsPerProgramCacheMutex};
  }

  template <typename T, class Predicate>
  void waitUntilBuilt(BuildResult<T> &BR, Predicate Pred) const {
    std::unique_lock<std::mutex> Lock(BR.MBuildResultMutex);

    BR.MBuildCV.wait(Lock, Pred);
  }

  template <typename T> void notifyAllBuild(BuildResult<T> &BR) const {
    BR.MBuildCV.notify_all();
  }

private:
  std::mutex MProgramCacheMutex;
  std::mutex MKernelsPerProgramCacheMutex;

  ProgramCacheT MCachedPrograms;
  KernelCacheT MKernelsPerProgramCache;
  ContextPtr MParentContext;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
