//==--- kernel_program_cache.hpp - Cache for kernel and program -*- C++-*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/locked.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <type_traits>

__SYCL_INLINE namespace cl {
namespace sycl {
namespace detail {
class KernelProgramCache {
public:
  /// Denotes pointer to some entity with its state.
  /// The pointer is not null if and only if the entity is usable.
  /// State of the entity is provided by the user of cache instance.
  /// Currently there is only a single user - ProgramManager class.
  template<typename T>
  struct EntityWithState {
    std::atomic<T *> Ptr;
    std::atomic<int> State;

    EntityWithState(T* P, int S)
      : Ptr{P}, State{S}
    {}
  };

  using PiProgramT = std::remove_pointer<RT::PiProgram>::type;
  using PiProgramPtrT = std::atomic<PiProgramT *>;
  using ProgramWithBuildStateT = EntityWithState<PiProgramT>;
  using ProgramCacheT = std::map<OSModuleHandle, ProgramWithBuildStateT>;

  using PiKernelT = std::remove_pointer<RT::PiKernel>::type;
  using PiKernelPtrT = std::atomic<PiKernelT *>;
  using KernelWithBuildStateT = EntityWithState<PiKernelT>;
  using KernelByNameT = std::map<string_class, KernelWithBuildStateT>;
  using KernelCacheT = std::map<RT::PiProgram, KernelByNameT>;

  ~KernelProgramCache();

  Locked<ProgramCacheT> acquireCachedPrograms() {
    return {MCachedPrograms, MProgramCacheMutex};
  }

  Locked<KernelCacheT> acquireKernelsPerProgramCache() {
    return {MKernelsPerProgramCache, MKernelsPerProgramCacheMutex};
  }

  template<class Predicate>
  void waitUntilBuilt(Predicate Pred) const {
    std::unique_lock<std::mutex> Lock(MBuildCVMutex);

    MBuildCV.wait(Lock, Pred);
  }

  void notifyAllBuild() const {
    MBuildCV.notify_all();
  }

private:
  std::mutex MProgramCacheMutex;
  std::mutex MKernelsPerProgramCacheMutex;

  mutable std::condition_variable MBuildCV;
  mutable std::mutex MBuildCVMutex;

  ProgramCacheT MCachedPrograms;
  KernelCacheT MKernelsPerProgramCache;
};
}
}
}
