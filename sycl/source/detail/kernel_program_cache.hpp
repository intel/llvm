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
  template<typename T> struct BuildResult {
    std::atomic<T *> Ptr;
    std::atomic<int> State;
    BuildError Error;

    BuildResult(T* P, int S) : Ptr{P}, State{S}, Error{"", 0} {}
  };

  using PiProgramT = std::remove_pointer<RT::PiProgram>::type;
  using PiProgramPtrT = std::atomic<PiProgramT *>;
  using ProgramWithBuildStateT = BuildResult<PiProgramT>;
  using ProgramCacheKeyT = std::pair<SerializedObj, KernelSetId>;
  using ProgramCacheT = std::map<ProgramCacheKeyT, ProgramWithBuildStateT>;
  using ContextPtr = context_impl *;

  using PiKernelT = std::remove_pointer<RT::PiKernel>::type;

  struct BuildResultKernel : public BuildResult<PiKernelT> {
    std::mutex MKernelMutex;

    BuildResultKernel(PiKernelT *P, int S) : BuildResult(P, S) {}
  };

  using PiKernelPtrT = std::atomic<PiKernelT *>;
  using KernelWithBuildStateT = BuildResultKernel;
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

  template <class Predicate> void waitUntilBuilt(Predicate Pred) const {
    std::unique_lock<std::mutex> Lock(MBuildCVMutex);

    MBuildCV.wait(Lock, Pred);
  }

  void notifyAllBuild() const { MBuildCV.notify_all(); }

private:
  std::mutex MProgramCacheMutex;
  std::mutex MKernelsPerProgramCacheMutex;

  mutable std::condition_variable MBuildCV;
  mutable std::mutex MBuildCVMutex;

  ProgramCacheT MCachedPrograms;
  KernelCacheT MKernelsPerProgramCache;
  ContextPtr MParentContext;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
