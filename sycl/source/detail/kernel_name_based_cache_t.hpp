//==-------------------- kernel_name_based_cache_t.hpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/kernel_arg_mask.hpp>
#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/ur.hpp>

#include <mutex>

#include <boost/unordered/unordered_flat_map.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
using FastKernelCacheKeyT = std::pair<ur_device_handle_t, ur_context_handle_t>;

struct FastKernelCacheVal {
  ur_kernel_handle_t MKernelHandle;    /* UR kernel handle pointer. */
  std::mutex *MMutex;                  /* Mutex guarding this kernel. */
  const KernelArgMask *MKernelArgMask; /* Eliminated kernel argument mask. */
  ur_program_handle_t MProgramHandle;  /* UR program handle corresponding to
                                     this kernel. */
  const Adapter &MAdapterPtr;          /* We can keep reference to the adapter
                                     because during 2-stage shutdown the kernel
                                     cache is destroyed deliberately before the
                                     adapter. */

  FastKernelCacheVal(ur_kernel_handle_t KernelHandle, std::mutex *Mutex,
                     const KernelArgMask *KernelArgMask,
                     ur_program_handle_t ProgramHandle,
                     const Adapter &AdapterPtr)
      : MKernelHandle(KernelHandle), MMutex(Mutex),
        MKernelArgMask(KernelArgMask), MProgramHandle(ProgramHandle),
        MAdapterPtr(AdapterPtr) {}

  ~FastKernelCacheVal() {
    if (MKernelHandle)
      MAdapterPtr.call<sycl::detail::UrApiKind::urKernelRelease>(
          MKernelHandle);
    if (MProgramHandle)
      MAdapterPtr.call<sycl::detail::UrApiKind::urProgramRelease>(
          MProgramHandle);
    MKernelHandle = nullptr;
    MMutex = nullptr;
    MKernelArgMask = nullptr;
    MProgramHandle = nullptr;
  }

  FastKernelCacheVal(const FastKernelCacheVal &) = delete;
  FastKernelCacheVal &operator=(const FastKernelCacheVal &) = delete;
};
using FastKernelCacheValPtr = std::shared_ptr<FastKernelCacheVal>;

using FastKernelSubcacheMapT =
    ::boost::unordered_flat_map<FastKernelCacheKeyT, FastKernelCacheValPtr>;

using FastKernelSubcacheMutexT = SpinLock;
using FastKernelSubcacheReadLockT = std::lock_guard<FastKernelSubcacheMutexT>;
using FastKernelSubcacheWriteLockT = std::lock_guard<FastKernelSubcacheMutexT>;

struct FastKernelSubcacheT {
  FastKernelSubcacheMapT Map;
  FastKernelSubcacheMutexT Mutex;
};

struct KernelNameBasedCacheT {
  FastKernelSubcacheT FastKernelSubcache;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
