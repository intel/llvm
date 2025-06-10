//==-------------------- kernel_name_based_cache_t.hpp ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <detail/hashers.hpp>
#include <detail/kernel_arg_mask.hpp>
#include <emhash/hash_table8.hpp>
#include <sycl/detail/spinlock.hpp>
#include <sycl/detail/ur.hpp>

#include <mutex>
#include <optional>

namespace sycl {
inline namespace _V1 {
namespace detail {
using FastKernelCacheKeyT = std::pair<ur_device_handle_t, ur_context_handle_t>;

struct FastKernelCacheVal {
  ur_kernel_handle_t MKernelHandle;    /* UR kernel handle pointer. */
  std::mutex *MMutex;                  /* Mutex guarding this kernel. When
                                     caching is disabled, the pointer is
                                     nullptr. */
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
      MAdapterPtr.call<sycl::detail::UrApiKind::urKernelRelease>(MKernelHandle);
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

using FastKernelSubcacheMutexT = SpinLock;
using FastKernelSubcacheReadLockT = std::lock_guard<FastKernelSubcacheMutexT>;
using FastKernelSubcacheWriteLockT = std::lock_guard<FastKernelSubcacheMutexT>;

struct FastKernelEntryT {
  FastKernelCacheKeyT Key;
  FastKernelCacheValPtr Value;

  FastKernelEntryT(FastKernelCacheKeyT Key, const FastKernelCacheValPtr &Value)
      : Key(Key), Value(Value) {}

  FastKernelEntryT(const FastKernelEntryT &) = default;
  FastKernelEntryT &operator=(const FastKernelEntryT &) = default;
  FastKernelEntryT(FastKernelEntryT &&) = default;
  FastKernelEntryT &operator=(FastKernelEntryT &&) = default;
};

using FastKernelSubcacheEntriesT = std::vector<FastKernelEntryT>;

struct FastKernelSubcacheT {
  FastKernelSubcacheEntriesT Entries;
  FastKernelSubcacheMutexT Mutex;
};

struct KernelNameBasedCacheT {
  FastKernelSubcacheT FastKernelSubcache;
  std::optional<bool> UsesAssert;
  // Implicit local argument position is represented by an optional int, this
  // uses another optional on top of that to represent lazy initialization of
  // the cached value.
  std::optional<std::optional<int>> ImplicitLocalArgPos;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
